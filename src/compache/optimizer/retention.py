"""Render causal checkpoints and evaluate historical evidence retained before recovery."""

from collections import defaultdict
from dataclasses import asdict, dataclass, replace

from compache.traces import ContractError

from .contracts import ModelSpec, number
from .edit_accounting import PartMeter
from .fit import label_index
from .gate import (
    Request,
    Source,
    placeholder,
    protected_sources,
    wire_part,
)
from .gate import (
    expand as reopen_part,
)
from .replay import assess

__all__ = [
    "RecipeView",
    "RetentionRecipe",
    "fit_recipe",
    "fit_recipes",
    "matched_prefix_tokens",
    "recipe_grid",
    "render_recipe",
    "reopen_part",
]


@dataclass(frozen=True)
class RetentionRecipe:
    """Counts of completed runs to retain per type; None keeps all of that type."""

    input_runs: int | None = None
    output_runs: int | None = None
    retrieval_runs: int | None = None
    protect_query_matches: bool = False

    def __post_init__(self):
        for name in ("input_runs", "output_runs", "retrieval_runs"):
            value = getattr(self, name)
            if value is not None:
                number(value, name, integer=True)
        if type(self.protect_query_matches) is not bool:
            raise ContractError("protect_query_matches must be boolean")


def recipe_grid(longest_thread_runs=None, *, protect_query_matches=(True, False)):
    """Candidate cuts: every pairing of a dialogue window with a retrieval window.

    A dialogue window keeps the inputs and outputs of that many completed runs.
    A retrieval window keeps that many completed retrieval groups, and zero keeps
    only the current one. Windows follow 1, 2, 5, 10, 20, 50 and so on, and stop
    below the longest training thread because a longer window never cuts there.
    The grid contains the narrowest combination of any two of its members, so
    checking a combined cut never needs a recipe outside it.
    """
    if longest_thread_runs is None:
        longest_thread_runs = 100
    number(longest_thread_runs, "longest_thread_runs", 1, integer=True)
    flags = tuple(protect_query_matches)
    if (
        not flags
        or len(set(flags)) != len(flags)
        or any(type(f) is not bool for f in flags)
    ):
        raise ContractError("protect_query_matches must list distinct booleans")
    windows, scale = [], 1
    while scale < longest_thread_runs:
        windows.extend(n * scale for n in (1, 2, 5) if n * scale < longest_thread_runs)
        scale *= 10
    dialogue = (None, *reversed(windows))
    retrieval = (None, *reversed(windows), 0)
    return tuple(
        RetentionRecipe(d, d, r, protect)
        for protect in flags
        for d in dialogue
        for r in retrieval
        if (d, r) != (None, None)
    )


@dataclass(frozen=True)
class RecipeView:
    thread_id: str
    run_id: str
    model: str
    parts: tuple[Source, ...]
    full_part_ids: frozenset[str]
    archived_ids: frozenset[str]
    wire: tuple[str, ...]
    part_tokens: tuple[int, ...]
    overhead_tokens: int
    tokens: int
    wire_bytes: int
    part_wire_bytes: tuple[int, ...] = ()


def _retrieval_turns(request):
    return sorted(
        {
            p.run_index
            for p in request.parts
            if p.kind == "retrieval" and p.run_index < request.run_index
        }
    )


def _checkpoint(request, recipe, archived, pins, *, matches=None, retrieval_turns=None):
    """Apply one cut to the mutable `archived` and `pins` sets and return `archived`.

    Rendering and fitting share this rule. A fit that scores many recipes on one
    request passes that request's query matches and retrieval turns once.
    """
    archived.difference_update(pins)
    if recipe.protect_query_matches:
        found = protected_sources(request) if matches is None else matches
        pins.update(found - archived)
    # Empty retrieval turns do not displace a retained group. Every exposed
    # chunk in one completed retrieval turn shares the same retention window.
    if retrieval_turns is None:
        retrieval_turns = _retrieval_turns(request)
    thresholds = {
        "input": request.run_index - recipe.input_runs
        if recipe.input_runs is not None
        else 0,
        "output": request.run_index - recipe.output_runs
        if recipe.output_runs is not None
        else 0,
        "retrieval": 0,
    }
    if recipe.retrieval_runs == 0:
        thresholds["retrieval"] = request.run_index
    elif (
        recipe.retrieval_runs is not None
        and len(retrieval_turns) >= recipe.retrieval_runs
    ):
        thresholds["retrieval"] = retrieval_turns[-recipe.retrieval_runs]
    archived.update(
        p.part_id
        for p in request.parts
        if p.part_id not in pins and p.run_index < thresholds[p.kind]
    )
    return archived


def render_recipe(
    request: Request,
    spec: ModelSpec,
    recipe: RetentionRecipe,
    *,
    archived_ids=frozenset(),
    protected_ids=frozenset(),
    cut=True,
    overhead_tokens=0,
    part_meter: PartMeter | None = None,
) -> RecipeView:
    """Render a preview. Commit its archived_ids only when the host chooses it.

    Cutting replaces removable old parts with version-bound, thread-scoped handles.
    Extending a checkpoint applies no new cuts. Explicit host pins may restore a
    prior source on a cut. A query match protects a still-visible source on a cut.
    Rendered parts stay in canonical order and the current sources remain verbatim.
    """
    if request.model != spec.model_id:
        raise ContractError("recipe token ruler belongs to a different model")
    number(overhead_tokens, "overhead_tokens", integer=True)
    if type(cut) is not bool:
        raise ContractError("cut must be boolean")
    ids = {p.part_id for p in request.parts}
    history = {p.part_id for p in request.parts if p.run_index < request.run_index}
    archived = set(archived_ids)
    pins = set(protected_ids)
    if not archived <= history or not pins <= ids:
        raise ContractError("checkpoint or protection references an unavailable source")
    if any(p.text is None for p in request.parts):
        raise ContractError("recipe projection requires complete visible source text")
    if pins & archived and not cut:
        raise ContractError("restoring an archived pin requires a new checkpoint")
    if cut:
        _checkpoint(request, recipe, archived, pins)
    if part_meter is not None:
        if part_meter.spec != spec:
            raise ContractError(
                "part meter and recipe have different model specifications"
            )
        blocks = [
            part_meter.measure(
                p, archived_in=request.thread_id if p.part_id in archived else None
            )
            for p in request.parts
        ]
        projected = tuple(b.source for b in blocks)
        counts = tuple(b.tokens for b in blocks)
        wire = tuple(b.wire for b in blocks)
        sizes = tuple(b.wire_bytes for b in blocks)
    else:
        projected = tuple(
            replace(p, text=placeholder(request.thread_id, p))
            if p.part_id in archived
            else p
            for p in request.parts
        )
        counts = tuple(spec.part_tokens(p.text) for p in projected)
        wire = tuple(wire_part(p) for p in projected)
        sizes = tuple(len(line.encode("utf-8")) for line in wire)
    return RecipeView(
        request.thread_id,
        request.run_id,
        request.model,
        tuple(projected),
        frozenset(ids - archived),
        frozenset(archived),
        wire,
        counts,
        overhead_tokens,
        overhead_tokens + sum(counts),
        sum(sizes),
        sizes,
    )


def matched_prefix_tokens(
    previous: RecipeView | None, current: RecipeView, *, stable_overhead_tokens=0
) -> int:
    """Whole-message prefix token estimate before provider minimum/block rounding.

    Call only for a matching provider prefix configuration. If framing changes,
    the caller must stop matching at its first differing token. Passing zero
    overhead means to exclude unmeasured framing from the estimate.
    """
    number(stable_overhead_tokens, "stable_overhead_tokens", integer=True)
    if previous is None or (previous.thread_id, previous.model) != (
        current.thread_id,
        current.model,
    ):
        return 0
    if stable_overhead_tokens > min(previous.overhead_tokens, current.overhead_tokens):
        raise ContractError("stable overhead exceeds the configured request framing")
    common = stable_overhead_tokens
    # Source equality includes document/version/chunk metadata that a host may
    # send alongside the wire body. Changing a citation can invalidate a prefix
    # even when the body text happens to be identical.
    for old, new, count in zip(previous.parts, current.parts, current.part_tokens):
        if old != new:
            break
        common += count
    return common


def _measure_recipes(
    threads,
    labels,
    spec,
    recipes,
    *,
    retention_floor,
    max_unknown_change_rate,
    unknowns="excluded",
):
    """Evaluate each irreversible checkpoint rule on training inputs and frozen outcomes.

    Each recipe cuts at every training turn and carries its own archive across
    turns. No future labels enter the cut. An unaltered full history is the
    fallback. The floor is an empirical coverage criterion.

    `unknowns` says how evidence nobody could assess enters the floor.
    With "excluded" the floor applies to assessed requirements only. With
    "worst_case" every unknown requirement on a labelled turn whose context the
    recipe changed counts as lost, and the floor applies to that lower bound.
    Both figures are always reported. A turn without a label is unsampled: it is
    counted in the report and enters neither side of the ratio.
    `max_unknown_change_rate` is an optional stricter cap on the share of
    unassessable turns, unlabelled ones included, that a recipe may change. Zero
    is the original rule, under which a single such turn rejects the recipe.
    """
    number(retention_floor, "retention_floor", 0, 1)
    if unknowns not in ("worst_case", "excluded"):
        raise ContractError("unknowns must be 'worst_case' or 'excluded'")
    if max_unknown_change_rate is not None:
        number(max_unknown_change_rate, "max_unknown_change_rate", 0, 1)
    recipes = list(recipes)
    if (
        not recipes
        or any(not isinstance(r, RetentionRecipe) for r in recipes)
        or len(set(recipes)) != len(recipes)
    ):
        raise ContractError("supply distinct retention recipes")
    threads = list(threads)
    if not threads:
        raise ContractError("fit requires training threads")
    by_thread = defaultdict(list)
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get("thread_id"), str):
            raise ContractError("invalid training label")
        by_thread[label["thread_id"]].append(label)
    thread_ids = [t.thread_id for t in threads]
    run_ids = [r.run_id for t in threads for r in t.runs]
    if (
        len(set(thread_ids)) != len(threads)
        or len(set(run_ids)) != len(run_ids)
        or set(by_thread) - set(thread_ids)
    ):
        raise ContractError(
            "fitting requires unique training identities and in-split labels"
        )
    reports = [
        {
            "recipe": asdict(recipe),
            "turns": 0,
            "historical_required": 0,
            "historical_retained": 0,
            "changed_historical_required": 0,
            "changed_historical_retained": 0,
            "direct_required": 0,
            "direct_retained": 0,
            "unknown_requirements": 0,
            "unknown_changed_requirements": 0,
            "unknown_runs": 0,
            "unknown_changed_runs": 0,
            "unlabeled_runs": 0,
            "unlabeled_changed_runs": 0,
            "changed_runs": 0,
            "canonical_tokens": 0,
            "projected_tokens": 0,
            "retention_floor": retention_floor,
            "max_unknown_change_rate": max_unknown_change_rate,
            "unknowns": unknowns,
        }
        for recipe in recipes
    ]
    protecting = any(r.protect_query_matches for r in recipes)
    for thread in threads:
        # Bound packet materialization to one thread and validate quote spans once.
        indexed = (
            label_index([thread], by_thread[thread.thread_id])
            if by_thread[thread.thread_id]
            else {}
        )
        canonical = Request.from_thread(thread, thread.runs[-1].run_id).parts
        # A recipe's projected size is the full request minus what its archive saves.
        # Count every part once per thread instead of once per turn and recipe.
        saving = {}
        archived = [set() for _ in recipes]
        saved = [0] * len(recipes)
        ids, counted, full_tokens = set(), 0, 0
        for run in thread.runs:
            count = len(run.generation.request_part_ids)
            request = Request(
                thread.thread_id,
                run.run_id,
                run.run_index,
                run.generation.model,
                canonical[:count],
            )
            if request.model != spec.model_id:
                raise ContractError("recipe token ruler belongs to a different model")
            if count < counted:
                raise ContractError(
                    "fitting requires each request to extend the previous one"
                )
            for part in canonical[counted:count]:
                if part.text is None:
                    raise ContractError(
                        "recipe projection requires complete visible source text"
                    )
                tokens = spec.part_tokens(part.text)
                full_tokens += tokens
                saving[part.part_id] = tokens - spec.part_tokens(
                    placeholder(thread.thread_id, part)
                )
                ids.add(part.part_id)
            counted = count
            label = indexed.get(run.run_id)
            reqs = label["task_requirements"] if label else []
            unknown_count = sum(
                r["historical_context_needed"] == "unknown" or not r["alternatives"]
                for r in reqs
            )
            known_direct = [
                r
                for r in reqs
                if r["historical_context_needed"] == "no" and r["alternatives"]
            ]
            matches = protected_sources(request) if protecting else None
            turns = _retrieval_turns(request)
            for i, (recipe, report) in enumerate(zip(recipes, reports)):
                before = len(archived[i])
                held = _checkpoint(
                    request,
                    recipe,
                    archived[i],
                    set(),
                    matches=matches,
                    retrieval_turns=turns,
                )
                if len(held) != before:
                    # Nothing is pinned during a fit, so an archive only grows.
                    saved[i] = sum(saving[part_id] for part_id in held)
                changed = bool(held)
                grade = assess(label, ids - held if changed else ids)
                missed = set(grade["missed_requirements"])
                required = grade["historical_requirements_assessed"]
                retained = grade["historical_requirements_retained"]
                report["turns"] += 1
                report["historical_required"] += required
                report["historical_retained"] += retained
                report["direct_required"] += len(known_direct)
                report["direct_retained"] += sum(
                    r["requirement_id"] not in missed for r in known_direct
                )
                report["unknown_requirements"] += unknown_count
                report["unknown_runs"] += int(not grade["assessable"])
                report["unlabeled_runs"] += int(label is None)
                if changed:
                    report["changed_runs"] += 1
                    report["changed_historical_required"] += required
                    report["changed_historical_retained"] += retained
                    if label is None:
                        report["unlabeled_changed_runs"] += 1
                    if not grade["assessable"]:
                        report["unknown_changed_runs"] += 1
                        if label is not None:
                            # A partial or unassessable label can hide a requirement
                            # that it does not list.
                            report["unknown_changed_requirements"] += max(
                                1, unknown_count
                            )
                report["canonical_tokens"] += full_tokens
                report["projected_tokens"] += full_tokens - saved[i]
    for recipe, report in zip(recipes, reports):
        required = report["historical_required"]
        retained = report["historical_retained"]
        coverage = retained / required if required else None
        at_risk = required + report["unknown_changed_requirements"]
        lower_bound = retained / at_risk if required else None
        unknown_rate = (
            report["unknown_changed_runs"] / report["unknown_runs"]
            if report["unknown_runs"]
            else 0.0
        )
        if all(
            v is None
            for v in (recipe.input_runs, recipe.output_runs, recipe.retrieval_runs)
        ):
            admission = "keeps_everything"
        elif coverage is None:
            admission = "no_assessed_requirements"
        elif coverage < retention_floor:
            admission = "below_retention_floor"
        elif unknowns == "worst_case" and lower_bound < retention_floor:
            admission = "unknown_requirements_could_break_floor"
        elif (
            max_unknown_change_rate is not None
            and unknown_rate > max_unknown_change_rate
        ):
            admission = "above_unknown_change_cap"
        else:
            admission = "passes"
        report.update(
            coverage=coverage,
            coverage_lower_bound=lower_bound,
            unknown_change_rate=unknown_rate,
            eligible=admission in ("keeps_everything", "passes"),
            admission=admission,
            compact_fraction=report["projected_tokens"] / report["canonical_tokens"]
            if report["canonical_tokens"]
            else None,
        )
    return reports


def _narrower(left, right):
    if left is None:
        return right
    return left if right is None else min(left, right)


_ENVELOPE_FIELDS = (
    "recipe",
    "coverage",
    "coverage_lower_bound",
    "historical_required",
    "historical_retained",
    "unknown_change_rate",
)


def fit_recipes(
    threads,
    labels,
    spec,
    recipes,
    *,
    retention_floor=0.95,
    max_unknown_change_rate=None,
    unknowns="excluded",
    select="supplied",
):
    """Approve an action set whose worst combined retention passed the training floor.

    A runtime that alternates independently adequate irreversible cuts may lose
    evidence from both. The shortest typed windows with only shared protections,
    applied at every turn, bound the surviving evidence for any mixture.

    `select="supplied"` admits candidates in the supplied order while that bound
    holds. `select="cheapest"` admits at most one recipe: the passing candidate
    with the fewest projected training tokens, among those that save tokens and
    whose cut met at least one assessed requirement. It sits at the floor by
    construction, so check its retention on held-out threads.
    """
    if select not in ("supplied", "cheapest"):
        raise ContractError("select must be 'supplied' or 'cheapest'")
    threads, labels, recipes = list(threads), list(labels), list(recipes)
    limits = {
        "retention_floor": retention_floor,
        "max_unknown_change_rate": max_unknown_change_rate,
        "unknowns": unknowns,
    }
    reports = _measure_recipes(threads, labels, spec, recipes, **limits)
    for report in reports:
        report["individual_eligible"] = report["eligible"]
        report["joint_envelope"] = None
    if select == "cheapest":
        passing = [
            (report["projected_tokens"], i)
            for i, report in enumerate(reports)
            if report["admission"] == "passes"
        ]

        def shortfall(report):
            if not report["changed_historical_required"]:
                return "cut_met_no_assessed_requirement"
            if report["projected_tokens"] >= report["canonical_tokens"]:
                # A placeholder can outweigh the short message it replaces.
                return "saves_no_tokens"
            return None

        usable = [pair for pair in passing if shortfall(reports[pair[1]]) is None]
        chosen = min(usable)[1] if usable else None
        for _, i in passing:
            report = reports[i]
            if i == chosen:
                report["joint_envelope"] = {k: report[k] for k in _ENVELOPE_FIELDS}
                report["admission"] = "admitted"
            else:
                report["eligible"] = False
                report["admission"] = shortfall(report) or "cheaper_recipe_selected"
        return reports
    metrics = dict(zip(recipes, reports))
    accepted = None
    for recipe, report in zip(recipes, reports):
        if not report["eligible"]:
            continue
        if accepted is None:
            envelope = recipe
        else:
            # A keep-all recipe drops nothing under any query, including when
            # its own protect_query_matches switch is disabled.
            protects = all(
                r.protect_query_matches
                or all(
                    n is None for n in (r.input_runs, r.output_runs, r.retrieval_runs)
                )
                for r in (accepted, recipe)
            )
            envelope = RetentionRecipe(
                _narrower(accepted.input_runs, recipe.input_runs),
                _narrower(accepted.output_runs, recipe.output_runs),
                _narrower(accepted.retrieval_runs, recipe.retrieval_runs),
                protects,
            )
        if envelope not in metrics:
            metrics[envelope] = _measure_recipes(
                threads, labels, spec, [envelope], **limits
            )[0]
        outcome = metrics[envelope]
        fits = outcome.get("individual_eligible", outcome["eligible"])
        report["joint_envelope"] = {key: outcome[key] for key in _ENVELOPE_FIELDS}
        report["eligible"] = fits
        report["admission"] = "admitted" if fits else "combined_cut_below_floor"
        if fits:
            accepted = envelope
    return reports


def fit_recipe(threads, labels, spec, recipe, **kwargs):
    return fit_recipes(threads, labels, spec, [recipe], **kwargs)[0]
