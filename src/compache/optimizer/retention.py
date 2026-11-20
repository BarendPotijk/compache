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
        archived.difference_update(pins)
        if recipe.protect_query_matches:
            pins.update(protected_sources(request) - archived)
        # Empty retrieval turns do not displace a retained group. Every exposed
        # chunk in one completed retrieval turn shares the same retention window.
        retrieval_turns = sorted(
            {
                p.run_index
                for p in request.parts
                if p.kind == "retrieval" and p.run_index < request.run_index
            }
        )
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
    threads, labels, spec, recipes, *, retention_floor, max_unknown_change_rate
):
    """Evaluate each irreversible checkpoint rule on training inputs and frozen outcomes.

    Each recipe cuts at every training turn and carries its own archive across
    turns. No future labels are passed into render_recipe. An unaltered full
    history is the fallback. The floor is an empirical coverage criterion.
    """
    number(retention_floor, "retention_floor", 0, 1)
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
            "direct_required": 0,
            "direct_retained": 0,
            "unknown_requirements": 0,
            "unknown_runs": 0,
            "unknown_changed_runs": 0,
            "changed_runs": 0,
            "canonical_tokens": 0,
            "projected_tokens": 0,
            "retention_floor": retention_floor,
            "max_unknown_change_rate": max_unknown_change_rate,
        }
        for recipe in recipes
    ]
    for thread in threads:
        # Bound packet materialization to one thread and validate quote spans once.
        indexed = (
            label_index([thread], by_thread[thread.thread_id])
            if by_thread[thread.thread_id]
            else {}
        )
        archived = [frozenset() for _ in recipes]
        canonical = Request.from_thread(thread, thread.runs[-1].run_id).parts
        for run in thread.runs:
            count = len(run.generation.request_part_ids)
            request = Request(
                thread.thread_id,
                run.run_id,
                run.run_index,
                run.generation.model,
                canonical[:count],
            )
            full_tokens = sum(spec.part_tokens(p.text) for p in request.parts)
            label = indexed.get(run.run_id)
            reqs = label["task_requirements"] if label else []
            unknown_count = sum(
                r["historical_context_needed"] == "unknown" or not r["alternatives"]
                for r in reqs
            )
            for i, (recipe, report) in enumerate(zip(recipes, reports)):
                view = render_recipe(request, spec, recipe, archived_ids=archived[i])
                archived[i] = view.archived_ids
                grade = assess(label, view.full_part_ids)
                known_direct = [
                    r
                    for r in reqs
                    if r["historical_context_needed"] == "no" and r["alternatives"]
                ]
                missed = set(grade["missed_requirements"])
                report["turns"] += 1
                report["historical_required"] += grade[
                    "historical_requirements_assessed"
                ]
                report["historical_retained"] += grade[
                    "historical_requirements_retained"
                ]
                report["direct_required"] += len(known_direct)
                report["direct_retained"] += sum(
                    r["requirement_id"] not in missed for r in known_direct
                )
                report["unknown_requirements"] += unknown_count
                report["unknown_runs"] += int(not grade["assessable"])
                report["unknown_changed_runs"] += int(
                    not grade["assessable"] and bool(view.archived_ids)
                )
                report["changed_runs"] += int(bool(view.archived_ids))
                report["canonical_tokens"] += full_tokens
                report["projected_tokens"] += view.tokens
    for recipe, report in zip(recipes, reports):
        required = report["historical_required"]
        coverage = report["historical_retained"] / required if required else None
        unknown_rate = (
            report["unknown_changed_runs"] / report["unknown_runs"]
            if report["unknown_runs"]
            else 0.0
        )
        full = all(
            v is None
            for v in (recipe.input_runs, recipe.output_runs, recipe.retrieval_runs)
        )
        report.update(
            coverage=coverage,
            unknown_change_rate=unknown_rate,
            eligible=full
            or (
                coverage is not None
                and coverage >= retention_floor
                and unknown_rate <= max_unknown_change_rate
            ),
            compact_fraction=report["projected_tokens"] / report["canonical_tokens"]
            if report["canonical_tokens"]
            else None,
        )
    return reports


def _narrower(left, right):
    if left is None:
        return right
    return left if right is None else min(left, right)


def fit_recipes(
    threads, labels, spec, recipes, *, retention_floor=0.95, max_unknown_change_rate=0.0
):
    """Approve an action set whose worst combined retention passed the training floor.

    A runtime that alternates independently adequate irreversible cuts may lose
    evidence from both. The shortest typed windows with only shared protections,
    applied at every turn, bound the surviving evidence for any mixture. Admit
    candidates in the supplied order. This empirical guard needs held-out validation.
    """
    threads, labels, recipes = list(threads), list(labels), list(recipes)
    limits = {
        "retention_floor": retention_floor,
        "max_unknown_change_rate": max_unknown_change_rate,
    }
    reports = _measure_recipes(threads, labels, spec, recipes, **limits)
    metrics = dict(zip(recipes, reports))
    accepted = None
    for recipe, report in zip(recipes, reports):
        report["individual_eligible"] = report["eligible"]
        if not report["eligible"]:
            report["joint_envelope"] = None
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
        report["joint_envelope"] = {
            key: outcome[key]
            for key in (
                "recipe",
                "coverage",
                "historical_required",
                "historical_retained",
                "unknown_change_rate",
            )
        }
        report["eligible"] = fits
        if fits:
            accepted = envelope
    return reports


def fit_recipe(threads, labels, spec, recipe, **kwargs):
    return fit_recipes(threads, labels, spec, [recipe], **kwargs)[0]
