"""Render causal checkpoints and evaluate historical evidence retained before recovery."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, replace

from compache.dependencies import label_index
from compache.traces import ContractError

from .contracts import ModelSpec, number
from .edit_accounting import PartMeter
from .parts import (
    Request,
    Source,
    placeholder,
    protected_sources,
    wire_part,
)
from .parts import (
    expand as reopen_part,
)

__all__ = [
    "RecipeView",
    "assess",
    "RetentionRecipe",
    "evidence_windows",
    "fit_recipe",
    "fit_recipes",
    "matched_prefix_tokens",
    "recipe_grid",
    "size_ladder",
    "render_recipe",
    "reopen_part",
]


@dataclass(frozen=True)
class RetentionRecipe:
    """Counts of completed runs to retain per type; None keeps all of that type.

    A part of at most `keep_under_tokens` tokens is never archived: a small part
    costs little to keep, and under a placeholder's size it costs less.
    """

    input_runs: int | None = None
    output_runs: int | None = None
    retrieval_runs: int | None = None
    protect_query_matches: bool = False
    keep_under_tokens: int = 0

    def __post_init__(self):
        number(self.keep_under_tokens, "keep_under_tokens", integer=True)
        for name in ("input_runs", "output_runs", "retrieval_runs"):
            value = getattr(self, name)
            if value is not None:
                number(value, name, integer=True)
        if type(self.protect_query_matches) is not bool:
            raise ContractError("protect_query_matches must be boolean")


def recipe_grid(
    longest_thread_runs=None,
    *,
    kinds=("input", "output", "retrieval"),
    protect_query_matches=(True, False),
    keep_under_tokens=(0,),
    extra_windows=None,
):
    """Candidate cuts: every combination of one window for each kind of part.

    A window keeps that kind's parts from that many completed runs, counted in
    retrieval groups for retrievals. Zero keeps only the current run's parts.
    Windows follow 1, 2, 5, 10, 20, 50 and so on, and stop below the longest
    training thread because a longer window never cuts there. A kind missing
    from `kinds` is always kept whole. Query-match protection only concerns
    retrievals, so it is varied only where a retrieval window can cut.
    `extra_windows` adds windows per kind, for example the look-backs that
    `evidence_windows` reads off the labels. Each combination is repeated for
    every size in `keep_under_tokens`, under which a part is never archived.
    The grid contains the narrowest combination of any two of its members, so
    checking a combined cut never needs a recipe outside it.
    """
    sizes = tuple(keep_under_tokens)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ContractError("keep_under_tokens must list distinct sizes")
    if longest_thread_runs is None:
        longest_thread_runs = 100
    number(longest_thread_runs, "longest_thread_runs", 1, integer=True)
    kinds, flags = set(kinds), tuple(protect_query_matches)
    if not kinds or kinds - {"input", "output", "retrieval"}:
        raise ContractError("kinds must name input, output or retrieval parts")
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
    extra = dict(extra_windows or {})
    if set(extra) - {"input", "output", "retrieval"}:
        raise ContractError("extra_windows must be keyed by input, output or retrieval")

    def cuts(kind):
        added = set(extra.get(kind, ()))
        for window in added:
            number(window, "extra window", integer=True)
        return tuple(sorted({0, *windows, *added}, reverse=True))

    inputs = (None, *cuts("input")) if "input" in kinds else (None,)
    outputs = (None, *cuts("output")) if "output" in kinds else (None,)
    retrievals = [(None, min(flags))]
    if "retrieval" in kinds:
        retrievals.extend(
            (window, flag) for flag in flags for window in cuts("retrieval")
        )
    return tuple(
        RetentionRecipe(i, o, r, flag, size)
        for size in sizes
        for r, flag in retrievals
        for i in inputs
        for o in outputs
        if (i, o, r) != (None, None, None)
    )


def evidence_windows(threads, labels, quantiles=(0.8, 0.85, 0.9, 0.95, 0.975, 1.0)):
    """Look-backs worth trying per kind, read off where labelled evidence lies.

    For every requirement that needs history, the smallest window on one kind
    that could still satisfy it is how far back its evidence of that kind lies.
    The windows returned are the look-backs that cover each quantile of those
    requirements. A fixed ladder can step straight over the range where
    retention moves from one floor to the other. These do not.
    """
    by_run = {}
    for label in labels:
        by_run[label["target_run_id"]] = label
    needs = {"input": [], "output": [], "retrieval": []}
    for thread in threads:
        origin, groups = {}, []
        for run in thread.runs:
            origin[run.input.part_id] = ("input", run.run_index)
            origin[run.output.part_id] = ("output", run.run_index)
            exposed = [p for p in run.retrievals if p.exposed_to_model]
            for part in exposed:
                origin[part.part_id] = ("retrieval", run.run_index)
            if exposed:
                groups.append(run.run_index)
        for run in thread.runs:
            label = by_run.get(run.run_id)
            for req in label["task_requirements"] if label else ():
                if req["historical_context_needed"] != "yes" or not req["alternatives"]:
                    continue
                best = {}
                for alternative in req["alternatives"]:
                    far = {"input": 0, "output": 0, "retrieval": 0}
                    for source in alternative["sources"]:
                        kind, turn = origin[source["part_id"]]
                        if turn >= run.run_index:
                            continue
                        if kind == "retrieval":
                            # Later retrieval groups before this run, plus its own.
                            lag = 1 + sum(turn < g < run.run_index for g in groups)
                        else:
                            lag = run.run_index - turn
                        far[kind] = max(far[kind], lag)
                    for kind, lag in far.items():
                        best[kind] = min(best.get(kind, lag), lag)
                for kind, lag in best.items():
                    needs[kind].append(lag)
    windows = {}
    for kind, lags in needs.items():
        lags.sort()
        windows[kind] = tuple(
            sorted({lags[min(len(lags) - 1, int(q * len(lags)))] for q in quantiles})
            if lags
            else ()
        )
    return windows


def size_ladder(threads, spec, quantiles=(0.25, 0.5, 0.75)):
    """Sizes worth trying for `keep_under_tokens`: zero and quantiles of the parts seen.

    Small parts are many and cheap, so keeping every part up to a size keeps far
    more of them than the same tokens spent on recent large ones.
    """
    sizes = sorted(
        spec.part_tokens(part.text)
        for thread in threads
        for run in thread.runs
        for part in (run.input, run.output, *run.retrievals)
        if part.text is not None
    )
    if not sizes:
        return (0,)
    picked = {sizes[int(q * (len(sizes) - 1))] for q in quantiles}
    # A size no part exceeds would keep everything, which is not a cut.
    return (0, *sorted(n for n in picked if 0 < n < sizes[-1]))


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


def _checkpoint(request, recipe, archived, pins, spec):
    """Apply one cut to the mutable `archived` and `pins` sets and return `archived`."""
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
    small = recipe.keep_under_tokens
    archived.update(
        p.part_id
        for p in request.parts
        if p.part_id not in pins
        and p.run_index < thresholds[p.kind]
        and (not small or spec.part_tokens(p.text) > small)
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
        _checkpoint(request, recipe, archived, pins, spec)
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


_NEVER = float("inf")
_COUNTERS = (
    "turns",
    "historical_required",
    "historical_retained",
    "changed_historical_required",
    "changed_historical_retained",
    "direct_required",
    "direct_retained",
    "unknown_requirements",
    "unknown_changed_requirements",
    "unknown_runs",
    "unknown_changed_runs",
    "unlabeled_runs",
    "unlabeled_changed_runs",
    "changed_runs",
    "canonical_tokens",
    "projected_tokens",
    "recovery_tokens",
    "recovery_view_tokens",
    "recovery_restored_tokens",
)


class _Window:
    """When each part of one kind is archived under one window, and what that saves."""

    def __init__(self, archived_at, saving, total):
        self.turn = {p: t for p, t in archived_at.items() if t <= total}
        self.first = min(self.turn.values(), default=_NEVER)
        gain = [0] * (total + 2)
        for part_id, turn in self.turn.items():
            gain[turn] += saving[part_id]
        # saved[n] is what the archive has removed from the request of run n.
        self.saved, running = [0] * (total + 1), 0
        for n in range(1, total + 1):
            running += gain[n]
            self.saved[n] = running
        self.total = sum(self.saved)


def _key(window, recipe):
    """A window that never cuts is the same table at every size."""
    return (window, recipe.keep_under_tokens if window is not None else 0)


def _retrieval_key(recipe):
    window = recipe.retrieval_runs
    if window is None:
        return (None, False, 0)
    return (window, recipe.protect_query_matches, recipe.keep_under_tokens)


def _thread_windows(thread, spec, recipes):
    """Archive turns for every window the recipes use, from one pass over the thread.

    A cut applied at every turn archives a part once it leaves its kind's window
    and never restores it. So a part's archive turn follows from how far back it
    lies: no turn-by-turn replay is needed, and the kinds do not interact.
    """
    runs, total = thread.runs, len(thread.runs)
    canonical = Request.from_thread(thread, runs[-1].run_id).parts
    full, saving, size, by_kind = [0] * (total + 2), {}, {}, defaultdict(list)
    for part in canonical:
        if part.text is None:
            raise ContractError(
                "recipe projection requires complete visible source text"
            )
        tokens = spec.part_tokens(part.text)
        size[part.part_id] = tokens
        saving[part.part_id] = tokens - spec.part_tokens(
            placeholder(thread.thread_id, part)
        )
        # An answer first appears in the request of the following run.
        full[part.run_index + (part.kind == "output")] += tokens
        by_kind[part.kind].append(part)
    for n in range(1, total + 1):
        full[n] += full[n - 1]
    table = {"input": {}, "output": {}, "retrieval": {}}

    def large(parts, small):
        return [p for p in parts if not small or size[p.part_id] > small]

    for kind, name in (("input", "input_runs"), ("output", "output_runs")):
        for window, small in {_key(getattr(r, name), r) for r in recipes}:
            table[kind][(window, small)] = _Window(
                {}
                if window is None
                else {
                    p.part_id: p.run_index + window + 1
                    for p in large(by_kind[kind], small)
                },
                saving,
                total,
            )
    groups = sorted({p.run_index for p in by_kind["retrieval"]})
    rank = {turn: j for j, turn in enumerate(groups)}
    wanted = {_retrieval_key(r) for r in recipes}
    leaving = {}
    for window in {w for w, _, _ in wanted if w is not None}:
        leaving[window] = {
            p.part_id: groups[rank[p.run_index] + window] + 1
            for p in by_kind["retrieval"]
            if rank[p.run_index] + window < len(groups)
        }
    matches = {}
    protected = [w for w, flag, _ in wanted if flag and leaving[w]]
    if protected:
        # A source the current question matches stays until a turn that does not.
        start = min(min(leaving[w].values()) for w in protected)
        for run in runs[start - 1 :]:
            request = Request(
                thread.thread_id,
                run.run_id,
                run.run_index,
                run.generation.model,
                canonical[: len(run.generation.request_part_ids)],
            )
            matches[run.run_index] = protected_sources(request)
    for window, flag, small in wanted:
        archived_at = (
            {}
            if window is None
            else {
                p: t for p, t in leaving[window].items() if not small or size[p] > small
            }
        )
        if flag:
            for part_id, turn in archived_at.items():
                while turn <= total and part_id in matches[turn]:
                    turn += 1
                archived_at[part_id] = turn
        table["retrieval"][(window, flag, small)] = _Window(archived_at, saving, total)
    kind_of = {p.part_id: p.kind for p in canonical}
    return table, full, saving, kind_of


def _measure_recipes(
    threads,
    labels,
    spec,
    recipes,
    *,
    retention_floor,
    max_unknown_change_rate,
    unknowns="excluded",
    tradeoff_floor=None,
):
    """Evaluate each irreversible checkpoint rule on training inputs and frozen outcomes.

    Each recipe cuts at every training turn and carries its own archive across
    turns. No future labels enter the cut. An unaltered full history is the
    fallback. The floor is an empirical coverage criterion. Threads are scored
    one at a time, so `threads` may be a generator over a corpus too large to hold.

    `unknowns` says how evidence nobody could assess enters the floor.
    With "excluded" the floor applies to assessed requirements only. With
    "worst_case" every unknown requirement on a labelled turn whose context the
    recipe changed counts as lost, and the floor applies to that lower bound.
    Both figures are always reported. A turn without a label is unsampled: it is
    counted in the report and enters neither side of the ratio.
    `max_unknown_change_rate` is an optional stricter cap on the share of
    unassessable turns, unlabelled ones included, that a recipe may change. Zero
    is the original rule, under which a single such turn rejects the recipe.

    `tradeoff_floor` admits a recipe between it and `retention_floor` as a
    candidate that must pay for what it loses. Every lost requirement is priced
    as one more request of that turn's size plus the archived sources it needs
    back, and that rate is carried from the labelled turns to all turns.
    """
    number(retention_floor, "retention_floor", 0, 1)
    minimum = retention_floor
    if tradeoff_floor is not None:
        number(tradeoff_floor, "tradeoff_floor", 0, retention_floor)
        minimum = tradeoff_floor
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
    by_thread = defaultdict(list)
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get("thread_id"), str):
            raise ContractError("invalid training label")
        by_thread[label["thread_id"]].append(label)
    reports = [
        {
            "recipe": asdict(recipe),
            **dict.fromkeys(_COUNTERS, 0),
            "retention_floor": retention_floor,
            "tradeoff_floor": tradeoff_floor,
            "max_unknown_change_rate": max_unknown_change_rate,
            "unknowns": unknowns,
        }
        for recipe in recipes
    ]
    weakest = [None] * len(recipes)
    seen_threads, seen_runs = set(), set()
    for thread in threads:
        run_ids = {r.run_id for r in thread.runs}
        if (
            thread.thread_id in seen_threads
            or len(run_ids) != len(thread.runs)
            or run_ids & seen_runs
        ):
            raise ContractError(
                "fitting requires unique training identities and in-split labels"
            )
        seen_threads.add(thread.thread_id)
        seen_runs |= run_ids
        if any(r.generation.model != spec.model_id for r in thread.runs):
            raise ContractError("recipe token ruler belongs to a different model")
        # Bound packet materialization to one thread and validate quote spans once.
        own = by_thread.get(thread.thread_id, [])
        indexed = label_index([thread], own) if own else {}
        table, full, saving, kind_of = _thread_windows(thread, spec, recipes)
        total = len(thread.runs)
        canonical_tokens = sum(full[1 : total + 1])
        labelled, unlabelled, unsure, unknown_total = [], [], [], 0
        for run in thread.runs:
            label = indexed.get(run.run_id)
            if label is None:
                unlabelled.append(run.run_index)
                continue
            needs, unknown = [], 0
            for req in label["task_requirements"]:
                if (
                    req["historical_context_needed"] == "unknown"
                    or not req["alternatives"]
                ):
                    unknown += 1
                    continue
                needs.append(
                    (
                        req["historical_context_needed"] == "yes",
                        [
                            [s["part_id"] for s in a["sources"]]
                            for a in req["alternatives"]
                        ],
                    )
                )
            unknown_total += unknown
            if label["status"] != "complete" or unknown:
                # A partial or unassessable label can hide a requirement that it
                # does not list.
                unsure.append((run.run_index, max(1, unknown)))
            labelled.append((run.run_index, needs))
        unsure_turns = [n for n, _ in unsure]
        for i, (recipe, report) in enumerate(zip(recipes, reports)):
            windows = (
                table["input"][_key(recipe.input_runs, recipe)],
                table["output"][_key(recipe.output_runs, recipe)],
                table["retrieval"][_retrieval_key(recipe)],
            )
            at = {
                "input": windows[0].turn,
                "output": windows[1].turn,
                "retrieval": windows[2].turn,
            }
            first = min(w.first for w in windows)
            required = retained = 0
            for n, needs in labelled:
                changed = n >= first
                for historical, alternatives in needs:
                    kept = any(
                        all(at[kind_of[p]].get(p, _NEVER) > n for p in sources)
                        for sources in alternatives
                    )
                    if not historical:
                        report["direct_required"] += 1
                        report["direct_retained"] += kept
                        continue
                    required += 1
                    retained += kept
                    if changed:
                        report["changed_historical_required"] += 1
                        report["changed_historical_retained"] += kept
                    if not kept:
                        # One more request of this size, plus the cheapest sources to reopen.
                        view = full[n] - sum(w.saved[n] for w in windows)
                        restored = min(
                            sum(
                                saving[p]
                                for p in sources
                                if at[kind_of[p]].get(p, _NEVER) <= n
                            )
                            for sources in alternatives
                        )
                        report["recovery_view_tokens"] += view
                        report["recovery_restored_tokens"] += restored
                        report["recovery_tokens"] += view + restored
            report["historical_required"] += required
            report["historical_retained"] += retained
            if required and (weakest[i] is None or retained / required < weakest[i]):
                weakest[i] = retained / required
            changed_unlabelled = len(unlabelled) - bisect_left(unlabelled, first)
            changed_unsure = unsure[bisect_left(unsure_turns, first) :]
            report["turns"] += total
            report["unknown_requirements"] += unknown_total
            report["unknown_runs"] += len(unlabelled) + len(unsure)
            report["unlabeled_runs"] += len(unlabelled)
            report["changed_runs"] += max(0, total - first + 1) if first <= total else 0
            report["unlabeled_changed_runs"] += changed_unlabelled
            report["unknown_changed_runs"] += changed_unlabelled + len(changed_unsure)
            report["unknown_changed_requirements"] += sum(c for _, c in changed_unsure)
            report["canonical_tokens"] += canonical_tokens
            report["projected_tokens"] += canonical_tokens - sum(
                w.total for w in windows
            )
    if not seen_threads:
        raise ContractError("fit requires training threads")
    if set(by_thread) - seen_threads:
        raise ContractError(
            "fitting requires unique training identities and in-split labels"
        )
    for recipe, report, worst in zip(recipes, reports, weakest):
        required = report["historical_required"]
        retained = report["historical_retained"]
        coverage = retained / required if required else None
        at_risk = required + report["unknown_changed_requirements"]
        lower_bound = retained / at_risk if required else None
        measure = lower_bound if unknowns == "worst_case" else coverage
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
        elif coverage < minimum:
            admission = "below_retention_floor"
        elif measure < minimum:
            admission = "unknown_requirements_could_break_floor"
        elif (
            max_unknown_change_rate is not None
            and unknown_rate > max_unknown_change_rate
        ):
            admission = "above_unknown_change_cap"
        elif measure < retention_floor:
            admission = "within_tradeoff"
        else:
            admission = "passes"
        labeled = report["turns"] - report["unlabeled_runs"]
        report.update(
            coverage=coverage,
            coverage_lower_bound=lower_bound,
            min_thread_coverage=worst,
            unknown_change_rate=unknown_rate,
            eligible=admission in ("keeps_everything", "passes", "within_tradeoff"),
            admission=admission,
            compact_fraction=report["projected_tokens"] / report["canonical_tokens"]
            if report["canonical_tokens"]
            else None,
            # Labelled turns stand for the unlabelled ones.
            expected_tokens=report["projected_tokens"]
            + (report["recovery_tokens"] * report["turns"] / labeled if labeled else 0),
        )
    return reports


def assess(label, full_ids):
    """Which labelled requirements a set of visible parts still satisfies."""
    if label is None:
        return {
            "assessable": False,
            "missed_requirements": [],
            "restore_alternatives": [],
            "historical_requirements_assessed": 0,
            "historical_requirements_retained": 0,
        }
    assessable = label["status"] == "complete" and all(
        t["historical_context_needed"] != "unknown" and t["alternatives"]
        for t in label["task_requirements"]
    )
    missed = []
    alternatives = []
    historical_total = historical_retained = 0
    for req in label["task_requirements"]:
        if req["historical_context_needed"] == "unknown" or not req["alternatives"]:
            continue
        sets = [{s["part_id"] for s in a["sources"]} for a in req["alternatives"]]
        retained = any(s <= full_ids for s in sets)
        if req["historical_context_needed"] == "yes":
            historical_total += 1
            historical_retained += int(retained)
        if not retained:
            missed.append(req["requirement_id"])
            alternatives.append([s - full_ids for s in sets])
    return {
        "assessable": assessable,
        "missed_requirements": missed,
        "restore_alternatives": alternatives,
        "historical_requirements_assessed": historical_total,
        "historical_requirements_retained": historical_retained,
    }


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
    tradeoff_floor=None,
    max_unknown_change_rate=None,
    unknowns="excluded",
    select="supplied",
):
    """Approve an action set whose worst combined retention passed the training floor.

    A runtime that alternates independently adequate irreversible cuts may lose
    evidence from both. The shortest typed windows with only shared protections,
    applied at every turn, bound the surviving evidence for any mixture.

    `select="supplied"` admits candidates in the supplied order while that bound
    holds. `select="cheapest"` admits at most one recipe: the cheapest candidate
    among those that save tokens and whose cut met at least one assessed
    requirement. Without `tradeoff_floor` every candidate passes `retention_floor`
    and the cheapest has the fewest projected training tokens. With it, a recipe
    down to that lower floor also competes, and the cheapest has the fewest
    expected tokens once each requirement it loses is charged a request to
    recover it. The winner sits near a floor by construction, so check its
    retention on held-out threads.
    """
    if select not in ("supplied", "cheapest"):
        raise ContractError("select must be 'supplied' or 'cheapest'")
    labels, recipes = list(labels), list(recipes)
    if select == "supplied":
        threads = list(threads)
    limits = {
        "retention_floor": retention_floor,
        "tradeoff_floor": tradeoff_floor,
        "max_unknown_change_rate": max_unknown_change_rate,
        "unknowns": unknowns,
    }
    reports = _measure_recipes(threads, labels, spec, recipes, **limits)
    for report in reports:
        report["individual_eligible"] = report["eligible"]
        report["joint_envelope"] = None
    if select == "cheapest":
        cost = "projected_tokens" if tradeoff_floor is None else "expected_tokens"
        passing = [
            (report[cost], i)
            for i, report in enumerate(reports)
            if report["admission"] in ("passes", "within_tradeoff")
        ]

        def shortfall(report):
            if not report["changed_historical_required"]:
                return "cut_met_no_assessed_requirement"
            if report[cost] >= report["canonical_tokens"]:
                # A placeholder can outweigh the short message it replaces, and
                # recovering lost evidence can outweigh what a cut removes.
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
                min(accepted.keep_under_tokens, recipe.keep_under_tokens),
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
