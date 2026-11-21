"""Choose the cut and the ping count by replayed cost on the caller's own threads."""

from compache.errors import ContractError
from compache.labels import label_index

from .contracts import digest, number, seconds
from .distribution import fit_distribution, measure_cache_reads, validate_calls_per_run
from .replay import _run_arm
from .retention import evidence_windows, fit_recipes, recipe_grid, size_ladder
from .session import WorkloadPolicy

__all__ = ["solve_workload"]


def _rank_coverage(row):
    # An assessed, history-independent cut has no historical denominator.
    return 1.0 if row["coverage"] is None else row["coverage"]


def _frontier(rows, limit):
    """Cuts worth pricing: each keeps more than every cheaper one."""
    usable = sorted(
        (
            r
            for r in rows
            # Replayed cost decides whether a cut pays, so the token estimate of
            # its recovery does not rule one out here.
            if r["individual_eligible"]
            and (r["changed_historical_required"] or r["assessed_independent_cut"])
            and r["projected_tokens"] < r["canonical_tokens"]
        ),
        key=lambda r: (r["projected_tokens"], -_rank_coverage(r)),
    )
    kept, best = [], None
    for row in usable:
        # A dearer cut earns a place only by keeping at least a point more.
        if best is None or _rank_coverage(row) >= best + 0.01:
            kept.append(row)
            best = _rank_coverage(row)
    if len(kept) > limit:
        # Keep both ends and spread the rest evenly between them.
        step = (len(kept) - 1) / (limit - 1)
        kept = [kept[round(i * step)] for i in range(limit)]
    return kept


def solve_workload(
    threads,
    labels,
    spec,
    *,
    observation_end,
    system_prompt,
    default_tools,
    retention_threads=None,
    retention_floor=0.95,
    tradeoff_floor=None,
    unknowns="excluded",
    ping_counts=(0, 1, 3),
    folds=3,
    max_cuts=6,
    horizon_seconds=86400,
    component_order=("system_prompt", "default_tools"),
    share_headers=True,
    provider="declared",
    cache_key="",
    tokenize=None,
    count_tokens=None,
    calls_per_run=None,
    next_endpoint_reuse_fraction=None,
    salt="compache-solve-v1",
    **planner_options,
):
    """Fit a policy and pick its cut and ping count by cost on these threads.

    Token counts cannot rank policies, because what a cut or a ping is worth
    depends on how often the cache is still warm when the user returns. So each
    candidate is replayed against the provider's cache at the threads' own
    timestamps. Candidates are the cuts on the retention frontier between
    `tradeoff_floor` and full retention, each with every ping count, plus
    keeping everything. A thread is always priced by a demand model fitted
    without it. A cut also pays for what it loses: every lost requirement is
    one more request that reads the cached view, writes the reopened sources
    and emits a tool call, at the rate seen on labelled turns.

    `calls_per_run` maps run IDs to the model calls that turn made. A call after
    the first re-sends the turn's request and reads it from the cache, in the
    fit of returns and in every candidate's replay alike.

    `threads` supply demand and cost. `retention_threads` supply the labelled
    runs when those differ, for example when graded questions are appended to a
    recorded conversation. Pass a function that returns a fresh iterator when
    they are too large to hold at once.
    """
    threads = list(threads)
    if len(threads) < 2:
        raise ContractError("solving needs at least two threads")
    number(folds, "folds", 2, integer=True)
    number(max_cuts, "max_cuts", 2, integer=True)
    ping_counts = sorted(set(ping_counts))
    if not ping_counts or any(type(n) is not int or n < 0 for n in ping_counts):
        raise ContractError("ping_counts must list nonnegative integers")
    if "max_pings" in planner_options:
        raise ContractError("the solver chooses max_pings; pass ping_counts instead")
    labels = list(labels)
    kinds = {"input", "output"}
    if any(p.exposed_to_model for t in threads for r in t.runs for p in r.retrievals):
        kinds.add("retrieval")
    # Labelled runs are read twice: once for where evidence lies, once to score.
    labelled = (
        (lambda: threads)
        if retention_threads is None
        else retention_threads
        if callable(retention_threads)
        else (lambda kept=list(retention_threads): kept)
    )
    grid = recipe_grid(
        max(len(t.runs) for t in threads),
        kinds=kinds,
        keep_under_tokens=size_ladder(threads, spec),
        extra_windows=evidence_windows(labelled(), labels),
    )
    rows = fit_recipes(
        labelled(),
        labels,
        spec,
        grid,
        select="cheapest",
        retention_floor=retention_floor,
        tradeoff_floor=tradeoff_floor,
        unknowns=unknowns,
    )
    cuts = _frontier(rows, max_cuts)

    cutoff = (
        seconds(observation_end)
        if isinstance(observation_end, str)
        else observation_end
    )
    replay = {
        "observation_end": cutoff,
        "system": system_prompt,
        "tools": default_tools,
        "tokenize": tokenize,
        "count_tokens": count_tokens,
        "component_order": component_order,
        "share_headers": share_headers,
        "provider": provider,
        "cache_key": cache_key,
        "calls_per_run": validate_calls_per_run(
            calls_per_run, {r.run_id for t in threads for r in t.runs}
        ),
        "modeled_ping_output_tokens": spec.ping_output_tokens,
    }
    counts = replay["calls_per_run"]

    def counted(group):
        """The call counts recorded for these threads' own runs."""
        own = {r.run_id for t in group for r in t.runs}
        return {run: n for run, n in counts.items() if run in own} or None

    fold_of = {
        t.thread_id: int(digest([salt, t.thread_id])[:8], 16) % folds for t in threads
    }
    if len(set(fold_of.values())) == 1:
        # A handful of threads can hash into one fold. Deal them out instead.
        ranked = sorted(threads, key=lambda t: digest([salt, t.thread_id]))
        fold_of = {
            t.thread_id: i % min(folds, len(threads)) for i, t in enumerate(ranked)
        }
    candidates = [(None, n) for n in ping_counts] + [
        (row, n) for row in cuts for n in ping_counts
    ]
    demand = fit_distribution(
        threads,
        spec,
        observation_end=observation_end,
        horizon_seconds=horizon_seconds,
        calls_per_run=counted(threads),
        next_endpoint_reuse_fraction=next_endpoint_reuse_fraction,
    )
    overlap = next_endpoint_reuse_fraction or {}
    full_cache_reads = measure_cache_reads(threads, spec, overlap)
    indexed = label_index(threads, labels) if retention_threads is None else {}
    recovery_totals = [0.0] * len(candidates)
    labelled_counts = [0] * len(candidates)
    cost = [0.0] * len(candidates)
    complete = [True] * len(candidates)
    turns = 0
    for fold in sorted(set(fold_of.values())):
        inside = [t for t in threads if fold_of[t.thread_id] == fold]
        outside = [t for t in threads if fold_of[t.thread_id] != fold]
        if not outside:
            raise ContractError("every fold needs threads outside it; use fewer folds")
        fold_demand = fit_distribution(
            outside,
            spec,
            observation_end=observation_end,
            horizon_seconds=horizon_seconds,
            calls_per_run=counted(outside),
            next_endpoint_reuse_fraction={
                r.run_id: overlap[r.run_id]
                for t in outside
                for r in t.runs[:-1]
                if r.run_id in overlap
            },
        )
        turns += sum(len(t.runs) for t in inside)
        read_estimate = measure_cache_reads(outside, spec, overlap)
        for i, (row, pings) in enumerate(candidates):
            policy = WorkloadPolicy(
                {
                    "schema_version": "compache-workload-planner-v1",
                    "model_spec": spec.to_dict(),
                    "demand": fold_demand,
                    "cache_reads": read_estimate,
                    "recipes": [{"recipe": row["recipe"], "eligible": True}]
                    if row
                    else [],
                    "retention_floor": retention_floor,
                    "training_threads": sorted(digest(t.thread_id) for t in outside),
                    "header_traffic": None,
                },
                max_pings=pings,
                **planner_options,
            )
            # With no cut and no ping the planner has nothing to decide.
            arm = _run_arm(
                inside, indexed, policy, candidate=bool(row or pings), **replay
            )
            for entry in arm["ledger"]:
                if entry["role"] == "answer" and entry["labeled"]:
                    labelled_counts[i] += 1
                    recovery_totals[i] += entry["projected_recovery_cost"]
            cost[i] += arm["totals"]["cost_input_equivalents"]
            complete[i] = complete[i] and arm["complete"]

    keep_coverage = 1.0 if rows and rows[0]["historical_required"] else None
    report = []
    for i, (row, pings) in enumerate(candidates):
        recovery = 0.0
        if labelled_counts[i]:
            recovery = recovery_totals[i] * turns / labelled_counts[i]
        elif row and retention_threads is not None:
            # Separately graded questions may be absent from the replay trajectories.
            labeled = row["turns"] - row["unlabeled_runs"]
            lost = row["historical_required"] - row["historical_retained"]
            share = full_cache_reads["share"]
            share = 1.0 if share is None else share
            view_rate = spec.read_multiplier * share + spec.write_multiplier * (
                1 - share
            )
            recovery = (
                turns
                / labeled
                * (
                    view_rate * row["recovery_view_tokens"]
                    + spec.write_multiplier * row["recovery_restored_tokens"]
                    + spec.output_multiplier * spec.expansion_output_tokens * lost
                )
            )
        report.append(
            {
                "recipe": row["recipe"] if row else None,
                "max_pings": pings,
                "coverage": row["coverage"] if row else keep_coverage,
                "min_thread_coverage": row["min_thread_coverage"]
                if row
                else keep_coverage,
                "replayed_cost": cost[i],
                "recovery_cost": recovery,
                "expected_cost": cost[i] + recovery,
                "complete": complete[i],
            }
        )
    usable = [r for r in report if r["complete"]]
    if not usable:
        raise ContractError(
            "no candidate policy fits the context window on these threads"
        )
    chosen = min(
        usable, key=lambda r: (r["expected_cost"], -_rank_coverage(r), r["max_pings"])
    )
    keep = report[ping_counts.index(0)] if 0 in ping_counts else None
    priced = {id(row) for row in cuts}
    for row in rows:
        if row["recipe"] == chosen["recipe"]:
            row["eligible"], row["admission"] = True, "admitted"
        elif row["individual_eligible"] and row["admission"] != "keeps_everything":
            row["eligible"] = False
            row["admission"] = (
                "not_cheapest_by_replayed_cost"
                if id(row) in priced
                else "off_the_retention_frontier"
                if (
                    row["changed_historical_required"]
                    or row["assessed_independent_cut"]
                )
                and row["projected_tokens"] < row["canonical_tokens"]
                else row["admission"]
                if row["admission"]
                in ("cut_met_no_assessed_requirement", "saves_no_tokens")
                else "saves_no_tokens"
            )
    return {
        "schema_version": "compache-workload-planner-v1",
        "header_traffic": None,
        "cache_reads": full_cache_reads,
        "model_spec": spec.to_dict(),
        "demand": demand,
        "recipes": rows,
        "retention_floor": retention_floor,
        "recipe_selection": {
            "select": "replayed_cost",
            "tradeoff_floor": tradeoff_floor,
            "unknowns": unknowns,
            "max_unknown_change_rate": None,
        },
        "planner_options": {"max_pings": chosen["max_pings"]},
        "solve": {
            "cost_unit": "fresh input token equivalents",
            "folds": len(set(fold_of.values())),
            "turns": turns,
            "chosen": chosen,
            "keep_everything_cost": keep["replayed_cost"]
            if keep and keep["complete"]
            else None,
            "saving_fraction": 1 - chosen["expected_cost"] / keep["replayed_cost"]
            if keep and keep["complete"] and keep["replayed_cost"]
            else None,
            "candidates": report,
        },
        "training_threads": sorted(digest(t.thread_id) for t in threads),
    }
