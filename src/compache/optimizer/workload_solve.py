"""Choose the cut and the ping count by replayed cost on the caller's own threads."""

from compache.traces import ContractError

from .contracts import digest, number, seconds
from .distribution import fit_distribution, validate_calls_per_run
from .retention import evidence_windows, fit_recipes, recipe_grid, size_ladder
from .workload import WorkloadPolicy
from .workload_replay import _run_arm

__all__ = ["solve_workload"]


def _frontier(rows, limit):
    """Cuts worth pricing: each keeps more than every cheaper one."""
    usable = sorted(
        (
            r
            for r in rows
            # Replayed cost decides whether a cut pays, so the token estimate of
            # its recovery does not rule one out here.
            if r["individual_eligible"]
            and r["changed_historical_required"]
            and r["projected_tokens"] < r["canonical_tokens"]
        ),
        key=lambda r: (r["projected_tokens"], -r["coverage"]),
    )
    kept, best = [], None
    for row in usable:
        # A dearer cut earns a place only by keeping at least a point more.
        if best is None or row["coverage"] >= best + 0.01:
            kept.append(row)
            best = row["coverage"]
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
    tradeoff_floor=0.8,
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
            None, {r.run_id for t in threads for r in t.runs}
        ),
        "modeled_ping_output_tokens": spec.ping_output_tokens,
    }
    fold_of = {
        t.thread_id: int(digest([salt, t.thread_id])[:8], 16) % folds for t in threads
    }
    candidates = [(None, n) for n in ping_counts] + [
        (row, n) for row in cuts for n in ping_counts
    ]
    cost = [0.0] * len(candidates)
    complete = [True] * len(candidates)
    turns = 0
    for fold in sorted(set(fold_of.values())):
        inside = [t for t in threads if fold_of[t.thread_id] == fold]
        outside = [t for t in threads if fold_of[t.thread_id] != fold]
        if not outside:
            raise ContractError("every fold needs threads outside it; use fewer folds")
        demand = fit_distribution(
            outside,
            spec,
            observation_end=observation_end,
            horizon_seconds=horizon_seconds,
        )
        turns += sum(len(t.runs) for t in inside)
        for i, (row, pings) in enumerate(candidates):
            policy = WorkloadPolicy(
                {
                    "schema_version": "compache-workload-planner-v1",
                    "model_spec": spec.to_dict(),
                    "demand": demand,
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
            arm = _run_arm(inside, {}, policy, candidate=bool(row or pings), **replay)
            cost[i] += arm["totals"]["cost_input_equivalents"]
            complete[i] = complete[i] and arm["complete"]

    report = []
    for i, (row, pings) in enumerate(candidates):
        recovery = 0.0
        if row:
            labeled = row["turns"] - row["unlabeled_runs"]
            lost = row["historical_required"] - row["historical_retained"]
            recovery = (
                turns
                / labeled
                * (
                    spec.read_multiplier * row["recovery_view_tokens"]
                    + spec.write_multiplier * row["recovery_restored_tokens"]
                    + spec.output_multiplier * spec.expansion_output_tokens * lost
                )
            )
        report.append(
            {
                "recipe": row["recipe"] if row else None,
                "max_pings": pings,
                "coverage": row["coverage"] if row else 1.0,
                "min_thread_coverage": row["min_thread_coverage"] if row else 1.0,
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
        usable, key=lambda r: (r["expected_cost"], -r["coverage"], r["max_pings"])
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
                if row["changed_historical_required"]
                and row["projected_tokens"] < row["canonical_tokens"]
                else row["admission"]
                if row["admission"]
                in ("cut_met_no_assessed_requirement", "saves_no_tokens")
                else "saves_no_tokens"
            )
    return {
        "schema_version": "compache-workload-planner-v1",
        "header_traffic": None,
        "model_spec": spec.to_dict(),
        "demand": fit_distribution(
            threads,
            spec,
            observation_end=observation_end,
            horizon_seconds=horizon_seconds,
        ),
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
