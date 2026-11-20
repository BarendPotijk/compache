"""Fit and evaluate a thread-local cache policy from an explicitly supplied numeric export.

Each export represents a single model and serial route. Identifiers are local
pseudonyms. A file reader and the host's permission to supply it live at the
calling boundary; these functions accept only the rows they are passed.
"""

from collections import defaultdict
from statistics import median
import re

from compache.traces import ContractError
from .optimizer.contracts import digest, number, seconds
from .optimizer.distribution import (
    DemandRun,
    fit_counted_distribution,
    plan_distribution_idle,
    repeat_input_usage,
    request_charge,
)
from .optimizer.workload import WorkloadPolicy


REQUIRED = frozenset(
    {
        "thread_id",
        "route_id",
        "model_id",
        "run_index",
        "started_at",
        "ended_at",
        "added_tokens",
        "output_tokens",
    }
)
OPTIONAL = frozenset(
    {
        "visible_output_tokens",
        "calls_per_run",
        "retrieved_tokens",
        "next_endpoint_reuse_fraction",
    }
)
_OPAQUE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}")


def validate_metadata_row(row):
    """Validate the content-free export contract; return only its numeric demand."""
    if not isinstance(row, dict) or not REQUIRED <= row.keys() <= REQUIRED | OPTIONAL:
        raise ContractError("metadata row has missing or unrecognized fields")
    for key in ("thread_id", "route_id"):
        if not isinstance(row[key], str) or not _OPAQUE.fullmatch(row[key]):
            raise ContractError("metadata requires opaque thread and route identifiers")
    if not isinstance(row["model_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}", row["model_id"]
    ):
        raise ContractError("metadata requires a model identifier")
    number(row["run_index"], "run_index", 1, integer=True)
    number(row["added_tokens"], "added_tokens", integer=True)
    if "retrieved_tokens" in row:
        number(
            row["retrieved_tokens"],
            "retrieved_tokens",
            0,
            row["added_tokens"],
            integer=True,
        )
    number(row["output_tokens"], "output_tokens", integer=True)
    visible = row.get("visible_output_tokens", row["output_tokens"])
    number(visible, "visible_output_tokens", integer=True)
    calls = row.get("calls_per_run")
    if calls is not None:
        number(calls, "calls_per_run", 1, integer=True)
    fraction = row.get("next_endpoint_reuse_fraction")
    if fraction is not None:
        number(fraction, "next_endpoint_reuse_fraction", 0, 1)
    start, end = seconds(row["started_at"]), seconds(row["ended_at"])
    if end < start:
        raise ContractError("metadata run end precedes its start")
    return DemandRun(
        start, end, row["added_tokens"], row["output_tokens"], visible, calls, fraction
    )


def _groups(rows, spec, cutoff):
    groups = defaultdict(dict)
    route = None
    assumed_visible = 0
    retrieval_counts = []
    for row in rows:
        run = validate_metadata_row(row)
        if row["model_id"] != spec.model_id:
            raise ContractError("metadata model differs from model spec")
        if route is not None and route != row["route_id"]:
            raise ContractError("fit one metadata route at a time")
        route = row["route_id"]
        if run.end > cutoff:
            raise ContractError("metadata observation cutoff precedes an exported run")
        pos = row["run_index"]
        group = groups[row["thread_id"]]
        if pos in group:
            raise ContractError("metadata has a duplicate thread position")
        group[pos] = run
        assumed_visible += int("visible_output_tokens" not in row)
        if "retrieved_tokens" in row:
            retrieval_counts.append(row["retrieved_tokens"])
    ordered = {}
    for tid in sorted(groups):
        group = groups[tid]
        if set(group) != set(range(1, len(group) + 1)):
            raise ContractError(
                "metadata thread positions must be consecutive from one"
            )
        runs = [group[i] for i in range(1, len(group) + 1)]
        if runs[-1].next_endpoint_reuse_fraction is not None:
            raise ContractError(
                "a terminal run has no next foreground endpoint observation"
            )
        if any(b.start < a.end for a, b in zip(runs, runs[1:])):
            raise ContractError("metadata must describe serial runs in each thread")
        ordered[tid] = runs
    return ordered, route, assumed_visible, retrieval_counts


def fit_metadata(
    rows, spec, *, observation_end, horizon_seconds=86400, retention_floor=0.95
):
    """Return an ordinary loadable caching-only planner artifact and aggregate report."""
    number(retention_floor, "retention_floor", 0, 1)
    groups, route, assumed_visible, retrieval_counts = _groups(
        rows, spec, seconds(observation_end)
    )
    if not groups:
        raise ContractError("metadata fitting needs training runs")
    runs = [r for group in groups.values() for r in group]
    demand = fit_counted_distribution(
        groups,
        spec,
        observation_end=observation_end,
        horizon_seconds=horizon_seconds,
        reported_output_runs=len(runs),
    )
    gaps = [
        b.start - a.end for group in groups.values() for a, b in zip(group, group[1:])
    ]
    positions = demand["by_position"].values()
    returns = sum(len(p["returns"]) for p in positions)
    observed = sum(p["no_return"] for p in positions) + returns
    return (
        {
            "schema_version": "compache-workload-planner-v1",
            "model_spec": spec.to_dict(),
            "metadata_route_digest": digest(route),
            "header_traffic": None,
            "demand": demand,
            "training_threads": sorted(digest(tid) for tid in groups),
            "recipes": [],
            "retention_floor": retention_floor,
        },
        {
            "threads": len(groups),
            "runs": len(runs),
            "median_thread_length": median(map(len, groups.values())),
            "median_added_tokens": median(r.added_tokens for r in runs),
            "median_output_tokens": median(r.output_tokens for r in runs),
            "median_duration_seconds": median(r.end - r.start for r in runs),
            "runs_with_retrieved_tokens": len(retrieval_counts),
            "median_retrieved_tokens": median(retrieval_counts)
            if retrieval_counts
            else None,
            "median_return_gap_seconds": median(gaps) if gaps else None,
            "return_within_horizon_fraction": returns / observed,
            "fully_observed_horizons": observed,
            "censored_horizons": len(runs) - observed,
            "observed_returns": returns,
            "runs_with_next_endpoint_reuse_fraction": sum(
                r.next_endpoint_reuse_fraction is not None for r in runs
            ),
            "runs_with_observed_call_count": sum(
                r.calls_per_run is not None for r in runs
            ),
            "runs_assuming_visible_equals_billed_output": assumed_visible,
        },
    )


def plan_metadata_idle(
    policy,
    *,
    run_index,
    input_tokens,
    added_tokens,
    visible_output_tokens,
    duration_seconds,
    reusable_input_tokens=None,
    ping_allowed=True,
):
    """Return the workload planner's timer advice using counts only.

    Pass a compiled caching-only WorkloadPolicy. Supply the foreground receipt's
    confirmed eligible prefix to enable pings. The host cancels timers on arrival,
    honors its cache-break latch and accounts for the actual maintenance receipt.
    Deadlines are relative to the foreground model request's dispatch time.
    """
    if policy.recipes:
        raise ContractError("metadata timer advice requires a caching-only policy")
    if type(ping_allowed) is not bool:
        raise ContractError("ping_allowed must be boolean")
    number(input_tokens, "input_tokens", integer=True)
    number(added_tokens, "added_tokens", 0, input_tokens, integer=True)
    if reusable_input_tokens is not None:
        number(
            reusable_input_tokens,
            "reusable_input_tokens",
            0,
            input_tokens,
            integer=True,
        )
    capacity = _capacity(reusable_input_tokens or 0, policy.spec)
    keep = {
        "name": "keep",
        "input_tokens": input_tokens,
        "shared_prefix_tokens": 0,
        "rewrite_prefix_tokens": input_tokens,
    }
    return plan_distribution_idle(
        policy.forecast,
        policy.spec,
        run_index=run_index,
        candidates=[keep],
        answer_tokens=visible_output_tokens,
        duration_seconds=duration_seconds,
        held_cache_tokens=capacity,
        current_arrival_tokens=added_tokens,
        ping_allowed=ping_allowed and capacity > 0,
        **policy.options,
    )


def evaluate_metadata(rows, artifact, *, observation_end):
    """Score fitted cache timers against no pings on held-out append-only timelines.

    Costs use fresh-input-token equivalents under the supplied model rates.
    An exact echo of the foreground items plus visible output is modeled as the
    maintenance request. The next user request can reuse that matching prefix.
    A timer tied with an actual return is cancelled. Every timer through the
    observation cutoff is priced, including those after the last response.
    """
    cutoff = seconds(observation_end)
    policy = WorkloadPolicy(artifact)
    if policy.recipes:
        raise ContractError("metadata alone cannot evaluate a compaction recipe")
    spec = policy.spec
    groups, route, assumed_visible, _ = _groups(rows, spec, cutoff)
    if groups and digest(route) != artifact.get("metadata_route_digest"):
        raise ContractError("held-out route differs from metadata fitting route")
    if any(digest(tid) in policy.training_threads for tid in groups):
        raise ContractError("held-out metadata overlaps fitted threads")
    result = {
        "status": "insufficient_evidence",
        "reason": None,
        "reference": "preserve_all_no_pings",
        "units": "fresh_input_token_equivalents",
        "threads": len(groups),
        "runs": sum(map(len, groups.values())),
        "baseline_cost": None,
        "candidate_cost": None,
        "savings_fraction": None,
        "ping_calls": None,
        "runs_assuming_visible_equals_billed_output": assumed_visible,
        "assumptions": [
            "One serial, append-only route with a stable token prefix and thread-local cache. Include static framing once in first-turn added_tokens.",
            "Each ping submits the prior input and its visible answer with full eligible reuse. Output and rates use the model spec. Cache reads expire at the exact TTL.",
            "Repeated foreground calls reread the initial input inside the TTL. Each run's exported output counts once. Unknown foreground call counts assume one.",
            "The observed period includes terminal pings through the cutoff. Host compaction, prefix breaks, additional tool suffixes and shared caches across threads are outside this estimate.",
        ],
    }
    if not groups:
        result["reason"] = "no_heldout_threads"
        return result
    baseline = candidate = 0.0
    ping_calls = 0
    for runs in groups.values():
        history = baseline_prefix = candidate_prefix = 0
        baseline_refresh = candidate_refresh = None
        for index, run in enumerate(runs):
            size = history + run.added_tokens
            history = size + run.visible_output_tokens
            if (
                size + run.output_tokens > spec.context_tokens
                or history > spec.context_tokens
            ):
                result.update(
                    status="context_limit",
                    reason="observed_append_only_path_exceeds_context",
                )
                return result
            base_match = (
                baseline_prefix
                if baseline_refresh is not None
                and run.start - baseline_refresh < spec.cache_ttl_seconds
                else 0
            )
            cand_match = (
                candidate_prefix
                if candidate_refresh is not None
                and run.start - candidate_refresh < spec.cache_ttl_seconds
                else 0
            )
            # This transition has now happened. It can price the held-out
            # foreground request, but must not enter the earlier timer decision.
            if index and runs[index - 1].next_endpoint_reuse_fraction is not None:
                fraction = runs[index - 1].next_endpoint_reuse_fraction
                base_match = int(base_match * fraction)
                cand_match = int(cand_match * fraction)
            repeats = repeat_input_usage(spec, size, (run.calls_per_run or 1) - 1)[
                "cost_input_equivalents"
            ]
            baseline += (
                request_charge(spec, size, base_match, run.output_tokens) + repeats
            )
            candidate += (
                request_charge(spec, size, cand_match, run.output_tokens) + repeats
            )
            baseline_refresh = candidate_refresh = run.start
            baseline_prefix = candidate_prefix = _capacity(size, spec)
            action = plan_metadata_idle(
                policy,
                run_index=index + 1,
                input_tokens=size,
                added_tokens=run.added_tokens,
                visible_output_tokens=run.visible_output_tokens,
                duration_seconds=run.end - run.start,
                reusable_input_tokens=candidate_prefix,
            )
            if action["status"] != "ok":
                result.update(status="planner_unavailable", reason=action["status"])
                return result
            for offset in action["due_after_start_seconds"]:
                due = run.start + offset
                if due > cutoff or (
                    index + 1 < len(runs) and due >= runs[index + 1].start
                ):
                    break
                match = (
                    candidate_prefix
                    if due - candidate_refresh < spec.cache_ttl_seconds
                    else 0
                )
                candidate += request_charge(
                    spec, history, match, spec.ping_output_tokens
                )
                ping_calls += 1
                candidate_refresh = due
                candidate_prefix = _capacity(history, spec)
    result.update(
        status="modeled",
        baseline_cost=baseline,
        candidate_cost=candidate,
        ping_calls=ping_calls,
        savings_fraction=1 - candidate / baseline if baseline else None,
    )
    return result


def _capacity(size, spec):
    whole = size // spec.cache_block_tokens * spec.cache_block_tokens
    return whole if spec.cache_ttl_seconds > 0 and whole >= spec.cache_min_tokens else 0
