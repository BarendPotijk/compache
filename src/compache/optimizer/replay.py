"""Replay the real gate/planner against an explicit, single-prefix cache model."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction

from compache.traces import ContractError, Thread
from .contracts import Policy, seconds, number
from .fit import freeze_profile, index_run_peaks, label_index, run_remaining_growth
from .gate import Request, Source, ThreadState, prepare, wire_part
from .planner import end_run
from .position import position_risk


def warm_outcome_validation(records, policy):
    """Score reached, supported warm entries by position using realized frozen growth."""
    groups = defaultdict(list)
    for row in records:
        if row["window"] != "warm":
            continue
        forecast = row.get("warm_overflow_risk", {}).get("before")
        if forecast is None and row["action"] == "observe_only":
            # Hidden input can suppress the gate evaluation. It is still a
            # missing outcome at a reached warm position with training support.
            forecast = position_risk(policy, row["run_index"], 0, "remaining_tokens")
        if forecast is not None and forecast["triggered"] is not None:
            groups[row["run_index"]].append(row.get("frozen_growth_outcome", {}))
    alpha = Fraction(str(policy.max_overflow_probability))
    minimum = max(policy.position_growth["min_threads"], math.ceil(1 / alpha) - 1)
    positions = {}
    for pos, outcomes in sorted(groups.items()):
        n = len(outcomes)
        x = sum(o.get("modeled_overflow") is True for o in outcomes)
        u = sum(o.get("modeled_overflow") is None for o in outcomes)
        if x > alpha * n:
            status = "exceeds_budget"
        elif x + u > alpha * n:
            status = "inconclusive_missing"
        elif n < minimum:
            status = "insufficient_validation_threads"
        else:
            status = "within_empirical_budget"
        positions[str(pos)] = {"opportunities": n, "assessed": n-u, "exceedances": x, "unresolved": u,
                               "lower_fraction": x/n, "upper_fraction": (x+u)/n, "status": status,
                               "sources": dict(Counter(o.get("source", "unassessed") for o in outcomes))}
    statuses = {v["status"] for v in positions.values()}
    status = next((s for s in ("exceeds_budget", "inconclusive_missing", "insufficient_validation_threads")
                   if s in statuses), "within_empirical_budget" if positions else "no_supported_warm_opportunities")
    return {"status": status, "max_overflow_probability": float(alpha), "min_validation_threads_per_position": minimum,
            "supported_warm_opportunities": sum(map(len, groups.values())), "by_run_index": positions,
            "basis": "Per-position empirical frozen-growth outcomes under the projected request. Unresolved outcomes count toward the upper fraction. Source labels distinguish whole-run sidecars and one-generation estimates. No population-rate confidence claim."}


class Cache:
    """Whole-part prefix matching, rounded down to configured cache blocks."""

    def __init__(self, spec, enabled=True, overhead_tokens=0):
        number(overhead_tokens, "overhead_tokens", integer=True)
        self.spec = spec
        self.enabled = enabled
        self.overhead_tokens = overhead_tokens
        self.lines = ()
        self.counts = ()
        self.refreshed = None

    def charge(self, parts, now, output_tokens):
        spec = self.spec
        lines = tuple(wire_part(p) for p in parts)
        counts = tuple(spec.part_tokens(p.text) for p in parts)
        if self.overhead_tokens:
            lines = ("__modeled_stable_runtime_prefix__",) + lines
            counts = (self.overhead_tokens,) + counts
        total = sum(counts)
        eligible = (
            self.enabled
            and spec.cache_ttl_seconds > 0
            and total >= spec.cache_min_tokens
        )
        common = 0
        if (
            eligible
            and self.refreshed is not None
            and now - self.refreshed < spec.cache_ttl_seconds
        ):
            for a, b, n in zip(lines, self.lines, counts):
                if a != b:
                    break
                common += n
        cached = (common // spec.cache_block_tokens) * spec.cache_block_tokens
        if cached < spec.cache_min_tokens:
            cached = 0
        fresh = total if not eligible else 0
        writes = total - cached if eligible else 0
        cost = spec.token_charge(fresh, writes, cached, output_tokens)
        if eligible:
            self.lines, self.counts, self.refreshed = lines, counts, now
        return {
            "input_tokens_estimated": total,
            "fresh_input_tokens": fresh,
            "cache_read_tokens": cached,
            "cache_write_tokens": writes,
            "output_tokens_estimated": output_tokens,
            "cost_input_equivalents": cost,
        }


def assess(label, full_ids):
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


def _cold_preview(
    request, state, policy, answer, now, overhead_tokens=0, *, token_estimator=None
):
    preview = Request(
        request.thread_id,
        request.run_id + ":preview",
        request.run_index + 1,
        request.model,
        request.parts + (answer,),
    )
    return prepare(
        preview,
        policy,
        copy.deepcopy(state),
        now,
        force_cold=True,
        overhead_tokens=overhead_tokens,
        token_estimator=token_estimator,
    )


def replay(
    threads: list[Thread],
    labels: list[dict],
    policy: Policy,
    profile: dict | None,
    *,
    observation_end: str,
    caching=True,
    include_ledger=True,
    recovery_overhead="priced",
    overhead_tokens=0,
    peak_observations=None,
) -> dict:
    number(overhead_tokens, "overhead_tokens", integer=True)
    if recovery_overhead not in ("priced", "zero"):
        raise ContractError("recovery_overhead must be priced or zero")
    if profile is not None:
        profile = freeze_profile(profile)
        if policy.profile_hash != profile.fingerprint:
            raise ContractError("policy/profile hash mismatch")
    end = seconds(observation_end)
    if any(seconds(r.ended_at) > end for t in threads for r in t.runs):
        raise ContractError("observation_end precedes trace")
    labels_by_id = label_index(threads, labels)
    peaks = index_run_peaks(peak_observations, {r.run_id for t in threads for r in t.runs})
    evaluate_growth = peaks is not None or policy.position_growth is not None
    spec = policy.spec
    if any(r.generation.model != spec.model_id for t in threads for r in t.runs):
        raise ContractError(
            "replay requires one model matching the supplied model spec"
        )
    ledger = []
    records = []
    skipped = 0
    unobserved_pings = 0
    for thread in sorted(threads, key=lambda t: t.thread_id):
        state = ThreadState(thread.thread_id)
        cache = Cache(spec, caching, overhead_tokens)
        for i, run in enumerate(thread.runs):
            request = Request.from_thread(thread, run.run_id)
            start, finish = seconds(run.generation.started_at), seconds(run.ended_at)
            prepared = prepare(
                request, policy, state, start, overhead_tokens=overhead_tokens
            )
            label = labels_by_id.get(run.run_id)
            assessment = assess(label, prepared.full_part_ids)
            record = {
                "thread_id": thread.thread_id,
                "run_id": run.run_id,
                "company_id": thread.company_id,
                "run_index": run.run_index,
                "estimated_full_input_tokens": sum(
                    spec.part_tokens(p.text) for p in request.parts
                ),
                "removed_part_ids": sorted(prepared.removed_part_ids),
                "dependency_assessable": assessment["assessable"],
                "missed_requirements": assessment["missed_requirements"],
                "historical_requirements_assessed": assessment[
                    "historical_requirements_assessed"
                ],
                "historical_requirements_retained": assessment[
                    "historical_requirements_retained"
                ],
                "unknown_context_changed": not assessment["assessable"]
                and bool(prepared.removed_part_ids),
                "recovered_part_ids": [],
                **prepared.telemetry(),
            }
            records.append(record)
            record["cost_coverage"] = (
                "visible_projection_only"
                if run.output.text is None or any(p.text is None for p in request.parts)
                else "complete_visible_text"
            )
            if not prepared.admitted:
                skipped += len(thread.runs) - i - 1
                break
            if evaluate_growth and prepared.action != "observe_only":
                # Score the frozen outcome after preparing the view. Holding the
                # original within-run growth fixed is a replay assumption.
                source, status, remaining = run_remaining_growth(run, spec, peaks)
                crossing = None if remaining is None else prepared.tokens + remaining > spec.context_tokens
                identified = (status == "complete" or (status == "censored" and crossing is True)
                              or (status == "upper_bound" and crossing is False))
                record["frozen_growth_outcome"] = {
                    "source": source, "status": status, "remaining_tokens": remaining,
                    "modeled_overflow": crossing if identified else None,
                }
            missing = bool(assessment["missed_requirements"])
            output = spec.text_tokens(run.output.text)
            usage = cache.charge(
                prepared.parts,
                start,
                spec.expansion_output_tokens
                if missing and recovery_overhead == "priced"
                else output,
            )
            ledger.append(
                {
                    "thread_id": thread.thread_id,
                    "run_id": run.run_id,
                    "event": "request",
                    "at": start,
                    **usage,
                }
            )
            if missing and recovery_overhead == "priced":
                sources = {p.part_id: p for p in request.parts}
                restore = set()
                for choices in assessment["restore_alternatives"]:
                    # One modeled recovery round trip; choose the cheapest sufficient alternative.
                    choice = min(
                        choices,
                        key=lambda ids: (
                            sum(
                                spec.part_tokens(sources[s].text) for s in ids - restore
                            ),
                            sorted(ids),
                        ),
                    )
                    restore |= choice
                expansions = state.expansions.setdefault(run.run_index, {})
                for pid in sorted(restore):
                    original = sources[pid]
                    if original.text is None:
                        raise ContractError("cannot recover unknown content")
                    expansions[pid] = replace(
                        original,
                        part_id=run.run_id + "/restored/" + pid,
                        run_index=run.run_index,
                    )
                prepared = prepare(
                    request, policy, state, start, overhead_tokens=overhead_tokens
                )
                if not prepared.admitted:
                    record["admitted"] = False
                    record["reason"] = "recovery_exceeds_estimated_limit"
                    skipped += len(thread.runs) - i - 1
                    break
                if assess(label, prepared.full_part_ids)["missed_requirements"]:
                    raise ContractError(
                        "modeled recovery failed to satisfy known requirements"
                    )
                record["recovered_part_ids"] = sorted(restore)
                ledger.append(
                    {
                        "thread_id": thread.thread_id,
                        "run_id": run.run_id,
                        "event": "expansion_request",
                        "at": start,
                        **cache.charge(prepared.parts, start, output),
                    }
                )
            answer = Source(
                run.output.part_id, "output", run.run_index, run.output.text
            )
            cold = _cold_preview(
                request, state, policy, answer, finish, overhead_tokens
            )
            decision = (
                end_run(
                    policy,
                    profile,
                    held_input=prepared.tokens,
                    answer_tokens=spec.part_tokens(answer.text),
                    cold_view=cold.tokens,
                    run_index=run.run_index,
                    request_started_at=start,
                    ended_at=finish,
                    reduces=bool(cold.removed_part_ids),
                )
                if caching and prepared.action != "observe_only"
                else None
            )
            record["retention_action"] = decision.action if decision else "lapse"
            record["planned_pings"] = decision.pings if decision else 0
            record["expected_ping_saving"] = decision.expected_saving if decision else 0
            if decision:
                if decision.next_run_overflow_risk is not None:
                    record["next_run_overflow_risk"] = decision.next_run_overflow_risk
                next_start = (
                    seconds(thread.runs[i + 1].started_at)
                    if i + 1 < len(thread.runs)
                    else end
                )
                ping_parts = prepared.parts + (answer,)
                for due in decision.due_times:
                    if due >= next_start:
                        if i + 1 == len(thread.runs):
                            unobserved_pings += sum(
                                t >= end for t in decision.due_times
                            )
                        break
                    usage = cache.charge(ping_parts, due, spec.ping_output_tokens)
                    ledger.append(
                        {
                            "thread_id": thread.thread_id,
                            "run_id": run.run_id,
                            "event": "ping",
                            "at": due,
                            **usage,
                        }
                    )
                    if (
                        usage["cache_read_tokens"] < prepared.tokens / 2
                        or spec.ping_output_tokens
                    ):
                        break
                    state.refreshed_at = due
    assessed = [r for r in records if r["dependency_assessable"]]
    unknown = [r for r in records if not r["dependency_assessable"]]
    misses = sum(bool(r["missed_requirements"]) for r in assessed)
    changes = sum(r["unknown_context_changed"] for r in unknown)
    historical_total = sum(r["historical_requirements_assessed"] for r in records)
    historical_retained = sum(r["historical_requirements_retained"] for r in records)
    summary = {
        "threads": len(threads),
        "recovery_overhead": recovery_overhead,
        "modeled_stable_runtime_overhead_tokens": overhead_tokens,
        "planned_runs": sum(len(t.runs) for t in threads),
        "replayed_runs": len(records),
        "skipped_runs": skipped,
        "cost_input_equivalents": math.fsum(
            r["cost_input_equivalents"] for r in ledger
        ),
        "request_cost": math.fsum(
            r["cost_input_equivalents"] for r in ledger if r["event"] == "request"
        ),
        "expansion_cost": math.fsum(
            r["cost_input_equivalents"]
            for r in ledger
            if r["event"] == "expansion_request"
        ),
        "ping_cost": math.fsum(
            r["cost_input_equivalents"] for r in ledger if r["event"] == "ping"
        ),
        "pings_sent": sum(r["event"] == "ping" for r in ledger),
        "unobserved_future_pings": unobserved_pings,
        "expansion_round_trips": sum(r["event"] == "expansion_request" for r in ledger),
        "assessable_runs": len(assessed),
        "unknown_runs": len(unknown),
        "miss_runs": misses,
        "miss_rate": misses / len(assessed) if assessed else None,
        "historical_requirements_assessed": historical_total,
        "historical_requirements_retained": historical_retained,
        "historical_dependency_coverage": historical_retained / historical_total
        if historical_total
        else None,
        "historical_coverage_basis": "Fraction of labeled historical requirements with one complete sufficient evidence alternative retained before recovery; AND within each alternative, OR across alternatives. Unknown requirements and direct lookup requirements are excluded, with unknown runs reported separately.",
        "unknown_changed_runs": changes,
        "unknown_change_rate": changes / len(unknown) if unknown else 0,
        "admission_failures": sum(not r["admitted"] for r in records),
        "compacted_runs": sum(bool(r["removed_part_ids"]) for r in records),
        "prefix_rewrites": sum(r["prefix_transition"] == "rewritten" for r in records),
        "mean_input_tokens_estimated": sum(r["tokens_estimated"] for r in records)
        / len(records)
        if records
        else 0,
        "ledger_totals": {
            k: sum(r[k] for r in ledger)
            for k in (
                "input_tokens_estimated",
                "fresh_input_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "output_tokens_estimated",
            )
        },
        "interpretation": "Modeled RAG-only cost in base-input-token equivalents. Byte-based token estimates; no Codex overhead or provider bill calibration. Recovery uses frozen task-label alternatives as an evaluation oracle, never runtime knowledge. No answer-quality replay or causal test.",
    }
    summary["runs_with_unavailable_text"] = sum(
        r["cost_coverage"] == "visible_projection_only" for r in records
    )
    if evaluate_growth:
        outcomes = [r.get("frozen_growth_outcome", {}).get("modeled_overflow") for r in records]
        summary["frozen_growth_overflow"] = {
            "exceedances": sum(x is True for x in outcomes),
            "assessed_runs": sum(x is not None for x in outcomes),
            "unassessed_or_skipped": summary["planned_runs"] - sum(x is not None for x in outcomes),
            "basis": "Initial projected input plus frozen run growth. Tool behavior and answers may change after a rewrite. Without a peak sidecar growth uses reported one-generation output tokens or a visible answer proxy.",
        }
    if policy.position_growth is not None:
        forecasts = [r["warm_overflow_risk"] for r in records if "warm_overflow_risk" in r]
        summary["warm_risk_gate"] = {
            "supported_runs": sum(f["before"]["triggered"] is not None for f in forecasts),
            "fixed_fallback_runs": sum(f["before"]["triggered"] is None for f in forecasts),
            "triggered_runs": sum(f["before"]["triggered"] is True for f in forecasts),
            "remaining_above_budget": sum(f["after"]["triggered"] is True for f in forecasts),
        }
        summary["warm_outcome_validation"] = warm_outcome_validation(records, policy)
    summary["cost_coverage"] = (
        "visible_projection_only"
        if summary["runs_with_unavailable_text"]
        else "complete_visible_text"
    )
    summary["interpretation"] += (
        " Unavailable bodies are unpriced; totals cover visible projection text only, not the hidden full workload."
    )
    if recovery_overhead == "zero":
        summary["interpretation"] += (
            " Recovery overhead is assumed zero: no expansion calls or restored content are inserted."
            " Known missing requirements are still reported; frozen answers are not quality evidence."
        )
    return {
        "schema_version": "compache-replay-v1",
        "policy_hash": policy.hash,
        "summary": summary,
        "runs": records,
        "ledger": ledger if include_ledger else [],
    }
