"""Content replay comparisons with shared economics and explicit evidence boundaries."""

from __future__ import annotations

import math
from dataclasses import replace

from compache.optimizer.contracts import digest, number, seconds
from compache.optimizer.fit import label_index
from compache.optimizer.gate import Request
from compache.optimizer.replay import Cache, assess, replay
from compache.traces import ContractError


def _message(part):
    message = {"role": {"input": "user", "output": "assistant", "retrieval": "tool"}[part.kind],
               "content": part.text}
    if part.kind == "retrieval":
        message.update(tool_call_id=part.part_id, name="retrieve_documents")
    return message


def replay_headroom(threads, labels, policy, backend, *, observation_end,
                    cold_refresh=False, overhead_tokens=0):
    """Replay actual compression; price with Compache's shared scenario ledger.

    Frozen answers are reused. Lossless verification is upstream's assertion,
    not an independent answer-quality score. Each canonical part maps to one
    message; this is a compression interface, not a provider wire transcript.
    """
    number(overhead_tokens, "overhead_tokens", integer=True)
    end = seconds(observation_end)
    spec = policy.spec
    if any(r.generation.model != spec.model_id for t in threads for r in t.runs):
        raise ContractError("comparison requires one model matching the model spec")
    if any(seconds(r.ended_at) > end for t in threads for r in t.runs):
        raise ContractError("observation_end precedes trace")
    indexed = label_index(threads, labels)
    records, ledger = [], []
    skipped = 0
    for thread in sorted(threads, key=lambda t: t.thread_id):
        cache = Cache(spec, overhead_tokens=overhead_tokens)
        projected, previous_fingerprints = {}, {}
        previous_started = None
        for i, run in enumerate(thread.runs):
            request = Request.from_thread(thread, run.run_id)
            start = seconds(run.generation.started_at)
            unavailable = run.output.text is None or any(p.text is None for p in request.parts)
            if unavailable:
                records.append({"thread_id": thread.thread_id, "run_id": run.run_id,
                                "status": "unavailable_text", "admitted": False})
                skipped += len(thread.runs) - i - 1
                break
            # Reset on any unexpected canonical edit, never reuse stale content.
            if any(previous_fingerprints.get(p.part_id, p.fingerprint) != p.fingerprint
                   for p in request.parts):
                projected = {}
            cold = previous_started is not None and (
                spec.cache_ttl_seconds == 0 or start - previous_started > spec.cache_ttl_seconds + 60
            )
            refresh = cold_refresh and cold
            messages = [_message(p) if refresh else projected.get(p.part_id, _message(p))
                        for p in request.parts]
            frozen = 0 if refresh else len(projected)
            output, measurement = backend.compress(
                messages, frozen=frozen, context_tokens=spec.context_tokens
            )
            # The adapter also checks this; guard all injected implementations.
            if len(output) != len(messages) or output[:frozen] != messages[:frozen]:
                raise ContractError("baseline returned an invalid or rewritten frozen prefix")
            if measurement.get("lossless_verified_by_headroom") is not True:
                raise ContractError("baseline did not verify lossless output")
            parts = tuple(replace(p, text=m["content"]) for p, m in zip(request.parts, output))
            tokens = sum(spec.part_tokens(p.text) for p in parts) + overhead_tokens
            admitted = (tokens <= spec.context_tokens * (1 - policy.reserve)
                        and sum(len(p.wire.encode()) for p in parts) <= spec.byte_limit)
            assessment = assess(indexed.get(run.run_id), {p.part_id for p in request.parts})
            record = {
                "thread_id": thread.thread_id, "run_id": run.run_id, "run_index": run.run_index,
                "status": "complete" if admitted else "admission_failed", "admitted": admitted,
                "cold_refresh": refresh, "frozen_messages": frozen,
                "tokens_estimated": tokens, "projection_hash": digest(output),
                "historical_requirements_assessed": assessment["historical_requirements_assessed"],
                "historical_requirements_retained": assessment["historical_requirements_retained"],
                "dependency_assessable": assessment["assessable"], **measurement,
            }
            records.append(record)
            if not admitted:
                skipped += len(thread.runs) - i - 1
                break
            ledger.append({"thread_id": thread.thread_id, "run_id": run.run_id,
                           "event": "request", "at": start,
                           **cache.charge(parts, start, spec.text_tokens(run.output.text))})
            projected = {p.part_id: m for p, m in zip(request.parts, output)}
            previous_fingerprints = {p.part_id: p.fingerprint for p in request.parts}
            previous_started = start
    complete = len(ledger) == sum(len(t.runs) for t in threads)
    return {
        "summary": {
            "planned_runs": sum(len(t.runs) for t in threads), "completed_runs": len(ledger),
            "skipped_runs": skipped, "complete": complete,
            "cost_input_equivalents": math.fsum(r["cost_input_equivalents"] for r in ledger),
            "admission_failures": sum(r["status"] == "admission_failed" for r in records),
            "unavailable_text_runs": sum(r["status"] == "unavailable_text" for r in records),
            "compression_seconds_measured": math.fsum(r.get("compression_seconds_measured", 0) for r in records),
            "dependency_coverage_basis": "Source availability from Headroom-verified lossless transformation; not independent semantic validation.",
        },
        "configuration": {"cold_refresh": cold_refresh, "cold_margin_seconds": 60,
                          "recovery_overhead": "zero", "pings": 0},
        "provenance": backend.provenance, "runs": records, "ledger": ledger,
    }


def compare_headroom(threads, labels, policy, profile, backend, *, observation_end,
                     overhead_tokens=0):
    """Evaluate a previously frozen Compache policy, without refitting to these traces."""
    kwargs = dict(observation_end=observation_end, overhead_tokens=overhead_tokens)
    compache = replay(threads, labels, policy, profile, recovery_overhead="zero", **kwargs)
    s = compache["summary"]
    compache_complete = not (s["admission_failures"] or s["skipped_runs"] or s["runs_with_unavailable_text"])
    arms = {}
    for name, cold in (("headroom-lossless-cache", False), ("headroom-lossless-cold", True)):
        result = replay_headroom(threads, labels, policy, backend, cold_refresh=cold, **kwargs)
        comparable = compache_complete and result["summary"]["complete"]
        base_cost = result["summary"]["cost_input_equivalents"]
        result["compache_saving_fraction"] = (
            1 - s["cost_input_equivalents"] / base_cost if comparable and base_cost > 0 else None
        )
        result["comparable_cost"] = comparable
        arms[name] = result
    return {
        "schema_version": "compache-headroom-comparison-v1", "policy_hash": policy.hash,
        "threads_hash": digest([t.to_dict() for t in threads]), "labels_hash": digest(labels),
        "observation_end": observation_end, "model_spec": policy.model_spec,
        "interpretation": (
            "Native Headroom lossless library compression with shared byte-based token estimates and "
            "modeled cache costs. Frozen answers, zero recovery-call overhead, no provider calls or "
            "answer-quality measurement. Native tokenizer counts and local compression time are separate "
            "diagnostics. Neither arm reproduces Headroom's default proxy/CCR/ML or opt-in net-cost gate. "
            "Cold refresh is this adapter's TTL-based reset of the frozen prefix, not Headroom's cold hook. "
            "Missing text or incomplete arms suppress savings. This evaluation does not fit a policy."
        ),
        "compache": compache, "baselines": arms,
    }
