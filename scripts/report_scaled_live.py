"""Audit a paired live sample, retaining admission-stopped arms and denominators."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compache.optimizer.cache_scenarios import reprice_prefix
from compache.optimizer.calibration import OnlineTokenCalibrator
from compache.optimizer import Policy
from report_live_comparison import audit_events


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if path.exists()
        else []
    )


def matched_prefix(rows, completed_answers):
    """Only compare calls needed for the same number of completed user turns."""
    n = {
        i: min(completed_answers[f"{a}-{i}"] for a in ("opencode", "compache"))
        for i in (1, 2, 3)
    }
    return [r for r in rows if r["run_index"] <= n[r["replicate"]]], n


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def audit_calibration(directory, ledger, plan, spec):
    """Rebuild estimates using only preceding actual calls, never final usage first."""
    config = plan.get("token_calibration")
    if config is None:
        return None
    ruler = OnlineTokenCalibrator(
        plan["overhead_tokens_for_planning_only"], config["safety_margin"]
    )
    gaps = []
    for row in ledger:
        view = (directory / f"call-{row['sequence']:04d}" / "prompt.txt").read_text()
        body = view.split("\nCONVERSATION\n", 1)[1].rsplit("\nTASK\n", 1)[0]
        payload = sum(
            spec.part_tokens(json.loads(line).get("text"))
            for line in body.splitlines()
            if line.strip()
        )
        before = ruler.to_dict()
        estimated = ruler.estimate_total(row["model"], payload)
        ruler.observe(
            row["model"],
            payload,
            row["usage"],
            successful=row["ok"],
            provider_observed=True,
        )
        expected = {
            "payload_tokens_estimated": payload,
            "input_tokens_planned": estimated,
            "state_before": before,
            "state_after": ruler.to_dict(),
            "provider_observed": True,
        }
        if row.get("token_calibration") != expected:
            raise ValueError("causal token calibration does not reconcile")
        if row["role"] == "answer" and row["tokens_estimated"] != estimated:
            raise ValueError("gate and rendered-payload estimates differ")
        gaps.append(estimated - row["usage"]["input_tokens"])
    return {
        "calls_audited": len(gaps),
        "state": ruler.to_dict(),
        "underestimated_calls": sum(gap < 0 for gap in gaps),
        "minimum_estimate_minus_reported_input": min(gaps, default=None),
        "scope": "Each logical request estimate versus its final reported provider invocation; internal invocations counted separately in costs.",
    }


def build(root):
    plan = read(root / "plan.json")
    spec = Policy.from_dict(read(root / "policy.json")).spec
    if (
        hashlib.sha256((root / "quality-design.json").read_bytes()).hexdigest()
        != plan["quality_design_sha256"]
    ):
        raise ValueError("quality design changed")
    design = read(root / "quality-design.json")
    authored = lines(root / "fixtures/labels.jsonl")
    unreported = [
        read(p)
        for p in sorted(
            (root / "unreported-transport-attempts").glob("*/unavailable-usage.json")
        )
    ]
    for name, expected in plan["source_hashes"].items():
        if (
            hashlib.sha256((root / "fixtures" / name).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("fixture changed")
    runs, expanded, quality = {}, [], []
    for arm in ("opencode", "compache"):
        for rep in (1, 2, 3):
            key = f"{arm}-{rep}"
            directory = root / "runs" / key
            ledger, answers = (
                lines(directory / "ledger.jsonl"),
                lines(directory / "answers.jsonl"),
            )
            marker = (
                read(directory / "complete.json")
                if (directory / "complete.json").exists()
                else None
            )
            stopped = (
                read(directory / "stopped.json")
                if (directory / "stopped.json").exists()
                else None
            )
            if marker is None and (
                stopped is None or stopped["status"] != "blocked_by_admission"
            ):
                raise ValueError(
                    f"{key}: neither completed nor a documented admission stop"
                )
            if [a["run_index"] for a in answers] != list(range(1, len(answers) + 1)):
                raise ValueError("answer sequence gap")
            if [r["sequence"] for r in ledger] != list(range(1, len(ledger) + 1)):
                raise ValueError("logical call sequence gap")
            expected = marker["counts"] if marker else stopped["counts"]
            actual = Counter(r["role"] for r in ledger)
            if any(actual[k] != v for k, v in expected.items()) or actual[
                "answer"
            ] != len(answers):
                raise ValueError("recorded call counts do not reconcile")
            if marker and len(answers) != plan["planned_answers_per_arm"]:
                raise ValueError("premature completion marker")
            if (
                stopped
                and not marker
                and (
                    stopped["completed_answers"] != len(answers)
                    or stopped["model_call_sent_for_blocked_run"]
                )
            ):
                raise ValueError("invalid admission stop marker")
            tid = plan["evaluation_thread_ids"][rep - 1]
            calibration = (
                audit_calibration(directory, ledger, plan, spec)
                if arm == "compache"
                else None
            )
            invocation = extra = 0
            usage = Counter()
            for row in ledger:
                if (
                    row["ok"] is not True
                    or row["thread_id"] != tid
                    or row["model"] != "gpt-6-luna"
                    or row["reasoning_effort"] != "low"
                ):
                    raise ValueError("failed call or wrong experiment identity")
                call_dir = directory / f"call-{row['sequence']:04d}"
                if (
                    hashlib.sha256((call_dir / "prompt.txt").read_bytes()).hexdigest()
                    != row["view_sha256"]
                ):
                    raise ValueError("prompt hash mismatch")
                if not (call_dir / "events.jsonl").exists():
                    raise ValueError("raw provider events missing")
                pieces = audit_events(call_dir / "events.jsonl", row)
                for j, part in enumerate(pieces):
                    invocation += 1
                    internal = j < len(pieces) - 1
                    extra += internal
                    usage.update(part)
                    expanded.append(
                        {
                            **row,
                            "logical_sequence": row["sequence"],
                            "sequence": invocation,
                            "role": "codex_overhead" if internal else row["role"],
                            "usage": part,
                            "wall_seconds": 0 if internal else row["wall_seconds"],
                        }
                    )
            by_id = {a["run_id"]: a for a in answers}
            for task in design[tid]:
                answer = by_id.get(task["run_id"])
                result = {
                    "arm": arm,
                    "replicate": rep,
                    "thread_id": tid,
                    "run_id": task["run_id"],
                    "family": task["family"],
                    "answered": answer is not None,
                    "scope": "Literal fact/quote screening only; not semantic answer accuracy.",
                }
                if answer is not None:
                    actual_text = answer["answer"]
                    expected_text = task["current_expected"]
                    result["current_fact_verbatim"] = expected_text in actual_text
                    result["historical_fact_verbatim"] = None
                    if task.get("actual_prior_output_required"):
                        prior = by_id.get(task["source_run_id"])
                        result["actual_prior_first_sentence_verbatim"] = (
                            first_sentence(prior["answer"]) in actual_text
                            if prior
                            else None
                        )
                    elif task.get("historical_expected"):
                        result["historical_fact_verbatim"] = (
                            task["historical_expected"] in actual_text
                        )
                quality.append(result)
            runs[key] = {
                "thread_id": tid,
                "complete": marker is not None,
                "planned_answers": plan["planned_answers_per_arm"],
                "completed_answers": len(answers),
                "unanswered_turns": plan["planned_answers_per_arm"] - len(answers),
                "stop": None if marker else stopped,
                "calls_by_role": dict(actual),
                "reported_provider_invocations": invocation,
                "internal_codex_invocations": extra,
                "reported_usage": dict(usage),
                "token_calibration_audit": calibration,
                "wall_seconds": sum(r["wall_seconds"] for r in ledger),
                "max_reported_input_tokens": max(
                    (
                        u["usage"]["input_tokens"]
                        for u in expanded
                        if u["arm"] == arm and u["replicate"] == rep
                    ),
                    default=None,
                ),
                "rewrites": sum(
                    r.get("prefix_transition") in ("rewrite", "rewritten")
                    for r in ledger
                ),
                "deterministic_recalled_parts": sum(
                    (a.get("diagnostic") or {}).get("recalled_parts", 0)
                    for a in answers
                )
                if arm == "compache"
                else None,
                "missing_authored_requirements_before_recovery": sum(
                    len(
                        (a.get("diagnostic") or {}).get(
                            "missing_authored_requirements", []
                        )
                    )
                    for a in answers
                )
                if arm == "compache"
                else None,
                "live_bound_historical_assessed": sum(
                    (a.get("diagnostic") or {})
                    .get("live_dependency_assessment", {})
                    .get("historical_requirements_assessed", 0)
                    for a in answers
                )
                if arm == "compache"
                else None,
                "live_bound_historical_retained": sum(
                    (a.get("diagnostic") or {})
                    .get("live_dependency_assessment", {})
                    .get("historical_requirements_retained", 0)
                    for a in answers
                )
                if arm == "compache"
                else None,
                "unsupported_output_alternatives_removed": sum(
                    (a.get("diagnostic") or {})
                    .get("live_output_binding", {})
                    .get("output_alternatives_removed", 0)
                    for a in answers
                )
                if arm == "compache"
                else None,
                "historical_requirements_on_unanswered_turns": sum(
                    req.get("historical_context_needed") == "yes"
                    and bool(req.get("alternatives"))
                    for label in authored
                    if label["thread_id"] == tid and label["target_run_id"] not in by_id
                    for req in label["task_requirements"]
                ),
            }
    common, n = matched_prefix(
        expanded, {k: v["completed_answers"] for k, v in runs.items()}
    )
    costs = reprice_prefix(expanded)
    comparable = reprice_prefix(common)
    all_complete = all(r["complete"] for r in runs.values())
    result = {
        "schema": "compache-scaled-live-report-v1",
        "all_arms_completed": all_complete,
        "fresh_runs": plan.get("fresh_runs"),
        "reused_runs": plan.get("reused_runs", []),
        "planned_answers": 6 * plan["planned_answers_per_arm"],
        "completed_answers": sum(r["completed_answers"] for r in runs.values()),
        "all_attempt_usage_available": not unreported,
        "unreported_transport_attempts": unreported,
        "runs": runs,
        "recorded_trajectory_costs": costs,
        "same_completed_prefix_costs": comparable,
        "matched_answer_counts_per_pair": n,
        "full_thread_savings_available": all_complete and not unreported,
        "quality_screen": quality,
        "limitations": [
            "Three distinct paired unseen-company cases, not three repeats of one conversation.",
            "A stopped arm does not receive credit for unexecuted work. Prefix costs are diagnostic, not full-thread savings.",
            "Provider token counts and wall time are observed; TTL costs use assumed return times and prefix reuse.",
            "Runtime scaffolding and reported internal Codex invocations are counted. Controller/platform overhead is unavailable.",
            "Three preserved network-timeout attempts have no reported token usage; recorded successful-call costs exclude that unknown overhead."
            if unreported
            else "No unreported transport attempts were recorded.",
            "Internal invocation wall seconds are allocated to the final invocation of their logical call; zero entries are not measured zero latency.",
            "Literal factual screening can reject valid paraphrases and cannot establish semantic answer quality.",
            "Authored dependency availability is not observed causal source use; prior-output checks use each arm's actual answer.",
        ],
    }
    (root / "scaled-live-report.json").write_text(json.dumps(result, indent=2) + "\n")
    lookup = {(x["arm"], x["replicate"], x["sequence"]): x for x in expanded}
    timeline = []
    for group in costs["replicates"]:
        total = 0
        for item in group["calls"]:
            row = lookup[group["arm"], group["replicate"], item["sequence"]]
            total += item["scenario_cost"]
            timeline.append(
                {
                    "arm": row["arm"],
                    "replicate": row["replicate"],
                    "thread_id": row["thread_id"],
                    "run_index": row["run_index"],
                    "sequence": row["sequence"],
                    "logical_sequence": row["logical_sequence"],
                    "role": row["role"],
                    "ttl_seconds": group["ttl_seconds"],
                    **row["usage"],
                    "modeled_cached_tokens": item["modeled_cached_input_tokens"],
                    "modeled_credits": item["scenario_cost"],
                    "cumulative_modeled_credits": total,
                    "wall_seconds": row["wall_seconds"],
                    "started_at_utc": row["started_at_utc"],
                    "ended_at_utc": row["ended_at_utc"],
                }
            )
    with (root / "timeline.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(timeline)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    result = build(parser.parse_args().root)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "all_arms_completed",
                    "planned_answers",
                    "completed_answers",
                    "matched_answer_counts_per_pair",
                    "full_thread_savings_available",
                )
            }
        )
    )
