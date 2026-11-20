"""Streaming, frozen-policy synthetic source-availability evaluation; no model calls."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from compache.optimizer import Policy, replay
from compache.optimizer.contracts import seconds
from compache.optimizer.gate import Request
from compache.optimizer.replay import Cache
from compache.traces import Thread
from scaled_rag_dataset import iter_threads


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline(thread, spec, overhead, limit=None):
    cache = Cache(spec, overhead_tokens=overhead)
    totals = defaultdict(float)
    for run in thread.runs[:limit]:
        req = Request.from_thread(thread, run.run_id)
        usage = cache.charge(
            req.parts,
            seconds(run.generation.started_at),
            spec.text_tokens(run.output.text),
        )
        for k, v in usage.items():
            totals[k] += v
    return dict(totals)


def summarize(thread, labels, result, metadata, target, context):
    byid = {x["target_run_id"]: x for x in labels}
    design = {x["run_id"]: x for x in metadata.get("run_design", [])}
    groups = defaultdict(lambda: defaultdict(int))
    runs = []
    for r in result["runs"]:
        label = byid.get(r["run_id"])
        groups["all"]["replayed_runs"] += 1
        if not r["dependency_assessable"]:
            groups["all"]["unknown_runs"] += 1
        if r["missed_requirements"]:
            groups["all"]["miss_runs"] += 1
        if label:
            for req in label["task_requirements"]:
                kind = req.get("historical_context_needed")
                if kind == "unknown" or not req.get("alternatives"):
                    groups["all"]["unknown_requirements"] += 1
                elif kind == "yes":
                    missed = req["requirement_id"] in r["missed_requirements"]
                    groups["all"]["historical_requirements"] += 1
                    groups["all"]["historical_missed"] += missed
                    q = req.get("question_quote", "")
                    family = design.get(r["run_id"], {}).get("family") or (
                        "input"
                        if "planning note in turn" in q
                        else "output"
                        if "quote the first sentence" in q
                        else "mixed"
                        if "planning allocation and case" in q
                        else "retrieval"
                    )
                    origins = [
                        int(e["source_run_id"].rsplit("-", 1)[-1])
                        for e in label.get("historical_dependencies", [])
                    ]
                    lag = r["run_index"] - min(origins) if origins else None
                    lag_group = (
                        (
                            "1-5"
                            if lag <= 5
                            else "6-20"
                            if lag <= 20
                            else "21-50"
                            if lag <= 50
                            else "51+"
                        )
                        if lag is not None
                        else "unknown"
                    )
                    position = (
                        "1-50"
                        if r["run_index"] <= 50
                        else "51-80"
                        if r["run_index"] <= 80
                        else "81+"
                    )
                    for group in (
                        "family:" + family,
                        "lag:" + lag_group,
                        "position:" + position,
                    ):
                        groups[group]["historical_requirements"] += 1
                        groups[group]["historical_missed"] += missed
        groups["all"]["rewrites"] += r["prefix_transition"] == "rewritten"
        groups["all"]["rewrite_target_attained"] += (
            r["prefix_transition"] == "rewritten"
            and r["tokens_estimated"] <= target * context
        )
        groups["all"]["cold_rewrites"] += (
            r["prefix_transition"] == "rewritten" and r.get("window") == "cold"
        )
        groups["all"]["warm_rewrites"] += (
            r["prefix_transition"] == "rewritten" and r.get("window") == "warm"
        )
        groups["all"]["emergency_rewrites"] += (
            r.get("action") == "replaced_warm_for_admission"
        )
        groups["all"]["warm_deferrals"] += "defer" in str(
            r.get("action", "")
        ) or "defer" in str(r.get("reason", ""))
        groups["all"]["admission_failures"] += not r["admitted"]
        groups["all"]["historical_requirements_on_failed_admission"] += (
            not r["admitted"]
        ) * r["historical_requirements_assessed"]
        groups["all"]["historical_retained_on_failed_admission"] += (
            not r["admitted"]
        ) * r["historical_requirements_retained"]
        runs.append(
            {
                k: r.get(k)
                for k in (
                    "run_id",
                    "run_index",
                    "admitted",
                    "action",
                    "reason",
                    "window",
                    "prefix_transition",
                    "tokens_estimated",
                    "dependency_assessable",
                    "missed_requirements",
                    "historical_requirements_assessed",
                    "historical_requirements_retained",
                )
            }
        )
    seen = {r["run_id"] for r in result["runs"]}
    for label in labels:
        if label["target_run_id"] not in seen:
            for req in label["task_requirements"]:
                if req.get("historical_context_needed") == "yes" and req.get(
                    "alternatives"
                ):
                    groups["all"]["historical_requirements_unassessed_skipped"] += 1
                    d = design.get(label["target_run_id"], {})
                    lag = d.get("lag")
                    lag_group = (
                        (
                            "1-5"
                            if lag <= 5
                            else "6-20"
                            if lag <= 20
                            else "21-50"
                            if lag <= 50
                            else "51+"
                        )
                        if lag
                        else "unknown"
                    )
                    idx = int(label["target_run_id"].rsplit("-", 1)[-1])
                    position = "1-50" if idx <= 50 else "51-80" if idx <= 80 else "81+"
                    for group in (
                        "family:" + str(d.get("family", "unknown")),
                        "lag:" + lag_group,
                        "position:" + position,
                    ):
                        groups[group]["historical_requirements_unassessed_skipped"] += 1
                elif req.get("historical_context_needed") == "unknown" or not req.get(
                    "alternatives"
                ):
                    groups["all"]["unknown_requirements_skipped"] += 1
    return {k: dict(v) for k, v in groups.items()}, runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(p): sha(p)
        for p in (args.policy, args.profile, args.dataset / "manifest.json")
    }
    policy = Policy.from_dict(json.loads(args.policy.read_text()))
    profile = json.loads(args.profile.read_text())
    totals = defaultdict(lambda: defaultdict(int))
    with (
        (args.output / "threads.jsonl").open("w") as tf,
        (args.output / "runs.jsonl").open("w") as rf,
    ):
        for data, labels, meta in iter_threads(args.dataset):
            for kind in ("trace", "labels"):
                if sha(args.dataset / meta[kind + "_path"]) != meta[kind + "_sha256"]:
                    raise ValueError(
                        "source shard hash mismatch: " + meta[kind + "_path"]
                    )
            thread = Thread.from_dict(data)
            result = replay(
                [thread],
                labels,
                policy,
                profile,
                observation_end=thread.runs[-1].ended_at,
                recovery_overhead="zero",
                overhead_tokens=16000,
                include_ledger=False,
            )
            groups, runs = summarize(
                thread, labels, result, meta, policy.target, policy.spec.context_tokens
            )
            split = meta["split"]
            for key, vals in groups.items():
                for scope in ("all", split):
                    for k, v in vals.items():
                        totals[scope + "/" + key][k] += v
            base = baseline(thread, policy.spec, 16000)
            s = result["summary"]
            admitted_count = s["replayed_runs"] - s["admission_failures"]
            prefix = (
                base
                if admitted_count == len(thread.runs)
                else baseline(thread, policy.spec, 16000, admitted_count)
            )
            for scope in ("all", split):
                for k in (
                    "planned_runs",
                    "skipped_runs",
                    "cost_input_equivalents",
                    "pings_sent",
                    "unobserved_future_pings",
                ):
                    totals[scope + "/all"][k] += s[k]
                totals[scope + "/all"]["cache_only_full_cost_input_equivalents"] += (
                    base["cost_input_equivalents"]
                )
                totals[scope + "/all"][
                    "cache_only_admitted_prefix_cost_input_equivalents"
                ] += prefix["cost_input_equivalents"]
                totals[scope + "/all"]["threads"] += 1
            tf.write(
                json.dumps(
                    {
                        "thread_id": thread.thread_id,
                        "company_id": thread.company_id,
                        "split": split,
                        "groups": groups,
                        "policy_summary": s,
                        "cache_only_full": base,
                        "cache_only_admitted_prefix": prefix,
                    }
                )
                + "\n"
            )
            for r in runs:
                rf.write(
                    json.dumps({"thread_id": thread.thread_id, "split": split, **r})
                    + "\n"
                )
    if any(sha(Path(p)) != h for p, h in source_hashes.items()):
        raise ValueError("frozen input changed during evaluation")
    manifest = {
        "policy_sha256": sha(args.policy),
        "profile_sha256": sha(args.profile),
        "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
        "recovery_overhead": "zero",
        "runtime_overhead_tokens": 16000,
        "note": "Synthetic source availability before recovery, not answer quality. Cache-only baseline uses replay cache model with full canonical input; not an OpenCode adapter. Costs are estimated input equivalents, not billing. Replay stops after admission failure; skipped runs are not assessed. Observation ends at last run per thread; future pings unobserved.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps({k: dict(v) for k, v in totals.items()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
