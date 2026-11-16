"""Offline frozen-fixture comparison. No model calls or new answer generation."""

from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def write(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def metrics(result, labels, spec, target):
    records = result["runs"]
    byid = {x["target_run_id"]: x for x in labels}
    denominator = miss = unknown = 0
    for r in records:
        label = byid.get(r["run_id"])
        if not label or label.get("status") != "complete":
            unknown += 1
            continue
        missed = set(r.get("missed_requirements", []))
        for req in label["task_requirements"]:
            if req.get("historical_context_needed") == "yes":
                if not req.get("alternatives"):
                    unknown += 1
                    continue
                denominator += 1
                miss += req["requirement_id"] in missed
            elif req.get("historical_context_needed") == "unknown":
                unknown += 1
    size = [r["tokens_estimated"] / spec.context_tokens for r in records]
    if "historical_requirements_assessed" in result["summary"]:
        denominator = result["summary"]["historical_requirements_assessed"]
        miss = denominator - result["summary"]["historical_requirements_retained"]
    rewrites = [
        r["tokens_estimated"] / spec.context_tokens
        for r in records
        if r.get("prefix_transition") == "rewritten"
    ]
    return {
        "rewrite_count": len(rewrites),
        "rewrites_target_attained": sum(x <= target for x in rewrites),
        "rewrite_context_fraction_min": min(rewrites, default=None),
        "rewrite_context_fraction_max": max(rewrites, default=None),
        "historical_total": denominator,
        "historical_retained": denominator - miss,
        "historical_coverage": (denominator - miss) / denominator
        if denominator
        else None,
        "unknown_historical_requirements": unknown,
        "one_miss_exceeds_5pct": bool(denominator and 1 / denominator > 0.05),
        "mean_context_fraction": sum(size) / len(size) if size else None,
        "min_context_fraction": min(size, default=None),
        "max_context_fraction": max(size, default=None),
        "cold_rewrites": sum(
            r.get("window") == "cold" and r.get("prefix_transition") == "rewritten"
            for r in records
        ),
        "warm_rewrites": sum(
            r.get("window") == "warm" and r.get("prefix_transition") == "rewritten"
            for r in records
        ),
        "deferred": sum(
            "defer" in str(r.get("action", "")) or "defer" in str(r.get("reason", ""))
            for r in records
        ),
        "safety": sum(
            r.get("action") == "replaced_warm_for_admission" for r in records
        ),
        "summary": result["summary"],
    }


def child(args):
    if args.engine == "old":
        from importlib.metadata import version

        if version("compache") != "0.3.0":
            raise RuntimeError("baseline interpreter must contain compache 0.3.0")
    from compache.optimizer import ModelSpec, Policy, replay
    from compache.traces import load_threads

    root = args.source
    data = root / "fixtures"
    manifest = load(data / "manifest.json")
    profile = load(root / "profile.json")
    old = Policy.from_dict(load(root / "policy.json"))
    spec = ModelSpec.from_dict(old.model_spec)
    threads = load_threads(data / "threads.jsonl")
    labels = lines(data / "labels.jsonl")
    lookup = {t.thread_id: t for t in threads}
    train = [lookup[x] for x in manifest["train_thread_ids"]]
    diagnostic = [lookup[x] for x in manifest["heldout_thread_ids"]]
    split = lambda ts: [
        x for x in labels if x["thread_id"] in {t.thread_id for t in ts}
    ]
    end = max(r.ended_at for t in threads for r in t.runs)

    def run(ts, ls, policy):
        result = replay(
            ts,
            ls,
            policy,
            profile,
            observation_end=end,
            recovery_overhead="zero",
            overhead_tokens=16000,
            include_ledger=False,
        )
        return metrics(result, ls, spec, policy.target)

    if args.engine == "old":
        out = {
            "engine": "archived-0.3.0",
            "frozen_policy": old.to_dict(),
            "training": run(train, split(train), old),
            "diagnostic": run(diagnostic, split(diagnostic), old),
        }
    else:
        grid = []
        for target, warm, grace in itertools.product(
            (0.10, 0.15, 0.20, 0.25), (0.50, 0.60), (0, 1, 29)
        ):
            policy = Policy(
                spec.to_dict(),
                old.profile_hash,
                retrieval_grace_runs=grace,
                target=target,
                cold_trigger=target + 0.02,
                warm_trigger=warm,
                reserve=old.reserve,
                max_pings=3,
            )
            m = run(train, split(train), policy)
            grid.append({"policy": policy.to_dict(), "training": m})
        selections = {}
        from compache.optimizer.solve import feasible
        from compache.optimizer.contracts import Constraints

        for threshold in (0.95, 0.90):
            constraints = Constraints(min_historical_dependency_coverage=threshold)
            eligible = [
                x for x in grid if feasible(x["training"]["summary"], constraints)
            ]
            if not eligible:
                selections[str(threshold)] = {
                    "selected": None,
                    "reason": "no feasible candidate",
                }
                continue
            winner = min(
                eligible,
                key=lambda x: (
                    round(x["training"]["summary"]["cost_input_equivalents"], 9),
                    -(
                        x["training"]["historical_coverage"]
                        if x["training"]["historical_coverage"] is not None
                        else 1.0
                    ),
                    -x["policy"]["target"],
                    -x["policy"]["warm_trigger"],
                ),
            )
            selections[str(threshold)] = {
                "selected": winner,
                "diagnostic": run(
                    diagnostic, split(diagnostic), Policy.from_dict(winner["policy"])
                ),
            }
        out = {"engine": "revised", "candidates": grid, "selections": selections}
    write(args.child_output, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--old-python", type=Path)
    p.add_argument("--engine", choices=["old", "new"])
    p.add_argument("--child-output", type=Path)
    args = p.parse_args()
    if args.engine:
        child(args)
        return
    if args.output is None or args.old_python is None:
        p.error("--output and --old-python are required for the parent evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        args.source / "fixtures" / x
        for x in ("threads.jsonl", "labels.jsonl", "manifest.json")
    ] + [args.source / "policy.json", args.source / "profile.json"]
    before = {str(x): sha(x) for x in files}
    write(args.output / "source-hashes.json", before)
    for engine, python in [("old", args.old_python), ("new", Path(sys.executable))]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src") if engine == "new" else ""
        subprocess.run(
            [
                str(python),
                str(HERE),
                "--source",
                str(args.source),
                "--engine",
                engine,
                "--child-output",
                str(args.output / f"{engine}.json"),
            ],
            env=env,
            check=True,
        )
    if before != {str(x): sha(x) for x in files}:
        raise RuntimeError("frozen source changed during evaluation")
    write(
        args.output / "manifest.json",
        {
            "source_hashes": before,
            "default_coverage": 0.95,
            "sensitivity_coverage": 0.90,
            "holdout": "previously used; diagnostic only, not fresh unseen",
            "cost": "offline byte-estimated input equivalents, not measured tokens or billing",
            "new_model_calls": 0,
        },
    )
    (args.output / "README.md").write_text(
        "# Offline cold compaction evaluation\n\nTraining only selected the revised candidate; the previously used held-out conversation is diagnostic, not a fresh unseen test. Estimates use byte-based token and cache models, frozen fixture answers and no new model calls. Historical requirement coverage is a source-availability proxy, not answer quality. The 95% default and 90% sensitivity are evaluated separately; small denominators may require 100% observed retention. See old.json and new.json for candidate metrics, coverage denominators and realized context fractions.\n"
    )


if __name__ == "__main__":
    main()
