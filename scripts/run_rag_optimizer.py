"""Reproduce the offline RAG evaluation; no network, API key or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compache.optimizer import ModelSpec, optimize
from compache.traces import load_threads


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/rag_workload",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    for filename, key in (
        ("threads.jsonl", "threads_sha256"),
        ("labels.jsonl", "labels_sha256"),
    ):
        if (
            hashlib.sha256((args.data / filename).read_bytes()).hexdigest()
            != manifest[key]
        ):
            raise SystemExit(f"Fixture hash mismatch: {filename}")
    threads = load_threads(args.data / "threads.jsonl")
    labels = [
        json.loads(line)
        for line in (args.data / "labels.jsonl").read_text().splitlines()
    ]
    # Coverage-only exclusion, declared before optimizing any secondary scenario.
    complete = [
        t
        for t in threads
        if all(
            p.text is not None
            for r in t.runs
            for p in [r.input, *r.retrievals, r.output]
        )
    ]
    excluded = [t.thread_id for t in threads if t not in complete]
    cases = [
        ("company-heldout-pressure", threads, "pressure", manifest["heldout_company"]),
        ("thread-heldout-pressure", threads, "pressure", None),
        ("thread-heldout-roomy", threads, "roomy", None),
        ("complete-text-thread-heldout-pressure", complete, "pressure", None),
    ]
    rows = []
    for name, cohort, model, heldout_company in cases:
        ids = {r.run_id for t in cohort for r in t.runs}
        cohort_labels = [row for row in labels if row["target_run_id"] in ids]
        spec = ModelSpec.from_dict(
            json.loads((args.data / f"{model}-model.json").read_text())
        )
        result = optimize(
            cohort,
            cohort_labels,
            spec,
            observation_end=manifest["observation_end"],
            timing_source=manifest["timing_source"],
            heldout_company=heldout_company,
        )
        directory = args.output / name
        for filename, key in (
            ("profile.json", "profile"),
            ("candidate-policy.json", "policy"),
            ("policy.json", "recommended_policy"),
            ("report.json", None),
        ):
            write(directory / filename, result[key] if key else result)
        (directory / "heldout-ledger.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in result["heldout"]["ledger"]
            )
        )
        baseline_training = next(
            e["training"]
            for e in result["selection"]["candidates"]
            if e["policy"]["retrieval_grace_runs"] is None
            and e["policy"]["max_pings"] == 0
        )
        row = {
            "scenario": name,
            "threads": len(cohort),
            "runs": sum(len(t.runs) for t in cohort),
            "excluded_threads": excluded if cohort is complete else [],
            "training_threads": len(result["split"]["training_threads"]),
            "heldout_threads": len(result["split"]["heldout_threads"]),
            "training_cost_baseline": baseline_training["cost_input_equivalents"],
            "training_cost_candidate": result["selection"]["training"][
                "cost_input_equivalents"
            ],
            "heldout_summary": result["heldout"]["summary"],
            "result": result["result"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "scenario": name,
                    "status": result["result"]["status"],
                    "candidate_saving_fraction": result["result"][
                        "modeled_saving_fraction_vs_cache_only"
                    ],
                }
            ),
            flush=True,
        )
    write(
        args.output / "summary.json",
        {
            "fixture": manifest,
            "scenarios": rows,
            "interpretation": "Development validation on fabricated customer cadence and actual agent answers. Multiple diagnostics, not an untouched external benchmark or production savings. Cost is estimated base-input-token equivalents; Codex overhead excluded. The complete-text diagnostic excludes whole threads solely for unavailable source text.",
        },
    )


if __name__ == "__main__":
    main()
