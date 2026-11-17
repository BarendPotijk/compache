"""Freeze three preselected, distinct unseen-company cases for paired live runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compache.optimizer.contracts import digest
from compache import __version__
from scaled_rag_dataset import load_thread

SELECTED = ["scaled-26-07", "scaled-24-07", "scaled-29-07"]


def prepare(dataset, reference, policy_path, output):
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    entries = []
    design = {}
    with (
        (fixtures / "threads.jsonl").open("w") as trace,
        (fixtures / "labels.jsonl").open("w") as labels,
    ):
        for tid in SELECTED:
            thread, truth, entry = load_thread(dataset, tid)
            if entry["split"] != "heldout" or entry["turn_count"] != 96:
                raise ValueError(
                    "preselected cases must be 96-turn unseen-company cases"
                )
            for path_key, hash_key in (
                ("trace_path", "trace_sha256"),
                ("labels_path", "labels_sha256"),
            ):
                if (
                    hashlib.sha256((dataset / entry[path_key]).read_bytes()).hexdigest()
                    != entry[hash_key]
                ):
                    raise ValueError("frozen dataset shard changed")
            trace.write(json.dumps(thread, ensure_ascii=False, sort_keys=True) + "\n")
            labels.writelines(
                json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n"
                for label in truth
            )
            entries.append({k: v for k, v in entry.items() if k != "run_design"})
            design[tid] = entry["run_design"]
    fixture_manifest = {
        "schema": "compache-scaled-live-cases-v1",
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "selected_cases": entries,
        "selection": "Company IDs fixed before the first policy evaluation; selected from original retrieval-size strata. They are not extrema of the revised full distribution.",
        "split": "Three distinct unseen-company conversations; not three stochastic repeats of one case.",
    }
    (fixtures / "manifest.json").write_text(
        json.dumps(fixture_manifest, indent=2) + "\n"
    )
    (output / "quality-design.json").write_text(json.dumps(design, indent=2) + "\n")
    shutil.copy2(reference / "profile.json", output / "profile.json")
    shutil.copy2(policy_path, output / "policy.json")
    plan = json.loads((reference / "plan.json").read_text())
    plan.pop("heldout_thread_id", None)
    plan.update(
        schema="compache-scaled-live-comparison-v1",
        evaluation_thread_ids=SELECTED,
        planned_answers_per_arm=96,
        policy_hash=digest(json.loads((output / "policy.json").read_text())),
        profile_hash=digest(json.loads((output / "profile.json").read_text())),
        source_hashes={
            name: hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
            for name in ("threads.jsonl", "labels.jsonl", "manifest.json")
        },
        quality_design_sha256=hashlib.sha256(
            (output / "quality-design.json").read_bytes()
        ).hexdigest(),
        frozen_at=datetime.now(timezone.utc).isoformat(),
        live_timing="Six paired arms launched concurrently; immediate successive calls, no simulated idle sleeps.",
        policy_scope="Original 0.4.0 selected policy and profile, not refitted on expanded corpus or test results.",
        sampling=fixture_manifest["selection"],
        comparison=fixture_manifest["split"],
        stop_rule="A denied Compache admission stops that arm without sending a model request; unexecuted turns remain in the coverage denominator. No full-thread savings claim for incomplete pairs.",
        package_version=__version__,
        token_calibration={
            "safety_margin": 0.1,
            "scope": "Per-arm causal provider-input calibration with the plan's fixed runtime allowance; empirical safeguard, not native tokenization.",
        },
        bind_live_output_dependencies=True,
        all_arms_external_tools_disabled=True,
        fresh_runs=[f"{a}-{i}" for a in ("opencode", "compache") for i in range(1, 4)],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = prepare(a.dataset, a.reference, a.policy, a.output)
    print(
        json.dumps(
            {
                "cases": plan["evaluation_thread_ids"],
                "planned_answers": 576,
                "policy_hash": plan["policy_hash"],
            }
        )
    )
