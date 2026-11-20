"""Reproduce the Headroom development comparison; no model or provider calls."""
import argparse
import hashlib
import json
from pathlib import Path

from compache.baselines.headroom import HeadroomLossless
from compache.comparison import compare_headroom
from compache.optimizer import ModelSpec, Policy, optimize
from compache.traces import load_threads


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "examples/rag_workload")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    for filename, key in (("threads.jsonl", "threads_sha256"), ("labels.jsonl", "labels_sha256")):
        if hashlib.sha256((args.data / filename).read_bytes()).hexdigest() != manifest[key]:
            raise SystemExit(f"Fixture hash mismatch: {filename}")
    threads = load_threads(args.data / "threads.jsonl")
    labels = [json.loads(line) for line in (args.data / "labels.jsonl").read_text().splitlines()]
    spec = ModelSpec.from_dict(json.loads((args.data / "roomy-model.json").read_text()))
    fitted = optimize(threads, labels, spec, observation_end=manifest["observation_end"],
                      timing_source=manifest["timing_source"])
    if fitted["recommended_policy"] is None:
        raise SystemExit("No admissible frozen Compache policy; comparison cannot proceed")
    policy = Policy.from_dict(fitted["recommended_policy"])
    heldout = set(fitted["split"]["heldout_threads"])
    evaluation = [t for t in threads if t.thread_id in heldout]
    complete = [t for t in evaluation if all(p.text is not None for r in t.runs
                                            for p in (r.input, *r.retrievals, r.output))]
    backend = HeadroomLossless()
    # Structured-output positive control, excluded from workload scores.
    control = [{"role": "user", "content": "Compare the approved cases."},
               {"role": "tool", "tool_call_id": "control", "name": "retrieve_documents",
                "content": json.dumps([{"case_id": i, "status": "approved", "region": "north",
                                        "budget_eur": 10000 + i, "owner": "operations"}
                                       for i in range(200)], indent=2)}]
    control_out, control_receipt = backend.compress(control, frozen=0, context_tokens=spec.context_tokens)
    write(args.output / "positive-control.json", {
        "scope": "Synthetic structured-output plumbing control; excluded from workload scores.",
        "input_sha256": hashlib.sha256(control[1]["content"].encode()).hexdigest(),
        "output_sha256": hashlib.sha256(control_out[1]["content"].encode()).hexdigest(),
        **control_receipt,
    })
    write(args.output / "fit-report.json", fitted)
    write(args.output / "policy.json", policy.to_dict())
    write(args.output / "profile.json", fitted["profile"])
    scenarios = {}
    for name, cohort in (("heldout-all", evaluation), ("heldout-complete-text", complete)):
        ids = {r.run_id for t in cohort for r in t.runs}
        result = compare_headroom(cohort, [r for r in labels if r["target_run_id"] in ids],
                                  policy, fitted["profile"], backend,
                                  observation_end=manifest["observation_end"])
        write(args.output / f"{name}.json", result)
        scenarios[name] = {
            "thread_ids": [t.thread_id for t in cohort],
            "excluded_unavailable_threads": [t.thread_id for t in evaluation if t not in cohort],
            "compache": result["compache"]["summary"],
            "baselines": {k: {"summary": v["summary"],
                              "compache_saving_fraction": v["compache_saving_fraction"],
                              "comparable_cost": v["comparable_cost"]}
                          for k, v in result["baselines"].items()},
        }
        print(json.dumps({"scenario": name, "threads": len(cohort)}), flush=True)
    write(args.output / "summary.json", {
        "schema_version": "compache-headroom-development-v1", "fixture": manifest,
        "provenance": backend.provenance,
        "scope": "Development replay on the declared held-out split. Corpus previously used in development. Actual Headroom lossless compression; modeled costs, frozen answers, zero recovery overhead. No full-proxy or answer-quality comparison.",
        "scenarios": scenarios,
    })


if __name__ == "__main__":
    main()
