"""Refit the preserved Fold A policy from the generated synthetic RAG data."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

from verify_fit import require, verify_fit, verify_generated

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source-0.8.0"))
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.header_traffic import (
    _bucket,
    _report,
    fit_shared_header_traffic,
)
from compache.optimizer.workload import DEFAULT_RECIPES, fit_workload
from compache.traces import Thread

HEADER = json.loads((HERE / "synthetic-header.json").read_bytes())
HEADER_TOKENS = json.loads((HERE / "header-token-ids.json").read_bytes())
SPEC = ModelSpec(
    "gpt-6-luna",
    context_tokens=828400,
    cache_ttl_seconds=300,
    cache_min_tokens=1024,
    cache_block_tokens=128,
    read_multiplier=0.1,
    write_multiplier=1.0,
    output_multiplier=5,
    ping_output_tokens=3,
)
# Later than any observation; all fabricated trajectories end in the recorded calendar.
CUTOFF = "2028-01-01T00:00:00Z"


def tokenize(wire):
    key = hashlib.sha256(wire).hexdigest()
    return HEADER_TOKENS[key]


def read_thread(root, tid):
    with gzip.open(root / "traces" / (tid + ".jsonl.gz"), "rt") as f:
        thread = Thread.from_dict(json.load(f))
    with gzip.open(root / "labels" / (tid + ".jsonl.gz"), "rt") as f:
        labels = [json.loads(line) for line in f]
    return thread, labels


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


def company_fit(company, *, corpus, out):
    path = out / f"company-{company:02d}.json"
    require(not path.exists(), f"Choose an empty fit output directory: {path}")
    ts, labels = [], []
    for i in range(1, 11):
        t, ls = read_thread(corpus, f"scaled-{company:02d}-{i:02d}")
        ts.append(t)
        labels.extend(ls)
    f = fit_workload(
        ts,
        labels,
        SPEC,
        observation_end=CUTOFF,
        recipes=DEFAULT_RECIPES,
        retention_floor=0.95,
        max_unknown_change_rate=0,
        header=dict(provider="modeled", **HEADER),
        tokenize=tokenize,
    )
    f["header_at_30"] = fit_shared_header_traffic(
        ts,
        replace(SPEC, cache_ttl_seconds=1800),
        provider="modeled",
        **HEADER,
        tokenize=tokenize,
    )
    write_json(path, f)
    print("Fitted", company, f["demand"]["training_runs"], flush=True)


def combine(arms, minutes):
    # Independently fitted recipe metrics are counts. Recompute eligibility
    # from the pooled numerator and denominator, including the unknown-change gate.
    result = deepcopy(arms[0])
    result.pop("header_at_30")
    result["model_spec"]["cache_ttl_seconds"] = 60 * minutes
    result["training_threads"] = sorted(
        v for arm in arms for v in arm["training_threads"]
    )
    assert len(result["training_threads"]) == len(set(result["training_threads"]))
    positions = {}
    for arm in arms:
        for k, obs in arm["demand"]["by_position"].items():
            target = positions.setdefault(
                k,
                {
                    "response": [],
                    "returns": [],
                    "calls_per_run": [],
                    "no_return": 0,
                    "censored": 0,
                },
            )
            for name in target:
                target[name] += obs[name]
    for k in ("training_threads", "training_runs", "reported_output_runs"):
        result["demand"][k] = sum(arm["demand"][k] for arm in arms)
    result["demand"]["by_position"] = positions
    for i, row in enumerate(result["recipes"]):
        for k in (
            "turns",
            "historical_required",
            "historical_retained",
            "direct_required",
            "direct_retained",
            "unknown_requirements",
            "unknown_runs",
            "unknown_changed_runs",
            "changed_runs",
            "canonical_tokens",
            "projected_tokens",
        ):
            row[k] = sum(a["recipes"][i][k] for a in arms)
        row["coverage"] = row["historical_retained"] / row["historical_required"]
        row["unknown_change_rate"] = (
            row["unknown_changed_runs"] / row["unknown_runs"]
            if row["unknown_runs"]
            else 0
        )
        row["compact_fraction"] = row["projected_tokens"] / row["canonical_tokens"]
        row["individual_eligible"] = (
            row["coverage"] >= row["retention_floor"]
            and row["unknown_change_rate"] <= row["max_unknown_change_rate"]
        )
        row["eligible"] = row["individual_eligible"]
        row["joint_envelope"] = None
        if row["eligible"]:
            # The first shipped recipe is strictly narrower in all fields;
            # both protect exact query matches. Its own score is the joint lower bound.
            chosen = (
                result["recipes"][0] if i and result["recipes"][0]["eligible"] else row
            )
            row["joint_envelope"] = {
                k: chosen[k]
                for k in (
                    "recipe",
                    "coverage",
                    "historical_required",
                    "historical_retained",
                    "unknown_change_rate",
                )
            }
    fk = "header_traffic" if minutes == 5 else "header_at_30"
    headers = [a[fk] for a in arms]
    header = deepcopy(headers[0])
    header["by_scope"] = {k: v for h in headers for k, v in h["by_scope"].items()}
    for k in ("training_threads", "reported_input_runs", "estimated_input_runs"):
        header[k] = sum(h[k] for h in headers)
    counts = {
        cohort: {k: sum(h[cohort][k] for h in headers) for k in _bucket()}
        for cohort in ("first_runs", "cold_returns", "warm_returns")
    }
    header.update(_report(counts))
    result["header_traffic"] = header
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--company", type=int, choices=range(1, 21),
                        help="Fit one training company as a smoke test")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")

    corpus, out = args.corpus.resolve(), args.out.resolve()
    require(out != HERE and not out.is_relative_to(HERE), "Choose an output outside the evidence folder")
    require(out != corpus and not out.is_relative_to(corpus), "Choose a separate fit output directory")
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            "Use a new or empty fit output directory")
    proof = json.loads((HERE / "proof.json").read_bytes())
    verify_fit(proof)
    verify_generated(corpus, proof)
    fit = partial(company_fit, corpus=corpus, out=out)
    first, last = proof["fit_company_range_inclusive"]
    companies = [args.company] if args.company else list(range(first, last + 1))
    if args.workers == 1:
        for company in companies:
            fit(company)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fit, companies))
    if args.company is not None:
        return
    arms = [json.loads((out / f"company-{n:02d}.json").read_bytes()) for n in companies]
    for minutes in (5, 30):
        path = out / f"policy-a-{minutes}m.json"
        write_json(path, combine(arms, minutes))
        require(hashlib.sha256(path.read_bytes()).hexdigest() == proof["policy_json_sha256"][str(minutes)],
                f"Regenerated policy differs: {path}")
        print("Reproduced", path.name, flush=True)


if __name__ == "__main__":
    main()
