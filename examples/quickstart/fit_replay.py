"""Fit on three complete conversations and evaluate on the fourth, offline."""

import gzip
import json
from pathlib import Path

import compache

HERE = Path(__file__).resolve().parent


def main():
    with gzip.open(HERE / "threads.jsonl.gz", "rt", encoding="utf-8") as stream:
        threads = [json.loads(line) for line in stream]
    with (HERE / "labels.jsonl").open(encoding="utf-8") as stream:
        labels = [json.loads(line) for line in stream]

    train, heldout = threads[:-1], threads[-1:]
    train_ids = {row["thread_id"] for row in train}
    heldout_ids = {row["thread_id"] for row in heldout}
    assert train_ids.isdisjoint(heldout_ids)
    policy = compache.fit(
        train,
        [row for row in labels if row["thread_id"] in train_ids],
        model="claude-sonnet-5-5",
    )
    report = compache.replay(
        policy, heldout, [row for row in labels if row["thread_id"] in heldout_ids]
    )
    print(f"Fit: {len(train)} threads, {policy.summary['turns']} turns")
    print(
        f"Training retention: {policy.summary['coverage']:.1%} "
        f"(minimum {policy.min_retention:.1%})"
    )
    print(f"Held out: {report['threads']} thread, {report['planned_turns']} turns")
    if not report["complete"]:
        raise RuntimeError("Replay did not cover every turn in both arms")
    counts = report["coverage_by_thread"][heldout[0]["thread_id"]]
    print(f"Historical requirements retained: {counts['retained']}/{counts['required']}")
    costs = report["cost_per_1000_turns"]
    print(f"Modeled {costs['unit']} per 1,000 turns, policy: {costs['policy']:.2f}")
    print(
        f"Modeled {costs['unit']} per 1,000 turns, full history: "
        f"{costs['keep_everything']:.2f}"
    )


if __name__ == "__main__":
    main()
