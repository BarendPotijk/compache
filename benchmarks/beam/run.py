#!/usr/bin/env python3
"""Fit a policy on BEAM's training conversations of one size and price it on the test ones.

    python benchmarks/beam/run.py --data beam-mixed --size 100K --ttl 300

BEAM's graded questions are appended to each chat as extra turns. Returns and cost
are therefore fitted and priced on the chat turns alone, while the questions supply
the labels: what must stay in view for them to be answerable.
"""

from __future__ import annotations

import argparse
import json

import compache
from beam import iter_threads

END = "2031-01-01T00:00:00Z"
SYSTEM = "You are a helpful assistant."


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="A build written by beam.py")
    parser.add_argument("--size", default="100K", choices=("100K", "500K", "1M"))
    parser.add_argument(
        "--ttl", type=int, default=300, help="Cache lifetime in seconds"
    )
    args = parser.parse_args(argv)
    spec = {
        "model_id": "replay",
        "context_tokens": 1_000_000,
        "cache_ttl_seconds": args.ttl,
        "cache_min_tokens": 1024,
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "output_multiplier": 5,
    }
    rows = [row for row in iter_threads(args.data) if row[2]["size"] == args.size]

    def chat(thread, entry):
        return thread | {"runs": thread["runs"][: entry["chat_runs"]]}

    train = [row for row in rows if row[2]["split"] == "train"]
    test = [row for row in rows if row[2]["split"] == "test"]
    policy = compache.fit(
        [chat(thread, entry) for thread, _, entry in train],
        [label for _, labels, _ in train for label in labels],
        model=spec,
        system_prompt=SYSTEM,
        observed_until=END,
        share_headers=False,
        retention_threads=compache.load_threads([thread for thread, _, _ in train]),
    )
    fitted = policy.artifact["solve"]
    held = compache.replay(
        policy,
        [chat(thread, entry) for thread, _, entry in test],
        observed_until=END,
        share_headers=False,
    )
    # The questions sit outside the chat turns, so what the cut loses is priced at the training rate.
    recovery = fitted["chosen"]["recovery_cost"] / fitted["turns"] * held["turns"]
    print(policy.describe())
    print(
        json.dumps(
            {
                "size": args.size,
                "cache_lifetime_seconds": args.ttl,
                "pings": policy.max_pings,
                "test_turns": held["turns"],
                "keep_everything_tokens_per_turn": round(
                    held["keep_everything_cost"] / held["turns"]
                ),
                "saving_on_test": round(
                    1
                    - (held["replayed_cost"] + recovery) / held["keep_everything_cost"],
                    3,
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
