#!/usr/bin/env python3
"""What one sample conversation sends on every turn, with and without the fitted policy.

    python scripts/request_sizes.py     # writes docs/assets/data/requests.json

Fits the quickstart's policy, then replays its 26-turn conversation through a
session, as `examples/quickstart/apply.py` does, and sends the pings it asks for.
Sizes are the package's estimates.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import compache

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "quickstart"
TARGET = ROOT / "docs" / "assets" / "data" / "requests.json"


def seconds(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def requests(policy, recorded):
    session, spec = policy.session("sample"), policy.spec
    lifetime = spec.cache_ttl_seconds
    rows, full, refreshed = [], 0, None
    due = None
    for run in recorded.runs:
        start, end = seconds(run.started_at), seconds(run.ended_at)
        # The pings that fall due before the user is back, each finding the cache warm.
        pings = 0
        while due is not None and due < start:
            ping = session.ping(at=due)
            if ping is None:
                break
            usage = {
                "input_tokens": 0,
                "cache_read_input_tokens": ping.tokens,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            }
            session.ping_sent(at=due, usage=usage)
            pings, refreshed, due = pings + 1, due, session.next_ping_at
        warm = refreshed is not None and start - refreshed < lifetime
        retrieved = [part.text for part in run.retrievals]
        turn = session.prepare(run.input.text, retrieved, at=start)
        due = session.complete(turn, run.output.text, at=end)
        full += sum(spec.part_tokens(text) for text in (run.input.text, *retrieved))
        rows.append(
            {
                "turn": run.run_index,
                "sent": turn.tokens,
                "full": full,
                "archived": len(turn.archived_ids),
                "cache": "first" if refreshed is None else "warm" if warm else "cold",
                "pings_before": pings,
            }
        )
        full += spec.part_tokens(run.output.text)
        refreshed = start
    return rows


def main():
    policy = compache.fit(
        SAMPLE / "threads.jsonl.gz", SAMPLE / "labels.jsonl", model="claude-sonnet-5-5"
    )
    recorded = compache.load_threads(SAMPLE / "threads.jsonl.gz")[1]
    data = {
        "thread_id": recorded.thread_id,
        "model": "Claude Sonnet 5.5",
        "cache_seconds": policy.spec.cache_ttl_seconds,
        "turns": requests(policy, recorded),
    }
    TARGET.write_text(json.dumps(data, indent=1) + "\n", "utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
