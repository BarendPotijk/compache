"""A policy for when there is nothing to fit on yet.

compache fits a policy to your own conversations. Before any are recorded,
`default_policy` gives one fitted to a default workload instead: the one the
cost calculator opens with. It is a place to start from while a record of your
own traffic builds up, not a substitute for it: a default is one point, and
your workload is somewhere else.

    policy = compache.default_policy("claude-sonnet-5-5")
    client = compache.wrap(client, policy, record="compache-record", sample=0.1)
    ...
    policy = compache.fit("compache-record")      # once there is enough of it
"""

from __future__ import annotations

import math
import random

from .api import Policy, fit
from .record import SIZES_ONLY

__all__ = ["WORKLOAD", "default_policy", "default_workload"]

# The calculator's default workload (site/engine.mjs). Sizes in tokens, delays in minutes.
# `needs` gives, per kind of part, the chance that a turn needs one from an earlier turn
# and how many turns back it lies: median and 95th percentile, as measured on DocChat.
WORKLOAD = {
    "threads": 40,
    "turns": (8, 20),
    "question_tokens": 200,
    "retrieved_tokens": (25_000, 45_000),
    "answer_tokens": 500,
    "system_tokens": 2_048,
    "away_minutes": (4, 60),
    "answer_seconds": 5,
    "needs": {
        "input": (0.15, 1, 20),
        "retrieval": (0.15, 3, 30),
        "output": (0.20, 1, 35),
    },
}
_Z95 = 1.6448536269514722
_START = 1_767_225_600  # 2026-01-01T00:00:00Z


def _lognormal(rng: random.Random, median: float, p95: float) -> float:
    sigma = math.log(p95 / median) / _Z95 if p95 > median else 0.0
    return median * math.exp(sigma * rng.gauss(0, 1))


def default_workload(seed: int = 42, **changes) -> tuple[list[dict], list[dict]]:
    """Threads and labels drawn from the default workload. `changes` replace entries of `WORKLOAD`.

    The threads carry sizes, not text, as a record of `compache.record` does.
    The same seed always gives the same threads.
    """
    given = {**WORKLOAD, **changes}
    rng = random.Random(seed)
    threads, labels = [], []
    for i in range(given["threads"]):
        name = f"default-{i:03d}"
        turns = max(1, min(160, round(_lognormal(rng, *given["turns"]))))
        at, runs = _START + i * 86_400, []
        for n in range(1, turns + 1):
            if n > 1:
                at += 60 * _lognormal(rng, *given["away_minutes"])
            end = at + given["answer_seconds"]
            runs.append(
                {
                    "started_at": round(at, 3),
                    "ended_at": round(end, 3),
                    "input": {"bytes": 4 * given["question_tokens"]},
                    "retrievals": [
                        {
                            "bytes": 4
                            * round(_lognormal(rng, *given["retrieved_tokens"]))
                        }
                    ],
                    "output": {"bytes": 4 * given["answer_tokens"]},
                }
            )
            at = end
            needed = []
            for kind, (chance, back, back_p95) in given["needs"].items():
                if n > 1 and rng.random() < chance:
                    lag = max(1, round(_lognormal(rng, back, back_p95)))
                    if lag < n:
                        suffix = "retrieval:1" if kind == "retrieval" else kind
                        needed.append(f"{name}:{n - lag}:{suffix}")
            labels.append(
                {"thread_id": name, "run": n, "needs": [needed] if needed else []}
            )
        threads.append(
            {
                "thread_id": name,
                "provenance": SIZES_ONLY,
                "system_bytes": 4 * given["system_tokens"],
                "runs": runs,
            }
        )
    return threads, labels


def default_policy(model, *, seed: int = 42, **changes) -> Policy:
    """The policy fitted to the default workload, for `model`.

    `model` is a known model name or a model spec, as for `fit`. `changes`
    replace entries of `WORKLOAD`, for a default that is nearer to what you
    expect, such as `retrieved_tokens=(4_000, 9_000)`.
    """
    threads, labels = default_workload(seed, **changes)
    return fit(threads, labels, model=model)
