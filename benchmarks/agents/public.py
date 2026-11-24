#!/usr/bin/env python3
"""What the agent's policy would have cost on public coding-agent runs, modeled.

    python public.py fetch runs      174 runs of nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0)
    python public.py price runs      fitted on every second one, priced on the others

The runs carry no times, so a pace is declared: eight seconds for a call, three for a tool.
A session puts three runs behind each other, with a user who comes back after a while that is
drawn around a median. Every price is the model's own, every call by the provider's cache rules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import ssl
import time
import urllib.request
from pathlib import Path

import compache
from compache import agent
from compache.models import cache_lifetimes, model_spec, pricing

ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=nebius%2FSWE-rebench-openhands-trajectories"
    "&config=default&split=train&offset={}&length=3"
)
MODELS = ("claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6-luna")
HEADER = 8_000


def fetch(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    for offset in random.Random(7).sample(range(0, 67_000, 5), 60):
        path = folder / f"rows-{offset}.json"
        for attempt in range(5):
            if path.exists():
                break
            try:
                with urllib.request.urlopen(
                    ROWS.format(offset), timeout=90, context=context
                ) as answer:
                    path.write_bytes(answer.read())
            except OSError:
                time.sleep(3 * (attempt + 1))
    print(len(list(folder.glob("rows-*.json"))), "files")


def runs(folder: Path) -> list[dict]:
    found = []
    for path in sorted(folder.glob("rows-*.json")):
        for item in json.loads(path.read_text())["rows"]:
            row = item["row"]
            messages = [
                {"content": "", **{k: v for k, v in m.items() if v is not None}}
                for m in row["trajectory"]
            ]
            found.append({"thread_id": row["trajectory_id"], "messages": messages})
    return found


def sessions(rows: list[dict], tasks: int, median: float, sigma: float = 1.2):
    """Runs behind each other as one conversation, at a declared pace."""
    r, rows, made = random.Random(3), list(rows), []
    r.shuffle(rows)
    for i in range(0, len(rows) - tasks + 1, tasks):
        at, messages = 1_790_000_000.0 + 1e7 * i, []
        for k, row in enumerate(rows[i : i + tasks]):
            if k:
                at += median * math.exp(r.gauss(0, sigma))
            for message in row["messages"]:
                if k and message["role"] == "system":
                    continue
                at += {"assistant": 8, "tool": 3}.get(message["role"], 0)
                one = {**message, "at": at}
                if message.get("tool_calls"):
                    one["tool_calls"] = [
                        {**c, "id": f"{k}-{c['id']}"} for c in message["tool_calls"]
                    ]
                messages.append(one)
        made.append({"thread_id": f"s{i}", "messages": messages})
    return made


def price(folder: Path) -> None:
    rows = runs(folder)
    cases = [("one task", 1, 0)]
    cases += [(f"three tasks, back after {m:g} s", 3, m) for m in (30, 240, 1800)]
    print(f"{len(rows)} runs. Dollars per 1,000 calls on runs the fit has not seen.\n")
    for name, tasks, median in cases:
        threads = compache.load_threads(sessions(rows, tasks, median), model=MODELS[0])
        train, test = threads[::2], threads[1::2]
        for model in MODELS:
            spec = model_spec(model)
            unit = 1000 * pricing(model)["input_price_per_million"] / 1e6
            fitted = agent.fit_agent(
                train,
                spec,
                lifetimes=cache_lifetimes(model),
                header_tokens=HEADER,
            )
            held = agent.replay_agent(fitted, test, header_tokens=HEADER)
            alone = fitted["solve"]["caching_alone"]
            asked = next(
                s
                for s in (spec, *cache_lifetimes(model))
                if s.cache_ttl_seconds == alone["cache_ttl_seconds"]
            )
            cached = agent.replay_agent(
                {
                    **fitted,
                    "recipe": alone["recipe"],
                    "model_spec": asked.to_dict(),
                    "standard_spec": spec.to_dict(),
                },
                test,
                header_tokens=HEADER,
            )
            calls = held["calls"]
            print(
                f"{name:32} {model:18} whole ${held['keep_everything_cost'] / calls * unit:6.2f}  "
                f"cached at its best ${cached['policy_cost'] / calls * unit:6.2f}  "
                f"policy ${held['policy_cost'] / calls * unit:6.2f}  {held['saving']:4.0%} less, "
                f"{held['coverage']:.0%} in view, {held['missed_calls'] / calls:.1%} of calls fetch back, "
                f"{fitted['recipe']['rounds']} rounds, ratio {fitted['recipe']['ratio']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "price"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    {"fetch": fetch, "price": price}[args.command](args.folder)
