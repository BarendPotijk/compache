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
HEADER = 18_000


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


def prompts(rows: list[dict], calls: int, median: float, sigma: float = 1.2):
    """Runs given in prompts of a few calls each, with a user in between: how people work with an agent."""
    r, made = random.Random(5), []
    for i, row in enumerate(rows):
        at, messages, made_calls = 1_790_000_000.0 + 1e7 * i, [], 0
        for message in row["messages"]:
            at += {"assistant": 8, "tool": 3}.get(message["role"], 0)
            if message["role"] == "assistant":
                made_calls += 1
                if made_calls % calls == 0 and messages[-1]["role"] == "tool":
                    said = {"role": "assistant", "content": "That part is done."}
                    messages.append({**said, "at": at})
                    at += median * math.exp(r.gauss(0, sigma))
                    messages.append({"role": "user", "content": "Go on.", "at": at})
            messages.append({**message, "at": at})
        made.append({"thread_id": f"p{i}", "messages": messages})
    return made


def price(folder: Path, output: Path | None) -> None:
    rows = runs(folder)
    cases = [("one task at a stretch", 0, sessions(rows, 1, 0))]
    for median in (30, 240, 1800):
        name = f"three tasks, back after {median} s"
        cases.append((name, median, sessions(rows, 3, median)))
    for median in (30, 240, 900):
        name = f"eight calls a prompt, back after {median} s"
        cases.append((name, median, prompts(rows, 8, median)))
    print(f"{len(rows)} runs. Dollars per 1,000 calls on runs the fit has not seen.\n")
    priced = []
    for name, median, made in cases:
        threads = compache.load_threads(made, model=MODELS[0])
        train, test = threads[::2], threads[1::2]
        for model in MODELS:
            spec, longer = model_spec(model), cache_lifetimes(model)
            fitted = agent.fit_agent(
                train, spec, lifetimes=longer, header_tokens=HEADER
            )
            held = agent.replay_agent(fitted, test, header_tokens=HEADER)
            # Everything kept, at the pings and lifetimes that were cheapest on what was fitted on.
            alone = fitted["solve"]["caching_alone"]
            specs = {s.cache_ttl_seconds: s.to_dict() for s in (spec, *longer)}
            cached = agent.replay_agent(
                {
                    "recipe": alone["recipe"],
                    "model_spec": specs[alone["cache_ttl_seconds"]],
                    "header_spec": specs[alone["header_ttl_seconds"]],
                    "standard_spec": spec.to_dict(),
                },
                test,
                header_tokens=HEADER,
            )
            calls, recipe = held["calls"], fitted["recipe"]
            unit = 1000 * pricing(model)["input_price_per_million"] / 1e6 / calls
            ttl = fitted["model_spec"]["cache_ttl_seconds"]
            row = {
                "case": name,
                "median": median,
                "model": model,
                "calls": calls,
                "whole": held["keep_everything_cost"] * unit,
                "cached": cached["policy_cost"] * unit,
                "policy": held["policy_cost"] * unit,
                "coverage": held["coverage"],
                "fetched_back": held["missed_calls"] / calls,
                **recipe,
                "ttl": ttl,
                "header_ttl": fitted.get("header_spec", {}).get(
                    "cache_ttl_seconds", ttl
                ),
            }
            priced.append(row)
            print(
                f"{name:38} {model:18} whole ${row['whole']:6.2f}  cached ${row['cached']:6.2f}  "
                f"policy ${row['policy']:6.2f}  {1 - row['policy'] / row['whole']:4.0%} less, "
                f"{row['coverage']:.0%} in view, {row['fetched_back']:.1%} of calls fetch back, "
                f"{row['rounds']} rounds, ratio {row['ratio']}, pings {row['tool_pings']}/{row['user_pings']}, "
                f"entries {ttl:.0f} s, header {row['header_ttl']:.0f} s"
            )
    if output is not None:
        shown = "claude-sonnet-5-5"
        shows = {
            "one task at a stretch": "One task at a stretch",
            "eight calls a prompt, back after 240 s": "A few steps, a 4-minute break",
            "three tasks, back after 240 s": "Three tasks, 4-minute breaks",
        }
        figure = [
            {**r, "heading": shows[r["case"]]}
            for name in shows
            for r in priced
            if r["model"] == shown and r["case"] == name
        ]
        data = {
            "model_name": pricing(shown)["name"],
            "runs": len(rows),
            "header_tokens": HEADER,
            "rows": priced,
            "figure": figure,
        }
        output.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "price"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, help="Where to write the figure's data")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.folder)
    else:
        price(args.folder, args.output)
