#!/usr/bin/env python3
"""Compare compache with other tools on DocChat's held-out conversations: what each would cost.

  shrink  run a compressor on every turn's retrieved text and keep what is left of it, as sizes
  price   price each tool's requests under the provider's cache rules, and write the figure's data

A compressor runs in an environment of its own, so `shrink` needs only that and a DocChat build:

  python run.py shrink --tool headroom   --data docchat-mixed --output headroom.json
  python run.py shrink --tool kompressor --data docchat-mixed --output kompressor.json --binary ./kompressor
  python run.py price  --data docchat-fast docchat-mixed docchat-slow --sizes headroom.json kompressor.json \
      --output docs/assets/data/compare.json

No model is called. The text of a conversation is the same in every build, so one build is enough to shrink.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def held_out(root):
    """The held-out conversations of a build: for each its ID and its turns."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for entry in manifest["threads"]:
        if entry["split"] == "test":
            with gzip.open(root / entry["trace_path"], "rt") as lines:
                thread = json.loads(lines.readline())
            yield thread["thread_id"], thread["runs"]


def found(run, shape):
    """What a turn's search returned, as one text or as the JSON many search tools return."""
    passages = [p["text"] for p in run["retrievals"]]
    if shape == "json":
        return json.dumps([{"id": i, "text": p} for i, p in enumerate(passages, 1)])
    return "\n\n".join(passages)


# -- what a compressor leaves of it ----------------------------------------------


def kompressor(args, shape):
    """OptScale Kompressor's `compress` command, with the settings its packaged configuration has for Claude."""
    settings = {
        "PATH": "/usr/bin:/bin",
        "HOME": args.home,
        "KOMPRESSOR__ENGINE__MIN_ELISION_SAVING_CHARS": "4000",
        "KOMPRESSOR__ENGINE__ENABLE_CODE_AWARE": "false",
    }

    def one(text):
        done = subprocess.run(
            [args.binary, "compress", "--json", "-"],
            input=text.encode(),
            capture_output=True,
            env=settings,
            check=True,
        )
        answer = json.loads(done.stdout)
        return answer["chars_in"], answer["chars_out"]

    rows = {}
    with ThreadPoolExecutor(8) as pool:
        for name, runs in held_out(args.data):
            rows[name] = list(pool.map(one, [found(r, shape) for r in runs]))
    return rows


def headroom(args, shape):
    """Headroom's `compress()` in process, turn by turn.

    The application keeps what it returned as its history and says how many messages the request
    before held, as Headroom's documentation asks of one that runs its own conversation loop.
    """
    from headroom import compress

    if args.text_model:
        from headroom.transforms.kompress_compressor import warm_kompress_model

        if not warm_kompress_model(device="auto"):
            raise SystemExit("Headroom's text model did not load")
    options = {"target_ratio": args.target_ratio} if args.target_ratio else {}
    rows = {}
    for name, runs in held_out(args.data):
        history, frozen, rows[name] = [], 0, []
        for n, run in enumerate(runs, 1):
            text, question = found(run, shape), run["input"]["text"]
            call = {
                "id": f"call_{n:04d}",
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": json.dumps({"query": question}),
                },
            }
            new = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": text},
            ]
            sent = compress(
                history + new,
                model="claude-sonnet-4-5-20250929",
                frozen_message_count=frozen,
                **options,
            ).messages
            # What was sent before has to go out unchanged, or the cache is lost from there on.
            if sent[: len(history)] != history:
                print(f"{name}: an earlier message was changed at turn {n}")
            rows[name].append((len(text), len(sent[-1]["content"])))
            history = [*sent, {"role": "assistant", "content": run["output"]["text"]}]
            frozen = len(sent)
        before, after = (sum(pair[i] for pair in rows[name]) for i in (0, 1))
        print(name, f"{1 - after / before:.1%} removed", flush=True)
    return rows


def shrink(args):
    tool = {"headroom": headroom, "kompressor": kompressor}[args.tool]
    rows = tool(args, args.shape)
    before = sum(a for turns in rows.values() for a, _ in turns)
    after = sum(b for turns in rows.values() for _, b in turns)
    Path(args.output).write_text(
        json.dumps(
            {
                "name": args.name or args.tool,
                "shape": args.shape,
                "removed": 1 - after / before,
                "left": {
                    name: [round(b / a, 5) if a else 1.0 for a, b in turns]
                    for name, turns in rows.items()
                },
            }
        )
    )
    print(
        f"{args.name or args.tool}, {args.shape}: {1 - after / before:.1%} of the retrieved text removed"
    )


# -- what its requests would cost ---------------------------------------------------


def five_minutes(gap, state):
    """The provider's default: an entry lives five minutes."""
    return gap is not None and gap < 300, 1.25


def adaptive(gap, state):
    """Kompressor's `cache_ttl = "adaptive"`: one-hour entries from a conversation's first pause past five minutes on.

    Its keep-warm requests go out while a tool runs, not while a user is away, so there are none here.
    """
    if not state and gap is not None and gap >= 300:
        state["hour"] = True
        return False, 2.0
    if state:
        return gap < 3600, 2.0
    return five_minutes(gap, state)


def price_whole(threads, spec, system, rule, left=None):
    """Dollars per 1,000 turns of sending the whole history, with what `left` says of each turn's retrieved text."""
    from compache.optimizer.contracts import seconds

    total = turns = 0
    for thread in threads:
        state, cached, last, size = {}, 0, None, spec.text_tokens(system)
        for k, run in enumerate(thread.runs):
            at = seconds(run.generation.started_at)
            retrieved = sum(spec.part_tokens(p.text) for p in run.retrievals)
            if left is not None:
                retrieved = round(retrieved * left[thread.thread_id][k])
            request = size + spec.part_tokens(run.input.text) + retrieved
            alive, write = rule(None if last is None else at - last, state)
            read = cached if alive else 0
            total += (
                read * spec.read_multiplier
                + (request - read) * write
                + spec.text_tokens(run.output.text) * spec.output_multiplier
            )
            turns, cached, last = turns + 1, request, at
            size = request + spec.part_tokens(run.output.text)
    return total / turns * 1000


def price(args):
    import dataclasses
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docchat"))
    import run as docchat

    import compache
    from compache.models import model_spec

    spec = model_spec(args.model)
    per_token = compache.models.pricing(args.model)["input_price_per_million"] / 1e6
    # An entry that lives an hour costs twice the input price to write instead of one and a quarter times.
    hour = dataclasses.replace(spec, cache_ttl_seconds=3600, write_multiplier=2.0)
    sizes = [json.loads(Path(p).read_text()) for p in args.sizes or ()]
    clocks = []
    for root in map(Path, args.data):
        manifest, _, train, train_labels, _ = docchat.load(root, "train")
        _, _, test, labels, _ = docchat.load(root, "test")
        labels = sum(labels, [])
        threads = compache.load_threads(test, model=spec.model_id)
        system = docchat.HEADER["system_prompt"]

        def replay(policy, test=test, labels=labels, **changed):
            held = compache.replay(
                compache.Policy(policy.artifact | changed),
                test,
                labels,
                observed_until=docchat.END,
                share_headers=False,
            )
            return held["policy_cost"] / held["turns"] * 1000 * per_token

        arms, warm = [], []
        for lifetime in (spec, hour):
            policy = compache.fit(
                train,
                sum(train_labels, []),
                model=dataclasses.asdict(lifetime),
                observed_until=docchat.END,
                **docchat.HEADER,
            )
            if lifetime is spec:
                fitted = replay(policy)
            for pings in args.pings:
                cost = replay(policy, recipes=[], planner_options={"max_pings": pings})
                warm.append((cost, lifetime.cache_ttl_seconds, pings))
                print(
                    manifest["clock"]["scenario"],
                    lifetime.cache_ttl_seconds,
                    pings,
                    f"${cost:.2f}",
                    flush=True,
                )
        whole = price_whole(threads, spec, system, five_minutes) * per_token
        # The same thing priced twice: here, and by compache's replay.
        if abs(whole / warm[0][0] - 1) > 0.001:
            raise SystemExit(f"prices disagree: {whole} and {warm[0][0]}")
        arms.append({"name": "Whole history, default cache", "cost": whole})
        for left in sizes:
            rule = adaptive if left["name"].startswith("Kompressor") else five_minutes
            cost = price_whole(threads, spec, system, rule, left["left"]) * per_token
            arms.append({"name": left["name"], "cost": cost})
        best = min(warm)
        arms.append(
            {
                "name": "Whole history, cache kept warm",
                "cost": best[0],
                "cache_seconds": best[1],
                "max_pings": best[2],
            }
        )
        arms.append({"name": "compache", "cost": fitted})
        clocks.append(
            {
                "away_seconds": manifest["clock"]["median_idle_seconds"],
                "turns": sum(len(t.runs) for t in threads),
                "arms": arms,
            }
        )
        data = {
            "model": compache.models.pricing(args.model)["name"],
            "removed": {left["name"]: left["removed"] for left in sizes},
            "clocks": clocks,
        }
        Path(args.output).write_text(json.dumps(data, indent=1) + "\n")
        print(json.dumps(clocks[-1]), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    smaller = commands.add_parser("shrink")
    smaller.add_argument("--tool", required=True, choices=("headroom", "kompressor"))
    smaller.add_argument("--data", required=True, help="A DocChat build")
    smaller.add_argument("--output", required=True)
    smaller.add_argument("--name", help="What the figure calls it")
    smaller.add_argument("--shape", choices=("text", "json"), default="text")
    smaller.add_argument("--binary", help="Kompressor's binary")
    smaller.add_argument("--home", default=os.environ.get("HOME", "/"))
    smaller.add_argument(
        "--text-model", action="store_true", help="Load Headroom's text model first"
    )
    smaller.add_argument("--target-ratio", type=float, help="Headroom's target_ratio")
    priced = commands.add_parser("price")
    priced.add_argument("--data", nargs="+", required=True, help="One build per clock")
    priced.add_argument("--sizes", nargs="*", help="What `shrink` wrote")
    priced.add_argument("--model", default="claude-sonnet-5-5")
    priced.add_argument("--pings", type=int, nargs="+", default=[0, 1, 3, 6, 8])
    priced.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"shrink": shrink, "price": price}[args.command](args)


if __name__ == "__main__":
    main()
