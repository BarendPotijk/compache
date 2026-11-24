#!/usr/bin/env python3
"""Other tools on live model calls: what the provider billed, and whether the answers hold.

  play    DocChat's held-out conversations through each tool, at their users' pace, and graded
  report  the table, and the figure's data

Every tool meets the same application, the tool loop of Anthropic's guide, and everything it
sends is recorded on its way to the provider:

  whole       the whole history, with the cache mark of the guide
  hour        the same, with a mark that asks for one hour
  headroom    Headroom's compress() in process before every request, at its setting for documents
  kompressor  OptScale Kompressor's proxy, with the settings its package ships for Claude
  compache    compache's proxy, under the policy that `benchmarks/live/run.py fit` wrote
  both        Headroom's compress() in process, and compache's proxy

Run it where headroom-ai[proxy], anthropic and openai are installed. The arguments of
`benchmarks/live/run.py` hold here too. A tool that runs as a process of its own is never given the key.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live"))

import apps  # noqa: E402
import run as harness  # noqa: E402

import compache  # noqa: E402
from compache.live import Live  # noqa: E402
from compache.proxy import serve  # noqa: E402
from compache.record import load_calls  # noqa: E402

ARMS = {
    "whole": "Whole history, default cache",
    "hour": "Whole history, one-hour entries",
    "headroom": "Headroom, target ratio 0.5",
    "kompressor": "Kompressor",
    "compache": "compache",
    "both": "Headroom and compache",
}
# The part of Kompressor 1.6.1's packaged configuration that Claude's requests meet.
KOMPRESSOR = """
[server]
listen = "127.0.0.1:{port}"

[engine]
mode = "recoverable"
max_ref_bytes = 1048576

[refs]
backend = "sqlite"
path = "{state}/refs.db"

[stats]
enabled = true
path = "{state}/stats.db"

[proxy]
retrieve_tool = "auto"
frozen_prefix = "marker"
cache_ttl = "default"

[proxy.providers.anthropic]
retrieve_tool = "always"
frozen_prefix = "first_seen"
cache_ttl = "adaptive"

[proxy.providers.anthropic.engine]
enable_code_aware = false
min_elision_saving_chars = 4000

[[proxy.upstreams]]
name = "anthropic"
path_prefix = "/v1/messages"
base_url = "{upstream}"
api_family = "anthropic"
forward_auth = true
"""


# -- the tools ---------------------------------------------------------------


class HeadroomLoop(apps.ClaudeLoop):
    """The loop with Headroom before every request.

    The application keeps what Headroom returned as its history and says how many messages the
    request before held, as Headroom's documentation asks of one that runs its own conversation loop.
    """

    lock = threading.Lock()
    frozen = 0

    def request(self):
        from headroom import compress

        plain = [
            {
                **message,
                "content": message["content"]
                if isinstance(message["content"], str)
                else [
                    block
                    if isinstance(block, dict)
                    else block.model_dump(exclude_none=True)
                    for block in message["content"]
                ],
            }
            for message in self.messages
        ]
        with self.lock:
            self.messages = compress(
                plain,
                model="claude-sonnet-4-5-20250929",
                frozen_message_count=self.frozen,
                target_ratio=0.5,
            ).messages
        self.frozen = len(self.messages)
        return self.messages


def with_key(base: str, key: str) -> ThreadingHTTPServer:
    """A server that passes requests on to `base` with the key put in."""
    import httpx

    client = httpx.Client(timeout=600)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            kept = {
                name: value
                for name, value in self.headers.items()
                if name.lower().startswith("anthropic-")
            }
            answer = client.post(
                base.rstrip("/") + self.path,
                content=body,
                headers={
                    **kept,
                    "Content-Type": "application/json",
                    "x-api-key": key,
                },
            )
            self.send_response(answer.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer.content)))
            self.end_headers()
            self.wfile.write(answer.content)

    return started(ThreadingHTTPServer(("127.0.0.1", 0), Handler))


def started(server: ThreadingHTTPServer) -> ThreadingHTTPServer:
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def at(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def kompressor(args, upstream: str, folder: Path):
    """Kompressor's proxy in front of `upstream`: its process, and where it listens."""
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = folder / "config.toml"
    config.write_text(
        KOMPRESSOR.format(port=port, state=folder.resolve(), upstream=upstream)
    )
    process = subprocess.Popen(
        [*shlex.split(args.kompressor), "serve", "--proxy", "--config", str(config)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(folder.resolve())},
        stdout=open(folder / "log", "wb"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return process, f"http://127.0.0.1:{port}"
        except OSError:
            if process.poll() is not None:
                break
            time.sleep(0.1)
    raise SystemExit(f"Kompressor did not start: {(folder / 'log').read_text()[-800:]}")


# -- play ----------------------------------------------------------------------


def play(args):
    key, work = harness.secret(args), Path(args.work)
    spend = harness.Spend(work, args.budget)
    test = harness.conversations(
        Path(args.data), "test", args.turns, tiers=tuple(args.tiers)
    )
    model, _ = harness.served(args, "claude")
    marks = harness.extra(args, model)
    hour = {**marks, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    if {"headroom", "both"} & set(args.arms):
        from headroom.transforms.kompress_compressor import warm_kompress_model

        if not warm_kompress_model(device="auto"):
            raise SystemExit("Headroom's text model did not load")
    lives, makers, servers, processes = {}, {}, [], []
    for arm in args.arms:
        folder = work / "play" / arm
        shutil.rmtree(folder, ignore_errors=True)
        ours = arm in ("compache", "both")
        lives[arm] = Live(
            compache.Policy.load(args.policy) if ours else None, record=folder
        )
        upstream, held = args.base, key
        if arm == "kompressor":
            servers.append(with_key(args.base, key))
            upstream, held = at(servers[-1]), "not-a-key"
        recorder = started(serve(lives[arm], port=0, upstream=upstream))
        servers.append(recorder)
        base = at(recorder)
        if arm == "kompressor":
            process, base = kompressor(args, base, work / "kompressor")
            processes.append(process)

        def make(name, system, arm=arm, base=base, held=held):
            client = harness.clients(args, held, base, {"x-compache-thread": name})
            kind = HeadroomLoop if arm in ("headroom", "both") else apps.ClaudeLoop
            return kind(
                client["anthropic"],
                model,
                system,
                extra=hour if arm == "hour" else marks,
            )

        makers[arm] = make
    played, lock = [], threading.Lock()

    def one(job):
        index, conversation, arm = job
        time.sleep(index * args.stagger)
        name = f"{arm}-{conversation['id']}"
        # A line of its own in the system prompt: no arm reads what another arm wrote.
        app = makers[arm](name, f"{conversation['system']} (Session {name}.)")
        turns = harness.play(
            app,
            conversation,
            name,
            lambda earlier, later: time.sleep(
                harness.gap(earlier, later, args.cap) * args.pace
            ),
            spend.check,
        )
        with lock:
            played.append(
                {"arm": arm, "conversation": conversation["id"], "turns": turns}
            )
        print(arm, conversation["id"], len(turns), "turns", flush=True)

    jobs = [(conversation, arm) for conversation in test for arm in lives]
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        list(pool.map(one, [(index, *job) for index, job in enumerate(jobs)]))
    waited = time.time()
    while time.time() - waited < args.tail * args.pace:
        # A policy cannot know that a conversation is over: its last pings are part of what it costs.
        if all(live.next_ping_at() is None for live in lives.values()):
            break
        time.sleep(2)
    for live in lives.values():
        if live.pinger is not None:
            live.pinger.stop()
    for server in servers:
        server.shutdown()
    for process in processes:
        process.terminate()
    by_id = {conversation["id"]: conversation for conversation in test}
    judge = harness.clients(args, key)["openai"]
    harness.graded(played, by_id, judge, args, spend)
    # Arms that were not played this time keep what they had.
    file = work / "played.json"
    before = json.loads(file.read_text()) if file.is_file() else []
    kept = [entry for entry in before if entry["arm"] not in lives]
    file.write_text(json.dumps(kept + played, indent=1))
    print(f"${spend.total():.4f} billed so far")


# -- report --------------------------------------------------------------------


def unlike(first: dict, second: dict) -> tuple[int, int, float]:
    """Turns only the one got right, only the other, and how likely a split as uneven is by chance."""
    both = first.keys() & second.keys()
    a = sum(first[k] and not second[k] for k in both)
    b = sum(second[k] and not first[k] for k in both)
    tail = sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)
    return a, b, min(1.0, 2 * tail)


def report(args):
    work, rows, verdicts = Path(args.work), {}, {}
    played = json.loads((work / "played.json").read_text())
    for arm in ARMS:
        mine = [entry for entry in played if entry["arm"] == arm]
        if not mine:
            continue
        turns = [turn for entry in mine for turn in entry["turns"]]
        verdicts[arm] = {
            (entry["conversation"], turn["run_id"]): turn["correct"]
            for entry in mine
            for turn in entry["turns"]
            if turn.get("correct") is not None
        }
        needing = [
            t["correct"]
            for t in turns
            if t.get("needs_history") == "yes" and t.get("correct") is not None
        ]
        calls = load_calls(work / "play" / arm)
        answered = sum("answer" in turn for turn in turns)
        cost = sum(call.get("cost") or 0 for call in calls)
        rows[arm] = {
            "name": ARMS[arm],
            "turns": answered,
            "errors": sum("error" in turn for turn in turns),
            "cost": round(1000 * cost / max(1, answered), 3),
            "correct": sum(verdicts[arm].values()),
            "graded": len(verdicts[arm]),
            "correct_needing": sum(needing),
            "graded_needing": len(needing),
            "requests": len(calls),
            "tokens": {
                kind: sum((call.get("usage") or {}).get(kind, 0) for call in calls)
                for kind in ("input", "output", "cache_read", "cache_write")
            },
        }
    print(
        f"{'':34}{'turns':>6}{'$/1000':>8}{'correct':>10}{'needing':>9}{'requests':>10}   against whole"
    )
    for arm, row in rows.items():
        split = ""
        if arm != "whole" and "whole" in verdicts:
            lost, won, p = unlike(verdicts["whole"], verdicts[arm])
            row["against_whole"] = {"only_whole": lost, "only_this": won, "p": p}
            saved = 1 - row["cost"] / (rows["whole"]["cost"] or 1)
            split = f"   {saved:.0%} less, -{lost} +{won}, p = {p:.2f}"
        print(
            f"{row['name']:34}{row['turns']:>6}{row['cost']:>8.2f}"
            f"{row['correct']:>6}/{row['graded']:<3}{row['correct_needing']:>5}/{row['graded_needing']:<3}"
            f"{row['requests']:>10}{split}"
        )
    (work / "report.json").write_text(json.dumps(rows, indent=1))
    if args.figure:
        # What `scripts/figures.py tools` draws.
        keys = ("name", "turns", "cost", "correct", "graded")
        shown = {
            "model": compache.models.pricing(args.prices)["name"],
            "conversations": len({entry["conversation"] for entry in played}),
            "arms": [{key: row[key] for key in keys} for row in rows.values()],
        }
        Path(args.figure).write_text(json.dumps(shown, indent=1) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("play", "report"))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--policy", help="The policy of compache's arms")
    parser.add_argument(
        "--kompressor",
        default="kompressor",
        help="The command that runs Kompressor, such as one that shuts it off from the network",
    )
    parser.add_argument("--figure", help="Where report writes the data of the figure")
    mine, rest = parser.parse_known_args(argv)
    # Everything else is what `benchmarks/live/run.py` takes.
    args = harness.parser().parse_args(["apply", *rest])
    for name, value in vars(mine).items():
        setattr(args, name, value)
    {"play": play, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
