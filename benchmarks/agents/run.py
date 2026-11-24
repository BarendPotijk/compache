#!/usr/bin/env python3
"""Coding agents on long tasks, with and without a tool between them and the model: cost and grade.

    python run.py play --agent claude --swe swe --session --env .env
    python run.py play --agent codex --tasks 1 2 3 --parts 4 --env .env
    python run.py report

Every arm gets the same agent, the same task (tasks.py) and the same model. What each call
cost is read off the provider's bill on its way back, and the work is graded by hidden tests.

  none        the agent as it is, with its own caching
  compache    compache's proxy as `compache wrap` starts it, learning from nothing
  learned     the same after `compache learn`, on the sessions of the first arm on other tasks
  headroom    Headroom's proxy with its defaults
  kompressor  OptScale Kompressor's proxy with the settings its package ships

The key is put in by the last server on the way, so neither the agent nor a tool is given it.
`--sandbox` is put ahead of every command of theirs.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import shlex
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import swe  # noqa: E402
import tasks  # noqa: E402

from compache import launch, stats  # noqa: E402
from compache.learn import Learner  # noqa: E402
from compache.live import Live  # noqa: E402
from compache.proxy import serve  # noqa: E402
from compache.threads import canonical_usage  # noqa: E402
from compache.transcripts import imported  # noqa: E402

ARMS = ("none", "compache", "learned", "headroom", "kompressor")
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
websocket_relay = true
[proxy.providers.anthropic]
retrieve_tool = "always"
frozen_prefix = "first_seen"
cache_ttl = "adaptive"
[proxy.providers.anthropic.engine]
enable_code_aware = false
min_elision_saving_chars = 4000
[[proxy.upstreams]]
name = "openai"
path_prefix = "/v1"
base_url = "{upstream}"
api_family = "openai"
forward_auth = true
[[proxy.upstreams]]
name = "anthropic"
path_prefix = "/v1/messages"
base_url = "{upstream}"
api_family = "anthropic"
forward_auth = true
"""


# -- the provider ------------------------------------------------------------


def secret(args) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and args.env:
        for line in Path(args.env).read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY in the environment or with --env")
    return key


class Bill:
    """What a run's calls cost, and whether there is money left for more."""

    def __init__(self, budget: float):
        self.budget, self.spent, self.lock = budget, 0.0, threading.Lock()

    def add(self, cost: float) -> None:
        with self.lock:
            self.spent += cost


def provider(key: str, bill: Bill, log: Path) -> ThreadingHTTPServer:
    """The provider as the last tool on the way sees it: puts the key in and reads the bill."""
    context = ssl.create_default_context()
    if not context.get_ca_certs():
        context.load_verify_locations("/etc/ssl/cert.pem")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def handle_one(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if bill.spent >= bill.budget:
                return self._send(402, b'{"error": {"message": "the budget is spent"}}')
            sent = None
            try:
                sent = json.loads(raw) if raw else None
            except ValueError:
                pass
            if isinstance(sent, dict) and "/" not in str(sent.get("model", "/")):
                # A tool may take the gateway's prefix off.
                name = str(sent["model"])
                maker = "anthropic/" if name.startswith("claude") else "openai/"
                sent["model"] = maker + name.replace("-5-5", "-5.5")
                raw = json.dumps(sent).encode()
            path = self.path if self.path.startswith("/v1") else "/v1" + self.path
            dropped = (
                "host",
                "authorization",
                "x-api-key",
                "content-length",
                "connection",
            )
            dropped += ("accept-encoding", "transfer-encoding", "content-encoding")
            head = {k: v for k, v in self.headers.items() if k.lower() not in dropped}
            head |= {
                "Authorization": f"Bearer {key}",
                "Accept-Encoding": "identity",
                "Content-Length": str(len(raw)),
            }
            begun = time.time()
            connection = http.client.HTTPSConnection(
                "openrouter.ai", timeout=300, context=context
            )
            connection.request(
                self.command, "/api" + path, body=raw or None, headers=head
            )
            answer = connection.getresponse()
            streams = "event-stream" in (answer.getheader("Content-Type") or "")
            self.send_response(answer.status)
            held = (
                "content-length",
                "transfer-encoding",
                "connection",
                "content-encoding",
            )
            for name, value in answer.getheaders():
                if name.lower() not in held:
                    self.send_header(name, value)
            got = b""
            if streams:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                gone = False
                while chunk := answer.read1(65536):
                    got += chunk
                    if not gone:
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        except OSError:
                            gone = self.close_connection = True
                if not gone:
                    self.wfile.write(b"0\r\n\r\n")
            else:
                got = answer.read()
                self.send_header("Content-Length", str(len(got)))
                self.end_headers()
                self.wfile.write(got)
            connection.close()
            usage, error = _usage(got, streams), None
            bill.add(usage.get("cost") or 0)
            if answer.status >= 400:
                error = got[:300].decode("utf-8", "replace")
            row = {
                "at": round(begun, 2),
                "seconds": round(time.time() - begun, 2),
                "path": self.path.split("?")[0],
                "status": answer.status,
                "cost": usage.get("cost") or 0,
                "usage": canonical_usage(usage),
                "limit": (sent or {}).get("max_tokens")
                if isinstance(sent, dict)
                else None,
                **({"error": error} if error else {}),
            }
            with bill.lock, open(log, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

        def do_POST(self):
            try:
                self.handle_one()
            except OSError as error:
                self.close_connection = True
                with open(log, "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps({"status": 599, "error": str(error)}) + "\n"
                    )

        do_GET = do_POST

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    return started(Server(("127.0.0.1", 0), Handler))


def _usage(got: bytes, streams: bool) -> dict:
    """The usage of a reply, whole or streamed, in any of the APIs."""
    found: dict = {}

    def look(data):
        for part in (data, data.get("response"), data.get("message")):
            if isinstance(part, dict) and isinstance(part.get("usage"), dict):
                found.update({k: v for k, v in part["usage"].items() if v is not None})

    lines = got.decode("utf-8", "replace").splitlines() if streams else []
    for text in [line[5:] for line in lines if line.startswith("data:")] or [got]:
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            look(data)
    return found


def started(server: ThreadingHTTPServer) -> ThreadingHTTPServer:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def at(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


# -- the tools ----------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def process(command: list[str], port: int, folder: Path, env: dict):
    """A tool that runs on its own, once it listens."""
    log = folder / "tool.log"
    running = subprocess.Popen(
        command,
        env={"PATH": "/usr/bin:/bin", "HOME": str(folder.resolve()), **env},
        stdout=open(log, "wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(900):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return running
        except OSError:
            if running.poll() is not None:
                break
            time.sleep(0.1)
    raise RuntimeError(f"it did not start: {log.read_text()[-800:]}")


def learn_from(args, which: str, learner: Learner) -> None:
    """Fit on the sessions that the agent kept of other tasks, with nothing in between."""
    kept = {"claude": ".claude/projects", "codex": ".codex/sessions"}[args.agent]
    for folder in sorted((Path(args.past or args.work) / "runs").iterdir()):
        sessions = folder / "home" / kept
        if folder.name.startswith(f"{args.agent}-none-") and sessions.is_dir():
            if f"-none-{which}-" not in folder.name:
                imported(learner.folder, args.agent, sessions, days=365)
    learner.refit(wait=True)
    if not learner.fitted:
        raise RuntimeError("too few sessions of other tasks to learn from")


class Between:
    """What stands between the agent and the provider in one arm: where it listens, and how to end it."""

    def __init__(self, arm: str, upstream: str, folder: Path, args, which: str):
        self.base, self.process, self.server, self.live = upstream, None, None, None
        self.folder = folder
        sandbox, port = shlex.split(args.sandbox), free_port()
        if arm in ("compache", "learned"):
            learner = Learner(folder / "compache")
            if arm == "learned":
                learn_from(args, which, learner)
            self.live = Live(learn=learner)
            self.server = started(serve(self.live, port=0, upstream=upstream))
            self.base = at(self.server)
        elif arm == "headroom":
            command = [*shlex.split(args.headroom), "proxy", "--port", str(port)]
            self.process = process(
                [*sandbox, *command, "--no-rate-limit"],
                port,
                folder,
                {
                    "ANTHROPIC_TARGET_API_URL": upstream,
                    "OPENAI_TARGET_API_URL": upstream,
                    "HF_HUB_OFFLINE": "1",
                    "HEADROOM_TELEMETRY": "off",
                    "DO_NOT_TRACK": "1",
                    **({"HF_HOME": args.hf_home} if args.hf_home else {}),
                },
            )
            self.base = f"http://127.0.0.1:{port}"
        elif arm == "kompressor":
            config = folder / "kompressor.toml"
            config.write_text(
                KOMPRESSOR.format(port=port, state=folder.resolve(), upstream=upstream)
            )
            command = shlex.split(args.kompressor)
            self.process = process(
                [*sandbox, *command, "serve", "--proxy", "--config", str(config)],
                port,
                folder,
                {},
            )
            self.base = f"http://127.0.0.1:{port}"

    def end(self) -> dict:
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.server is not None:
            if self.live.pinger is not None:
                self.live.pinger.stop()
            self.server.shutdown()
            self.server.server_close()
            found = stats.savings(self.folder / "compache")
            policies = self.live.learner._policies.values()
            recipes = [asdict(p.recipe) for p in policies if p is not None]
            return {
                "stats": self.live.stats,
                "estimate": found["without"],
                "recipes": recipes,
            }
        return {}


# -- the agents ---------------------------------------------------------------


def command(args, base: str, prompt: str, again: bool, home: Path) -> tuple[list, dict]:
    """An agent's command line and its environment."""
    env = {
        "HOME": str(home),
        "PATH": f"{args.path}:/usr/bin:/bin" if args.path else os.environ["PATH"],
        "TMPDIR": str(home / "tmp"),
        "TERM": "dumb",
        "LANG": "en_US.UTF-8",
        "ANTHROPIC_API_KEY": "none",
        "OPENAI_API_KEY": "none",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if args.agent == "claude":
        model = args.model or "anthropic/claude-haiku-5.5"
        for name in ("MODEL", "SMALL_FAST_MODEL", "DEFAULT_HAIKU_MODEL"):
            env[f"ANTHROPIC_{name}"] = model
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        env["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
        words = ["-p", prompt, "--dangerously-skip-permissions"]
        words += ["--output-format", "json", "--max-turns", str(args.turns)]
        words += ["--continue"] if again else []
    else:
        env["CODEX_HOME"] = str(home / ".codex")
        (home / ".codex").mkdir(parents=True, exist_ok=True)
        words = ["exec", *(["resume", "--last"] if again else [])]
        words += ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
        words += ["-m", args.model or "openai/gpt-6-luna"]
        words += ["-c", "model_reasoning_effort=low", "--json", prompt]
    changes, ahead, _ = launch.prepared(launch.AGENTS[args.agent], base, env)
    return [args.agent, *ahead, *words], {**env, **changes}


# What follows the corrections in a long session.
EXTRA = (
    "Now write a test file `more_<module>.py` for each of {0}: at least ten unittest cases a module, read "
    "off its section in SPEC.md, its edge cases first. Run them, and correct the module where one shows "
    "that it still departs from SPEC.md. Reply with the single word DONE."
)
REVIEW = (
    "Read SPEC.md once more, and for each of {0} say in one line which edge case of its section is the "
    "easiest to get wrong. Run `python3 -m unittest -v` and correct what fails."
)
GUIDE = (
    "Write docs/<module>.md for each of {0}: every rule of its section with an example call and what it "
    "returns. Run every example and print what it returns, and correct the page where it says otherwise. "
    "Reply with the single word DONE."
)
RANDOM = (
    "For each of {0}, write a file `random_<module>.py` that checks it on 200 random inputs against a "
    "second, plain implementation of its section that you write in the test file. Run them, and where the "
    "two differ, read the section again and correct the one that is wrong. Reply with the single word DONE."
)


class Repository:
    """A generated task (tasks.py), in one prompt, in several, or as a long session."""

    def __init__(self, seed: int, modules: int, long: bool = False):
        self.name, self.seed, self.modules = str(seed), seed, modules
        self.long = long
        self.hidden: dict = {}

    def make(self, tree: Path) -> None:
        self.hidden = tasks.make(tree, self.seed, self.modules)

    def prompts(self, parts: int) -> list[str]:
        names = sorted(self.hidden)
        if self.long:
            parts = -(-len(names) // 3)
        if parts == 1:
            return [tasks.PROMPT]
        size = -(-len(names) // parts)
        groups = [names[i : i + size] for i in range(0, len(names), size)]
        first = tasks.PROMPT.replace(
            "When every module does",
            "Leave the others for later. When every one of these does",
        )
        said = [f"{first} Start with {', '.join(groups[0])}."]
        said += [f"Now go on with {', '.join(g)}." for g in groups[1:]]
        if self.long:
            # The same modules again, from other sides.
            for ask in (EXTRA, REVIEW, GUIDE, RANDOM):
                halves = (names[: len(names) // 2], names[len(names) // 2 :])
                said += [ask.format(", ".join(half)) for half in halves]
        return said

    def grade(self, tree: Path) -> dict:
        return tasks.grade(tree, self.hidden)


def task_of(args, name: str):
    if args.swe:
        kind = swe.Session if args.session else swe.Issue
        return kind(Path(args.swe), name)
    return Repository(int(name), args.modules, args.long)


def play_one(args, arm: str, which: str, repeat: int, key: str, bill: Bill) -> dict:
    work, asked = Path(args.work).resolve(), task_of(args, which)
    name = f"{args.agent}-{arm}-{which}-{repeat}" + (
        f"-p{args.parts}" if args.parts > 1 else ""
    )
    folder = work / "runs" / name
    shutil.rmtree(folder, ignore_errors=True)
    home = folder / "home"
    (home / "tmp").mkdir(parents=True)
    task = folder / "task"
    asked.make(task)
    for words in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *words], cwd=task, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "task"],
        cwd=task,
        check=True,
    )
    log = folder / "provider.jsonl"
    server = provider(key, bill, log)
    between = Between(arm, at(server), folder, args, which)
    status, begun, said, pauses = None, time.time(), [], args.pause
    try:
        for k, prompt in enumerate(asked.prompts(args.parts)):
            if k:
                time.sleep(pauses[(k - 1) % len(pauses)])
            words, env = command(args, between.base, prompt, k > 0, home)
            with open(folder / f"out-{k}.txt", "wb") as out:
                running = subprocess.Popen(
                    [*shlex.split(args.sandbox), *words],
                    cwd=task,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    status = running.wait(args.limit)
                except subprocess.TimeoutExpired:
                    os.killpg(running.pid, signal.SIGKILL)
                    status = "timeout"
            said.append(
                (folder / f"out-{k}.txt").read_bytes()[-300:].decode("utf-8", "replace")
            )
    finally:
        told = between.end()
        server.shutdown()
        server.server_close()
    rows = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    billed = [r for r in rows if r.get("usage")]
    result = {
        "agent": args.agent,
        "arm": arm,
        "task": which,
        "repeat": repeat,
        "parts": args.parts,
        "long": args.long,
        "pause": args.pause if args.parts > 1 or args.session or args.long else [],
        "status": status,
        "seconds": round(time.time() - begun, 1),
        "cost": sum(r["cost"] for r in rows if r.get("cost")),
        "calls": len(billed),
        "errors": sum(r["status"] >= 400 for r in rows),
        "input": sum(r["usage"]["input"] for r in billed),
        "cache_read": sum(r["usage"]["cache_read"] for r in billed),
        "cache_write": sum(r["usage"]["cache_write"] for r in billed),
        "output": sum(r["usage"]["output"] for r in billed),
        "largest": max((r["usage"]["input"] for r in billed), default=0),
        **asked.grade(task),
        **told,
        "said": said[-1][-160:] if said else "",
    }
    if args.swe:
        # A tree of sympy is large. What the agent changed in it is enough to keep.
        shutil.rmtree(task, ignore_errors=True)
    with bill.lock, open(work / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    print(
        f"{name}: {result['right']} of {result['of']}, {result['calls']} calls, "
        f"${result['cost']:.4f}, {result['seconds']:.0f} s, {result['status']}, "
        f"{result['errors']} errors, spent ${bill.spent:.3f}",
        flush=True,
    )
    return result


def play(args):
    Path(args.work).mkdir(parents=True, exist_ok=True)
    key, bill = secret(args), Bill(args.budget)
    names = args.tasks or ["1"]
    if args.swe and not args.tasks:
        root = Path(args.swe)
        names = (
            list(json.loads((root / "sessions.json").read_text()))
            if args.session
            else swe.issues(root)
        )
    jobs = [
        (arm, name, repeat)
        for repeat in range(args.repeats)
        for name in names
        for arm in args.arms
    ]
    done = set()
    results = Path(args.work) / "results.jsonl"
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if r["agent"] == args.agent and r["parts"] == args.parts:
                done.add((r["arm"], r["task"], r["repeat"]))
    jobs = [job for job in jobs if job not in done]
    # The arms of one task run one after the other, in an order that changes, so none is always first.
    random.Random(7).shuffle(jobs)
    with ThreadPoolExecutor(args.workers) as pool:
        for made in [pool.submit(play_one, args, *job, key, bill) for job in jobs]:
            try:
                made.result()
            except Exception as error:
                print(f"a run failed: {error!r}", flush=True)


# -- report ---------------------------------------------------------------------


def report(args):
    rows = [
        json.loads(line)
        for line in (Path(args.work) / "results.jsonl").read_text().splitlines()
    ]
    for agent in sorted({r["agent"] for r in rows}):
        for parts in sorted({r["parts"] for r in rows}):
            mine = [r for r in rows if r["agent"] == agent and r["parts"] == parts]
            if not mine:
                continue
            print(f"\n{agent}, {parts} part{'s' if parts > 1 else ''}")
            base = {(r["task"], r["repeat"]): r for r in mine if r["arm"] == "none"}
            print(
                "arm          runs    cost   vs none  right        calls  largest  minutes"
            )
            for arm in ARMS:
                some = [r for r in mine if r["arm"] == arm]
                if not some:
                    continue
                pairs = [
                    (r["cost"], base[r["task"], r["repeat"]]["cost"])
                    for r in some
                    if (r["task"], r["repeat"]) in base
                ]
                change = (
                    f"{sum(a for a, _ in pairs) / sum(b for _, b in pairs) - 1:+.0%}"
                    if pairs
                    else ""
                )
                right, of = sum(r["right"] for r in some), sum(r["of"] for r in some)
                print(
                    f"{arm:12} {len(some):4} ${mean(r['cost'] for r in some):.4f} {change:>8}  "
                    f"{right:3} of {of:3} {right / of:4.0%} {mean(r['calls'] for r in some):6.1f} "
                    f"{mean(r['largest'] for r in some):8,.0f} {mean(r['seconds'] for r in some) / 60:8.1f}"
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("play")
    sub.add_argument("--agent", choices=["claude", "codex"], default="claude")
    sub.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    sub.add_argument("--swe", help="The folder of swe.py, for issues instead of seeds")
    sub.add_argument("--tasks", nargs="+", help="Seeds, or names of issues")
    sub.add_argument("--repeats", type=int, default=1)
    sub.add_argument("--modules", type=int, default=14)
    sub.add_argument("--parts", type=int, default=1, help="Prompts a task is given in")
    sub.add_argument(
        "--session", action="store_true", help="Sessions of several issues"
    )
    sub.add_argument(
        "--long", action="store_true", help="A long session on a generated task"
    )
    sub.add_argument(
        "--pause",
        type=float,
        nargs="+",
        default=[60, 20, 420, 120, 780, 30, 330],
        help="Seconds before each prompt after the first, over and over",
    )
    sub.add_argument("--model")
    sub.add_argument("--env", help="A file of NAME=value lines with OPENROUTER_API_KEY")
    sub.add_argument(
        "--budget", type=float, default=1.0, help="Dollars for this command"
    )
    sub.add_argument("--workers", type=int, default=2)
    sub.add_argument("--turns", type=int, default=150, help="Most turns of an agent")
    sub.add_argument(
        "--limit", type=float, default=900, help="Most seconds of a prompt"
    )
    sub.add_argument("--sandbox", default="", help="Put ahead of what is under test")
    sub.add_argument("--path", default="", help="Where the agents are")
    sub.add_argument("--headroom", default="headroom")
    sub.add_argument("--hf-home", default="")
    sub.add_argument("--kompressor", default="kompressor")
    sub.add_argument("--work", default="agents-out")
    sub.add_argument(
        "--past", help="The folder whose sessions `learned` learns from, if another"
    )
    sub = commands.add_parser("report")
    sub.add_argument("--work", default="agents-out")
    args = parser.parse_args(argv)
    {"play": play, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
