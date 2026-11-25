"""`compache doctor`: what is here, and what to run next.

It looks at this machine only: which agents and SDKs are installed, what is set,
what is on record. A request is sent through a proxy of its own to a model of its
own, so that the way through compache is known to work. Keys are named, never read.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import ssl
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata

from . import record
from .launch import AGENTS, home, prepared
from .learn import Learner
from .live import Live
from .proxy import serve

__all__ = ["checked", "main"]

_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)
# What can carry compache in a Python program, and how.
_PACKAGES = {
    "anthropic": "compache.wrap(Anthropic(), learn=...)",
    "openai": "compache.wrap(OpenAI(), learn=...)",
    "google-genai": "compache.wrap(genai.Client(), learn=...)",
    "langchain": "compache.langchain.middleware()",
    "litellm": "compache.litellm.install(live)",
}


def _way_through() -> str | None:
    """Send one request through a proxy to a model made here. None if it came back right."""

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            message = {"role": "assistant", "content": "here"}
            body = json.dumps(
                {"choices": [{"message": message, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    servers = []
    try:
        model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
        servers.append(model)
        proxy = serve(
            Live(), port=0, upstream=f"http://127.0.0.1:{model.server_address[1]}"
        )
        servers.append(proxy)
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        asked = {"model": "m", "messages": [{"role": "user", "content": "there?"}]}
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/v1/chat/completions",
            json.dumps(asked).encode(),
            {"Content-Type": "application/json"},
        )
        # A proxy that is set for the machine is no part of this.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=10) as reply:
            said = json.load(reply)["choices"][0]["message"]["content"]
        return None if said == "here" else f"the reply came back as {said!r}"
    except Exception as error:
        return str(error) or type(error).__name__
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def _running(port: int) -> dict | None:
    """What a proxy on this machine says of itself."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"http://127.0.0.1:{port}/compache/health", timeout=2
        ) as reply:
            found = json.load(reply)
    except Exception:
        return None
    return found if isinstance(found, dict) and "calls" in found else None


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def checked(env: dict | None = None, port: int = 8787) -> list[dict]:
    """Every finding, as its level (`ok`, `note` or `problem`) and what it says."""
    env = dict(os.environ if env is None else env)
    found: list[dict] = []

    def say(level, text):
        found.append({"level": level, "text": text})

    version = _installed("compache") or "from source"
    say("ok", f"compache {version} on Python {platform.python_version()}")
    trouble = _way_through()
    if trouble is None:
        say("ok", "a request goes through a local proxy and back")
    else:
        say("problem", f"a request through a local proxy failed: {trouble}")
    if not ssl.create_default_context().get_ca_certs():
        say(
            "note",
            "Python has no certificates of its own here: the system's are used instead",
        )
    here = [(n, a) for n, a in AGENTS.items() if shutil.which(a.command)]
    for name, agent in here:
        try:
            _, _, upstreams = prepared(agent, "http://127.0.0.1:1", env)
        except ValueError as error:
            say("note", f"{agent.title}: {error}")
            continue
        goes = ", ".join(sorted(set(upstreams.values())))
        more = f", which goes on to {goes}" if goes else ""
        say("ok", f"{agent.title}: compache wrap {name}{more}")
    if not here:
        say(
            "note",
            f"none of {', '.join(AGENTS)} is on the PATH. Any other agent: compache wrap -- <command>",
        )
    for package, how in _PACKAGES.items():
        if _installed(package):
            say("ok", f"{package} {_installed(package)}: {how}")
    keys = [name for name in _KEYS if env.get(name)]
    say("note", f"keys set: {', '.join(keys)}" if keys else "no API key is set here")
    running = _running(port)
    if running is not None:
        say("ok", f"a proxy on port {port} has carried {running['calls']:,} calls")
    for folder in sorted(f.parent for f in home().glob("*/turns.jsonl")):
        try:
            rows, lost = record.load_calls(folder), record.unreadable(folder)
        except OSError as error:
            say("problem", f"{folder} could not be read: {error}")
            continue
        if lost:
            # A process that dies leaves a line unfinished. Nothing that can be read is no record.
            level = "note" if rows else "problem"
            say(level, f"{folder}: {lost:,} lines cannot be read and are passed over")
        last = max((row["at"] for row in rows), default=None)
        when = (
            f", last on {datetime.fromtimestamp(last, timezone.utc).date()}"
            if last
            else ""
        )
        learned = Learner(folder).fitted
        taught = f", policies for {', '.join(learned)}" if learned else ""
        say("ok", f"{folder}: {len(rows):,} calls on record{when}{taught}")
        if last and last > time.time() + 86_400:
            say("problem", f"{folder} holds calls from the future: check the clock")
    return found


def main(args) -> tuple[dict, str]:
    found = checked(port=args.port)
    problems = sum(f["level"] == "problem" for f in found)
    marks = {"ok": "ok      ", "note": "        ", "problem": "problem "}
    lines = [marks[f["level"]] + f["text"] for f in found]
    return {"problems": problems, "findings": found}, "\n".join(lines)
