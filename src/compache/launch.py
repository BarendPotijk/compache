"""`compache wrap`: run an agent through compache.

    compache wrap claude
    compache wrap codex -- exec "fix the failing test"
    compache wrap -- my-agent --its-own-flags

A proxy is started on a free local port, the agent is started with its base URL
pointing there, and when it ends a line says what was saved. Nothing has to be
set up and no file of the agent's is changed. What it learns is kept in
`~/.compache/<agent>`, sizes and times and no text, and the policy follows from
that (`compache.learn`).

Every agent is told of a base URL in its own way. One that is not known here is
given the variables that SDKs read.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import stats
from .learn import Learner
from .live import Live
from .proxy import DEFAULTS, serve
from .record import Recorder

__all__ = ["AGENTS", "Agent", "home", "main", "prepared"]

# What an SDK is told its base URL with, by the API it speaks.
_VARIABLES = {
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "gemini": ("GOOGLE_GEMINI_BASE_URL",),
}
# What a base URL of that API ends on, and so what is set when none is.
_PATHS = {"anthropic": "", "openai": "/v1", "gemini": ""}


def home() -> Path:
    """Where records and what was learned from them are kept: `COMPACHE_HOME`, or `~/.compache`."""
    return Path(os.environ.get("COMPACHE_HOME") or Path.home() / ".compache")


@dataclass(frozen=True)
class Agent:
    """How one agent is started and told where its model is."""

    command: str
    title: str
    # Changes to its environment and the arguments put ahead of its own, for base URLs by API.
    setup: Callable[[dict, dict], tuple[dict, list]]
    # The APIs whose own base URL variable the agent reads, which then say where requests go on to.
    apis: tuple[str, ...] = ("anthropic", "openai", "gemini")


def _plain(*apis: str):
    """An agent that reads the SDKs' own variables."""

    def setup(bases: dict, env: dict) -> tuple[dict, list]:
        return {name: bases[api] for api in apis for name in _VARIABLES[api]}, []

    return setup


def _claude(bases: dict, env: dict) -> tuple[dict, list]:
    changes = {"ANTHROPIC_BASE_URL": bases["anthropic"]}
    if not env.get("ENABLE_TOOL_SEARCH", "").strip():
        # Behind a base URL of one's own, Claude Code would send every tool's definition up front.
        changes["ENABLE_TOOL_SEARCH"] = "true"
    return changes, []


def _codex_plan(env: dict) -> bool:
    """Whether Codex is signed in to a ChatGPT plan. Only the kind of sign-in is looked at."""
    folder = Path(env.get("CODEX_HOME") or Path(env.get("HOME", "~")) / ".codex")
    try:
        found = json.loads((folder.expanduser() / "auth.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(found, dict):
        return False
    mode = found.get("auth_mode")
    if mode:
        return str(mode).lower() not in ("apikey", "api_key", "api-key")
    return bool(found.get("tokens")) and not found.get("OPENAI_API_KEY")


def _codex(bases: dict, env: dict) -> tuple[dict, list]:
    # Codex reads no variable for this: what it is told on its command line holds for this run only.
    # Its bodies come compressed otherwise, and Python reads that only from 3.14 on.
    said = ["features.enable_request_compression=false"]
    if _codex_plan(env):
        said.append(f"openai_base_url={json.dumps(bases['chatgpt'])}")
    else:
        stored = not env.get("OPENAI_API_KEY")
        said += [
            'model_provider="compache"',
            'model_providers.compache.name="compache"',
            f"model_providers.compache.base_url={json.dumps(bases['openai'])}",
            'model_providers.compache.wire_api="responses"',
            "model_providers.compache.supports_websockets=false",
            # A key that Codex keeps itself is not in the environment.
            "model_providers.compache.requires_openai_auth=true"
            if stored
            else 'model_providers.compache.env_key="OPENAI_API_KEY"',
        ]
    return {}, [word for line in said for word in ("-c", line)]


def _opencode(bases: dict, env: dict) -> tuple[dict, list]:
    try:
        given = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
    except ValueError:
        given = {}
    providers = dict(given.get("provider") or {})
    for name, base in (
        ("anthropic", bases["anthropic"].rstrip("/") + "/v1"),
        ("openai", bases["openai"]),
        ("google", bases["gemini"].rstrip("/") + "/v1beta"),
    ):
        own = dict(providers.get(name) or {})
        own["options"] = {"baseURL": base, **(own.get("options") or {})}
        providers[name] = own
    content = {**given, "provider": providers}
    return {"OPENCODE_CONFIG_CONTENT": json.dumps(content)}, []


def _goose(bases: dict, env: dict) -> tuple[dict, list]:
    hosts = {"OPENAI_HOST": "openai", "ANTHROPIC_HOST": "anthropic"}
    return {name: _origin(bases[api]) for name, api in hosts.items()}, []


AGENTS = {
    "claude": Agent("claude", "Claude Code", _claude, ("anthropic",)),
    "codex": Agent("codex", "Codex", _codex, ("openai",)),
    "opencode": Agent("opencode", "opencode", _opencode),
    "gemini": Agent("gemini", "Gemini CLI", _plain("gemini"), ("gemini",)),
    "qwen": Agent("qwen", "Qwen Code", _plain("openai"), ("openai",)),
    "aider": Agent("aider", "aider", _plain("openai", "anthropic")),
    "goose": Agent("goose", "goose", _goose, ("openai", "anthropic")),
}


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _upstreams(env: dict, apis) -> tuple[dict, dict]:
    """Where each API's requests go on to, and the path its base URL has, by what is set already."""
    upstreams, paths = {}, {}
    for api in DEFAULTS:
        given = next((env[n] for n in _VARIABLES[api] if env.get(n)), None)
        if given and api in apis:
            upstreams[api] = _origin(given)
            paths[api] = urllib.parse.urlsplit(given).path.rstrip("/")
        else:
            paths[api] = _PATHS[api]
    return upstreams, paths


def prepared(agent: Agent, origin: str, env: dict) -> tuple[dict, list, dict]:
    """What to start an agent with, for a proxy at `origin`.

    Returns the changes to its environment, the arguments put ahead of its own
    and where the proxy has to forward to.
    """
    upstreams, paths = _upstreams(env, agent.apis)
    bases = {api: origin + path for api, path in paths.items()}
    # A ChatGPT plan is served at a place of its own, under the same API.
    bases["chatgpt"] = origin + "/backend-api/codex"
    changes, ahead = agent.setup(bases, env)
    if agent.command == "codex" and _codex_plan(env) and "openai" not in upstreams:
        upstreams["openai"] = "https://chatgpt.com"
    return changes, ahead, upstreams


class _Kept(Recorder):
    """A record that also remembers the calls of this session."""

    def __init__(self, path):
        super().__init__(path)
        self.calls: list[dict] = []

    def _append(self, file, row):
        super()._append(file, row)
        if file == "calls.jsonl":
            self.calls.append(row)


def _said(live: Live, calls: list[dict]) -> list[str]:
    """What to say when the agent has ended."""
    found = stats.savings(calls)
    if not found["calls"]:
        return []
    told = f"compache: {found['calls']:,} call{'' if found['calls'] == 1 else 's'}"
    if not found["without"]:
        entries = found["unpriced"].values()
        left = sum(e["left_out_tokens"] for e in entries)
        whole = left + sum(e["input_tokens"] for e in entries)
        share = f"{left / whole:.0%} of the input left out" if whole else "recorded"
        return [f"{told}, {share}. There is no price sheet for this model."]
    spent, without = found["spent"], found["without"]
    if live.learner is not None and found["left_alone"] == 1 and spent == without:
        return [
            f"{told}, ${spent:,.4f}. This session went out as written: one in ten does, "
            "to learn what replies rely on."
        ]
    change = (
        f"{1 - spent / without:.0%} less"
        if without > spent
        else f"{spent / without - 1:.0%} more"
    )
    more = ""
    fetched = sum(e["reopened"] for e in found["models"].values())
    if fetched:
        more = f" {fetched:,} result{'' if fetched == 1 else 's'} fetched back."
    return [
        f"{told}, ${spent:,.4f} instead of ${without:,.4f} with everything in view and cached: "
        f"{change}.{more}"
    ]


def run(
    agent: Agent,
    words: list[str],
    live: Live,
    *,
    port: int = 0,
    upstream: str | None = None,
    env: dict | None = None,
) -> int:
    """Run an agent behind `live` until it ends. Returns its exit status."""
    env = dict(os.environ if env is None else env)
    found = shutil.which(agent.command, path=env.get("PATH"))
    if found is None:
        raise FileNotFoundError(f"{agent.command} is not on the PATH")
    server = serve(live, port=port)
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    changes, ahead, upstreams = prepared(agent, origin, env)
    server.forward(upstream, upstreams)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Ctrl-C belongs to the agent.
    before = signal.getsignal(signal.SIGINT)
    child = subprocess.Popen([found, *ahead, *words], env={**env, **changes})
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        status = child.wait()
    finally:
        signal.signal(signal.SIGINT, before)
        live.drain(5)
        server.shutdown()
        server.server_close()
    return status


def _given(words: list[str]) -> tuple[str, Agent, list[str]]:
    """The agent that a command line names, and its own arguments."""
    words = [w for i, w in enumerate(words) if w != "--" or "--" in words[:i]]
    if not words:
        raise ValueError(
            f"name the agent to run: {', '.join(AGENTS)}, or any command after --"
        )
    name = Path(words[0]).name
    agent = AGENTS.get(name)
    if agent is None or words[0] != name:
        agent = Agent(
            words[0],
            name,
            agent.setup if agent else _plain("anthropic", "openai", "gemini"),
            agent.apis if agent else ("anthropic", "openai", "gemini"),
        )
    return name, agent, words[1:]


def main(args) -> int:
    """`compache wrap`."""
    name, agent, words = _given(list(args.words))
    folder = args.folder or home() / name
    if args.print:
        origin = f"http://127.0.0.1:{args.port or 8787}"
        changes, ahead, upstreams = prepared(agent, origin, dict(os.environ))
        print(
            f"compache proxy --port {args.port or 8787} --learn {folder}"
            + "".join(f" --{api}-url {url}" for api, url in upstreams.items())
        )
        for key, value in changes.items():
            print(f"export {key}='{value}'")
        print(" ".join([agent.command, *(f"'{a}'" for a in ahead)]))
        return 0
    kept = _Kept(folder)
    if args.policy is not None:
        live = Live(args.policy, record=kept)
    elif args.watch:
        live = Live(record=kept)
    else:
        live = Live(learn=Learner(folder))
        live.recorder = kept
    status = run(agent, words, live, port=args.port or 0, upstream=args.upstream)
    if not args.quiet:
        for line in _said(live, kept.calls):
            print(line, file=sys.stderr)
    if live.learner is not None and kept.calls:
        # Fitting takes a few seconds, and the user has the terminal back by then.
        subprocess.Popen(
            [sys.executable, "-m", __package__, "learn", str(folder)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return status
