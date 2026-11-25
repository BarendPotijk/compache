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
import re
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
import urllib.parse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import stats
from .learn import Learner
from .live import Live
from .proxy import DEFAULTS, serve
from .record import Recorder
from .transcripts import AGENTS as TRANSCRIPTS

__all__ = ["AGENTS", "Agent", "home", "main", "prepared"]

# What an SDK is told its base URL with, by the API it speaks.
_VARIABLES = {
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "gemini": ("GOOGLE_GEMINI_BASE_URL",),
}
# What a base URL of that API ends on, and so what is set when none is.
_PATHS = {"anthropic": "", "openai": "/v1", "gemini": ""}
_GOOSE = {"openai": ("OPENAI_HOST",), "anthropic": ("ANTHROPIC_HOST",)}
_CRUSH = {
    "openai": ("OPENAI_API_ENDPOINT",),
    "anthropic": ("ANTHROPIC_API_ENDPOINT",),
}
# What says that this machine reaches others through a proxy.
_PROXIES = ("http_proxy", "https_proxy", "all_proxy")


def home() -> Path:
    """Where records and what was learned from them are kept: `COMPACHE_HOME`, or `~/.compache`."""
    return Path(os.environ.get("COMPACHE_HOME") or Path.home() / ".compache")


@dataclass(frozen=True)
class Agent:
    """How one agent is started and told where its model is."""

    command: str
    title: str
    # For base URLs by API and an environment: the changes to it, the arguments put ahead of the
    # agent's own, and where its requests go on to, as far as its own settings say.
    setup: Callable[[dict, dict], tuple[dict, list, dict]]
    # The variables the agent itself takes a base URL from, by API. What they say already is where
    # requests go on to.
    reads: dict


def _reading(*apis: str) -> dict:
    return {api: _VARIABLES[api] for api in apis}


def _plain(*apis: str):
    """An agent that reads the SDKs' own variables."""

    def setup(bases: dict, env: dict) -> tuple[dict, list, dict]:
        return {name: bases[api] for api in apis for name in _VARIABLES[api]}, [], {}

    return setup


def _claude(bases: dict, env: dict) -> tuple[dict, list, dict]:
    changes = {"ANTHROPIC_BASE_URL": bases["anthropic"]}
    if not env.get("ENABLE_TOOL_SEARCH", "").strip():
        # Behind a base URL of one's own, Claude Code would send every tool's definition up front.
        changes["ENABLE_TOOL_SEARCH"] = "true"
    return changes, [], {}


def _codex_home(env: dict) -> Path:
    folder = env.get("CODEX_HOME") or Path(env.get("HOME", "~")) / ".codex"
    return Path(folder).expanduser()


def _codex_plan(env: dict) -> bool:
    """Whether Codex is signed in to a ChatGPT plan. Only the kind of sign-in is looked at."""
    try:
        found = json.loads((_codex_home(env) / "auth.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(found, dict):
        return False
    mode = found.get("auth_mode")
    if mode:
        return str(mode).lower() not in ("apikey", "api_key", "api-key")
    return bool(found.get("tokens")) and not found.get("OPENAI_API_KEY")


def _codex_settings(env: dict) -> dict:
    """Codex's own settings, with those of the profile it uses."""
    try:
        found = tomllib.loads((_codex_home(env) / "config.toml").read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    profiles = found.get("profiles")
    named = profiles.get(found.get("profile")) if isinstance(profiles, dict) else None
    return {**found, **(named if isinstance(named, dict) else {})}


def _codex(bases: dict, env: dict) -> tuple[dict, list, dict]:
    # Codex reads no variable for this: what it is told on its command line holds for this run only.
    # Its bodies come compressed otherwise, and Python reads that only from 3.14 on.
    said = ["features.enable_request_compression=false"]
    settings, origin = _codex_settings(env), _origin(bases["openai"])
    providers = settings.get("model_providers")
    name = settings.get("model_provider")
    own = providers.get(name) if isinstance(providers, dict) else None
    given = own.get("base_url") if isinstance(own, dict) else None
    if given and re.fullmatch(r"[A-Za-z0-9_-]+", name):
        # A provider it has been given: everything about it stays, but for where it is.
        base = json.dumps(origin + _path(given))
        said.append(f"model_providers.{name}.base_url={base}")
        said.append(f"model_providers.{name}.supports_websockets=false")
        return {}, [w for line in said for w in ("-c", line)], {"openai": given}
    given = settings.get("openai_base_url")
    if _codex_plan(env):
        # A ChatGPT plan is served at a place of its own, under the same API. Its sign-in goes there
        # and nowhere else, whatever other tools have been told.
        given = given or "https://chatgpt.com/backend-api/codex"
        said.append(f"openai_base_url={json.dumps(origin + _path(given))}")
    else:
        given = given or "https://api.openai.com/v1"
        stored = not env.get("OPENAI_API_KEY")
        said += [
            'model_provider="compache"',
            'model_providers.compache.name="compache"',
            f"model_providers.compache.base_url={json.dumps(origin + _path(given))}",
            'model_providers.compache.wire_api="responses"',
            "model_providers.compache.supports_websockets=false",
            # A key that Codex keeps itself is not in the environment.
            "model_providers.compache.requires_openai_auth=true"
            if stored
            else 'model_providers.compache.env_key="OPENAI_API_KEY"',
        ]
    return {}, [w for line in said for w in ("-c", line)], {"openai": given}


def _configured(variable: str):
    """opencode, or an agent made from it, which takes its configuration from a variable."""

    def setup(bases: dict, env: dict) -> tuple[dict, list, dict]:
        try:
            given = json.loads(env.get(variable) or "{}")
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
        return {variable: json.dumps({**given, "provider": providers})}, [], {}

    return setup


def _goose(bases: dict, env: dict) -> tuple[dict, list, dict]:
    hosts = {"OPENAI_HOST": "openai", "ANTHROPIC_HOST": "anthropic"}
    return {name: _origin(bases[api]) for name, api in hosts.items()}, [], {}


def _crush(bases: dict, env: dict) -> tuple[dict, list, dict]:
    return {names[0]: bases[api] for api, names in _CRUSH.items()}, [], {}


def _copilot(bases: dict, env: dict) -> tuple[dict, list, dict]:
    given = env.get("COPILOT_PROVIDER_BASE_URL")
    if not given:
        raise ValueError(
            "what GitHub hosts does not go through compache. For a model of your own, set "
            "COPILOT_PROVIDER_BASE_URL and COPILOT_MODEL (copilot help providers-env)"
        )
    kind = (env.get("COPILOT_PROVIDER_TYPE") or "openai").lower()
    api = {"anthropic": "anthropic", "google": "gemini"}.get(kind, "openai")
    base = _origin(bases["openai"]) + _path(given)
    changes = {"COPILOT_PROVIDER_BASE_URL": base, "COPILOT_PROVIDER_TRANSPORT": "http"}
    return changes, [], {api: given}


def _openhands(bases: dict, env: dict) -> tuple[dict, list, dict]:
    given = env.get("LLM_BASE_URL")
    # Without the version: what it calls for Claude has it on its own path, and the proxy puts it
    # where it is missing.
    base = _origin(bases["openai"]) + (_path(given) if given else "")
    told = {"openai": given, "anthropic": given} if given else {}
    return {"LLM_BASE_URL": base}, ["--override-with-envs"], told


# What pi loads to be told: its own way of sending a provider's models elsewhere.
_PI = """// Written by compache wrap: what pi asks of these providers goes through it.
export default function (pi) {
  const bases = JSON.parse(process.env.COMPACHE_BASES || "{}");
  for (const [name, baseUrl] of Object.entries(bases)) pi.registerProvider(name, { baseUrl });
}
"""


def _pi(bases: dict, env: dict) -> tuple[dict, list, dict]:
    file = home() / "pi" / "compache.js"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(_PI, "utf-8")
    told = {
        "anthropic": bases["anthropic"],
        "openai": bases["openai"],
        "google": bases["gemini"].rstrip("/") + "/v1beta",
    }
    return {"COMPACHE_BASES": json.dumps(told)}, ["-e", str(file)], {}


AGENTS = {
    "claude": Agent("claude", "Claude Code", _claude, _reading("anthropic")),
    "codex": Agent("codex", "Codex", _codex, {}),
    "opencode": Agent(
        "opencode", "opencode", _configured("OPENCODE_CONFIG_CONTENT"), _VARIABLES
    ),
    "kilo": Agent("kilo", "Kilo", _configured("KILO_CONFIG_CONTENT"), _VARIABLES),
    "gemini": Agent("gemini", "Gemini CLI", _plain("gemini"), _reading("gemini")),
    "qwen": Agent("qwen", "Qwen Code", _plain("openai"), _reading("openai")),
    "aider": Agent(
        "aider",
        "aider",
        _plain("openai", "anthropic"),
        _reading("openai", "anthropic"),
    ),
    "goose": Agent("goose", "goose", _goose, _GOOSE),
    "crush": Agent("crush", "Crush", _crush, _CRUSH),
    "copilot": Agent("copilot", "Copilot CLI", _copilot, {}),
    "pi": Agent("pi", "pi", _pi, {}),
    "openhands": Agent("openhands", "OpenHands", _openhands, {}),
}


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _path(url: str) -> str:
    return urllib.parse.urlsplit(url).path.rstrip("/")


def _reached(env: dict) -> dict:
    """What lets an agent on a machine that reaches others through a proxy reach this one directly."""
    if not any(env.get(n) or env.get(n.upper()) for n in _PROXIES):
        return {}
    given = env.get("no_proxy") or env.get("NO_PROXY") or ""
    ours = ",".join(filter(None, [given, "127.0.0.1", "localhost"]))
    return {"no_proxy": ours, "NO_PROXY": ours}


def prepared(agent: Agent, origin: str, env: dict) -> tuple[dict, list, dict]:
    """What to start an agent with, for a proxy at `origin`.

    Returns the changes to its environment, the arguments put ahead of its own
    and where the proxy has to forward to.
    """
    upstreams, bases = {}, {}
    for api in DEFAULTS:
        names = agent.reads.get(api, ())
        given = next((env[n] for n in names if env.get(n)), None)
        if given:
            upstreams[api] = _origin(given)
        bases[api] = origin + (_path(given) if given else _PATHS[api])
    changes, ahead, told = agent.setup(bases, env)
    upstreams.update({api: _origin(url) for api, url in told.items()})
    return {**changes, **_reached(env)}, ahead, upstreams


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
    known = AGENTS.get(name)
    if known is None:
        known = Agent(words[0], name, _plain(*_VARIABLES), _VARIABLES)
    # A path to a known agent is that agent.
    return name, replace(known, command=words[0]), words[1:]


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
        if live.learner is not None and kept.calls and not live.learner.fitted:
            if name in TRANSCRIPTS:
                print(
                    f"compache: to start from your past sessions, as sizes and times: "
                    f"compache learn {name} --from {name}",
                    file=sys.stderr,
                )
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
