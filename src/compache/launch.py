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
import time
import urllib.parse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import places, stats
from .learn import Learner
from .live import Live
from .proxy import DEFAULTS, serve
from .record import Recorder
from .transcripts import AGENTS as TRANSCRIPTS

__all__ = ["AGENTS", "Agent", "home", "main", "prepared"]

# What an SDK is told its base URL with, by the API it speaks, and LiteLLM beside those.
_VARIABLES = {
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "openai": ("OPENAI_BASE_URL",),
    "gemini": ("GOOGLE_GEMINI_BASE_URL",),
}
_LITELLM = {
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
}
# What a base URL of that API ends on, and so what is set when none is.
_PATHS = {"anthropic": "", "openai": "/v1", "gemini": ""}
_CRUSH = {
    "openai": ("OPENAI_API_ENDPOINT",),
    "anthropic": ("ANTHROPIC_API_ENDPOINT",),
}
# After how many seconds without a model call it is worth saying that there was none.
_WHILE = 60
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
    # For base URLs by API, an environment, the base URLs the agent would use itself and its arguments:
    # the changes to the environment, the arguments put ahead of the agent's own, and where its requests
    # go on to, as far as that says more.
    setup: Callable[[dict, dict, dict, list], tuple[dict, list, dict]]
    # The variables the agent itself takes a base URL from, by API, in its order. What they say is
    # where requests go on to. One that it does not read must not be among them.
    reads: dict
    # The one API it speaks, where it is one.
    speaks: str | None = None
    # What its own files say of where its models are, for an environment and its arguments.
    says: Callable[[dict, list], list] = lambda env, words: []
    # Whether its files come before its variables.
    files_first: bool = False
    # What has to be on disk before it starts.
    needs: Callable[[], None] = lambda: None
    # What may be why it failed without a single model call, for an environment.
    hint: Callable[[dict], str | None] = lambda env: None


# -- how agents are told ---------------------------------------------------------------


def _reading(*apis: str) -> dict:
    return {api: _VARIABLES[api] for api in apis}


def _plain(variables: dict):
    """An agent that reads variables such as the SDKs' own."""

    def setup(
        bases: dict, env: dict, given: dict, words: list
    ) -> tuple[dict, list, dict]:
        return (
            {n: bases[api] for api, names in variables.items() for n in names},
            [],
            {},
        )

    return setup


def _any(bases: dict, env: dict, given: dict, words: list) -> tuple[dict, list, dict]:
    """Whatever it is: told by all of them, but for one of LiteLLM's that is set already."""
    changes, _, _ = _plain(_VARIABLES)(bases, env, given, words)
    more = {n: bases[api] for api, names in _LITELLM.items() for n in names}
    return {**{n: url for n, url in more.items() if not env.get(n)}, **changes}, [], {}


def _claude(
    bases: dict, env: dict, given: dict, words: list
) -> tuple[dict, list, dict]:
    origin, changes, told = _origin(bases["anthropic"]), {}, {}
    env, kept = places.claude_env(env, words)
    uses = places.claude_uses(env)
    if any("AWS" in use or use in ("BEDROCK", "MANTLE") for use in uses):
        raise ValueError(
            "Claude Code signs what it sends to AWS, so nothing can be changed in it. compache "
            "carries what it sends to Anthropic, Vertex AI and Foundry"
        )
    if "ANTHROPIC_GOOGLE_CLOUD" in uses:
        raise ValueError(
            "compache carries what Claude Code sends to Anthropic, Vertex AI and Foundry, and nothing "
            "that goes to Anthropic's own service on Google Cloud"
        )
    if "VERTEX" in uses:
        region = env.get("CLOUD_ML_REGION") or "global"
        host = "aiplatform.googleapis.com"
        if region in ("us", "eu"):
            host = f"aiplatform.{region}.rep.googleapis.com"
        elif region != "global":
            host = f"{region}-{host}"
        base = env.get("ANTHROPIC_VERTEX_BASE_URL") or f"https://{host}/v1"
        changes["ANTHROPIC_VERTEX_BASE_URL"] = origin + _path(base)
        told["anthropic"] = base
    elif "FOUNDRY" in uses:
        resource = env.get("ANTHROPIC_FOUNDRY_RESOURCE", "")
        base = (
            env.get("ANTHROPIC_FOUNDRY_BASE_URL")
            or f"https://{resource}.services.ai.azure.com/anthropic"
        )
        changes["ANTHROPIC_FOUNDRY_BASE_URL"] = origin + _path(base)
        if resource:
            # It takes one of the two.
            changes["ANTHROPIC_FOUNDRY_RESOURCE"] = ""
        told["anthropic"] = base
    else:
        changes["ANTHROPIC_BASE_URL"] = bases["anthropic"]
        if not env.get("ENABLE_TOOL_SEARCH", "").strip():
            # Behind a base URL of one's own, Claude Code would send every tool's definition up front.
            changes["ENABLE_TOOL_SEARCH"] = "true"
    # What is said on its command line comes before all its files.
    over = {name: value for name, value in changes.items() if name in kept}
    ahead = ["--settings", json.dumps({"env": over})] if over else []
    return changes, ahead, told


def _codex(bases: dict, env: dict, given: dict, words: list) -> tuple[dict, list, dict]:
    # Codex reads no variable for this: what it is told on its command line holds for this run only.
    # Its bodies come compressed otherwise, and Python reads that only from 3.14 on.
    said = ["features.enable_request_compression=false"]
    settings, origin = places.codex_settings(env, words), _origin(bases["openai"])
    name = places.url(settings.get("model_provider"))
    given = places.url(places.at(settings, "model_providers", name, "base_url"))
    if given and re.fullmatch(r"[A-Za-z0-9_-]+", name):
        # A provider it has been given: everything about it stays, but for where it is.
        base = json.dumps(origin + _path(given))
        said.append(f"model_providers.{name}.base_url={base}")
        said.append(f"model_providers.{name}.supports_websockets=false")
        return {}, [w for line in said for w in ("-c", line)], {"openai": given}
    if name not in (None, "openai"):
        raise ValueError(
            f"Codex is set to its own provider {name!r}, which does not go through compache. One "
            "with a base_url under [model_providers] does"
        )
    given = places.url(settings.get("openai_base_url"))
    key = places.codex_key(env, words)
    if key == "plan":
        # A ChatGPT plan is served at a place of its own, under the same API. Its sign-in goes there
        # and nowhere else, whatever other tools have been told.
        given = given or "https://chatgpt.com/backend-api/codex"
        said.append(f"openai_base_url={json.dumps(origin + _path(given))}")
    else:
        given = given or "https://api.openai.com/v1"
        said += [
            'model_provider="compache"',
            'model_providers.compache.name="compache"',
            f"model_providers.compache.base_url={json.dumps(origin + _path(given))}",
            'model_providers.compache.wire_api="responses"',
            "model_providers.compache.supports_websockets=false",
            # One that Codex keeps itself comes before one in the environment.
            f"model_providers.compache.env_key={json.dumps(key)}"
            if key in env
            else "model_providers.compache.requires_openai_auth=true",
        ]
    return {}, [w for line in said for w in ("-c", line)], {"openai": given}


def _configured(variable: str):
    """opencode, or an agent made from it, which takes its configuration from a variable."""

    def setup(
        bases: dict, env: dict, given: dict, words: list
    ) -> tuple[dict, list, dict]:
        try:
            content = json.loads(env.get(variable) or "{}")
        except ValueError:
            content = {}
        content = content if isinstance(content, dict) else {}
        providers = content.get("provider")
        providers = dict(providers) if isinstance(providers, dict) else {}
        for name, api, version in (
            ("anthropic", "anthropic", "/v1"),
            ("openai", "openai", ""),
            ("google", "gemini", "/v1beta"),
        ):
            # What it uses itself has the version already.
            base = bases[api].rstrip("/") + ("" if given.get(api) else version)
            own = providers.get(name)
            own = dict(own) if isinstance(own, dict) else {}
            options = own.get("options")
            options = options if isinstance(options, dict) else {}
            own["options"] = {"baseURL": base, **options}
            providers[name] = own
        return {variable: json.dumps({**content, "provider": providers})}, [], {}

    return setup


def _goose(bases: dict, env: dict, given: dict, words: list) -> tuple[dict, list, dict]:
    # A host stands without the version, a base URL with it.
    hosts = {"OPENAI_HOST": "openai", "ANTHROPIC_HOST": "anthropic"}
    changes = {name: bases[api].removesuffix("/v1") for name, api in hosts.items()}
    if given.get("openai", "").rstrip("/").endswith("/v1"):
        changes["OPENAI_BASE_URL"] = bases["openai"]
    return changes, [], {}


def _crush(bases: dict, env: dict, given: dict, words: list) -> tuple[dict, list, dict]:
    for api, (name,) in _CRUSH.items():
        for other in _VARIABLES[api]:
            if env.get(other) and not env.get(name):
                raise ValueError(f"{other} is set. Set {name} to the same for Crush")
    return {name: bases[api] for api, (name,) in _CRUSH.items()}, [], {}


def _copilot(
    bases: dict, env: dict, given: dict, words: list
) -> tuple[dict, list, dict]:
    own = env.get("COPILOT_PROVIDER_BASE_URL")
    if not own:
        raise ValueError(
            "what GitHub hosts does not go through compache. For a model of your own, set "
            "COPILOT_PROVIDER_BASE_URL and COPILOT_MODEL (copilot help providers-env)"
        )
    kind = (env.get("COPILOT_PROVIDER_TYPE") or "openai").lower()
    api = {"anthropic": "anthropic", "google": "gemini"}.get(kind, "openai")
    base = _origin(bases["openai"]) + _path(own)
    changes = {"COPILOT_PROVIDER_BASE_URL": base, "COPILOT_PROVIDER_TRANSPORT": "http"}
    return changes, [], {api: own}


def _openhands(
    bases: dict, env: dict, given: dict, words: list
) -> tuple[dict, list, dict]:
    try:
        kept = places.read(places.home(env) / ".openhands" / "agent_settings.json")
    except places.Unclear:
        kept = {}
    own = env.get("LLM_BASE_URL") or places.url(places.at(kept, "llm", "base_url"))
    model = str(env.get("LLM_MODEL") or places.at(kept, "llm", "model") or "")
    if not own and model.startswith("openrouter/"):
        own = "https://openrouter.ai/api/v1"
    # Without the version: what it calls for Claude has it on its own path, and the proxy puts it
    # where it is missing.
    base = _origin(bases["openai"]) + (_path(own) if own else "")
    told = {"openai": own, "anthropic": own} if own else {}
    return {"LLM_BASE_URL": base}, ["--override-with-envs"], told


# What pi loads to be told: its own way of sending a provider's models elsewhere.
_PI = """// Written by compache wrap: what pi asks of these providers goes through it.
export default function (pi) {
  const bases = JSON.parse(process.env.COMPACHE_BASES || "{}");
  for (const [name, baseUrl] of Object.entries(bases)) pi.registerProvider(name, { baseUrl });
}
"""


def _pi_file() -> Path:
    return home() / "pi" / "compache.js"


def _pi_needs() -> None:
    _pi_file().parent.mkdir(parents=True, exist_ok=True)
    _pi_file().write_text(_PI, "utf-8")


def _pi(bases: dict, env: dict, given: dict, words: list) -> tuple[dict, list, dict]:
    version = "" if given.get("gemini") else "/v1beta"
    told = {
        "anthropic": bases["anthropic"],
        "openai": bases["openai"],
        "google": bases["gemini"].rstrip("/") + version,
    }
    return {"COMPACHE_BASES": json.dumps(told)}, ["-e", str(_pi_file())], {}


AGENTS = {
    "claude": Agent(
        "claude",
        "Claude Code",
        _claude,
        _reading("anthropic"),
        "anthropic",
        places.claude,
        files_first=True,
    ),
    "codex": Agent("codex", "Codex", _codex, {}, "openai"),
    "opencode": Agent(
        "opencode",
        "opencode",
        _configured("OPENCODE_CONFIG_CONTENT"),
        _reading("openai", "anthropic"),
        says=places.opencode,
        files_first=True,
    ),
    "kilo": Agent(
        "kilo",
        "Kilo",
        _configured("KILO_CONFIG_CONTENT"),
        _reading("openai", "anthropic"),
    ),
    "gemini": Agent(
        "gemini",
        "Gemini CLI",
        _plain(_reading("gemini")),
        _reading("gemini"),
        "gemini",
        places.gemini,
        hint=places.gemini_hint,
    ),
    "qwen": Agent(
        "qwen",
        "Qwen Code",
        _plain(_reading("openai")),
        _reading("openai"),
        "openai",
        places.qwen,
    ),
    "aider": Agent("aider", "aider", _plain(_LITELLM), _LITELLM),
    "goose": Agent("goose", "goose", _goose, places.GOOSE, says=places.goose),
    "crush": Agent("crush", "Crush", _crush, _CRUSH),
    "copilot": Agent("copilot", "Copilot CLI", _copilot, {}),
    "pi": Agent("pi", "pi", _pi, {}, says=places.pi, needs=_pi_needs),
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


def prepared(
    agent: Agent, origin: str, env: dict, words=(), decided: bool = False
) -> tuple[dict, list, dict]:
    """What to start an agent with, for a proxy at `origin`. `words` are its own arguments.

    Returns the changes to its environment, the arguments put ahead of its own
    and where the proxy has to forward to. Where that is `decided`, what cannot
    be told from its variables and files does not matter.
    """
    given: dict = {}
    try:
        for api, names in agent.reads.items():
            said = {n: env[n] for n in names if env.get(n)}
            if len({_origin(url) for url in said.values()}) > 1:
                names = " and ".join(said)
                raise places.Unclear(
                    f"{names} say different things: set them to the same"
                )
            given.update({api: url for url in said.values()})
        kept = places.settled(agent.says(env, words), given, agent.files_first)
        given = {**given, **kept} if agent.files_first else {**kept, **given}
    except places.Unclear:
        if not decided:
            raise
        given = {}
    bases = {
        api: origin + (_path(given[api]) if given.get(api) else _PATHS[api])
        for api in DEFAULTS
    }
    changes, ahead, told = agent.setup(bases, env, given, list(words))
    upstreams = {api: _origin(url) for api, url in {**given, **told}.items()}
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
    changes, ahead, upstreams = prepared(
        agent, origin, env, words, decided=upstream is not None
    )
    agent.needs()
    # An agent of one API sends its credentials to that and to no other.
    other = agent.speaks or "openai"
    server.forward(upstream, upstreams, other, careful=True)
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
        known = Agent(words[0], name, _any, _VARIABLES)
    # A path to a known agent is that agent.
    return name, replace(known, command=words[0]), words[1:]


def main(args) -> int:
    """`compache wrap`."""
    name, agent, words = _given(list(args.words))
    folder = args.folder or home() / name
    if args.print:
        origin = f"http://127.0.0.1:{args.port or 8787}"
        decided = args.upstream is not None
        changes, ahead, upstreams = prepared(
            agent, origin, dict(os.environ), words, decided=decided
        )
        agent.needs()
        goes = [f" --{api}-url {url}" for api, url in upstreams.items()]
        goes = [f" --upstream {args.upstream}"] if decided else [*goes, " --careful"]
        print(
            f"compache proxy --port {args.port or 8787} --learn {folder}{''.join(goes)}"
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
    begun = time.monotonic()
    status = run(agent, words, live, port=args.port or 0, upstream=args.upstream)
    hint = agent.hint(dict(os.environ)) if status and not kept.calls else None
    if hint and not args.quiet:
        print(f"compache: {hint}", file=sys.stderr)
    elif not args.quiet and not kept.calls and time.monotonic() - begun > _WHILE:
        print(
            "compache: no model call came through. If there were some, the agent has another place "
            "for them: compache doctor",
            file=sys.stderr,
        )
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
