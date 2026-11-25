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
_GOOSE = {
    "openai": ("OPENAI_BASE_URL", "OPENAI_HOST"),
    "anthropic": ("ANTHROPIC_HOST",),
}
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
    # For base URLs by API, an environment, and the base URLs the agent would use itself: the changes
    # to the environment, the arguments put ahead of the agent's own, and where its requests go on to,
    # as far as that says more.
    setup: Callable[[dict, dict, dict], tuple[dict, list, dict]]
    # The variables the agent itself takes a base URL from, by API, in its order. What they say is
    # where requests go on to. One that it does not read must not be among them.
    reads: dict
    # The one API it speaks, where it is one.
    speaks: str | None = None
    # The base URLs that its own files name, by API. Its variables come first.
    keeps: Callable[[dict], dict] = lambda env: {}


# -- what agents keep in files -------------------------------------------------------


def _home(env: dict) -> Path:
    return Path(env.get("HOME") or Path.home())


def _config(env: dict) -> Path:
    return Path(env.get("XDG_CONFIG_HOME") or _home(env) / ".config")


def _text(file: Path) -> str:
    try:
        return file.read_text("utf-8", "replace")
    except OSError:
        return ""


def _json(file: Path) -> dict:
    """A JSON file, which may hold comments. Empty where there is none."""
    quoted = r'("(?:\\.|[^"\\])*")'
    text = _text(file)
    text = re.sub(
        quoted + r"|//[^\n]*|/\*.*?\*/", lambda m: m[1] or "", text, flags=re.S
    )
    text = re.sub(quoted + r"|,(?=\s*[}\]])", lambda m: m[1] or "", text)
    try:
        found = json.loads(text or "{}")
    except ValueError:
        raise ValueError(f"{file} could not be read") from None
    return found if isinstance(found, dict) else {}


def _in_env_file(folder: str, api: str, variable: str):
    """An agent that takes variables from the first `.env` file it meets: its own, or one of the project."""

    def keeps(env: dict) -> dict:
        at, home = Path.cwd(), _home(env)
        for place in (at, *at.parents, home):
            for file in (place / folder / ".env", place / ".env"):
                if not file.is_file():
                    continue
                said = re.search(
                    rf"^\s*(?:export\s+)?{variable}\s*=\s*[\"']?([^\"'\s#]+)",
                    _text(file),
                    re.M,
                )
                if said and not env.get(variable) and place != home:
                    # Whether the agent trusts it is not known here.
                    raise ValueError(
                        f"{file} says where the model is. Set {variable} in the environment to that"
                    )
                return {api: said[1]} if said else {}
        return {}

    return keeps


def _opencode_keeps(env: dict) -> dict:
    at, own = Path.cwd(), _config(env) / "opencode"
    names = ("opencode.json", "opencode.jsonc")
    files = [own / name for name in ("config.json", *names)]
    files += [Path(env["OPENCODE_CONFIG"])] if env.get("OPENCODE_CONFIG") else []
    # The project's come after the user's, the nearest last.
    files += [p / name for p in (*reversed(at.parents), at) for name in names]
    found: dict = {}
    for file in files:
        providers = _json(file).get("provider")
        for name, api in (
            ("anthropic", "anthropic"),
            ("openai", "openai"),
            ("google", "gemini"),
        ):
            given = (providers or {}).get(name)
            options = given.get("options") if isinstance(given, dict) else None
            base = options.get("baseURL") if isinstance(options, dict) else None
            if isinstance(base, str):
                found[api] = re.sub(
                    r"\{env:([^}]+)\}", lambda m: env.get(m[1], ""), base
                )
    return found


def _goose_keeps(env: dict) -> dict:
    text, found = _text(_config(env) / "goose" / "config.yaml"), {}
    for api, names in _GOOSE.items():
        for name in names:
            said = re.search(rf"^{name}:\s*[\"']?([^\"'\s#]+)", text, re.M)
            if said and api not in found:
                found[api] = said[1]
    return found


def _pi_keeps(env: dict) -> dict:
    folder = env.get("PI_CODING_AGENT_DIR") or _home(env) / ".pi" / "agent"
    providers = _json(Path(folder) / "models.json").get("providers") or {}
    names = {"anthropic": "anthropic", "openai": "openai", "google": "gemini"}
    return {
        api: providers[name]["baseUrl"]
        for name, api in names.items()
        if isinstance(providers.get(name), dict) and providers[name].get("baseUrl")
    }


def _qwen_keeps(env: dict) -> dict:
    found = _in_env_file(".qwen", "openai", "OPENAI_BASE_URL")(env)
    security = _json(_home(env) / ".qwen" / "settings.json").get("security") or {}
    given = (security.get("auth") or {}).get("baseUrl")
    return found or ({"openai": given} if given else {})


def _claude_settings(env: dict) -> Path:
    return Path(env.get("CLAUDE_CONFIG_DIR") or _home(env) / ".claude") / "settings.json"


# -- how agents are told ---------------------------------------------------------------


def _reading(*apis: str) -> dict:
    return {api: _VARIABLES[api] for api in apis}


def _plain(variables: dict):
    """An agent that reads variables such as the SDKs' own."""

    def setup(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
        return {n: bases[api] for api, names in variables.items() for n in names}, [], {}

    return setup


def _any(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    """Whatever it is: told by all of them, but for one of LiteLLM's that is set already."""
    changes, _, _ = _plain(_VARIABLES)(bases, env, given)
    more = {n: bases[api] for api, names in _LITELLM.items() for n in names}
    return {**{n: url for n, url in more.items() if not env.get(n)}, **changes}, [], {}


def _set(env: dict, name: str) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _claude(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    origin, changes, told = _origin(bases["anthropic"]), {}, {}
    ahead: list = []
    if any(_set(env, f"CLAUDE_CODE_USE_{name}") for name in ("BEDROCK", "MANTLE")):
        raise ValueError(
            "Claude Code signs what it sends to AWS, so nothing can be changed in it. compache "
            "carries what it sends to Anthropic, Vertex AI and Foundry"
        )
    if _set(env, "CLAUDE_CODE_USE_VERTEX"):
        region = env.get("CLOUD_ML_REGION") or "global"
        host = "aiplatform.googleapis.com"
        if region in ("us", "eu"):
            host = f"aiplatform.{region}.rep.googleapis.com"
        elif region != "global":
            host = f"{region}-{host}"
        given = env.get("ANTHROPIC_VERTEX_BASE_URL") or f"https://{host}/v1"
        changes["ANTHROPIC_VERTEX_BASE_URL"] = origin + _path(given)
        told["anthropic"] = given
    elif _set(env, "CLAUDE_CODE_USE_FOUNDRY"):
        resource = env.get("ANTHROPIC_FOUNDRY_RESOURCE", "")
        given = (
            env.get("ANTHROPIC_FOUNDRY_BASE_URL")
            or f"https://{resource}.services.ai.azure.com/anthropic"
        )
        changes["ANTHROPIC_FOUNDRY_BASE_URL"] = origin + _path(given)
        if resource:
            # It takes one of the two.
            changes["ANTHROPIC_FOUNDRY_RESOURCE"] = ""
        told["anthropic"] = given
    else:
        changes["ANTHROPIC_BASE_URL"] = base = bases["anthropic"]
        kept = _json(_claude_settings(env)).get("env")
        kept = kept.get("ANTHROPIC_BASE_URL") if isinstance(kept, dict) else None
        if kept:
            # Its own settings come before its environment, and these before those.
            base = origin + _path(kept)
            said = json.dumps({"env": {"ANTHROPIC_BASE_URL": base}})
            ahead, told["anthropic"] = ["--settings", said], kept
        if not env.get("ENABLE_TOOL_SEARCH", "").strip():
            # Behind a base URL of one's own, Claude Code would send every tool's definition up front.
            changes["ENABLE_TOOL_SEARCH"] = "true"
    return changes, ahead, told


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


def _codex(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
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
    if name not in (None, "openai"):
        raise ValueError(
            f"Codex is set to its own provider {name!r}, which does not go through compache. One "
            "with a base_url under [model_providers] does"
        )
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

    def setup(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
        try:
            content = json.loads(env.get(variable) or "{}")
        except ValueError:
            content = {}
        providers = dict(content.get("provider") or {})
        for name, api, version in (
            ("anthropic", "anthropic", "/v1"),
            ("openai", "openai", ""),
            ("google", "gemini", "/v1beta"),
        ):
            # What it uses itself has the version already.
            base = bases[api].rstrip("/") + ("" if given.get(api) else version)
            own = dict(providers.get(name) or {})
            own["options"] = {"baseURL": base, **(own.get("options") or {})}
            providers[name] = own
        return {variable: json.dumps({**content, "provider": providers})}, [], {}

    return setup


def _goose(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    # A host stands without the version, a base URL with it.
    hosts = {"OPENAI_HOST": "openai", "ANTHROPIC_HOST": "anthropic"}
    changes = {name: bases[api].removesuffix("/v1") for name, api in hosts.items()}
    if given.get("openai", "").rstrip("/").endswith("/v1"):
        changes["OPENAI_BASE_URL"] = bases["openai"]
    return changes, [], {}


def _crush(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    for api, (name,) in _CRUSH.items():
        for other in _VARIABLES[api]:
            if env.get(other) and not env.get(name):
                raise ValueError(f"{other} is set. Set {name} to the same for Crush")
    return {name: bases[api] for api, (name,) in _CRUSH.items()}, [], {}


def _copilot(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
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


def _openhands(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    kept = _json(_home(env) / ".openhands" / "agent_settings.json").get("llm") or {}
    own = env.get("LLM_BASE_URL") or kept.get("base_url")
    model = str(env.get("LLM_MODEL") or kept.get("model") or "")
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


def _pi(bases: dict, env: dict, given: dict) -> tuple[dict, list, dict]:
    file = home() / "pi" / "compache.js"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(_PI, "utf-8")
    version = "" if given.get("gemini") else "/v1beta"
    told = {
        "anthropic": bases["anthropic"],
        "openai": bases["openai"],
        "google": bases["gemini"].rstrip("/") + version,
    }
    return {"COMPACHE_BASES": json.dumps(told)}, ["-e", str(file)], {}


AGENTS = {
    "claude": Agent(
        "claude", "Claude Code", _claude, _reading("anthropic"), "anthropic"
    ),
    "codex": Agent("codex", "Codex", _codex, {}, "openai"),
    "opencode": Agent(
        "opencode",
        "opencode",
        _configured("OPENCODE_CONFIG_CONTENT"),
        _reading("openai", "anthropic"),
        keeps=_opencode_keeps,
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
        _in_env_file(".gemini", "gemini", "GOOGLE_GEMINI_BASE_URL"),
    ),
    "qwen": Agent(
        "qwen",
        "Qwen Code",
        _plain(_reading("openai")),
        _reading("openai"),
        "openai",
        _qwen_keeps,
    ),
    "aider": Agent("aider", "aider", _plain(_LITELLM), _LITELLM),
    "goose": Agent("goose", "goose", _goose, _GOOSE, keeps=_goose_keeps),
    "crush": Agent("crush", "Crush", _crush, _CRUSH),
    "copilot": Agent("copilot", "Copilot CLI", _copilot, {}),
    "pi": Agent("pi", "pi", _pi, {}, keeps=_pi_keeps),
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
    given: dict = {}
    for api, names in agent.reads.items():
        said = {n: env[n] for n in names if env.get(n)}
        if len({_origin(url) for url in said.values()}) > 1:
            raise ValueError(f"{' and '.join(said)} say different things")
        given.update({api: url for url in said.values()})
    given = {**agent.keeps(env), **given}
    bases = {
        api: origin + (_path(given[api]) if given.get(api) else _PATHS[api])
        for api in DEFAULTS
    }
    changes, ahead, told = agent.setup(bases, env, given)
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
    changes, ahead, upstreams = prepared(agent, origin, env)
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
