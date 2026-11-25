"""What agents' own files and arguments say of where their models are.

`compache wrap` sends an agent's requests, and the key in them, on to where the
agent itself would have sent them. Agents read many files for that, in orders
that change. So a file is followed only where nothing speaks against it: where
files say different things, or a project's says what nothing of the user's does,
it is for the user to say.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Said", "Unclear", "settled"]


class Unclear(ValueError):
    """Where an agent's requests go on to is for its user to say."""

    def __init__(self, what: str):
        super().__init__(f"{what}, or say where requests go with --upstream URL")


@dataclass(frozen=True)
class Said:
    """A base URL that a file names for an API."""

    file: Path
    api: str
    base: str
    # Whether the file is the user's own, and none of a project's.
    own: bool = True
    # The variable that says the same.
    name: str = ""

    @property
    def same_as(self) -> str:
        return self.base.rstrip("/")


def asked(file: Path, name: str) -> Unclear:
    """A project's file may come with somebody else's repository. Whether it is trusted is not known here."""
    return Unclear(
        f"{file} says where the model is. If that is yours, set {name} to the same in the environment"
    )


def settled(said: list[Said], given: dict, files_first: bool = False) -> dict:
    """What files say of where the models are, by API, beside what the environment has `given`.

    `files_first` says that they come before that.
    """
    found: dict[str, Said] = {}
    said = [one for one in said if files_first or not given.get(one.api)]
    for one in said:
        if one.own:
            first = found.setdefault(one.api, one)
            if first.same_as != one.same_as:
                raise Unclear(
                    f"{first.file} and {one.file} say different things of where the model is. Let "
                    "them say the same"
                )
    for one in said:
        if not one.own:
            known = found[one.api].base if one.api in found else given.get(one.api)
            if not known or known.rstrip("/") != one.same_as:
                raise asked(one.file, one.name)
    return {api: one.base for api, one in found.items()}


# -- files ---------------------------------------------------------------------------


def home(env: dict) -> Path:
    return Path(env.get("HOME") or Path.home())


def config(env: dict) -> Path:
    return Path(env.get("XDG_CONFIG_HOME") or home(env) / ".config")


def text(file: Path) -> str:
    try:
        return file.read_text("utf-8", "replace")
    except OSError:
        # The agent cannot read it either.
        return ""


def read(file: Path) -> dict:
    """A JSON file, which may hold comments. Empty where there is none."""
    quoted = r'("(?:\\.|[^"\\])*")'
    said = re.sub(
        quoted + r"|//[^\n]*|/\*.*?\*/", lambda m: m[1] or "", text(file), flags=re.S
    )
    said = re.sub(quoted + r"|,(?=\s*[}\]])", lambda m: m[1] or "", said)
    try:
        found = json.loads(said or "{}")
    except ValueError:
        raise Unclear(f"{file} could not be read: correct it") from None
    return found if isinstance(found, dict) else {}


def at(value, *names):
    """What stands at a path in a file that somebody wrote. None where it is something else."""
    for name in names:
        value = value.get(name) if isinstance(value, dict) else None
    return value


def url(value) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _real(path: Path) -> str:
    return os.path.realpath(path)


def _above(place: Path) -> tuple[list[Path], Path | None]:
    """A folder and those above it, up to the root of its repository. That of the main checkout too.

    Without a repository, up to the root of everything.
    """
    found = [place, *place.parents]
    for k, folder in enumerate(found):
        git = folder / ".git"
        if not os.path.lexists(git):
            continue
        if os.path.isfile(git):
            # A worktree: its file names a folder of the main checkout's own.
            named = re.search(r"^gitdir:\s*(.+?)\s*$", text(git), re.M)
            inside = re.split(r"[/\\]\.git[/\\]", named[1])[0] if named else ""
            if inside and os.path.isdir(inside):
                return found[: k + 1], Path(inside)
        return found[: k + 1], None
    return found, None


def _env_line(name: str) -> re.Pattern:
    return re.compile(
        rf"^[ \t]*(?:export[ \t]+)?{name}[ \t]*=[ \t]*[\"'`]?([^\"'`\s#]*)", re.M
    )


def _in_env_file(file: Path, name: str) -> str:
    """A variable of a `.env` file. The last one counts, and one that is left empty says nothing."""
    return ["", *_env_line(name).findall(text(file))][-1]


def _first(folders, names) -> Path | None:
    for folder in folders:
        for name in names:
            if os.path.isfile(folder / name):
                return folder / name
    return None


def toml(file: Path) -> dict:
    try:
        return tomllib.loads(file.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def over(under: dict, more: dict) -> dict:
    """Settings with others laid over them."""
    found = dict(under)
    for name, value in more.items():
        before = found.get(name)
        both = isinstance(before, dict) and isinstance(value, dict)
        found[name] = over(before, value) if both else value
    return found


# -- agents --------------------------------------------------------------------------


def gemini(env: dict, words=()) -> list[Said]:
    """The first `.env` file it meets on its way up, or the user's where it meets none."""
    name, names = "GOOGLE_GEMINI_BASE_URL", (Path(".gemini") / ".env", ".env")
    here = Path.cwd()
    file = _first((here, *here.parents, home(env)), names)
    if file is None:
        return []
    said = _in_env_file(file, name)
    own = _real(file) in {_real(home(env) / n) for n in names}
    return [Said(file, "gemini", said, own, name)] if said else []


def qwen(env: dict, words=()) -> list[Said]:
    """The first `.env` file it meets, the user's, and the settings of both."""
    name, names = "OPENAI_BASE_URL", (Path(".qwen") / ".env", ".env")
    here, mine = Path.cwd(), home(env)
    folder = Path(env.get("QWEN_HOME") or mine / ".qwen")
    own = [folder / ".env", *(mine / n for n in names)]
    met = _first((here, *here.parents), names)
    files = dict.fromkeys([*([met] if met else []), *own])
    said = [(file, _in_env_file(file, name)) for file in files]
    own.append(folder / "settings.json")
    for file in dict.fromkeys((here / ".qwen" / "settings.json", own[-1])):
        found = read(file)
        said.append((file, url(at(found, "env", name))))
        said.append((file, url(at(found, "security", "auth", "baseUrl"))))
    ours = {_real(file) for file in own}
    return [
        Said(file, "openai", base, _real(file) in ours, name)
        for file, base in said
        if base
    ]


def opencode(env: dict, words=()) -> list[Said]:
    names, mine = ("opencode.json", "opencode.jsonc"), home(env)
    folders = [config(env) / "opencode", mine / ".opencode"]
    if env.get("OPENCODE_CONFIG_DIR"):
        folders.append(Path(env["OPENCODE_CONFIG_DIR"]))
    own = [folders[0] / "config.json"]
    own += [folder / name for folder in folders for name in names]
    own += [Path(env["OPENCODE_CONFIG"])] if env.get("OPENCODE_CONFIG") else []
    files = []
    if not truthy(env.get("OPENCODE_DISABLE_PROJECT_CONFIG")):
        for folder in _above(Path.cwd())[0]:
            files += [folder / name for name in names]
            files += [folder / ".opencode" / name for name in names]
    # What lies in the user's home is the user's, whatever way it is met.
    ours = {_real(f) for f in (*own, *(mine / name for name in names))}
    said = []
    for file in dict.fromkeys((*own, *files)):
        providers = read(file).get("provider")
        for named, api, variable in (
            ("anthropic", "anthropic", "ANTHROPIC_BASE_URL"),
            ("openai", "openai", "OPENAI_BASE_URL"),
            ("google", "gemini", "GOOGLE_GEMINI_BASE_URL"),
        ):
            base = url(at(providers, named, "options", "baseURL"))
            base = base or url(at(providers, named, "api"))
            if base and "{file:" in base:
                raise Unclear(f"{file} has where the model is in another file")
            base = re.sub(
                r"\{env:([^}]+)\}", lambda m: env.get(m[1], ""), base or ""
            ).strip()
            if base:
                said.append(Said(file, api, base, _real(file) in ours, variable))
    return said


GOOSE = {
    "openai": ("OPENAI_BASE_URL", "OPENAI_HOST"),
    "anthropic": ("ANTHROPIC_HOST",),
}


def goose(env: dict, words=()) -> list[Said]:
    file = config(env) / "goose" / "config.yaml"
    found, said = text(file), {}
    for api, names in GOOSE.items():
        for name in names:
            line = re.search(rf"^{name}:[ \t]*[\"']?([^\"'\s#]+)", found, re.M)
            if line and api not in said:
                said[api] = Said(file, api, line[1])
    return list(said.values())


def pi(env: dict, words=()) -> list[Said]:
    folder = env.get("PI_CODING_AGENT_DIR") or home(env) / ".pi" / "agent"
    file = Path(folder) / "models.json"
    providers = read(file).get("providers")
    names = {"anthropic": "anthropic", "openai": "openai", "google": "gemini"}
    said = {api: url(at(providers, name, "baseUrl")) for name, api in names.items()}
    return [Said(file, api, base) for api, base in said.items() if base]


# Where Claude Code goes instead of to Anthropic, and what says where that is.
_CLAUDE_USES = {
    "BEDROCK": (),
    "MANTLE": (),
    "ANTHROPIC_AWS": (),
    "ANTHROPIC_GOOGLE_CLOUD": (),
    "VERTEX": ("ANTHROPIC_VERTEX_BASE_URL", "CLOUD_ML_REGION"),
    "FOUNDRY": ("ANTHROPIC_FOUNDRY_BASE_URL", "ANTHROPIC_FOUNDRY_RESOURCE"),
}


def claude_uses(env: dict) -> list[str]:
    return [n for n in _CLAUDE_USES if truthy(env.get(f"CLAUDE_CODE_USE_{n}"))]


def _claude_goes(env: dict) -> dict:
    """What decides where Claude Code's requests go."""
    uses = claude_uses(env)
    names = [name for use in uses for name in _CLAUDE_USES[use]]
    names = names if uses else ["ANTHROPIC_BASE_URL"]
    found = {f"CLAUDE_CODE_USE_{use}": "1" for use in uses}
    return found | {name: env[name] for name in names if env.get(name)}


def _claude_sources(words) -> tuple[set, bool]:
    """Which of its settings files Claude Code reads, and whether that is known."""
    sources = {"user", "project", "local"}
    words = [str(word) for word in words]
    for k, word in enumerate(words):
        if word == "--":
            break
        name, has, value = word.partition("=")
        if name == "--setting-sources":
            value = value if has else "".join(words[k + 1 : k + 2])
            sources = {source.strip() for source in value.split(",")}
        if word == "--restricted":
            sources = set()
    return sources, not {"--bare", "--safe-mode"} & set(words)


def claude_settings(env: dict, strict: bool = True) -> Iterator[tuple[Path, str, dict]]:
    """The variables that Claude Code's settings files set, the user's file first."""
    folder = Path(env.get("CLAUDE_CONFIG_DIR") or home(env) / ".claude")
    above, main = _above(Path.cwd())
    in_repository = os.path.lexists(above[-1] / ".git")
    places = [*(above if in_repository else above[:1]), *([main] if main else [])]
    files = [(folder / "settings.json", "user")]
    for place in places:
        files.append((place / ".claude" / "settings.json", "project"))
        files.append((place / ".claude" / "settings.local.json", "local"))
    seen: set = set()
    for file, source in files:
        if _real(file) in seen:
            continue
        seen.add(_real(file))
        try:
            said = read(file).get("env")
        except Unclear:
            if strict:
                raise
            said = None
        said = said if isinstance(said, dict) else {}
        yield file, source, {k: str(v) for k, v in said.items() if v is not None}


def claude_env(env: dict, words=()) -> tuple[dict, set]:
    """Claude Code's environment with what its settings set in it, and all that they set."""
    sources, names = _claude_sources(words)[0], set()
    for _, source, said in claude_settings(env, strict=False):
        env = {**env, **said} if source in sources else env
        names |= set(said)
    return env, names


def claude(env: dict, words=()) -> list[Said]:
    """What its settings name, which come before its environment."""
    sources, known = _claude_sources(words)
    ours, own = dict(env), None
    for file, source, said in claude_settings(env):
        if source == "user":
            if "user" in sources:
                ours, own = {**env, **said}, (file, said)
            if not known and _claude_goes(ours) != _claude_goes(env):
                raise Unclear(
                    f"which settings count beside its arguments is not known, and {file} says where the "
                    "model is. Leave the arguments out"
                )
        elif source in sources:
            mine, theirs = _claude_goes(ours), _claude_goes({**ours, **said})
            if theirs != mine:
                names = [n for n in {**theirs, **mine} if theirs.get(n) != mine.get(n)]
                raise asked(file, names[0])
    base = url(own[1].get("ANTHROPIC_BASE_URL")) if own else None
    return [Said(own[0], "anthropic", base)] if base else []


# -- Codex ---------------------------------------------------------------------------

# What holds for everybody on the machine.
CODEX_SYSTEM = Path("/etc/codex/config.toml")


def codex_home(env: dict) -> Path:
    folder = env.get("CODEX_HOME") or Path(env.get("HOME", "~")) / ".codex"
    return Path(folder).expanduser()


def codex_key(env: dict, words=()) -> str | None:
    """What Codex is signed in with: `plan`, `stored`, the variable that holds a key, or nothing that is known."""
    if env.get("CODEX_API_KEY") and any(w in ("exec", "e") for w in words):
        return "CODEX_API_KEY"
    try:
        found = json.loads((codex_home(env) / "auth.json").read_text("utf-8"))
    except (OSError, ValueError):
        found = {}
    found = found if isinstance(found, dict) else {}
    mode, key = str(found.get("auth_mode") or "").lower(), found.get("OPENAI_API_KEY")
    if mode and mode not in ("apikey", "api_key", "api-key"):
        return "plan"
    if not mode and found.get("tokens") and not key:
        return "plan"
    if key:
        return "stored"
    return "OPENAI_API_KEY" if env.get("OPENAI_API_KEY") else None


def _codex_words(words) -> tuple[dict, str | None, set]:
    """What Codex is told on its command line: settings, a profile, and which flags are there."""
    settings: dict = {}
    profile, flags, waiting = None, set(), None
    for word in map(str, words):
        name, _, value = word.partition("=")
        if waiting is None and word == "--":
            break
        if waiting is None:
            flags.add(name)
            if word in ("-c", "--config", "-p", "--profile"):
                waiting = word
                continue
            if name not in ("-c", "--config", "-p", "--profile"):
                if word[:2] not in ("-c", "-p") or len(word) < 3:
                    continue
                name, value = word[:2], word[2:]
            word, waiting = value, name
        told, waiting = waiting, None
        if told in ("-p", "--profile"):
            profile = word
            continue
        path, _, value = word.partition("=")
        try:
            said = tomllib.loads(f"it = {value}")["it"]
        except ValueError:
            said = value
        for name in reversed(path.strip().split(".")):
            said = {name.strip().strip('"'): said}
        settings = over(settings, said)
    return settings, profile, flags


def codex_settings(env: dict, words=()) -> dict:
    """Codex's own settings, with those of the profile it uses and those on its command line."""
    told, profile, flags = _codex_words(words)
    folder, found = codex_home(env), toml(CODEX_SYSTEM)
    if "--ignore-user-config" not in flags:
        found = over(found, toml(folder / "config.toml"))
    name = profile or told.get("profile") or found.get("profile")
    if isinstance(name, str) and re.fullmatch(r"[\w.-]+", name):
        found = over(found, toml(folder / f"{name}.config.toml"))
        # As it was until it had files for them.
        named = at(found, "profiles", name)
        found = over(found, named if isinstance(named, dict) else {})
    found = over(found, told)
    return over(found, {"model_provider": "oss"}) if "--oss" in flags else found
