"""Conversation threads: the record a policy is fitted to.

A thread is a list of runs. A run is one user message, the text retrieved for it
and the answer. One JSON object per line, in any of three shapes:

- a chat log:    {"thread_id": ..., "messages": [{"role", "content", "at"}, ...]}
- explicit runs: {"thread_id": ..., "runs": [{"started_at", "ended_at", "input",
                  "retrievals", "output"}, ...]}
- the canonical form, in which every identity is spelled out.

`normalize_thread` turns the first two into the third. `Thread` validates it.
The first two may also say which `model` answered, which `tools` every request
carried and until when the thread was observed. `recorded` reads those.
"""

from __future__ import annotations

import copy
import gzip
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ContractError

__all__ = [
    "ContractError",
    "Thread",
    "canonical_usage",
    "load_threads",
    "normalize_thread",
    "read_jsonl",
    "recorded",
    "system_prompt_of",
    "write_threads",
]

RETRIEVAL_ROLES = frozenset({"tool", "retrieval", "context", "document", "function"})
_TIME_KEYS = ("at", "created_at", "timestamp", "time")
_THREAD_FIELDS = {"company_id", "thread_id", "runs", "scenario_id", "provenance"}
_RUN_FIELDS = {
    "run_id",
    "run_index",
    "started_at",
    "ended_at",
    "history_policy",
    "input",
    "output",
    "retrievals",
    "generation",
    "calls",
}
_GENERATION_FIELDS = {
    "observation_id",
    "started_at",
    "ended_at",
    "model",
    "usage",
    "request_part_ids",
    "cache_read_reported",
}
_USAGE_FIELDS = {"input", "output", "cache_read", "cache_write"}
_RETRIEVAL_FIELDS = {"document_id", "version", "chunk_id", "exposed_to_model"}


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{name}: expected nonempty trimmed string")
    return value


def _time(value: Any, name: str) -> datetime:
    _id(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError()
        return result
    except ValueError as exc:
        raise ContractError(f"{name}: expected timezone-aware ISO timestamp") from exc


def _status(part: dict) -> str:
    return part.get(
        "content_status", "available" if part.get("text") is not None else "missing"
    )


def _check_part(part: Any, kind: str) -> None:
    if not isinstance(part, dict):
        raise ContractError(f"{kind}: expected object")
    allowed = {"part_id", "text", "content_status"} | (
        _RETRIEVAL_FIELDS if kind == "retrieval" else set()
    )
    if set(part) - allowed:
        raise ContractError(f"{kind}: unknown fields: {sorted(set(part) - allowed)}")
    _id(part.get("part_id"), "part_id")
    text = part.get("text")
    if text is not None and not isinstance(text, str):
        raise ContractError("part text must be string or null")
    status = _status(part)
    if status not in ("available", "missing", "masked"):
        raise ContractError("invalid content_status")
    if status == "available" and text is None:
        raise ContractError("available content must have text")
    if status != "available" and text is not None:
        raise ContractError("masked or missing content must have null text")
    if kind == "retrieval":
        for key in ("document_id", "version", "chunk_id"):
            _id(part.get(key), key)
        if type(part.get("exposed_to_model")) is not bool:
            raise ContractError("exposed_to_model must be boolean")


@dataclass(frozen=True)
class Part:
    part_id: str
    text: str | None
    content_status: str


@dataclass(frozen=True)
class RetrievalPart(Part):
    document_id: str
    version: str
    chunk_id: str
    exposed_to_model: bool


@dataclass(frozen=True)
class Generation:
    observation_id: str
    model: str
    started_at: str
    ended_at: str
    usage: tuple[tuple[str, int], ...] | None
    request_part_ids: tuple[str, ...]
    cache_read_reported: bool = True


@dataclass(frozen=True)
class Run:
    run_id: str
    run_index: int
    started_at: str
    ended_at: str
    history_policy: str
    input: Part
    output: Part
    retrievals: tuple[RetrievalPart, ...]
    generation: Generation
    # Model calls the turn made, when the record says. None means not recorded.
    calls: int | None = None


def _typed_part(part: dict, retrieval: bool = False) -> Part:
    args = (part["part_id"], part.get("text"), _status(part))
    if not retrieval:
        return Part(*args)
    return RetrievalPart(
        *args,
        part["document_id"],
        part["version"],
        part["chunk_id"],
        part["exposed_to_model"],
    )


def _check_usage(usage: Any) -> None:
    if usage is None:
        return
    if (
        not isinstance(usage, dict)
        or set(usage) != _USAGE_FIELDS
        or any(type(v) is not int or v < 0 for v in usage.values())
        or usage["cache_read"] + usage["cache_write"] > usage["input"]
    ):
        raise ContractError(
            "usage must contain nonnegative integer input/output/cache_read/cache_write, with cache within input"
        )


class Thread:
    """One validated conversation in the canonical form."""

    def __init__(self, data: dict):
        self._data = self._validate(data)
        d = self._data
        self.company_id, self.thread_id = d["company_id"], d["thread_id"]
        runs = []
        for r in d["runs"]:
            g = r["generation"]
            usage = g["usage"]
            generation = Generation(
                g["observation_id"],
                g["model"],
                g["started_at"],
                g["ended_at"],
                tuple(sorted(usage.items())) if usage is not None else None,
                tuple(g["request_part_ids"]),
                g.get("cache_read_reported", True),
            )
            runs.append(
                Run(
                    r["run_id"],
                    r["run_index"],
                    r["started_at"],
                    r["ended_at"],
                    r["history_policy"],
                    _typed_part(r["input"]),
                    _typed_part(r["output"]),
                    tuple(_typed_part(p, True) for p in r["retrievals"]),
                    generation,
                    r.get("calls"),
                )
            )
        self.runs = tuple(runs)

    @classmethod
    def from_dict(cls, data: dict) -> Thread:
        return cls(data)

    @cached_property
    def part_origin(self) -> dict[str, tuple[str, int]]:
        """Kind and run index of every part, by part ID."""
        origin = {}
        for run in self.runs:
            origin[run.input.part_id] = ("input", run.run_index)
            origin[run.output.part_id] = ("output", run.run_index)
            for part in run.retrievals:
                origin[part.part_id] = ("retrieval", run.run_index)
        return origin

    @staticmethod
    def _validate(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ContractError("thread must be object")
        if set(data) - _THREAD_FIELDS:
            raise ContractError(
                f"unknown thread fields: {sorted(set(data) - _THREAD_FIELDS)}"
            )
        if "provenance" in data:
            _id(data["provenance"], "provenance")
        _id(data.get("company_id"), "company_id")
        _id(data.get("thread_id"), "thread_id")
        runs = data.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ContractError("runs must be a nonempty array")
        seen: dict[str, tuple[int, str, dict]] = {}
        run_ids, observations = set(), set()
        last_end = None
        expected_history: list[str] = []
        for i, run in enumerate(runs, 1):
            if not isinstance(run, dict):
                raise ContractError("run must be object")
            if set(run) - _RUN_FIELDS:
                raise ContractError("unknown run fields")
            run_id = _id(run.get("run_id"), "run_id")
            if run_id in run_ids:
                raise ContractError("duplicate run_id")
            run_ids.add(run_id)
            if type(run.get("run_index")) is not int or run["run_index"] != i:
                raise ContractError("run_index must be consecutive starting at 1")
            start = _time(run.get("started_at"), "run.started_at")
            end = _time(run.get("ended_at"), "run.ended_at")
            if start > end or (last_end is not None and start < last_end):
                raise ContractError("run timestamps not chronological")
            last_end = end
            if run.get("history_policy") != "full_canonical":
                raise ContractError("unsupported history_policy")
            calls = run.get("calls")
            if calls is not None and (type(calls) is not int or calls < 1):
                raise ContractError("calls must be a positive integer")
            retrievals = run.get("retrievals")
            if not isinstance(retrievals, list):
                raise ContractError("retrievals must be array")
            for kind, parts in (
                ("input", [run.get("input")]),
                ("retrieval", retrievals),
                ("output", [run.get("output")]),
            ):
                for part in parts:
                    _check_part(part, kind)
                    part_id = part["part_id"]
                    if part_id in seen:
                        raise ContractError("duplicate part_id: " + part_id)
                    seen[part_id] = (i, kind, part)
            g = run.get("generation")
            if not isinstance(g, dict):
                raise ContractError("exactly one generation object required")
            if type(g.get("cache_read_reported", True)) is not bool:
                raise ContractError("cache_read_reported must be boolean")
            if set(g) - _GENERATION_FIELDS:
                raise ContractError("unknown generation fields")
            observation = _id(g.get("observation_id"), "observation_id")
            _id(g.get("model"), "model")
            if observation in observations:
                raise ContractError("duplicate observation_id")
            observations.add(observation)
            sent = _time(g.get("started_at"), "generation.started_at")
            received = _time(g.get("ended_at"), "generation.ended_at")
            if not start <= sent <= received <= end:
                raise ContractError("generation timestamps outside run")
            if "usage" not in g:
                raise ContractError(
                    "generation usage must be explicit, null if unavailable"
                )
            _check_usage(g.get("usage"))
            ids = g.get("request_part_ids")
            if (
                not isinstance(ids, list)
                or any(not isinstance(x, str) or not x for x in ids)
                or len(ids) != len(set(ids))
            ):
                raise ContractError("request_part_ids must be unique IDs")
            expected = (
                expected_history
                + [run["input"]["part_id"]]
                + [p["part_id"] for p in retrievals if p["exposed_to_model"]]
            )
            if ids != expected:
                raise ContractError(
                    "full_canonical request does not match ordered visible history and current sources"
                )
            expected_history = expected + [run["output"]["part_id"]]
        return copy.deepcopy(data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)


# -- simple shapes ---------------------------------------------------------


def _iso(value: Any, name: str) -> str:
    """A timezone-aware ISO timestamp from ISO text or seconds since the epoch."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Thirteen digits is milliseconds.
        moment = datetime.fromtimestamp(
            value / 1000 if value > 1e11 else value, timezone.utc
        )
        return moment.isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError(
                f"{name}: expected an ISO timestamp or epoch seconds"
            ) from exc
        if moment.tzinfo is None:
            # A time without a zone is read as UTC.
            return (
                moment.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            )
        return value
    raise ContractError(f"{name}: expected an ISO timestamp or epoch seconds")


def cache_read_reported(usage: Any) -> bool:
    """Whether a receipt supplied a read meter, including an explicit zero."""
    if not isinstance(usage, dict):
        dump = getattr(usage, "model_dump", None)
        usage = dump() if callable(dump) else None
    if not isinstance(usage, dict):
        return False
    counts = [
        v for k, v in usage.items() if "cache" in k and ("read" in k or "cached" in k)
    ]
    for key in ("input_tokens_details", "prompt_tokens_details"):
        if isinstance(usage.get(key), dict):
            counts.append(usage[key].get("cached_tokens"))
    return any(type(n) is int and n >= 0 for n in counts)


def canonical_usage(usage: Any) -> dict | None:
    """Token counters in the canonical form, from that form or a provider's.

    Takes a dict or an SDK usage object. `input` includes cache reads and
    writes. Anthropic reports the three apart, OpenAI reports cached tokens
    inside the prompt total. An unknown shape is treated as no usage.
    """
    if not isinstance(usage, dict):
        # A provider SDK's usage object.
        dump = getattr(usage, "model_dump", None)
        usage = dump() if callable(dump) else None
        if not isinstance(usage, dict):
            return None
    if set(usage) == _USAGE_FIELDS:
        return dict(usage)

    def count(*path):
        value = usage
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value if type(value) is int and value >= 0 else None

    output = count("output_tokens")
    if output is None:
        output = count("completion_tokens")
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        read = count("cache_read_input_tokens") or 0
        write = count("cache_creation_input_tokens") or 0
        fresh = count("input_tokens")
        total = None if fresh is None else fresh + read + write
    else:
        total = count("input_tokens")
        if total is None:
            total = count("prompt_tokens")
        read = (
            count("input_tokens_details", "cached_tokens")
            or count("prompt_tokens_details", "cached_tokens")
            or 0
        )
        write = count("input_tokens_details", "cache_write_tokens") or 0
    if total is None or output is None or read + write > total:
        return None
    return {"input": total, "output": output, "cache_read": read, "cache_write": write}


def _content(content: Any) -> tuple[str, list[str]]:
    """A message's text, and the tool results it carries inside its blocks."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise ContractError("message content must be text or a list of blocks")
    texts, results = [], []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "tool_result":
                results.append(_content(block.get("content"))[0])
            elif isinstance(block.get("text"), str):
                texts.append(block["text"])
            # Tool calls, images and other blocks add no text to the history.
    return "\n\n".join(t for t in texts if t), [r for r in results if r]


def system_prompt_of(data: Any) -> str | None:
    """The first system message of a chat-log thread, when it has one."""
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None
    for message in data["messages"]:
        if isinstance(message, dict) and message.get("role") == "system":
            return _content(message.get("content"))[0] or None
    return None


def recorded(rows: Iterable[Any]) -> dict:
    """What a set of thread rows says about itself, beyond the conversations.

    The system prompt, the tool definitions and the model that most of them
    carry, and the latest `observed_until`: when the record was read, for
    threads that may still have been going. Each is None where no row says.
    """
    prompts, tools, models = Counter(), Counter(), Counter()
    until = None
    for row in rows:
        if isinstance(row, Thread):
            row = row.to_dict()
        if not isinstance(row, dict):
            continue
        prompt = system_prompt_of(row)
        if prompt:
            prompts[prompt] += 1
        if isinstance(row.get("tools"), list) and row["tools"]:
            tools[json.dumps(row["tools"], ensure_ascii=False, sort_keys=True)] += 1
        default = row.get("model")
        for message in row.get("messages") or ():
            if isinstance(message, dict) and message.get("role") == "assistant":
                models[message.get("model") or default] += 1
        for run in row.get("runs") or ():
            if isinstance(run, dict):
                generation = run.get("generation")
                named = (
                    generation.get("model") if isinstance(generation, dict) else None
                )
                models[run.get("model") or named or default] += 1
        if row.get("observed_until"):
            moment = _iso(row["observed_until"], "observed_until")
            if until is None or _time(moment, "observed_until") > _time(
                until, "observed_until"
            ):
                until = moment
    models.pop(None, None)
    models.pop("unspecified", None)

    def most(counter):
        return counter.most_common(1)[0][0] if counter else None

    return {
        "system_prompt": most(prompts),
        "tools": json.loads(most(tools)) if tools else None,
        "model": most(models),
        "models": dict(models.most_common()),
        "observed_until": until,
    }


def _turn_usage(usages: list) -> dict | None:
    """One usage for a turn of several calls.

    The input is the last call's, which holds the whole turn. The output is
    all of it. The cache read is the first call's: what the turn found in the
    cache when it arrived.
    """
    counted = [u for u in map(canonical_usage, usages) if u is not None]
    if not counted:
        return None
    return {
        "input": counted[-1]["input"],
        "output": sum(u["output"] for u in counted),
        "cache_read": counted[0]["cache_read"],
        "cache_write": min(
            counted[-1]["cache_write"],
            counted[-1]["input"] - counted[0]["cache_read"],
        ),
    }


def _runs_from_messages(messages: Any) -> list[dict]:
    """Group a chat log into runs: a user message, what was retrieved, the answer."""
    if not isinstance(messages, list):
        raise ContractError("messages must be an array")
    runs: list[dict] = []
    current: dict | None = None
    waiting: list[dict] = []

    def close():
        nonlocal current
        if current is not None and current["answers"]:
            runs.append(
                {
                    "started_at": current["started_at"],
                    "ended_at": current["ended_at"] or current["started_at"],
                    "input": "\n\n".join(current["questions"]),
                    "retrievals": current["retrievals"],
                    "output": "\n\n".join(current["answers"]),
                    # Every assistant message is one model call.
                    "calls": current["calls"],
                    **(
                        {"generation": {"cache_read_reported": False}}
                        if current["usages"] and not current["read_reported"]
                        else {}
                    ),
                    **(
                        {"usage": _turn_usage(current["usages"])}
                        if current["usages"]
                        else {}
                    ),
                    **({"model": current["model"]} if current["model"] else {}),
                }
            )
        current = None

    for number, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            raise ContractError(f"message {number}: expected object")
        role = message.get("role")
        text, results = _content(message.get("content"))
        at = next((message[k] for k in _TIME_KEYS if message.get(k) is not None), None)
        if role == "system":
            continue
        found = [{"text": r} for r in results]
        if role in RETRIEVAL_ROLES and text:
            part = {"text": text}
            if isinstance(message.get("id"), str) and message["id"].strip():
                part["id"] = message["id"].strip()
            found.append(part)
        if role == "user" and text:
            if current is not None and current["answers"]:
                close()
            if current is None:
                if at is None:
                    raise ContractError(
                        f"message {number}: a user message needs a time (`at`)"
                    )
                current = {
                    "started_at": at,
                    "ended_at": None,
                    "questions": [],
                    "retrievals": waiting,
                    "answers": [],
                    "usages": [],
                    "calls": 0,
                    "model": None,
                }
                waiting = []
            current["questions"].append(text)
        elif role == "assistant" and current is not None:
            if text:
                current["answers"].append(text)
            if at is not None:
                current["ended_at"] = at
            if current["calls"] == 0:
                current["read_reported"] = message.get(
                    "cache_read_reported", cache_read_reported(message.get("usage"))
                )
            current["calls"] += 1
            if message.get("usage") is not None:
                current["usages"].append(message["usage"])
            current["model"] = message.get("model") or current["model"]
        elif role not in ("user", "assistant") and role not in RETRIEVAL_ROLES:
            raise ContractError(f"message {number}: unknown role {role!r}")
        # Retrieved text belongs to the run of the last user message. Before the
        # first user message it waits for that run.
        if current is not None:
            current["retrievals"].extend(found)
        else:
            waiting.extend(found)
    close()
    if not runs:
        raise ContractError("messages hold no answered user message")
    return runs


def _named_part(value: Any, part_id: str) -> dict:
    if value is None or isinstance(value, str):
        return {"part_id": part_id, "text": value}
    if isinstance(value, dict):
        return value if "part_id" in value else {"part_id": part_id, **value}
    raise ContractError("a part is text or an object")


def _retrieval(value: Any, part_id: str) -> dict:
    part = dict(_named_part(value, part_id))
    if "id" in part:
        part.setdefault("document_id", part.pop("id"))
    part.setdefault("document_id", part["part_id"])
    part.setdefault("version", "1")
    part.setdefault("chunk_id", part["part_id"])
    part.setdefault("exposed_to_model", True)
    return part


def normalize_thread(data: Any, *, model: str | None = None) -> dict:
    """Spell out what a thread leaves implicit. A canonical thread comes back equal.

    Missing identities are numbered: run `<thread>:<n>`, parts `<run>:input`,
    `<run>:retrieval:<k>` and `<run>:output`. `model` overrides the recorded
    model of every run, which is what a fit for one price sheet needs.
    """
    if not isinstance(data, dict):
        raise ContractError("thread must be object")
    data = dict(data)
    thread_id = _id(data.get("thread_id"), "thread_id")
    default_model = data.pop("model", None)
    # What `recorded` reads: about the record, not part of the conversation.
    data.pop("tools", None)
    data.pop("observed_until", None)
    if "messages" in data:
        if "runs" in data:
            raise ContractError("a thread has either messages or runs")
        data["runs"] = _runs_from_messages(data.pop("messages"))
    data.setdefault("company_id", "default")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ContractError("runs must be a nonempty array")
    filled, history = [], []
    for index, run in enumerate(runs, 1):
        if not isinstance(run, dict):
            raise ContractError("run must be object")
        run = dict(run)
        run_id = run.setdefault("run_id", f"{thread_id}:{index}")
        if not isinstance(run_id, str):
            raise ContractError("run_id: expected nonempty trimmed string")
        run.setdefault("run_index", index)
        run.setdefault("history_policy", "full_canonical")
        run["input"] = _named_part(run.get("input"), f"{run_id}:input")
        run["output"] = _named_part(run.get("output"), f"{run_id}:output")
        retrievals = run.get("retrievals", [])
        if not isinstance(retrievals, list):
            raise ContractError("retrievals must be array")
        run["retrievals"] = [
            _retrieval(part, f"{run_id}:retrieval:{k}")
            for k, part in enumerate(retrievals, 1)
        ]
        run["started_at"] = _iso(run.get("started_at"), "run.started_at")
        run["ended_at"] = _iso(run.get("ended_at", run["started_at"]), "run.ended_at")
        generation = dict(run.get("generation") or {})
        usage, recorded = run.pop("usage", None), run.pop("model", None)
        generation.setdefault("observation_id", f"{run_id}:generation")
        generation.setdefault("started_at", run["started_at"])
        generation.setdefault("ended_at", run["ended_at"])
        if usage is not None and not cache_read_reported(usage):
            generation.setdefault("cache_read_reported", False)
        generation.setdefault("usage", canonical_usage(usage))
        generation.setdefault("model", recorded or default_model or "unspecified")
        if model is not None:
            generation["model"] = model
        current = [run["input"].get("part_id")] + [
            p.get("part_id") for p in run["retrievals"] if p.get("exposed_to_model")
        ]
        generation.setdefault("request_part_ids", history + current)
        history = history + current + [run["output"].get("part_id")]
        run["generation"] = generation
        filled.append(run)
    data["runs"] = filled
    return data


# -- files -----------------------------------------------------------------


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(
            p for p in path.iterdir() if p.name.endswith((".jsonl", ".jsonl.gz"))
        )
        if not found:
            raise ContractError(f"{path}: no .jsonl or .jsonl.gz files")
        return found
    return [path]


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """JSON objects from a JSONL file, a gzipped one, or a folder of either."""
    for file in _files(Path(path)):
        opener = gzip.open if file.name.endswith(".gz") else open
        with opener(file, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(
                        f"{file}:{number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ContractError(f"{file}:{number}: expected a JSON object")
                yield row


def _unique(threads: Iterable[Thread]) -> None:
    seen = {
        "thread_id": set(),
        "run_id": set(),
        "observation_id": set(),
        "part_id": set(),
    }

    def add(name, value):
        if value in seen[name]:
            raise ContractError(f"duplicate {name}: {value}")
        seen[name].add(value)

    for thread in threads:
        add("thread_id", thread.thread_id)
        for run in thread.runs:
            add("run_id", run.run_id)
            add("observation_id", run.generation.observation_id)
            for part in (run.input, run.output, *run.retrievals):
                add("part_id", part.part_id)


def load_threads(source: Any, *, model: str | None = None) -> list[Thread]:
    """Validated threads from a file, a folder, or an iterable of dicts or threads.

    Every shape in the module docstring is accepted. `model` stamps one model on
    every run. Identities must be unique across the whole set.
    """
    if isinstance(source, (str, Path)):
        rows: Iterable[Any] = read_jsonl(source)
    else:
        rows = source
    threads = []
    for number, row in enumerate(rows, 1):
        if isinstance(row, Thread) and model is None:
            threads.append(row)
            continue
        try:
            data = row.to_dict() if isinstance(row, Thread) else row
            threads.append(Thread(normalize_thread(data, model=model)))
        except (ValueError, TypeError) as exc:
            raise ContractError(f"thread {number}: {exc}") from exc
    _unique(threads)
    return threads


def write_threads(threads: Iterable[Thread], path: str | Path) -> None:
    """Write canonical threads as JSONL, gzipped when the name ends in .gz."""
    threads = list(threads)
    _unique(threads)
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for thread in threads:
            stream.write(
                json.dumps(thread.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
