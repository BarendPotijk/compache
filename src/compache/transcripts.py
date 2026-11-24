"""Threads from the sessions that coding agents keep on disk.

    compache pull claude -o threads.jsonl.gz      # Claude Code
    compache pull codex -o threads.jsonl.gz       # Codex
    compache learn ~/.compache/claude --from claude

They hold what was said, when, and what the provider counted, which is all a
policy for an agent is fitted on. Nothing leaves the machine. `learn` keeps
sizes, times and usage, and no text.

A conversation that its agent compacted goes on as another thread, since its
requests did. Agents that an agent started are threads of their own.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .errors import ContractError
from .record import Recorder, thread_name
from .threads import load_threads

__all__ = ["AGENTS", "imported", "pull"]


def _rows(path: Path) -> Iterator[dict]:
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    # A session that is being written, or was cut off.
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _opened(messages: list[dict]) -> int:
    """Bytes of instructions and tools ahead of a conversation, by what its first call was counted at."""
    question = 0
    for message in messages:
        if message["role"] == "assistant":
            usage = message.get("usage") or {}
            total = sum(
                usage.get(k) or 0
                for k in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            )
            return max(0, 4 * total - question)
        question += len(json.dumps(message["content"]))
    return 0


def _thread(name: str, messages: list[dict]) -> dict | None:
    """Messages as a thread, if anything in it was answered."""
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]
    if not any(m["role"] == "assistant" for m in messages):
        return None
    thread: dict[str, Any] = {"thread_id": name, "messages": messages}
    opened = _opened(messages)
    if opened:
        thread["system_bytes"] = opened
    return thread


# -- Claude Code -----------------------------------------------------------------


def _claude(path: Path) -> Iterator[dict]:
    """The threads of one of Claude Code's session files."""
    talks: dict[str, list[list[dict]]] = {}
    said: dict[str, dict] = {}
    session = path.stem
    for row in _rows(path):
        message = row.get("message")
        if row.get("type") not in ("user", "assistant") or not isinstance(
            message, dict
        ):
            continue
        session = row.get("sessionId") or session
        at, content = row.get("timestamp"), message.get("content")
        if at is None or content is None or row.get("isMeta"):
            continue
        talk = talks.setdefault(row.get("agentId") or "", [[]])
        if row["type"] == "user":
            if row.get("isCompactSummary"):
                talk.append([])
            talk[-1].append({"role": "user", "content": content, "at": at})
            continue
        # A reply is written block by block, with the results of its first tools in between.
        blocks = content if isinstance(content, list) else [content]
        known = said.get(message.get("id") or "")
        if known is not None:
            known["content"] += blocks
            known["at"], known["usage"] = at, message.get("usage") or known["usage"]
            continue
        made = {
            "role": "assistant",
            "content": list(blocks),
            "at": at,
            "usage": message.get("usage"),
            "model": message.get("model"),
        }
        if message.get("id"):
            said[message["id"]] = made
        talk[-1].append(made)
    for agent, parts in talks.items():
        for k, messages in enumerate(parts):
            name = f"{session}~{agent}" if agent else session
            thread = _thread(f"{name}#{k + 1}" if k else name, _unnamed(messages))
            if thread is not None:
                yield thread


def _unnamed(messages: list[dict]) -> list[dict]:
    """Messages without a model that is none, as replies Claude Code wrote itself have."""
    made = []
    for message in messages:
        if message["role"] == "assistant":
            if message["model"] == "<synthetic>":
                continue
            message = {k: v for k, v in message.items() if v is not None}
        made.append(message)
    return made


# -- Codex -----------------------------------------------------------------------


def _codex_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = content if isinstance(content, list) else ()
    texts = [p.get("text") for p in parts if isinstance(p, dict)]
    return "\n\n".join(t for t in texts if isinstance(t, str))


def _codex_usage(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "input_tokens_details": {
            "cached_tokens": usage.get("cached_input_tokens") or 0,
            "cache_write_tokens": usage.get("cache_write_input_tokens") or 0,
        },
        "output_tokens": usage.get("output_tokens") or 0,
        "output_tokens_details": {
            "reasoning_tokens": usage.get("reasoning_output_tokens") or 0
        },
    }


def _codex(path: Path) -> Iterator[dict]:
    """The threads of one of Codex's session files."""
    parts: list[list[dict]] = [[]]
    session, model = path.stem, None
    waiting: list[dict] = []
    reply: dict | None = None

    def put(message):
        # What came in while a reply was being made follows it.
        (waiting if reply is not None else parts[-1]).append(message)

    def close(usage=None):
        nonlocal reply, waiting
        if reply is not None:
            if usage is not None:
                reply["usage"] = _codex_usage(usage)
            if model:
                reply["model"] = model
            parts[-1].append(reply)
        parts[-1].extend(waiting)
        reply, waiting = None, []

    for row in _rows(path):
        payload, at = row.get("payload"), row.get("timestamp")
        if not isinstance(payload, dict) or at is None:
            continue
        kind, what = row.get("type"), payload.get("type")
        if kind == "session_meta":
            session = payload.get("id") or session
        elif kind == "turn_context":
            model = payload.get("model") or model
        elif kind == "compacted":
            close()
            parts.append([])
        elif kind == "token_usage_record" and isinstance(payload.get("usage"), dict):
            close(payload["usage"])
        elif kind != "response_item":
            continue
        elif what == "message" and payload.get("role") == "user":
            if waiting or reply is not None:
                close()
            parts[-1].append(
                {
                    "role": "user",
                    "content": _codex_text(payload.get("content")),
                    "at": at,
                }
            )
        elif str(what).endswith("_call_output"):
            text = _codex_text(payload.get("output"))
            put(
                {
                    "role": "tool",
                    "tool_call_id": payload.get("call_id"),
                    "content": text,
                    "at": at,
                }
            )
        elif (
            what == "reasoning"
            or str(what).endswith("_call")
            or (what == "message" and payload.get("role") == "assistant")
        ):
            if reply is None:
                reply = {"role": "assistant", "content": [], "at": at}
            reply["at"] = at
            if what == "reasoning":
                reply["content"].append({"type": "thinking", "thinking": ""})
            elif what == "message":
                text = _codex_text(payload.get("content"))
                reply["content"].append({"type": "text", "text": text})
            else:
                given = payload.get("arguments", payload.get("input"))
                reply["content"].append({"type": "tool_use", "input": given})
    close()
    for k, messages in enumerate(parts):
        thread = _thread(f"{session}#{k + 1}" if k else session, messages)
        if thread is not None:
            yield thread


# -- both ------------------------------------------------------------------------


def _home(variable: str, name: str) -> Path:
    return Path(os.environ.get(variable) or Path.home() / name)


AGENTS = {
    "claude": (
        lambda: _home("CLAUDE_CONFIG_DIR", ".claude") / "projects",
        "**/*.jsonl",
        _claude,
    ),
    "codex": (
        lambda: _home("CODEX_HOME", ".codex") / "sessions",
        "**/rollout-*.jsonl",
        _codex,
    ),
}


def _moment(value) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    found = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (found if found.tzinfo else found.replace(tzinfo=timezone.utc)).timestamp()


def pull(
    agent: str,
    folder: str | Path | None = None,
    *,
    days: float = 30,
    since=None,
    until=None,
) -> tuple[list[dict], dict]:
    """The sessions an agent kept on this machine, as thread rows for `fit`, and a report.

    `folder` is where it keeps them, if not where it does by default. The last
    `days` are read, or what lies between `since` and `until`.
    """
    if agent not in AGENTS:
        raise ContractError(f"agent: expected {' or '.join(AGENTS)}")
    place, pattern, read = AGENTS[agent]
    root = Path(folder) if folder is not None else place()
    if not root.is_dir():
        raise ContractError(f"no sessions at {root}")
    start = (
        _moment(since)
        if since is not None
        else (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    )
    end = _moment(until) if until is not None else float("inf")
    threads, files, unread = [], 0, 0
    for path in sorted(root.glob(pattern)):
        if path.stat().st_mtime < start:
            continue
        files += 1
        for thread in read(path):
            begun = _moment(thread["messages"][0]["at"])
            if not start <= begun <= end:
                continue
            try:
                # One that cannot be read as turns is left out, and the others are not.
                load_threads([thread])
            except ContractError:
                unread += 1
                continue
            threads.append(thread)
    models: dict[str, int] = {}
    for thread in threads:
        for message in thread["messages"]:
            if message["role"] == "assistant" and message.get("model"):
                models[message["model"]] = models.get(message["model"], 0) + 1
    calls = sum(m["role"] == "assistant" for t in threads for m in t["messages"])
    return threads, {
        "files": files,
        "sessions": len(threads),
        "model_calls": calls,
        "threads": len(threads),
        "turns": sum(len(t.runs) for t in (load_threads(threads) if threads else ())),
        "sessions_unreadable": unread,
        "models": dict(sorted(models.items(), key=lambda pair: -pair[1])),
    }


def _size(text: str | None) -> int:
    return len((text or "").encode("utf-8"))


def imported(folder: str | Path, agent: str, source=None, **window) -> dict:
    """Add an agent's past sessions to a record, as sizes and times. Returns the report.

    Sessions the record holds already, such as those that went through compache,
    are left as they are.
    """
    found, report = pull(agent, source, **window)
    path = Path(folder)
    known = set()
    for row in _rows(path / "turns.jsonl"):
        known.add(row.get("thread"))
    recorder, added = Recorder(path), 0
    opened = {t["thread_id"]: t.get("system_bytes", 0) for t in found}
    rows = [t for t in found if thread_name(t["thread_id"]) not in known]
    for thread in load_threads(rows) if rows else ():
        added += 1
        for run in thread.runs:
            steps = [
                {
                    "started_at": _moment(step.started_at),
                    "ended_at": _moment(step.ended_at),
                    "results": step.results,
                    "bytes": step.bytes,
                    "output_tokens": step.output_tokens,
                    "needs": None
                    if step.needs is None
                    else [
                        "R{}.{}".format(*need.split(":")[-3::2])
                        for need in step.needs
                        if ":retrieval:" in need
                    ],
                    "input_tokens": step.input_tokens,
                    "thought_tokens": step.thought_tokens,
                }
                for step in run.steps
            ]
            usage = run.generation.usage
            recorder.turn(
                thread.thread_id,
                run.run_index,
                started_at=_moment(run.started_at),
                ended_at=_moment(run.ended_at),
                model=run.generation.model,
                sizes={
                    "input": _size(run.input.text),
                    "retrievals": [_size(p.text) for p in run.retrievals],
                    "output": _size(run.output.text),
                    # Which part of it is tools is not on record. An agent has some.
                    "system": 0,
                    "tools": max(1, opened.get(thread.thread_id, 0)),
                },
                usage=dict(usage) if usage else None,
                calls=run.calls or max(1, len(steps)),
                steps=steps,
                carries=True,
            )
    return {**report, "added": added}
