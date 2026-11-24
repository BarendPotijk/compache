"""A record of live conversations that keeps no text.

Past traces can be protected or out of reach. A record is made going forward
instead, by `compache.live` as requests pass. For every turn it holds when the
turn started and ended, how large its question, its retrieved texts and its
answer were, what the provider's usage said, and, for the turns that were
asked, which earlier parts the reply named. A turn of many calls, as an
agent's, holds them one by one, with the earlier results each reply quoted.
Sizes, times, counters and labels: no word of any conversation, and
conversation IDs only as a hash.

A record is a folder. `compache fit <folder>` reads it like any other threads,
and takes its labels from it.

    turns.jsonl   one line per answered turn: what a fit reads
    calls.jsonl   one line per model call, pings included: what it cost
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = ["SIZES_ONLY", "Recorder", "is_record", "load", "load_calls"]

_TURNS, _CALLS = "turns.jsonl", "calls.jsonl"
# What a thread says of itself when its parts are given by size and not by text.
SIZES_ONLY = "sizes only"
_VERSION = 1
_LABEL_KINDS = {"Q": "input", "A": "output"}


def _torn(file: Path) -> bool:
    """Whether a file ends in the middle of a line, as a process that died leaves it."""
    try:
        with open(file, "rb") as stream:
            stream.seek(-1, os.SEEK_END)
            return stream.read(1) != b"\n"
    except OSError:
        return False


def thread_name(thread_id: str) -> str:
    """The name a conversation is recorded under: a hash, so an ID that says who it is stays out."""
    return hashlib.sha256(thread_id.encode()).hexdigest()[:16]


class Recorder:
    """Appends turns and calls to a record folder. Safe to share between threads."""

    def __init__(self, path: str | Path, *, hashed: bool = True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hashed = hashed
        self._lock = threading.Lock()
        # The files whose last line is known to have been ended.
        self._ended: set[str] = set()

    def name(self, thread_id: str) -> str:
        return thread_name(thread_id) if self.hashed else thread_id

    def _append(self, file: str, row: dict) -> None:
        line = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        # Written in one go, since other processes may append to the same file.
        with self._lock, open(self.path / file, "ab", buffering=0) as stream:
            if file not in self._ended:
                self._ended.add(file)
                if _torn(self.path / file):
                    line = "\n" + line
            stream.write(line.encode("utf-8"))

    def turn(
        self,
        thread_id: str,
        turn: int,
        *,
        started_at: float,
        ended_at: float,
        model: str | None,
        sizes: dict,
        requested: str | None = None,
        usage: dict | None = None,
        read_reported: bool = True,
        calls: int = 1,
        asked: bool = False,
        refs=None,
        older: bool = False,
        steps=(),
        carries: bool = False,
        cut: bool = False,
    ) -> None:
        """One answered turn. `sizes` are bytes: `input`, `retrievals`, `output`, `system`, `tools`.

        `model` is the one that answered, and `requested` the name it was asked
        for by, where that is another. `asked` says the reply was asked to name what it needed, and `refs`
        are the labels it named: an empty list for none, None for no answer.
        `steps` are the turn's calls, where it made more than one, `carries`
        says that requests hold what the model reasoned before, and `cut` that a
        policy had taken something out of view.
        """
        row = {
            "v": _VERSION,
            "thread": self.name(thread_id),
            "turn": turn,
            "started_at": round(started_at, 3),
            "ended_at": round(ended_at, 3),
            "model": model,
            "calls": calls,
            "bytes": sizes,
        }
        if requested and requested != model:
            row["requested"] = requested
        if usage is not None:
            row["usage"] = usage
            row["cache_read_reported"] = bool(read_reported)
        if asked:
            row["asked"] = True
            if refs is not None:
                row["refs"] = list(refs)
                if older:
                    row["older"] = True
        if cut:
            row["cut"] = True
        if steps:
            row["steps"] = [
                [
                    round(step["started_at"], 3),
                    round(step["ended_at"], 3),
                    step["results"],
                    step["bytes"],
                    step["output_tokens"],
                    None if step["needs"] is None else list(step["needs"]),
                    step["input_tokens"],
                    step["thought_tokens"] if carries else 0,
                ]
                for step in steps
            ]
        self._append(_TURNS, row)

    def call(
        self,
        thread_id: str,
        turn: int,
        kind: str,
        *,
        at: float,
        model: str | None = None,
        requested: str | None = None,
        usage: dict | None = None,
        cost: float | None = None,
        archived=None,
        left=None,
        lifetime=None,
        header_lifetime=None,
        long_write: int = 0,
        left_alone: bool = False,
    ) -> None:
        """One model call: an `answer`, a `reopen`, a `ping`, a `note` asked apart, or a `retry`.

        `archived` are the labels of the parts that went out as a notice, and
        `left` how many bytes of text compache left out of the request, less
        what it added, and how many were sent. `lifetime` is the cache lifetime
        it asked for, where that is not the provider's default, and
        `header_lifetime` the one for instructions and tools alone, with
        `long_write` the tokens that were written for an hour. `left_alone`
        marks a conversation that a learner leaves as the application wrote it.
        """
        row = {
            "v": _VERSION,
            "thread": self.name(thread_id),
            "turn": turn,
            "kind": kind,
            "at": round(at, 3),
            "model": model,
        }
        if requested and requested != model:
            row["requested"] = requested
        if usage is not None:
            row["usage"] = usage
        if cost is not None:
            row["cost"] = cost
        if archived:
            row["archived"] = list(archived)
        if left is not None:
            row["left_bytes"], row["sent_bytes"] = left
        if lifetime is not None:
            row["ttl"] = lifetime
        if header_lifetime is not None:
            row["header_ttl"] = header_lifetime
        if long_write:
            row["long_write"] = long_write
        if left_alone:
            row["left_alone"] = True
        self._append(_CALLS, row)


def is_record(source: Any) -> bool:
    """Whether `source` is a record folder."""
    return isinstance(source, (str, Path)) and (Path(source) / _TURNS).is_file()


def _read(file: Path, last: int | None = None) -> tuple[list[dict], int]:
    """The rows of a file, or of its last so many bytes, and how many lines of it are none."""
    rows, others = [], 0
    if not file.is_file():
        return rows, others
    with open(file, "rb") as stream:
        if last is not None and file.stat().st_size > last:
            stream.seek(-last, os.SEEK_END)
            # The line that was cut in two.
            stream.readline()
        for raw in stream:
            line = raw.decode("utf-8", "replace")
            try:
                row = json.loads(line) if line.strip() else None
            except ValueError:
                others += 1
                continue
            if isinstance(row, dict) and row.get("v") == _VERSION:
                rows.append(row)
    return rows, others


def _lines(file: Path) -> list[dict]:
    """The rows of a file. A line that a process which died left unfinished is passed over."""
    return _read(file)[0]


def load_calls(path: str | Path, last: int | None = None) -> list[dict]:
    """Every model call of a record, as written, or those in its last so many bytes."""
    return _read(Path(path) / _CALLS, last)[0]


def unreadable(path: str | Path) -> int:
    """How many lines of a record cannot be read."""
    return sum(_read(Path(path) / file)[1] for file in (_TURNS, _CALLS))


def _ids(name: str, labels, last: int, counts: list[int]) -> tuple[list, list]:
    """Part IDs that labels such as R2.1 or A1 name: those of earlier turns, and the retrieved ones."""
    needs, cited = [], []
    for label in labels:
        text = str(label).strip("{} ")
        kind, (turn, _, item) = text[:1].upper(), text[1:].partition(".")
        if not turn.isdigit() or not 1 <= int(turn) <= last:
            continue
        n = int(turn)
        if kind == "R" and item.isdigit() and 1 <= int(item) <= counts[n - 1]:
            part = f"{name}:{n}:retrieval:{int(item)}"
            cited.append(part)
        elif kind in _LABEL_KINDS and not item and n < last:
            part = f"{name}:{n}:{_LABEL_KINDS[kind]}"
        else:
            continue
        if n < last and part not in needs:
            needs.append(part)
    return needs, list(dict.fromkeys(cited))


def _steps(name: str, rows, counts: list[int], cut: bool) -> list[dict]:
    """A turn's calls, with what each quoted as part IDs.

    Where a policy kept things out of view, what replies would have relied on is not known.
    """
    made, seen = [], 0
    for row in rows:
        started, ended, results, size, output, needs, counted, thought = [
            *row,
            *[None] * (8 - len(row)),
        ][:8]
        # An application may have dropped results while its turn went on.
        seen = min(max(seen, results), counts[-1])
        step = {
            "started_at": started,
            "ended_at": max(ended, started),
            "results": seen,
            "bytes": size,
            "needs": None if cut or needs is None else [],
        }
        if output is not None:
            step["output_tokens"] = output
        if counted is not None:
            step["input_tokens"] = counted
        if thought:
            step["thought_tokens"] = thought
        for label in () if cut or needs is None else needs:
            turn, _, item = str(label)[1:].partition(".")
            if turn.isdigit() and item.isdigit() and 1 <= int(turn) <= len(counts):
                if 1 <= int(item) <= counts[int(turn) - 1]:
                    step["needs"].append(f"{name}:{int(turn)}:retrieval:{int(item)}")
        made.append(step)
    return made


def _follow_up(turns: dict[int, list[dict]]):
    """The turns of a conversation in order, up to the first gap.

    A turn may have been written more than once: after a retry, with a note that came late, or by
    a request of the host's own that branched off there. What was written last counts, of what
    lies between the turns around it.
    """
    after = float("-inf")
    for n in range(1, len(turns) + 1):
        rows = [r for r in turns.get(n, ()) if r["started_at"] >= after]
        if not rows:
            return
        before = max(
            (r["started_at"] for r in turns.get(n + 1, ())), default=float("inf")
        )
        row = next((r for r in reversed(rows) if r["ended_at"] <= before), rows[-1])
        yield n, row
        after = max(row["ended_at"], row["started_at"])


def _usage(row: dict) -> dict:
    """A turn's usage. A policy that cut while the turn went on left less than was read before."""
    usage = dict(row["usage"])
    usage["cache_read"] = min(usage["cache_read"], usage["input"])
    usage["cache_write"] = min(
        usage["cache_write"], usage["input"] - usage["cache_read"]
    )
    return usage


def load(path: str | Path) -> tuple[list[dict], list[dict]]:
    """A record as thread rows and label rows, ready for `fit`, `replay` and `profile`.

    A conversation is used from its first turn up to the first gap in what was
    recorded. One that was already going when recording began is left out.
    Every first turn is labelled as needing nothing, since nothing came before.
    """
    by_thread: dict[str, dict[int, list[dict]]] = {}
    for row in _lines(Path(path) / _TURNS):
        if isinstance(row.get("thread"), str) and type(row.get("turn")) is int:
            turns = by_thread.setdefault(row["thread"], {})
            turns.setdefault(row["turn"], []).append(row)
    threads, labels = [], []
    for name, turns in by_thread.items():
        runs, counts = [], []
        requested: Counter = Counter()
        cut = any(row.get("cut") for rows in turns.values() for row in rows)
        for n, row in _follow_up(turns):
            sizes = row.get("bytes") or {}
            found = [int(b) for b in sizes.get("retrievals") or ()]
            counts.append(len(found))
            run = {
                "started_at": row["started_at"],
                "ended_at": max(row["ended_at"], row["started_at"]),
                "input": {"bytes": int(sizes.get("input") or 0)},
                "retrievals": [{"bytes": b} for b in found],
                "output": {"bytes": int(sizes.get("output") or 0)},
                "calls": max(1, int(row.get("calls") or 1)),
            }
            if row.get("steps"):
                run["steps"] = _steps(name, row["steps"], counts, cut)
            if row.get("model"):
                run["model"] = row["model"]
            if row.get("requested") or row.get("model"):
                requested[row.get("requested") or row["model"]] += 1
            if row.get("usage") is not None:
                run["usage"] = _usage(row)
                if row.get("cache_read_reported") is False:
                    run["generation"] = {"cache_read_reported": False}
            if n == 1:
                labels.append({"thread_id": name, "run": 1, "needs": []})
            elif row.get("asked") and row.get("refs") is not None:
                needs, cited = _ids(name, row["refs"], n, counts)
                label = {"thread_id": name, "run": n, "needs": [needs] if needs else []}
                if row.get("older"):
                    # The reply pointed past what it was shown a list of.
                    label["needs"] = None
                labels.append(label)
                if cited:
                    run["cited_part_ids"] = cited
            runs.append(run)
        if not runs:
            continue
        first = turns[1][-1].get("bytes") or {}
        thread = {"thread_id": name, "provenance": SIZES_ONLY, "runs": runs}
        if requested:
            thread["requested_model"] = requested.most_common(1)[0][0]
        if first.get("system"):
            thread["system_bytes"] = int(first["system"])
        if first.get("tools"):
            thread["tools_bytes"] = int(first["tools"])
        threads.append(thread)
    return threads, labels
