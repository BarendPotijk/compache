"""A policy that keeps up with the traffic it is applied to.

A `Learner` is a record folder and the policies fitted on it, one per model and
kind of conversation. It starts with what public runs taught, and fits again in
the background as the record grows. No text is kept, and nothing is asked of
the user: the folder is all there is to it.

    live = compache.Live(learn="compache-record")

A policy that takes results out of view hides that they would have been needed.
So one conversation in ten is left as the application wrote it. Those say what
replies rely on when everything is there, and what the others save.

Until there is a fit, instructions and tools are cached for an hour from the
first time that would have helped: when somebody comes back, or begins anew,
after the short entry has gone and before the long one would have.

    turns.jsonl, calls.jsonl      the record (`compache.record`)
    policies/<model>.<kind>.json  what was fitted on it
    policies/state.json           on how much of it, and when
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import record
from .agent import default_header
from .api import Policy, fit
from .defaults import default_policy
from .errors import ContractError
from .models import cache_lifetimes, known_models, model_spec
from .optimizer.contracts import ModelSpec, seconds
from .threads import load_threads

__all__ = ["Learner"]

logger = logging.getLogger(__name__)

# Calls on record before a first fit, and how much the record has to grow for the next one.
FIRST_CALLS = 200
_GROWN = 1.5
# A fit looks at no more than the newest calls.
_MOST_CALLS = 20_000
# One conversation in this many is left as the application wrote it.
CONTROL_EVERY = 10


# A model without a price sheet.
_OTHER = ModelSpec(
    "other",
    context_tokens=1_000_000,
    cache_ttl_seconds=300,
    cache_min_tokens=1024,
    read_multiplier=0.1,
    write_multiplier=1.0,
)


def _known(spec: ModelSpec) -> bool:
    return spec.model_id in known_models()


def _named(spec: ModelSpec):
    """A model as `fit` takes it: by name where there is a price sheet, or by its spec."""
    return spec.model_id if _known(spec) else spec


def _asked(row: dict):
    """The name a recorded request gave its model."""
    return row.get("requested") or row.get("model")


class Learner:
    """The policies a record teaches, for the models it has seen. Safe to share between threads."""

    def __init__(
        self,
        folder: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        background: bool = True,
    ):
        self.folder = Path(folder)
        self.clock, self.background = clock, background
        (self.folder / "policies").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._policies: dict[str, Policy | None] = {}
        # How many conversations have begun, by key: those on record, then those of this process.
        self._begun: dict[str, int] | None = None
        self._state = self._read_state()
        # When a request last went out, by key.
        self._opened: dict[str, float | None] = {}
        self._since = 0
        self._worker: threading.Thread | None = None
        # Set where a provider takes a request only under its default cache lifetime.
        self.default_lifetime = False

    # -- files -----------------------------------------------------------------

    def _read_state(self) -> dict:
        try:
            found = json.loads(
                (self.folder / "policies" / "state.json").read_text("utf-8")
            )
        except (OSError, ValueError):
            return {}
        return found if isinstance(found, dict) else {}

    def _write(self, name: str, text: str) -> None:
        # Other processes may read the folder at this moment, and write to it.
        path = self.folder / "policies" / name
        aside = path.with_name(f"{name}.{uuid.uuid4().hex}.part")
        try:
            aside.write_text(text, "utf-8")
            aside.replace(path)
        finally:
            aside.unlink(missing_ok=True)

    def _keep(self, changed: dict) -> None:
        """Add to the state. Another process may have added to it meanwhile."""
        self._state = {**self._read_state(), **changed}
        self._write("state.json", json.dumps(self._state, indent=2) + "\n")

    def _load(self, key: str) -> Policy | None:
        try:
            return Policy.load(self.folder / "policies" / f"{key}.json")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, ContractError):
            logger.warning("compache: a learned policy was not read", exc_info=True)
            return None

    # -- what a conversation is given -------------------------------------------

    @staticmethod
    def _key(model, tools: bool, answered=None) -> tuple[str, ModelSpec] | None:
        """The name a model's policy goes by, and the model's spec. None where no model is named.

        `model` is the name that requests give, and `answered` the one their replies gave.
        """
        name = str(model or "").strip()
        if not name:
            return None
        spec = None
        for known in (name, str(answered or "").strip()):
            try:
                spec = spec or model_spec(known)
            except ContractError:
                pass
        if spec is None:
            # One without a price sheet: what most providers charge, and no pings at a guess.
            spec = replace(_OTHER, model_id=name)
        try:
            name = model_spec(name).model_id
        except ContractError:
            pass
        plain = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return f"{plain}.{'agent' if tools else 'chat'}", spec

    def _on_record(self) -> dict[str, int]:
        """How many conversations the record holds, by key. Only their first turns are looked at."""
        counts: Counter = Counter()
        try:
            with open(
                self.folder / "turns.jsonl", encoding="utf-8", errors="replace"
            ) as stream:
                for line in stream:
                    if '"turn":1,' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    tools = (row.get("bytes") or {}).get("tools")
                    found = self._key(_asked(row), bool(tools))
                    if found is not None:
                        counts[found[0]] += 1
        except OSError:
            pass
        return dict(counts)

    def policy(self, model, tools: bool) -> Policy | None:
        """The policy for a conversation that begins now, or None to leave it as it is."""
        return self.choose(model, tools)[0]

    def choose(self, model, tools: bool) -> tuple[Policy | None, bool]:
        """The policy for a conversation that begins now, and whether it is left alone on purpose.

        `tools` says whether its requests offer the model any: an agent's do.
        A conversation without starts with no policy until labels teach one.
        """
        found = self._key(model, tools)
        if found is None:
            return None, False
        key, spec = found
        with self._lock:
            if key not in self._policies:
                policy = self._load(key)
                if policy is None and tools:
                    policy = default_policy(_named(spec), agent=True)
                    # Whether its header is to live longer shows as it is used.
                    artifact = dict(policy.artifact)
                    artifact.pop("header_spec", None)
                    if not _known(spec):
                        artifact["recipe"] = {**artifact["recipe"], "tool_pings": 0}
                    policy = Policy(artifact)
                self._policies[key] = policy
            policy = self._policies[key]
            if policy is None:
                return None, False
            if self.default_lifetime:
                policy = policy.at_default_lifetime()
            if not policy.agentic:
                return policy, False
            if self._begun is None:
                self._begun = self._on_record()
            begun = self._begun[key] = self._begun.get(key, 0) + 1
        return (None, True) if begun % CONTROL_EVERY == 0 else (policy, False)

    def _last_call(self, key: str) -> float | None:
        """When the record last saw a call of a model."""
        try:
            with open(self.folder / "calls.jsonl", "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65_536))
                lines = stream.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            found = self._key(_asked(row), key.endswith(".agent"))
            if found is not None and found[0] == key:
                return row.get("at")
        return None

    def header_lifetime(self, model, at: float) -> float | None:
        """How long to have instructions and tools cached, for a request that goes out now and finds none.

        None for the provider's default.
        """
        found = self._key(model, True)
        if found is None or self.default_lifetime:
            return None
        key, spec = found
        longer = default_header(spec, cache_lifetimes(spec.model_id))
        with self._lock:
            policy = self._policies.get(key)
            if key in self._state or policy is None or longer is None:
                return policy.header_lifetime if policy is not None else None
            last = self._opened.get(key)
            if key not in self._opened:
                last = self._last_call(key)
            lasting = self._state.get("lasting", [])
            if key not in lasting:
                if last is None or not (
                    spec.cache_ttl_seconds < at - last < longer.cache_ttl_seconds
                ):
                    return None
                others = self._read_state().get("lasting", [])
                self._keep({"lasting": sorted({*others, *lasting, key})})
        return longer.cache_ttl_seconds

    def opened(self, model, at: float) -> None:
        """A request went out at that time."""
        found = self._key(model, True)
        if found is not None:
            with self._lock:
                self._opened[found[0]] = at

    @property
    def fitted(self) -> list[str]:
        """The policies that were fitted on the record, by model and kind."""
        return sorted(key for key in self._state if key != "lasting")

    def fitted_on(self, model, tools: bool = True) -> dict | None:
        """On how much a model's policy was fitted and when, or None while it is the default."""
        found = self._key(model, tools)
        return self._state.get(found[0]) if found else None

    # -- fitting again ------------------------------------------------------------

    def heard(self, calls: int) -> None:
        """A turn of that many calls went on record."""
        with self._lock:
            self._since += calls
            if self._since < FIRST_CALLS // 2:
                return
            self._since = 0
        self.refit(wait=not self.background)

    def refit(self, *, wait: bool = False) -> None:
        """Fit again what the record has grown enough for, in the background unless told to wait."""
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._refit, name="compache-learn", daemon=True
                )
                self._worker.start()
            worker = self._worker
        if wait:
            worker.join()

    def _groups(self) -> tuple[dict, list]:
        rows, labels = record.load(self.folder)
        groups: dict = {}
        for row in rows:
            named = Counter(run["model"] for run in row["runs"] if run.get("model"))
            answered = named.most_common(1)[0][0] if named else None
            asked = row.get("requested_model") or answered
            found = self._key(asked, bool(row.get("tools_bytes")), answered)
            if found is not None:
                groups.setdefault(found[0], (found[1], []))[1].append(row)
        return groups, labels

    def _refit(self) -> None:
        try:
            groups, labels = self._groups()
        except Exception:
            logger.warning("compache: a record was not read", exc_info=True)
            return
        for key, (spec, rows) in groups.items():
            try:
                self._fit(key, spec, rows, labels)
            except Exception:
                logger.warning("compache: a policy was not fitted", exc_info=True)

    def _fit(self, key: str, spec: ModelSpec, rows: list, labels: list) -> None:
        agentic = key.endswith(".agent")
        calls = sum(run.get("calls") or 1 for row in rows for run in row["runs"])
        before = self._state.get(key, {}).get("calls", 0)
        if calls < FIRST_CALLS or calls < before * _GROWN:
            return
        kept, held = [], 0
        for row in sorted(rows, key=lambda r: -seconds_of(r)):
            try:
                load_threads([row])
            except ContractError:
                logger.warning("compache: a conversation was not read", exc_info=True)
                continue
            kept.append(row)
            held += sum(run.get("calls") or 1 for run in row["runs"])
            if held >= _MOST_CALLS:
                break
        if not kept:
            return
        names = {row["thread_id"] for row in kept}
        mine = [row for row in labels if row["thread_id"] in names]
        asked = [row for row in mine if row["run"] > 1]
        if not agentic and not asked:
            # Without labels nothing can be left out, and a chat is given nothing else here.
            return
        # Where a cache's lifetime is a guess, nothing is paid to keep one alive.
        more = {"pings": _known(spec)} if agentic else {}
        policy = fit(kept, mine, model=_named(spec), agent=agentic, **more)
        text = json.dumps(policy.artifact, ensure_ascii=False)
        self._write(f"{key}.json", text)
        summary = policy.summary
        entry = {
            "at": round(self.clock(), 3),
            "calls": calls,
            "threads": len(rows),
            "saving": summary["saving"],
        }
        with self._lock:
            self._keep({key: entry})
            self._policies[key] = policy
        logger.info("compache: fitted %s on %d calls", key, calls)


def seconds_of(row: dict) -> float:
    """When a recorded conversation last went on."""
    ended = row["runs"][-1]["ended_at"]
    return ended if isinstance(ended, (int, float)) else seconds(ended)
