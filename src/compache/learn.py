"""A policy that keeps up with the traffic it is applied to.

A `Learner` is a record folder and the policies fitted on it, one per model and
kind of conversation. It starts with what public runs taught, and fits again in
the background as the record grows. No text is kept, and nothing is asked of
the user: the folder is all there is to it.

    live = compache.Live(learn="compache-record")

A policy that takes results out of view hides that they would have been needed.
So one conversation in ten is left as the application wrote it. Those say what
replies rely on when everything is there, and what the others save.

    turns.jsonl, calls.jsonl      the record (`compache.record`)
    policies/<model>.<kind>.json  what was fitted on it
    policies/state.json           on how much of it, and when
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from . import record
from .api import Policy, fit
from .defaults import default_policy
from .errors import ContractError
from .models import model_spec
from .optimizer.contracts import seconds

__all__ = ["Learner"]

logger = logging.getLogger(__name__)

# Calls on record before a first fit, and how much the record has to grow for the next one.
FIRST_CALLS = 200
_GROWN = 1.5
# A fit looks at no more than the newest calls.
_MOST_CALLS = 20_000
# One conversation in this many is left as the application wrote it.
CONTROL_EVERY = 10


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
        self._begun: dict[str, int] = {}
        self._unknown: set[str] = set()
        self._state = self._read_state()
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
        # Other processes may read the folder at this moment.
        path = self.folder / "policies" / name
        aside = path.with_name(f"{name}.{os.getpid()}.part")
        aside.write_text(text, "utf-8")
        aside.replace(path)

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
    def _key(model, tools: bool) -> tuple[str, str] | None:
        """The name a model's policy goes by, and the model's ID. None for a model without a price sheet."""
        try:
            model_id = model_spec(str(model or "")).model_id
        except ContractError:
            return None
        return f"{model_id}.{'agent' if tools else 'chat'}", model_id

    def policy(self, model, tools: bool) -> Policy | None:
        """The policy for a conversation that begins now, or None to leave it as it is.

        `tools` says whether its requests offer the model any: an agent's do.
        A conversation without starts with no policy until labels teach one.
        """
        found = self._key(model, tools)
        if found is None:
            if model and model not in self._unknown:
                self._unknown.add(model)
                logger.info(
                    "compache: no price sheet for %s, so it is only recorded", model
                )
            return None
        key, model_id = found
        with self._lock:
            if key not in self._policies:
                policy = self._load(key)
                if policy is None and tools:
                    policy = default_policy(model_id, agent=True)
                self._policies[key] = policy
            policy = self._policies[key]
            if policy is None or not policy.agentic:
                return policy
            begun = self._begun.get(key, self._state.get(key, {}).get("threads", 0)) + 1
            self._begun[key] = begun
        if begun % CONTROL_EVERY == 0:
            return None
        return policy.at_default_lifetime() if self.default_lifetime else policy

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
            if not named:
                continue
            found = self._key(named.most_common(1)[0][0], bool(row.get("tools_bytes")))
            if found is not None:
                groups.setdefault(found, []).append(row)
        return groups, labels

    def _refit(self) -> None:
        try:
            groups, labels = self._groups()
        except Exception:
            logger.warning("compache: a record was not read", exc_info=True)
            return
        for (key, model_id), rows in groups.items():
            try:
                self._fit(key, model_id, rows, labels)
            except Exception:
                logger.warning("compache: a policy was not fitted", exc_info=True)

    def _fit(self, key: str, model_id: str, rows: list, labels: list) -> None:
        agentic = key.endswith(".agent")
        calls = sum(run.get("calls") or 1 for row in rows for run in row["runs"])
        before = self._state.get(key, {}).get("calls", 0)
        if calls < FIRST_CALLS or calls < before * _GROWN:
            with self._lock:
                self._begun[key] = max(self._begun.get(key, 0), len(rows))
            return
        kept, held = [], 0
        for row in sorted(rows, key=lambda r: -seconds_of(r)):
            kept.append(row)
            held += sum(run.get("calls") or 1 for run in row["runs"])
            if held >= _MOST_CALLS:
                break
        names = {row["thread_id"] for row in kept}
        mine = [row for row in labels if row["thread_id"] in names]
        asked = [row for row in mine if row["run"] > 1]
        if not agentic and not asked:
            # Without labels nothing can be left out, and a chat is given nothing else here.
            return
        policy = fit(kept, mine, model=model_id, agent=agentic)
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
            # Another process may have fitted another model meanwhile.
            self._state = {**self._read_state(), key: entry}
            self._write("state.json", json.dumps(self._state, indent=2) + "\n")
            self._policies[key] = policy
        logger.info("compache: fitted %s on %d calls", key, calls)


def seconds_of(row: dict) -> float:
    """When a recorded conversation last went on."""
    ended = row["runs"][-1]["ended_at"]
    return ended if isinstance(ended, (int, float)) else seconds(ended)
