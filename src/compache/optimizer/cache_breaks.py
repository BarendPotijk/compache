"""Observe prefix reuse on user returns and suspend speculative cache refreshes.

Record successful user requests in the same provider cache namespace. Compare
provider-visible system/tool bytes, preserving order. Byte strings are best
when SDK serialization differs from the compact JSON used for Python values.
The host supplies eligible block counts in provider token units. Observation,
latched suppression, and the warning audit commit in one SQLite transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import sqlite3
from collections import deque
from threading import RLock

from compache.errors import ContractError
from .contracts import digest, number
from .distribution import modeled_read_tokens


logger = logging.getLogger(__name__)


def _id(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _fingerprint(value):
    """Hash wire bytes directly, or serialize an ordered JSON value once."""
    try:
        if isinstance(value, bytes):
            data = value
        elif isinstance(value, str):
            data = value.encode("utf-8")
        else:
            data = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ContractError(
            "headers must be bytes, text, or an ordered JSON value"
        ) from exc
    return hashlib.sha256(data).hexdigest()


def _price_ping_receipt(spec, input_tokens, reads, writes, output_tokens):
    """Price the exact read count, without rounding it to a forecast cache block."""
    if input_tokens is None or reads is None or output_tokens is None:
        return None, "incomplete_counters"
    uncached = input_tokens - reads
    if writes is None:
        eligible = spec.cache_ttl_seconds > 0 and input_tokens >= spec.cache_min_tokens
        writes = uncached if eligible else 0
        source = "model_spec_uncached_input_assumption"
    else:
        source = "explicit_cache_writes"
    cost = spec.token_charge(uncached - writes, writes, reads, output_tokens)
    return cost, source


@dataclass(frozen=True)
class CacheScope:
    installation: str
    provider: str
    model: str
    thread_id: str
    cache_key: str = ""

    def __post_init__(self):
        for key in ("installation", "provider", "model", "thread_id"):
            _id(getattr(self, key), key)
        if not isinstance(self.cache_key, str):
            raise ContractError("cache_key must be a string")

    @property
    def fingerprint(self):
        return digest(
            [
                self.installation,
                self.provider,
                self.model,
                self.thread_id,
                self.cache_key,
            ]
        )


def _classify(
    prior,
    *,
    started_at,
    ttl,
    reusable,
    reads,
    shared,
    static_tokens,
    intentional,
    system,
    tools,
):
    """Return outcome, changed components, age and eligible prior reuse."""
    if prior is None:
        return "first_observation", [], None, 0
    changed = [
        name
        for name, current in (("system_prompt", system), ("tools", tools))
        if prior[name] != current
    ]
    age = started_at - prior["refreshed_at"]
    expected = min(prior["reusable"], reusable)
    if prior["foreground_bound"] is not None:
        expected = min(expected, prior["foreground_bound"])
    if prior["ttl"] != ttl:
        outcome = "cache_ttl_changed"
    elif age < 0:
        outcome = "refresh_after_request_started"
    elif age >= ttl:
        outcome = "expired"
    elif not expected:
        outcome = "below_cache_minimum"
    elif intentional:
        outcome = "intentional_change"
    elif reads is not None and reads >= expected:
        outcome = "expected_reuse_observed"
    elif reads == 0:
        outcome = "on_time_cache_miss"
    elif (
        reads is not None
        and static_tokens is not None
        and reads <= static_tokens < expected
    ):
        outcome = "on_time_history_cache_miss"
    elif shared is not None and shared < expected:
        outcome = "on_time_prefix_invalidated"
    elif changed:
        outcome = "header_change_unconfirmed"
    elif reads is None:
        outcome = "cache_reads_unknown"
    else:
        outcome = "partial_reuse_observed"
    return outcome, changed, age, expected


_TABLES = (
    "compache_cache_guard_v1",
    "compache_cache_arrival_v1",
    "compache_cache_reset_v1",
    "compache_ping_stop_v1",
    "compache_ping_foreground_v1",
    "compache_cache_warning_v1",
)


class CacheBreakChecker:
    """Per-thread cache guard. Pass a database path for cross-process persistence."""

    def __init__(self, path=":memory:"):
        self._lock = RLock()
        # Sessions end wherever they are collected, perhaps in the middle of something here.
        self._gone: deque[str] = deque()
        self._db = sqlite3.connect(
            path, timeout=10, isolation_level=None, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS compache_cache_guard_v1 (
                scope TEXT PRIMARY KEY, run TEXT NOT NULL, started_at REAL NOT NULL,
                refreshed_at REAL NOT NULL, ttl REAL NOT NULL, reusable INTEGER NOT NULL,
                system_prompt TEXT NOT NULL, tools TEXT NOT NULL, disabled INTEGER NOT NULL,
                observation TEXT NOT NULL, event TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compache_cache_arrival_v1 (
                scope TEXT PRIMARY KEY, run TEXT NOT NULL, started_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compache_cache_reset_v1 (
                scope TEXT PRIMARY KEY, through_time REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compache_ping_stop_v1 (
                scope TEXT PRIMARY KEY, run TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compache_ping_foreground_v1 (
                scope TEXT PRIMARY KEY, run TEXT NOT NULL, reusable INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compache_cache_warning_v1 (
                id INTEGER PRIMARY KEY, scope TEXT NOT NULL, event TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS compache_cache_warning_scope_v1
                ON compache_cache_warning_v1(scope, id);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            self._db.close()

    def forget(self, scope_id: str) -> None:
        """Nothing will be asked of a scope any more. It goes with the next request of another."""
        self._gone.append(scope_id)

    def _clear(self) -> None:
        while self._gone:
            scope_id = self._gone.popleft()
            for table in _TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE scope=?", (scope_id,))

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()

    def _prior(self, scope_id):
        return self._db.execute(
            """SELECT g.*, f.reusable AS foreground_bound FROM compache_cache_guard_v1 g
               LEFT JOIN compache_ping_foreground_v1 f ON f.scope=g.scope AND f.run=g.run
               WHERE g.scope=?""",
            (scope_id,),
        ).fetchone()

    def _pending(self, scope_id):
        return self._db.execute(
            "SELECT run, started_at FROM compache_cache_arrival_v1 WHERE scope=?",
            (scope_id,),
        ).fetchone()

    def _before_reset(self, scope_id, started_at):
        return (
            self._db.execute(
                "SELECT 1 FROM compache_cache_reset_v1 WHERE scope=? AND through_time>=?",
                (scope_id, started_at),
            ).fetchone()
            is not None
        )

    def begin_return(self, scope: CacheScope, *, run_id, started_at):
        """Immediately cancel the old timer; only a successful response refreshes the cache."""
        scope_id = scope.fingerprint
        run = digest(_id(run_id, "run_id"))
        number(started_at, "started_at", -1e20)
        with self._transaction():
            self._clear()
            if self._before_reset(scope_id, started_at):
                return False
            prior = self._prior(scope_id)
            if prior is not None and (
                prior["run"] == run or started_at <= prior["started_at"]
            ):
                return False
            pending = self._pending(scope_id)
            if pending is not None and started_at <= pending["started_at"]:
                return pending["run"] == run and started_at == pending["started_at"]
            self._db.execute(
                "INSERT OR REPLACE INTO compache_cache_arrival_v1 VALUES (?, ?, ?)",
                (scope_id, run, started_at),
            )
        return True

    def prefix_changed(self, scope: CacheScope, *, system_prompt, tools):
        """Return changed component names relative to the last successful observation."""
        with self._lock:
            prior = self._prior(scope.fingerprint)
        if prior is None:
            return []
        return [
            name
            for name, value in (("system_prompt", system_prompt), ("tools", tools))
            if prior[name] != _fingerprint(value)
        ]

    def observe_return(
        self,
        scope: CacheScope,
        *,
        run_id,
        started_at,
        system_prompt,
        tools,
        ttl_seconds,
        reusable_input_tokens,
        request_eligible_input_tokens=None,
        cached_input_tokens=None,
        intentional_change=False,
        shared_prefix_tokens=None,
        verified_shared_static_tokens=None,
    ):
        """Record the first successful serving call on a user run, once.

        `reusable_input_tokens` is the block-rounded reusable input capacity of
        this request on a normal append. Use zero for ineligible requests.
        When the new publication covers fewer tokens than the request offered,
        pass its independent request_eligible_input_tokens for the read comparison.
        Set
        `cached_input_tokens` only from the provider's actual cache-read counter.
        `shared_prefix_tokens`, if supplied, is the independently established
        block-rounded upper bound on unchanged prefix tokens between this and
        the previous request (actual ordered wire). An estimate is insufficient.
        `verified_shared_static_tokens` is the known common static header length
        for a provider with a single contiguous cached-prefix read counter.
        A read that ends inside that header did not reach private thread history.
        Requests start in strictly increasing timestamp order within a scope.
        Set `intentional_change` for known history rewrites, required tool or
        system changes, or branching. Failed requests must not be recorded.
        """
        if not isinstance(scope, CacheScope):
            raise ContractError("CacheScope required")
        scope_id = scope.fingerprint
        run = digest(_id(run_id, "run_id"))
        number(started_at, "started_at", -1e20)
        number(ttl_seconds, "ttl_seconds")
        number(reusable_input_tokens, "reusable_input_tokens", integer=True)
        for label, value in [
            ("cached_input_tokens", cached_input_tokens),
            ("shared_prefix_tokens", shared_prefix_tokens),
            ("verified_shared_static_tokens", verified_shared_static_tokens),
        ]:
            if value is not None:
                number(value, label, integer=True)
        if type(intentional_change) is not bool:
            raise ContractError("intentional_change must be a boolean")
        system, toolset = _fingerprint(system_prompt), _fingerprint(tools)
        eligible = (
            reusable_input_tokens
            if request_eligible_input_tokens is None
            else request_eligible_input_tokens
        )
        number(
            eligible,
            "request_eligible_input_tokens",
            reusable_input_tokens,
            integer=True,
        )
        observation = digest(
            [
                float(started_at),
                float(ttl_seconds),
                reusable_input_tokens,
                cached_input_tokens,
                shared_prefix_tokens,
                verified_shared_static_tokens,
                intentional_change,
                system,
                toolset,
            ]
            + ([eligible] if eligible != reusable_input_tokens else [])
        )
        with self._transaction():
            prior = self._prior(scope_id)
            if prior is not None and prior["run"] == run:
                if prior["observation"] != observation:
                    raise ContractError("conflicting observation for this run")
                return json.loads(prior["event"])
            pending = self._pending(scope_id)
            if (
                pending is not None
                and pending["run"] == run
                and pending["started_at"] < started_at
            ):
                raise ContractError("conflicting start time for pending run")
            superseded = pending is not None and (
                started_at < pending["started_at"]
                or (started_at == pending["started_at"] and pending["run"] != run)
            )
            if (
                superseded
                or self._before_reset(scope_id, started_at)
                or (prior is not None and started_at <= prior["started_at"])
            ):
                return {
                    "event": "compache.cache_observation",
                    "scope_hash": scope_id,
                    "run_hash": run,
                    "outcome": "out_of_order_ignored",
                    "flagged": False,
                    "pings_disabled": bool(prior is not None and prior["disabled"]),
                }
            outcome, changed, age, expected = _classify(
                prior,
                started_at=started_at,
                ttl=ttl_seconds,
                reusable=eligible,
                reads=cached_input_tokens,
                shared=shared_prefix_tokens,
                static_tokens=verified_shared_static_tokens,
                intentional=intentional_change,
                system=system,
                tools=toolset,
            )
            suppress = outcome in (
                "on_time_cache_miss",
                "on_time_prefix_invalidated",
                "on_time_history_cache_miss",
            )
            flagged = suppress or outcome == "header_change_unconfirmed"
            previously_disabled = bool(prior is not None and prior["disabled"])
            result = {
                "event": "compache.cache_break"
                if flagged
                else "compache.cache_observation",
                "scope_hash": scope_id,
                "run_hash": run,
                "started_at": started_at,
                "outcome": outcome,
                "changed_components": changed,
                "idle_since_refresh_seconds": age,
                "ttl_seconds": ttl_seconds,
                "expected_reusable_tokens": expected,
                "observed_cached_tokens": cached_input_tokens,
                "shared_prefix_tokens": shared_prefix_tokens,
                "verified_shared_static_tokens": verified_shared_static_tokens,
                "flagged": flagged,
                "pings_disabled": previously_disabled or suppress,
                "newly_disabled": suppress and not previously_disabled,
            }
            serialized = json.dumps(result, separators=(",", ":"))
            self._db.execute(
                """INSERT OR REPLACE INTO compache_cache_guard_v1
                (scope, run, started_at, refreshed_at, ttl, reusable, system_prompt, tools,
                 disabled, observation, event) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scope_id,
                    run,
                    started_at,
                    started_at,
                    ttl_seconds,
                    reusable_input_tokens,
                    system,
                    toolset,
                    int(result["pings_disabled"]),
                    observation,
                    serialized,
                ),
            )
            self._db.execute(
                "DELETE FROM compache_cache_arrival_v1 WHERE scope=?", (scope_id,)
            )
            self._db.execute(
                "DELETE FROM compache_ping_foreground_v1 WHERE scope=?", (scope_id,)
            )
            if flagged:
                self._db.execute(
                    "INSERT INTO compache_cache_warning_v1(scope, event) VALUES (?, ?)",
                    (scope_id, serialized),
                )
        if flagged:
            logger.warning("%s", serialized, extra={"compache_cache_break": result})
        return result

    def can_ping(self, scope: CacheScope, *, run_id=None, at=None):
        """Check immediately before dispatch with current run ID and clock time."""
        if not isinstance(scope, CacheScope):
            raise ContractError("CacheScope required")
        if at is not None:
            number(at, "at", -1e20)
        run = digest(_id(run_id, "run_id")) if run_id is not None else None
        with self._lock:
            row = self._db.execute(
                """SELECT guard.*, EXISTS (SELECT 1 FROM compache_cache_arrival_v1 a
                WHERE a.scope=guard.scope) AS pending,
                EXISTS (SELECT 1 FROM compache_ping_stop_v1 s
                WHERE s.scope=guard.scope AND s.run=guard.run) AS stopped
                FROM compache_cache_guard_v1 guard
                WHERE guard.scope=?""",
                (scope.fingerprint,),
            ).fetchone()
        if run is None and at is None:
            return row is None or not row["disabled"]
        return bool(
            row is not None
            and not row["disabled"]
            and not row["pending"]
            and not row["stopped"]
            and row["ttl"] > 0
            and row["reusable"] > 0
            and (run is None or run == row["run"])
            and (at is None or 0 <= at - row["refreshed_at"] < row["ttl"])
        )

    def note_refresh(
        self, scope: CacheScope, *, run_id, refreshed_at, reusable_input_tokens=None
    ):
        """Apply a successful refresh and its optional known block-rounded capacity."""
        if not isinstance(scope, CacheScope):
            raise ContractError("CacheScope required")
        run = digest(_id(run_id, "run_id"))
        number(refreshed_at, "refreshed_at", -1e20)
        if reusable_input_tokens is not None:
            number(reusable_input_tokens, "reusable_input_tokens", integer=True)
        with self._transaction():
            cursor = self._db.execute(
                """UPDATE compache_cache_guard_v1 SET refreshed_at=?, reusable=COALESCE(?, reusable)
                WHERE scope=? AND run=? AND disabled=0 AND reusable>0
                  AND refreshed_at<=? AND ?-refreshed_at<ttl
                  AND NOT EXISTS (SELECT 1 FROM compache_ping_stop_v1
                                  WHERE scope=? AND run=?)
                  AND NOT EXISTS (SELECT 1 FROM compache_cache_arrival_v1 WHERE scope=?)""",
                (
                    refreshed_at,
                    reusable_input_tokens,
                    scope.fingerprint,
                    run,
                    refreshed_at,
                    refreshed_at,
                    scope.fingerprint,
                    run,
                    scope.fingerprint,
                ),
            )
            return cursor.rowcount == 1

    def _observe_ping(
        self,
        scope,
        *,
        run_id,
        at,
        completed_at,
        previous_refresh,
        status,
        request_eligible,
        unverified_layout,
        capacity,
        reads,
        input_tokens,
        output_tokens,
        spec,
        verified_static,
        modeled,
        modeled_read_share=1.0,
        write_tokens=None,
        ping_has_private_suffix=False,
    ):
        """Commit a normalized receipt only against the exact outstanding refresh."""
        scope_id, run = scope.fingerprint, digest(_id(run_id, "run_id"))
        with self._transaction():
            prior = self._prior(scope_id)
            if (
                prior is None
                or prior["run"] != run
                or prior["disabled"]
                or prior["reusable"] <= 0
                or prior["refreshed_at"] != previous_refresh
                or not 0 <= at - previous_refresh < prior["ttl"]
                or self._pending(scope_id) is not None
                or self._db.execute(
                    "SELECT 1 FROM compache_ping_stop_v1 WHERE scope=? AND run=?",
                    (scope_id, run),
                ).fetchone()
            ):
                return None

            # The first ping adds the new answer to the foreground input. Only
            # the previously held block-rounded prefix is expected to be read.
            expected = prior["reusable"]
            # Replay charges a fitted fraction of an otherwise matching prefix.
            # The physical capacity remains the reference for provider receipts.
            expected_reads = (
                modeled_read_tokens(spec, expected, modeled_read_share)
                if modeled
                else expected
            )
            if status != "completed":
                reuse = "unverified"
            elif unverified_layout:
                reuse = "prefix_capacity_unknown"
            elif request_eligible == 0:
                reuse = "below_cache_minimum"
            elif reads is None:
                reuse = "unknown"
            elif reads >= expected:
                reuse = "expected_reuse"
            elif reads == 0:
                reuse = "zero_read"
            elif verified_static is not None and reads <= verified_static < expected:
                reuse = "static_only"
            elif modeled and 0 < expected_reads <= reads:
                reuse = "expected_reuse"
            else:
                reuse = "partial_read"
            if status != "completed":
                outcome = "response_unsuccessful"
            elif reuse != "expected_reuse":
                outcome = reuse
            elif capacity is None:
                outcome = "input_unknown"
            elif output_tokens is None:
                outcome = "output_unknown"
            else:
                outcome = "refresh_confirmed"
            confirmed = outcome == "refresh_confirmed"
            remaining = None
            if reuse == "below_cache_minimum":
                remaining = 0
            elif reuse in ("zero_read", "static_only", "partial_read"):
                rounded = reads // spec.cache_block_tokens * spec.cache_block_tokens
                remaining = rounded if rounded >= spec.cache_min_tokens else 0
            # A repeated maintenance request can read its private suffix. A
            # foreground return can count only the earlier, shared prefix.
            private = ping_has_private_suffix or prior["foreground_bound"] is not None
            foreground = capacity if confirmed else remaining
            if private:
                bound = prior["foreground_bound"]
                if bound is None:
                    bound = expected
                foreground = bound if foreground is None else min(bound, foreground)
                self._db.execute(
                    "INSERT OR REPLACE INTO compache_ping_foreground_v1 VALUES (?, ?, ?)",
                    (scope_id, run, foreground),
                )
            # Model scenarios can stop their local chain, but cannot establish
            # a provider cache break for this namespace.
            suppress = not modeled and reuse in ("zero_read", "static_only")
            static_read = (
                min(reads, verified_static)
                if status == "completed"
                and reads is not None
                and verified_static is not None
                and not modeled
                else None
            )
            cost, cost_source = (
                _price_ping_receipt(
                    spec, input_tokens, reads, write_tokens, output_tokens
                )
                if status != "model_mismatch"
                else (None, "model_mismatch")
            )
            event = {
                "event": "compache.ping_receipt",
                "scope_hash": scope_id,
                "run_hash": run,
                "at": at,
                "completed_at": completed_at,
                "response_status": status,
                "outcome": outcome,
                "reuse_outcome": reuse,
                "evidence_source": "modeled" if modeled else "provider",
                "expected_reusable_tokens": expected,
                "expected_cache_read_tokens": expected_reads,
                "ping_has_private_suffix": private,
                "foreground_reusable_tokens": foreground,
                "input_tokens": None if modeled else input_tokens,
                "observed_cached_tokens": None if modeled else reads,
                "output_tokens": None if modeled else output_tokens,
                "cache_write_input_tokens": None if modeled else write_tokens,
                "modeled_input_tokens": input_tokens if modeled else None,
                "modeled_cached_tokens": reads if modeled else None,
                "modeled_cache_write_input_tokens": write_tokens if modeled else None,
                "modeled_output_tokens": output_tokens if modeled else None,
                "verified_static_read_tokens": static_read,
                "verified_private_read_tokens": reads - static_read
                if static_read is not None
                else None,
                "remaining_reuse_limit_tokens": remaining,
                "output_limit_tokens": spec.ping_output_tokens,
                "output_budget_exceeded": None
                if output_tokens is None
                else output_tokens > spec.ping_output_tokens,
                "output_within_budget": None
                if output_tokens is None
                else output_tokens <= spec.ping_output_tokens,
                "occupancy_anchored": status == "completed"
                and not modeled
                and not private
                and input_tokens is not None,
                "refresh_confirmed": confirmed,
                "refreshed_at": at if confirmed else None,
                "refreshed_reusable_tokens": capacity if confirmed else None,
                "receipt_cost_input_equivalents": cost,
                "receipt_cost_basis": cost_source,
                "flagged": suppress,
                "pings_disabled": suppress,
            }
            if confirmed:
                self._db.execute(
                    "UPDATE compache_cache_guard_v1 SET refreshed_at=?, reusable=? WHERE scope=?",
                    (at, capacity, scope_id),
                )
            else:
                event["continuation"] = {
                    "decision": "stop",
                    "reason": outcome,
                    "due_times": [],
                }
                if remaining is not None:
                    self._db.execute(
                        "UPDATE compache_cache_guard_v1 SET reusable=? WHERE scope=?",
                        (remaining, scope_id),
                    )
            # Another worker with an old timer must wait for repricing. Record
            # the refreshed prefix and close the run's timer gate atomically.
            if not confirmed or event["output_budget_exceeded"]:
                self._db.execute(
                    "INSERT OR REPLACE INTO compache_ping_stop_v1 VALUES (?, ?)",
                    (scope_id, run),
                )
            if suppress:
                self._db.execute(
                    "UPDATE compache_cache_guard_v1 SET disabled=1 WHERE scope=?",
                    (scope_id,),
                )
                serialized = json.dumps(event, separators=(",", ":"))
                self._db.execute(
                    "INSERT INTO compache_cache_warning_v1(scope, event) VALUES (?, ?)",
                    (scope_id, serialized),
                )
        if suppress:
            logger.warning("%s", serialized, extra={"compache_cache_break": event})
        return event

    def _resume_repriced_pings(self, scope, *, run_id, refreshed_at):
        """Resume timers only for the same accepted receipt, if the user is still idle."""
        scope_id, run = scope.fingerprint, digest(run_id)
        with self._transaction():
            cursor = self._db.execute(
                """DELETE FROM compache_ping_stop_v1 WHERE scope=? AND run=?
                   AND EXISTS (SELECT 1 FROM compache_cache_guard_v1
                               WHERE scope=? AND run=? AND refreshed_at=? AND disabled=0)
                   AND NOT EXISTS (SELECT 1 FROM compache_cache_arrival_v1 WHERE scope=?)""",
                (scope_id, run, scope_id, run, refreshed_at, scope_id),
            )
            return cursor.rowcount == 1

    def reset(self, scope: CacheScope):
        """Forget carry state after an operator fixes instability, keeping the audit."""
        scope_id = scope.fingerprint
        with self._transaction():
            # A delayed completion from before the reset cannot re-arm an old timer.
            times = [
                row["started_at"]
                for row in (self._prior(scope_id), self._pending(scope_id))
                if row is not None
            ]
            if times:
                self._db.execute(
                    """INSERT INTO compache_cache_reset_v1 VALUES (?, ?)
                    ON CONFLICT(scope) DO UPDATE SET through_time=max(through_time, excluded.through_time)""",
                    (scope_id, max(times)),
                )
            self._db.execute(
                "DELETE FROM compache_cache_guard_v1 WHERE scope=?", (scope_id,)
            )
            self._db.execute(
                "DELETE FROM compache_cache_arrival_v1 WHERE scope=?", (scope_id,)
            )
            self._db.execute(
                "DELETE FROM compache_ping_stop_v1 WHERE scope=?", (scope_id,)
            )
            self._db.execute(
                "DELETE FROM compache_ping_foreground_v1 WHERE scope=?", (scope_id,)
            )

    def events(self, scope: CacheScope, limit=100):
        """Return the newest warning observations for one cache namespace."""
        number(limit, "limit", 1, 10_000, integer=True)
        with self._lock:
            rows = self._db.execute(
                """SELECT event FROM compache_cache_warning_v1
                WHERE scope=? ORDER BY id DESC LIMIT ?""",
                (scope.fingerprint, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
