"""Pure RAG request projection. The gate never sees labels or a target answer."""

from __future__ import annotations

import math
import re
import copy
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache

from compache.traces import ContractError, Thread
from .contracts import Policy, canonical, digest, number


@dataclass(frozen=True)
class Source:
    part_id: str
    kind: str
    run_index: int
    text: str | None
    document_id: str = ""
    version: str = ""
    chunk_id: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.part_id, str)
            or not self.part_id
            or self.kind not in ("input", "retrieval", "output")
        ):
            raise ContractError("invalid source identity/kind")
        number(self.run_index, "source run_index", 1, integer=True)
        if self.text is not None and not isinstance(self.text, str):
            raise ContractError("source text must be string or null")

    @cached_property
    def fingerprint(self):
        return digest({k: getattr(self, k) for k in self.__dataclass_fields__})

    @cached_property
    def wire(self):
        return (
            canonical({"id": self.part_id, "kind": self.kind, "text": self.text}) + "\n"
        )


@dataclass(frozen=True)
class Request:
    thread_id: str
    run_id: str
    run_index: int
    model: str
    parts: tuple[Source, ...]

    @classmethod
    def from_thread(cls, thread: Thread, run_id: str):
        parts = []
        for r in thread.runs:
            parts.append(Source(r.input.part_id, "input", r.run_index, r.input.text))
            parts.extend(
                Source(
                    p.part_id,
                    "retrieval",
                    r.run_index,
                    p.text,
                    p.document_id,
                    p.version,
                    p.chunk_id,
                )
                for p in r.retrievals
                if p.exposed_to_model
            )
            if r.run_id == run_id:
                return cls(
                    thread.thread_id,
                    r.run_id,
                    r.run_index,
                    r.generation.model,
                    tuple(parts),
                )
            parts.append(Source(r.output.part_id, "output", r.run_index, r.output.text))
        raise ContractError("unknown run for request")

    def __post_init__(self):
        for value in (self.thread_id, self.run_id, self.model):
            if not isinstance(value, str) or not value:
                raise ContractError("request identities must be nonempty strings")
        number(self.run_index, "request run_index", 1, integer=True)
        ids = [p.part_id for p in self.parts]
        if len(ids) != len(set(ids)) or any(
            p.run_index > self.run_index for p in self.parts
        ):
            raise ContractError("request contains duplicate or future parts")
        if any(
            p.kind not in ("input", "retrieval", "output")
            or (p.kind == "output" and p.run_index == self.run_index)
            for p in self.parts
        ):
            raise ContractError("invalid source kind or current answer in request")


@dataclass
class ThreadState:
    thread_id: str
    policy: Policy | None = None
    removed: set[str] = field(default_factory=set)
    fingerprints: dict[str, str] = field(default_factory=dict)
    expansions: dict[int, dict[str, Source]] = field(default_factory=dict)
    last_wire: tuple[str, ...] = ()
    last_started_at: float | None = None
    last_model: str | None = None
    refreshed_at: float | None = None


@dataclass(frozen=True)
class Prepared:
    parts: tuple[Source, ...]
    full_part_ids: frozenset[str]
    removed_part_ids: frozenset[str]
    tokens: int
    wire: tuple[str, ...]
    window: str
    action: str
    prefix_transition: str
    policy_hash: str
    admitted: bool
    reason: str | None

    def telemetry(self):
        return {
            "window": self.window,
            "action": self.action,
            "prefix_transition": self.prefix_transition,
            "tokens_estimated": self.tokens,
            "removed_parts": len(self.removed_part_ids),
            "policy_hash": self.policy_hash,
            "admitted": self.admitted,
            "reason": self.reason,
        }


def handle(thread_id: str, source: Source) -> str:
    return "ctx:" + digest([thread_id, source.part_id, source.fingerprint])[:24]


def placeholder(thread_id: str, source: Source) -> str:
    return f"[Compacted passage; expand {handle(thread_id, source)} before relying on its contents.]"


def wire_part(source: Source) -> str:
    return source.wire


@lru_cache(maxsize=1024)
def terms(text: str):
    stop = {
        "what",
        "which",
        "does",
        "that",
        "this",
        "their",
        "there",
        "with",
        "from",
        "have",
        "would",
        "could",
        "should",
        "about",
        "after",
        "before",
        "within",
        "much",
        "many",
        "each",
        "know",
        "please",
        "customer",
        "company",
    }
    result = set()
    for word in re.findall(r"[a-z]{4,}", text.lower()):
        if word in stop:
            continue
        for suffix in ("ing", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        result.add(word)
    return frozenset(result)


def protected_sources(request: Request):
    """Conservative lexical guard, not a semantic dependency classifier."""
    old = [
        p
        for p in request.parts
        if p.kind == "retrieval" and p.run_index < request.run_index
    ]
    current = [
        p
        for p in request.parts
        if p.kind == "retrieval" and p.run_index == request.run_index
    ]
    query = " ".join(
        p.text or ""
        for p in request.parts
        if p.kind == "input" and p.run_index == request.run_index
    )
    q = terms(query)
    # Company/product words shared by most passages are not distinctive guards.
    vocab = [terms((p.text or "") + " " + p.document_id) for p in old + current]
    common = {
        w for w in q if len(vocab) >= 3 and sum(w in v for v in vocab) > len(vocab) / 2
    }
    q -= common
    protected = set()
    for p in old:
        duplicated = any(
            (p.document_id, p.version, p.chunk_id, p.text)
            == (c.document_id, c.version, c.chunk_id, c.text)
            for c in current
        )
        cited = any(
            p.document_id and p.document_id in (a.text or "")
            for a in request.parts
            if a.kind == "output"
        )
        if cited or (
            not duplicated and q & terms((p.text or "") + " " + p.document_id)
        ):
            protected.add(p.part_id)
    return protected


def _project(request: Request, state: ThreadState, removed: set[str]):
    result = []
    inserted = set()
    for p in request.parts:
        if p.kind == "output" and p.run_index not in inserted:
            result.extend(state.expansions.get(p.run_index, {}).values())
            inserted.add(p.run_index)
        result.append(
            replace(p, text=placeholder(request.thread_id, p))
            if p.part_id in removed
            else p
        )
    result.extend(state.expansions.get(request.run_index, {}).values())
    return tuple(result)


def _prepare(
    request: Request,
    policy: Policy,
    state: ThreadState,
    now: float,
    *,
    force_cold=False,
    protected_ids=frozenset(),
    overhead_tokens=0,
) -> Prepared:
    number(now, "now", -1e20)
    number(overhead_tokens, "overhead_tokens", integer=True)
    if request.thread_id != state.thread_id:
        raise ContractError("state belongs to another thread")
    if state.last_started_at is not None and now < state.last_started_at:
        raise ContractError("request clock moved backwards")
    old_policy = state.policy or policy
    last_use = (
        max(t for t in (state.last_started_at, state.refreshed_at) if t is not None)
        if state.last_started_at is not None
        else None
    )
    cold = (
        force_cold
        or last_use is None
        or state.last_model != request.model
        or now - last_use
        >= old_policy.spec.cache_ttl_seconds - old_policy.freshness_margin_seconds
    )
    # Policy changes latch only at cold starts, preserving a warm prefix.
    active = policy if cold else old_policy
    state.policy = active
    spec = active.spec
    existing = {p.part_id: p.fingerprint for p in request.parts}
    usable = tuple(existing.items())[: len(state.fingerprints)] == tuple(
        state.fingerprints.items()
    )
    if not usable or state.last_model not in (None, request.model):
        state.removed.clear()
        state.expansions.clear()
    observe_only = request.model != spec.model_id or any(
        p.text is None for p in request.parts
    )
    protected = set(protected_ids)
    if active.protect_query_matches:
        protected |= protected_sources(request)
    action = (
        "applied_warm"
        if not cold and usable
        else ("planned_cold" if cold else "planned_prefix_changed")
    )
    old_removed = set(state.removed)
    if observe_only:
        # Retain previous placeholders if already visible; no new cuts on incomplete data.
        action = "observe_only"
    else:
        projected = _project(request, state, state.removed)
        size = overhead_tokens + sum(spec.part_tokens(p.text) for p in projected)
        wire_bytes = sum(len(wire_part(p).encode()) for p in projected)
        threshold = active.cold_trigger if cold or not usable else active.warm_trigger
        pressure = max(
            size,
            math.ceil(wire_bytes * spec.context_tokens / spec.byte_limit),
        )
        if (
            pressure > threshold * spec.context_tokens
            and active.retrieval_grace_runs is not None
        ):
            for p in sorted(request.parts, key=lambda p: (p.run_index, p.part_id)):
                if pressure <= active.target * spec.context_tokens:
                    break
                age = request.run_index - p.run_index
                if (
                    p.kind != "retrieval"
                    or age <= active.retrieval_grace_runs
                    or p.part_id in protected
                    or p.part_id in state.removed
                ):
                    continue
                saving = spec.part_tokens(p.text) - spec.part_tokens(
                    placeholder(request.thread_id, p)
                )
                if saving > 0:
                    state.removed.add(p.part_id)
                    size -= saving
                    compacted = replace(p, text=placeholder(request.thread_id, p))
                    wire_bytes -= len(wire_part(p).encode()) - len(
                        wire_part(compacted).encode()
                    )
                    pressure = max(
                        size,
                        math.ceil(wire_bytes * spec.context_tokens / spec.byte_limit),
                    )
            if not cold and usable and state.removed != old_removed:
                action = "replaced_warm"
    projected = _project(request, state, state.removed)
    wire = tuple(wire_part(p) for p in projected)
    size = overhead_tokens + sum(spec.part_tokens(p.text) for p in projected)
    admitted = observe_only or (
        size <= (1 - active.reserve) * spec.context_tokens
        and sum(len(x.encode()) for x in wire) <= spec.byte_limit
    )
    reason = (
        (
            "unknown_model_or_masked_content_admission_not_verified"
            if observe_only
            else None
        )
        if admitted
        else "protected_or_current_context_exceeds_estimated_limit"
    )
    transition = (
        "new"
        if not state.last_wire
        else (
            "extended"
            if wire[: len(state.last_wire)] == state.last_wire
            else "rewritten"
        )
    )
    originals_restored = {
        part_id for values in state.expansions.values() for part_id in values
    }
    full = ({p.part_id for p in request.parts} - state.removed) | originals_restored
    result = Prepared(
        projected,
        frozenset(full),
        frozenset(state.removed),
        size,
        wire,
        "cold" if cold else "warm",
        action,
        transition,
        active.hash,
        admitted,
        reason,
    )
    # Admission failures do not record an unsent request or permanently cut sources.
    if admitted:
        state.fingerprints = existing
        state.last_wire, state.last_started_at, state.last_model = (
            wire,
            now,
            request.model,
        )
        state.refreshed_at = None
    else:
        state.removed = old_removed
    return result


def prepare(
    request: Request,
    policy: Policy,
    state: ThreadState,
    now: float,
    *,
    force_cold=False,
    protected_ids=frozenset(),
    overhead_tokens=0,
) -> Prepared:
    """Plan deterministically; commit explicit state only after successful admission."""
    proposed = copy.deepcopy(state)
    result = _prepare(
        request,
        policy,
        proposed,
        now,
        force_cold=force_cold,
        protected_ids=protected_ids,
        overhead_tokens=overhead_tokens,
    )
    if result.admitted:
        state.__dict__.update(proposed.__dict__)
    return result


def expand(request: Request, source_handle: str, offset=0, limit=48_000) -> dict:
    number(offset, "offset", integer=True)
    number(limit, "limit", 1, 48_000, integer=True)
    source = next(
        (
            p
            for p in request.parts
            if p.kind == "retrieval" and handle(request.thread_id, p) == source_handle
        ),
        None,
    )
    if source is None or source.text is None:
        raise ContractError("unknown or unavailable expansion handle in this request")
    if offset > len(source.text):
        raise ContractError("expansion offset exceeds content length")
    end = min(len(source.text), offset + limit)
    return {
        "part_id": source.part_id,
        "text": source.text[offset:end],
        "next_offset": end if end < len(source.text) else None,
    }
