"""Pure RAG request projection. The gate never sees labels or a target answer."""

from __future__ import annotations

import math
import re
import copy
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache

from compache.traces import ContractError, Thread
from .contracts import Policy, canonical, digest, number
from .ingress import Delta, make_delta
from .position import position_risk


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
    deltas: dict[str, Delta] = field(default_factory=dict)
    last_run_id: str | None = None


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
    recalled_parts: int = 0
    referenced_part_ids: frozenset[str] = frozenset()
    ingress_enabled: bool = False
    warm_overflow_risk: dict | None = None

    def telemetry(self):
        return {
            "window": self.window,
            "action": self.action,
            "prefix_transition": self.prefix_transition,
            "tokens_estimated": self.tokens,
            "removed_parts": len(self.removed_part_ids),
            "recalled_parts": self.recalled_parts,
            "policy_hash": self.policy_hash,
            "admitted": self.admitted,
            "reason": self.reason,
            **({"warm_overflow_risk": self.warm_overflow_risk} if self.warm_overflow_risk else {}),
            **({"verbatim_referenced_retrievals": len(self.referenced_part_ids),
                "literal_retrievals": sum(p.kind == "retrieval"
                                         and p.part_id not in self.removed_part_ids
                                         and p.part_id not in self.referenced_part_ids
                                         for p in self.parts)}
               if self.ingress_enabled else {}),
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


def _contains(text: str, value: str) -> bool:
    return bool(value) and bool(
        re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", text, re.I)
    )


@lru_cache(maxsize=1024)
def _identifiers(text: str):
    return frozenset(
        token.lower()
        for token in re.findall(
            r"(?<![\w-])[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*(?![\w-])", text
        )
        if len(token) >= 5
        and any(c.isalpha() for c in token)
        and (
            any(c.isdigit() for c in token)
            or (token.isupper() and ("-" in token or "_" in token))
        )
    )


def protected_sources(request: Request):
    """Protect plausible relevant evidence; this is not a semantic guarantee."""
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
    vocab = [terms((p.text or "") + " " + p.document_id) for p in old + current]
    rare = {
        w
        for w in q
        if len(w) >= 5 and sum(w in v for v in vocab) <= max(1, len(vocab) / 2)
    }
    qids = _identifiers(query)
    versions = {
        p.version for p in old + current if p.version and _contains(query, p.version)
    }
    chunks = {
        p.chunk_id for p in old + current if p.chunk_id and _contains(query, p.chunk_id)
    }
    protected = set()

    def select(p):
        duplicated = any(
            (p.document_id, p.version, p.chunk_id, p.text)
            == (c.document_id, c.version, c.chunk_id, c.text)
            for c in current
        )
        if not duplicated or _contains(query, p.part_id):
            protected.add(p.part_id)

    for p, words in zip(old, vocab):
        content = " ".join((p.text or "", p.document_id, p.chunk_id, p.part_id))
        explicit = any(
            _contains(query, x) for x in (p.part_id, p.chunk_id, p.document_id)
        )
        if explicit:
            if (not versions or p.version in versions) and (
                not chunks or p.chunk_id in chunks
            ):
                select(p)
        elif qids and qids & _identifiers(content):
            select(p)
        elif not qids and len(rare & words) >= 2:
            select(p)

    outputs = [
        p
        for p in request.parts
        if p.kind == "output" and p.run_index < request.run_index and p.text
    ]
    turns = {int(n) for n in re.findall(r"\bturn\s+(\d+)\b", query, re.I)}
    relevant = [
        a
        for a in outputs
        if a.run_index in turns
        or (qids and qids & _identifiers(a.text or ""))
        or (not qids and len(rare & terms(a.text or "")) >= 2)
    ]
    if (
        not relevant
        and not qids
        and not turns
        and outputs
        and re.search(r"\b(?:previous|earlier|prior|above|last)\b", query, re.I)
    ):
        relevant = [max(outputs, key=lambda a: a.run_index)]
    for a in relevant:
        text = a.text or ""
        documents = {p.document_id for p in old if p.document_id}
        # Brackets delimit a citation group. Associate each document with the
        # identifiers following it, rather than applying every version/chunk in
        # the answer to every document. The preceding prose provides claim
        # context even when several citations are coalesced in one bracket.
        groups = list(re.finditer(r"\[([^\]]+)\]", text))
        if not groups:
            groups = [re.match(r"(.*)", text, re.S)]
        previous_end = 0
        for group in groups:
            citation = group.group(1)
            claim = text[previous_end:group.start()]
            previous_end = group.end()
            # Bare bracket entries may identify a part or chunk without a
            # document. Match whole entries only: prose is not a citation.
            if groups and group.group(0).startswith("["):
                claim_ids = _identifiers(claim)
                claim_relevant = not (
                    qids and claim_ids and not qids & claim_ids
                    and a.run_index not in turns
                )
                if claim_relevant:
                    for entry in re.split(r"[;,]", citation):
                        key = entry.strip()
                        visible = [p for p in old if p.run_index <= a.run_index]
                        parts = [p for p in visible if p.part_id.lower() == key.lower()]
                        chunks = [p for p in visible if p.chunk_id and p.chunk_id.lower() == key.lower()]
                        # A chunk name shared by documents cannot identify which
                        # document was cited. Repetitions within one document
                        # remain possible occurrences and are all protected.
                        if not parts and len({p.document_id for p in chunks}) == 1:
                            parts = chunks
                        for p in parts:
                            if (qids and len(claim_ids) > 1 and a.run_index not in turns
                                and _identifiers(p.text or "")
                                and not qids & _identifiers(p.text or "")):
                                continue
                            select(p)
            mentions = sorted(
                (m.start(), m.end(), doc)
                for doc in documents
                for m in re.finditer(
                    r"(?<![\w-])" + re.escape(doc) + r"(?![\w-])",
                    citation, re.I,
                )
            )
            for i, (start, end, doc) in enumerate(mentions):
                scope = citation[start:mentions[i + 1][0] if i + 1 < len(mentions) else len(citation)]
                candidates = [
                    p for p in old
                    if p.document_id == doc and p.run_index <= a.run_index
                ]
                cited_versions = {p.version for p in candidates if p.version and _contains(scope, p.version)}
                cited_chunks = {p.chunk_id for p in candidates if p.chunk_id and _contains(scope, p.chunk_id)}
                cited_parts = {p.part_id for p in candidates if _contains(scope, p.part_id)}
                if cited_versions:
                    candidates = [p for p in candidates if p.version in cited_versions]
                if cited_chunks:
                    candidates = [p for p in candidates if p.chunk_id in cited_chunks]
                if cited_parts:
                    candidates = [p for p in candidates if p.part_id in cited_parts]
                if not candidates:
                    continue
                claim_ids = _identifiers(claim)
                if qids and claim_ids and not qids & claim_ids and a.run_index not in turns:
                    continue
                # A chunk or occurrence citation is an explicit reference. Keep
                # every matching occurrence because provenance remains ambiguous.
                if cited_chunks or cited_parts:
                    for p in candidates:
                        select(p)
                    continue
                matches = [p for p in candidates if qids & _identifiers(p.text or "")]
                if not qids:
                    matches = [p for p in candidates if len(rare & terms(p.text or "")) >= 2]
                if matches:
                    for p in matches:
                        select(p)
                    continue
                # With multiple entities in a coalesced claim there is no safe
                # basis for assigning this document to the requested one.
                if qids and len(claim_ids) > 1 and a.run_index not in turns:
                    continue
                # The answer's own retrievals are the strongest available
                # provenance. If none exist, retain the most recent visible
                # occurrence rather than fanning out through the entire history.
                latest = max(p.run_index for p in candidates)
                for p in candidates:
                    if p.run_index == latest:
                        select(p)
    return protected


def _placeholder_for(request: Request, state: ThreadState, source: Source):
    # Expansion aliases are not in the canonical request. Their placeholder
    # must carry the canonical source's reopen handle, not an unusable alias.
    original_id = next(
        (
            original
            for values in state.expansions.values()
            for original, alias in values.items()
            if alias.part_id == source.part_id
        ),
        None,
    )
    original = next((p for p in request.parts if p.part_id == original_id), source)
    return placeholder(request.thread_id, original)


def _plan_ingress(request, state, min_chars):
    # A source is encoded once, before its first send. Only literal, visible
    # sources may supply lines so a reference can always be resolved in one hop.
    anchors = {}
    for p in request.parts:
        if (p.kind != "retrieval" or not p.text or not p.document_id or not p.version
                or p.part_id in state.removed or p.part_id in state.deltas):
            continue
        family = anchors.setdefault((p.document_id, p.version), [])
        if p.part_id not in state.fingerprints and family:
            delta = make_delta(p, family, handle(request.thread_id, p), min_chars)
            if delta:
                state.deltas[p.part_id] = delta
                continue
        family.append(p)


def _project(request: Request, state: ThreadState, removed: set[str]):
    result = []
    inserted = set()

    def project(p):
        if p.part_id in removed:
            return replace(p, text=_placeholder_for(request, state, p))
        if delta := state.deltas.get(p.part_id):
            if delta.anchor_id in removed:
                raise ContractError("referenced retrieval lost its visible anchor")
            return replace(p, text=delta.text)
        return p

    for p in request.parts:
        if p.kind == "output" and p.run_index not in inserted:
            result.extend(
                project(x) for x in state.expansions.get(p.run_index, {}).values()
            )
            inserted.add(p.run_index)
        result.append(project(p))
    result.extend(
        project(x) for x in state.expansions.get(request.run_index, {}).values()
    )
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
    token_estimator=None,
) -> Prepared:
    number(now, "now", -1e20)
    number(overhead_tokens, "overhead_tokens", integer=True)

    def estimate(payload):
        value = (
            payload + overhead_tokens
            if token_estimator is None
            else token_estimator(payload, overhead_tokens)
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("token estimator must return a nonnegative integer")
        return value

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
        or old_policy.spec.cache_ttl_seconds <= 0
        or now - last_use
        >= old_policy.spec.cache_ttl_seconds + old_policy.freshness_margin_seconds
    )
    # Policy changes latch only at cold starts, preserving a warm prefix.
    active = policy if cold else old_policy
    state.policy = active
    if cold and active.position_growth is not None:
        state.deltas.clear()
    spec = active.spec
    existing = {p.part_id: p.fingerprint for p in request.parts}
    usable = tuple(existing.items())[: len(state.fingerprints)] == tuple(
        state.fingerprints.items()
    )
    if not usable or state.last_model not in (None, request.model):
        state.removed.clear()
        state.expansions.clear()
        state.deltas.clear()
        state.fingerprints.clear()
    observe_only = request.model != spec.model_id or any(
        p.text is None for p in request.parts
    )
    protected = set(protected_ids)
    if active.protect_query_matches and state.removed:
        protected |= protected_sources(request)
    action = (
        "applied_warm"
        if not cold and usable
        else ("planned_cold" if cold else "planned_prefix_changed")
    )
    old_removed = set(state.removed)
    deferred_warm = False
    warm_risk = None
    recalled = 0
    if observe_only:
        # Retain previous placeholders if already visible; no new cuts on incomplete data.
        action = "observe_only"
    else:
        if active.ingress_match_min_chars is not None:
            _plan_ingress(request, state, active.ingress_match_min_chars)
        originals = {p.part_id: p for p in request.parts if p.kind == "retrieval"}
        expanded_full = {
            original
            for values in state.expansions.values()
            for original, source in values.items()
            if source.part_id not in state.removed
            and original in originals
            and source.text is not None
            and source.text == originals[original].text
        }
        used_ids = {p.part_id for p in request.parts} | {
            source.part_id
            for values in state.expansions.values()
            for source in values.values()
        }
        for original_id in sorted(
            protected & state.removed & originals.keys() - expanded_full
        ):
            original = originals[original_id]
            if original_id in state.expansions.get(request.run_index, {}):
                raise ContractError(
                    "current-run expansion already occupies recall identity"
                )
            alias = (
                request.run_id
                + "/recall/"
                + digest([original_id, original.fingerprint])[:24]
            )
            if alias in used_ids:
                raise ContractError("recall expansion identity collision")
            state.expansions.setdefault(request.run_index, {})[original_id] = replace(
                original, part_id=alias, run_index=request.run_index
            )
            used_ids.add(alias)
            recalled += 1
        projected = _project(request, state, state.removed)
        payload = sum(spec.part_tokens(p.text) for p in projected)
        size = estimate(payload)
        wire_bytes = sum(len(wire_part(p).encode()) for p in projected)
        threshold = active.cold_trigger if cold or not usable else active.warm_trigger
        pressure = max(
            size,
            math.ceil(wire_bytes * spec.context_tokens / spec.byte_limit),
        )
        needs_admission_reduction = (
            size > (1 - active.reserve) * spec.context_tokens
            or wire_bytes > spec.byte_limit
        )
        target_tokens = active.target * spec.context_tokens
        should_replan = pressure > threshold * spec.context_tokens
        if active.position_growth is not None and not cold and usable:
            if state.last_run_id == request.run_id:
                # Reopening a source is a later call in the same run. Its current
                # input can already contain part of the fitted start-to-peak growth.
                should_replan = False
            else:
                forecast = position_risk(active, request.run_index, size, "remaining_tokens")
                warm_risk = {"before": forecast, "admission_override": needs_admission_reduction}
                if forecast["triggered"] is not None:
                    should_replan = forecast["triggered"]
                    ceiling = forecast["warm_ceiling_tokens"]
                    if should_replan and ceiling is not None:
                        target_tokens = min(target_tokens, max(0, ceiling))
        if (
            (should_replan or needs_admission_reduction)
            and active.retrieval_grace_runs is not None
        ):
            if active.protect_query_matches and not old_removed:
                protected |= protected_sources(request)
            expansion_original = {
                source.part_id: original
                for values in state.expansions.values()
                for original, source in values.items()
            }
            candidates = list(request.parts) + [
                source
                for values in state.expansions.values()
                for source in values.values()
            ]
            dependents = {}
            for part_id, delta in state.deltas.items():
                dependents.setdefault(delta.anchor_id, set()).add(part_id)
            # Consider dependent views first. Their anchor becomes removable
            # only after every remaining reference to it has been removed.
            for p in sorted(candidates, key=lambda p: (p.part_id in dependents, p.run_index, p.part_id)):
                if pressure <= target_tokens:
                    break
                age = request.run_index - p.run_index
                if (
                    p.kind != "retrieval"
                    or age <= active.retrieval_grace_runs
                    or p.part_id in protected
                    or expansion_original.get(p.part_id) in protected
                    or p.part_id in state.removed
                    or dependents.get(p.part_id, set()) - state.removed
                ):
                    continue
                placeholder_text = _placeholder_for(request, state, p)
                current = replace(p, text=state.deltas[p.part_id].text) if p.part_id in state.deltas else p
                saving = spec.part_tokens(current.text) - spec.part_tokens(placeholder_text)
                if saving > 0:
                    state.removed.add(p.part_id)
                    payload -= saving
                    size = estimate(payload)
                    compacted = replace(p, text=placeholder_text)
                    wire_bytes -= len(wire_part(current).encode()) - len(
                        wire_part(compacted).encode()
                    )
                    pressure = max(
                        size,
                        math.ceil(wire_bytes * spec.context_tokens / spec.byte_limit),
                    )
            if not cold and usable and state.removed != old_removed:
                risk_reduced = False
                if warm_risk and warm_risk["before"]["triggered"] is True:
                    after_cut = position_risk(active, request.run_index, size, "remaining_tokens")
                    risk_reduced = (after_cut["rank_adjusted_exceedance"]
                                    < warm_risk["before"]["rank_adjusted_exceedance"])
                if pressure > target_tokens and not needs_admission_reduction and not risk_reduced:
                    # A warm rewrite is costly. If protected context prevents
                    # reaching the target, keep the cached view until a cold
                    # return or an actual admission constraint requires a cut.
                    # Cold compaction can still take every safe available saving.
                    state.removed = old_removed
                    deferred_warm = True
                    action = "deferred_warm"
                else:
                    action = (
                        "replaced_warm"
                        if not needs_admission_reduction
                        else "replaced_warm_for_admission"
                    )
    projected = _project(request, state, state.removed)
    wire = tuple(wire_part(p) for p in projected)
    size = estimate(sum(spec.part_tokens(p.text) for p in projected))
    if warm_risk is not None:
        warm_risk["after"] = position_risk(active, request.run_index, size, "remaining_tokens")
    admitted = observe_only or (
        size <= (1 - active.reserve) * spec.context_tokens
        and sum(len(x.encode()) for x in wire) <= spec.byte_limit
    )
    reason = (
        (
            "unknown_model_or_masked_content_admission_not_verified"
            if observe_only
            else "warm_compaction_target_unreachable"
            if deferred_warm
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
    canonical_retrievals = {
        p.part_id: p for p in request.parts if p.kind == "retrieval"
    }
    originals_restored = {
        part_id
        for values in state.expansions.values()
        for part_id, source in values.items()
        if source.part_id not in state.removed
        and part_id in canonical_retrievals
        and source.text is not None
        and source.text == canonical_retrievals[part_id].text
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
        recalled,
        frozenset(state.deltas.keys() - state.removed),
        active.ingress_match_min_chars is not None or bool(state.deltas),
        warm_risk,
    )
    # Admission failures do not record an unsent request or permanently cut sources.
    if admitted:
        state.last_run_id = request.run_id
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
    token_estimator=None,
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
        token_estimator=token_estimator,
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
            if handle(request.thread_id, p) == source_handle
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
