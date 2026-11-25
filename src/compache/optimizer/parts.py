"""A request, the parts it is made of, and handles to parts that were archived."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

from compache.threads import ContractError, Thread, citation_ids

from .contracts import canonical, digest, number


@dataclass(frozen=True)
class Source:
    part_id: str
    kind: str
    run_index: int
    text: str | None
    document_id: str = ""
    version: str = ""
    chunk_id: str = ""
    # Adapter metadata, never part of the rendered prefix or expansion handle.
    cited_part_ids: tuple[str, ...] = field(default=(), compare=False)
    # Whether it holds more than its text, and whether no notice can stand for that.
    more: bool = field(default=False, compare=False)
    stays: bool = field(default=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.part_id, str)
            or not self.part_id
            or self.kind not in ("input", "retrieval", "output")
        ):
            raise ContractError("invalid source identity/kind")
        number(self.run_index, "source run_index", 1, integer=True)
        if self.cited_part_ids != ():
            object.__setattr__(
                self, "cited_part_ids", citation_ids(self.cited_part_ids)
            )
        if self.cited_part_ids and self.kind != "output":
            raise ContractError("only an observed output can cite retrieved parts")
        if self.text is not None and not isinstance(self.text, str):
            raise ContractError("source text must be string or null")

    @cached_property
    def fingerprint(self):
        return digest(
            {
                k: getattr(self, k)
                for k in self.__dataclass_fields__
                if k != "cited_part_ids"
            }
        )

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
            parts.append(
                Source(
                    r.output.part_id,
                    "output",
                    r.run_index,
                    r.output.text,
                    cited_part_ids=r.cited_part_ids,
                )
            )
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
        if any(p.cited_part_ids for p in self.parts):
            available = {}
            for p in self.parts:
                if p.kind == "retrieval":
                    available[p.part_id] = p.run_index
                if any(
                    pid not in available or available[pid] > p.run_index
                    for pid in p.cited_part_ids
                ):
                    raise ContractError(
                        "answer citations must reference earlier exposed retrievals"
                    )
        if any(
            p.kind not in ("input", "retrieval", "output")
            or (p.kind == "output" and p.run_index == self.run_index)
            for p in self.parts
        ):
            raise ContractError("invalid source kind or current answer in request")


def handle(thread_id: str, source: Source) -> str:
    return "ctx:" + digest([thread_id, source.part_id, source.fingerprint])[:24]


def notice(reference: str, said: bool = False) -> str:
    """What stands in the place of an archived part, and how to see it again.

    It says that the part was seen in full. A model that only finds a reference where text
    should be takes what it wrote from that text for made up, and takes it back.
    """
    seen = "said" if said else "read"
    return f'[You {seen} this in full. Cleared since: reopen("{reference}") shows it.]'


def placeholder(thread_id: str, source: Source) -> str:
    return notice(handle(thread_id, source), source.kind == "output")


# Every notice is this long: a handle is always 28 characters.
NOTICE_BYTES = len(notice("")) + 28


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
            claim = text[previous_end : group.start()]
            previous_end = group.end()
            # Bare bracket entries may identify a part or chunk without a
            # document. Match whole entries only: prose is not a citation.
            if groups and group.group(0).startswith("["):
                claim_ids = _identifiers(claim)
                claim_relevant = not (
                    qids
                    and claim_ids
                    and not qids & claim_ids
                    and a.run_index not in turns
                )
                if claim_relevant:
                    for entry in re.split(r"[;,]", citation):
                        key = entry.strip()
                        visible = [p for p in old if p.run_index <= a.run_index]
                        parts = [p for p in visible if p.part_id.lower() == key.lower()]
                        chunks = [
                            p
                            for p in visible
                            if p.chunk_id and p.chunk_id.lower() == key.lower()
                        ]
                        # A chunk name shared by documents cannot identify which
                        # document was cited. Repetitions within one document
                        # remain possible occurrences and are all protected.
                        if not parts and len({p.document_id for p in chunks}) == 1:
                            parts = chunks
                        for p in parts:
                            if (
                                qids
                                and len(claim_ids) > 1
                                and a.run_index not in turns
                                and _identifiers(p.text or "")
                                and not qids & _identifiers(p.text or "")
                            ):
                                continue
                            select(p)
            mentions = sorted(
                (m.start(), m.end(), doc)
                for doc in documents
                for m in re.finditer(
                    r"(?<![\w-])" + re.escape(doc) + r"(?![\w-])",
                    citation,
                    re.I,
                )
            )
            for i, (start, end, doc) in enumerate(mentions):
                scope = citation[
                    start : mentions[i + 1][0]
                    if i + 1 < len(mentions)
                    else len(citation)
                ]
                candidates = [
                    p
                    for p in old
                    if p.document_id == doc and p.run_index <= a.run_index
                ]
                cited_versions = {
                    p.version
                    for p in candidates
                    if p.version and _contains(scope, p.version)
                }
                cited_chunks = {
                    p.chunk_id
                    for p in candidates
                    if p.chunk_id and _contains(scope, p.chunk_id)
                }
                cited_parts = {
                    p.part_id for p in candidates if _contains(scope, p.part_id)
                }
                if cited_versions:
                    candidates = [p for p in candidates if p.version in cited_versions]
                if cited_chunks:
                    candidates = [p for p in candidates if p.chunk_id in cited_chunks]
                if cited_parts:
                    candidates = [p for p in candidates if p.part_id in cited_parts]
                if not candidates:
                    continue
                claim_ids = _identifiers(claim)
                if (
                    qids
                    and claim_ids
                    and not qids & claim_ids
                    and a.run_index not in turns
                ):
                    continue
                # A chunk or occurrence citation is an explicit reference. Keep
                # every matching occurrence because provenance remains ambiguous.
                if cited_chunks or cited_parts:
                    for p in candidates:
                        select(p)
                    continue
                matches = [p for p in candidates if qids & _identifiers(p.text or "")]
                if not qids:
                    matches = [
                        p for p in candidates if len(rare & terms(p.text or "")) >= 2
                    ]
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


def expand(request: Request, source_handle: str, offset=0, limit=48_000) -> dict:
    number(offset, "offset", integer=True)
    number(limit, "limit", 1, 48_000, integer=True)
    source = next(
        (p for p in request.parts if handle(request.thread_id, p) == source_handle),
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
