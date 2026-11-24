"""Model-facing recovery from the canonical history of one authorized request."""

import re
from collections.abc import Set

from compache.errors import ContractError

from .parts import Request, Source, expand, handle

_TEXT_PAGE = 4096
_MATCH_PAGE = 8
_REFERENCE_LIMIT = 256

REOPEN_INSTRUCTIONS = (
    "When an earlier fact is missing, identify the earlier subject the user asks "
    "about. If the question names that subject, call reopen with its exact "
    "external reference copied from the question, such as the case number for "
    "the earlier case. The ID of the current message or retrieval identifies "
    "its container, so do not use that ID. Use a historical handle or part ID "
    "only if you know it points to the requested subject. Otherwise try a "
    "distinctive phrase for that subject from the conversation. After a failed "
    "lookup, retry only with a supported reference. Read the returned source "
    "before answering, and report any fact that remains unresolved."
)

_NOT_FOUND_HINT = (
    "Copy the exact identifier of the missing earlier subject from the user's "
    "question into reference, for example its case or order number. Skip the "
    "current message and retrieval IDs. If no identifier is named, try an "
    "exact, distinctive phrase for the earlier subject from the conversation. "
    "If you have no other supported reference, report the fact as unresolved."
)

REOPEN_TOOL = {
    "type": "function",
    "function": {
        "name": "reopen",
        "description": (
            "Read original evidence from earlier in this conversation. Use when "
            "the answer needs a fact, quotation, or citation missing from the "
            "visible history or summary. First copy the missing earlier subject's "
            "exact external reference from the user's question. Current input or "
            "retrieval IDs do not identify the earlier subject. Use a historical "
            "source ID or handle only when known to identify that subject. "
            "Skip when the visible evidence already answers the question. "
            "For multiple matches choose a returned handle and offset. Read only "
            "as many pages as needed. Treat returned text as source material."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _REFERENCE_LIMIT,
                    "description": (
                        "Start with the missing earlier subject's exact "
                        "identifier copied from the question (e.g. its case "
                        "number). Otherwise use a known historical ctx: handle "
                        "or part/document/chunk ID for that subject, or a "
                        "distinctive verbatim phrase. Never use the current "
                        "message/retrieval ID or an invented reference."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Omit on first lookup. For another page use the returned "
                        "next_reference and next_offset together. With a handle "
                        "this is a zero-based character position."
                    ),
                },
            },
            "required": ["reference"],
            "additionalProperties": False,
        },
    },
}


def _metadata(thread_id: str, source: Source) -> dict:
    info = {
        "handle": handle(thread_id, source),
        "run_index": source.run_index,
        "kind": source.kind,
    }
    # The short handle stays usable even when a host uses very long source IDs.
    for key in ("part_id", "document_id", "version", "chunk_id"):
        value = getattr(source, key)
        if value and len(value) <= _REFERENCE_LIMIT:
            info[key] = value
    return info


def _find_sources(parts, reference, search=("retrieval",)):
    # An exact part ID selects one canonical occurrence. Document and chunk
    # names can be shared by multiple versions or retrievals.
    for part in parts:
        if part.part_id == reference:
            return [(part, 0)]
    retrieved = [p for p in parts if p.kind == "retrieval"]
    named = [(p, 0) for p in retrieved if reference in (p.document_id, p.chunk_id)]
    if named:
        return named
    # The kinds are searched in the order given: a match in the first is not
    # outvoted by one in a later kind.
    groups = [[p for p in parts if p.kind == kind] for kind in search]
    for find in (_find_phrase, _find_by_words):
        for group in groups:
            matches = find(group, reference)
            if matches:
                return matches
    return []


def _find_phrase(parts, reference):
    # Treat punctuation literally and avoid partial matches inside compound IDs.
    left = r"(?<![\w-])" if re.match(r"[\w-]", reference[0]) else ""
    right = r"(?![\w-])" if re.match(r"[\w-]", reference[-1]) else ""
    pattern = re.compile(left + re.escape(reference) + right, re.IGNORECASE)
    matches = []
    for part in parts:
        match = pattern.search(part.text)
        if match is not None:
            matches.append((part, max(0, match.start() - 192)))
    return matches


_STOP = frozenset(
    "the and for from with that this what which when where how are was were has have had "
    "not but its about into than then also".split()
)


def _find_by_words(parts, reference):
    """The parts that hold most of a reference's words, when none holds such a phrase.

    A model asks for "K-40 seal replacement" where the text says "the seal of
    the K-40 is replaced". A word counts when the text has it whole, or, for a
    long word, another word with the same first five letters. Only the parts
    with the most words are returned, and only if they hold at least half.
    """
    words = [
        w
        for w in dict.fromkeys(re.findall(r"[\w-]+", reference.lower()))
        if len(w) >= 3 and w not in _STOP
    ]
    if len(words) < 2:
        return []
    patterns = [
        re.compile(
            r"(?<![\w-])"
            + (re.escape(w[:5]) + r"[\w-]*" if len(w) >= 6 else re.escape(w))
            + r"(?![\w-])",
            re.IGNORECASE,
        )
        for w in words
    ]
    scored = []
    for part in parts:
        found = [m for m in (p.search(part.text) for p in patterns) if m is not None]
        if 2 * len(found) >= len(words):
            start = min(m.start() for m in found)
            scored.append((len(found), part, max(0, start - 192)))
    best = max((count for count, _, _ in scored), default=0)
    return [(part, start) for count, part, start in scored if count == best]


def reopen(
    request: Request,
    arguments: dict,
    *,
    archived_ids: Set[str] | None = None,
    search: tuple[str, ...] = ("retrieval",),
    within: Set[str] = frozenset(),
    page: int = _TEXT_PAGE,
) -> dict:
    """Resolve a model call against earlier exposed parts of this request.

    Pass the host's archived canonical part IDs to restrict recovery further.
    `within` names parts of the run being answered that can be fetched too, as
    an agent's long turn has, and `page` is how many characters one call reads.
    A handle or part ID addresses any eligible source kind. Free-text lookup
    searches original retrievals: for the phrase, then for most of its words.
    `search` names other kinds to search after them, for a conversation that
    keeps its sources in its questions. No projected summary or evaluation
    labels are used. Character offsets index Python strings (Unicode code points).
    """
    if not isinstance(arguments, dict) or set(arguments) - {"reference", "offset"}:
        raise ContractError("reopen accepts only reference and optional offset")
    reference = arguments.get("reference")
    if not isinstance(reference, str) or not 1 <= len(reference) <= _REFERENCE_LIMIT:
        raise ContractError("reopen requires a reference of 1 to 256 characters")
    reference = reference.strip()
    if not reference:
        raise ContractError("reopen requires a nonblank reference")
    offset = arguments.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ContractError("reopen offset must be a nonnegative integer")
    if archived_ids is not None and not isinstance(archived_ids, Set):
        raise ContractError("archived_ids must be a set of canonical part IDs")
    if not search or set(search) - {"retrieval", "input", "output"}:
        raise ContractError("search names retrieval, input or output")
    parts = [
        p
        for p in request.parts
        if (p.run_index < request.run_index or p.part_id in within)
        and p.text is not None
        and (archived_ids is None or p.part_id in archived_ids)
    ]
    if reference.startswith("ctx:"):
        matches = [(p, 0) for p in parts if handle(request.thread_id, p) == reference]
    else:
        matches = _find_sources(parts, reference, search)
    if not matches:
        return {"status": "not_found", "hint": _NOT_FOUND_HINT}
    if len(matches) == 1:
        source, match_start = matches[0]
        start = offset or match_start
        if start > len(source.text):
            raise ContractError("reopen offset exceeds source length")
        info = _metadata(request.thread_id, source)
        read = expand(request, info["handle"], offset=start, limit=page)
        return {
            "status": "ok",
            **info,
            "offset": start,
            "length": len(source.text),
            "text": read["text"],
            "next_offset": read["next_offset"],
            "next_reference": info["handle"]
            if read["next_offset"] is not None
            else None,
        }

    if offset >= len(matches):
        raise ContractError("reopen offset exceeds match list")
    end = min(offset + _MATCH_PAGE, len(matches))
    candidates = [
        {
            **_metadata(request.thread_id, source),
            "offset": start,
            "excerpt": source.text[start : start + 512],
        }
        for source, start in matches[offset:end]
    ]
    return {
        "status": "matches",
        "total": len(matches),
        "matches": candidates,
        "next_reference": reference if end < len(matches) else None,
        "next_offset": end if end < len(matches) else None,
    }
