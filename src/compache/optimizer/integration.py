"""A bounded, request-scoped recovery protocol for coding-agent integrations."""

from __future__ import annotations

import json
from compache.traces import ContractError
from .gate import Request, expand

RETRIEVE_SCHEMA = {
    "name": "retrieve",
    "description": "Read a document already introduced in this conversation.",
    "parameters": {
        "document_id": "string",
        "version": "string",
        "offset": "integer, default 0",
    },
}
REOPEN_SCHEMA = {
    "name": "reopen",
    "description": "Reopen a compacted passage using its ctx: handle; follow next_offset for more.",
    "parameters": {"handle": "string", "offset": "integer, default 0"},
}


def decode_response(text: str) -> dict:
    """Accept an answer or a controller tool call, never silently repair JSON."""
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    elif value.startswith("```\n") and value.endswith("\n```"):
        value = value[4:-4]
    try:
        obj = json.loads(value)
    except ValueError as exc:
        raise ContractError(
            "model response must be a JSON answer or recovery call"
        ) from exc
    if not isinstance(obj, dict):
        raise ContractError("model response must be an object")
    if (
        set(obj) == {"answer"}
        and isinstance(obj["answer"], str)
        and obj["answer"].strip()
    ):
        return obj
    if obj.get("tool") == "reopen" and set(obj) <= {"tool", "handle", "offset"}:
        if not isinstance(obj.get("handle"), str):
            raise ContractError("reopen requires a handle")
        return obj
    if obj.get("tool") == "retrieve" and set(obj) <= {
        "tool",
        "document_id",
        "version",
        "offset",
    }:
        if not all(
            isinstance(obj.get(k), str) and obj[k] for k in ("document_id", "version")
        ):
            raise ContractError("retrieve requires a document identity and version")
        return obj
    raise ContractError("invalid response protocol")


def recover(request: Request, call: dict, *, allow_reopen: bool) -> dict:
    """No corpus-wide or future lookup: only sources actually introduced so far."""
    offset = call.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ContractError("invalid recovery offset")
    if call["tool"] == "reopen":
        if not allow_reopen:
            raise ContractError("reopen is unavailable in this integration")
        return expand(request, call["handle"], offset=offset, limit=48_000)
    matches = [
        p
        for p in request.parts
        if p.kind == "retrieval"
        and p.document_id == call["document_id"]
        and p.version == call["version"]
    ]
    if not matches:
        raise ContractError("document was not introduced in this request history")
    # Each chunk is returned once, preserving source order and version.
    seen = set()
    selected = []
    for p in matches:
        if p.chunk_id not in seen:
            seen.add(p.chunk_id)
            selected.append(
                f"[{p.part_id}; chunk={p.chunk_id}]\n{p.text or '[unavailable]'}"
            )
    body = "\n\n".join(selected)
    if offset > len(body):
        raise ContractError("retrieval offset exceeds document length")
    end = min(len(body), offset + 48_000)
    return {
        "document_id": call["document_id"],
        "version": call["version"],
        "text": body[offset:end],
        "next_offset": end if end < len(body) else None,
    }
