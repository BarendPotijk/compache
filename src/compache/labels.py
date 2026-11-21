"""Dependency labels: which earlier parts of a thread a turn needed.

One JSON object per labelled run:

    {"thread_id": "t-17", "run": 7, "needs": [["t-17:3:output"], ["t-17:2:retrieval:1"]]}

`needs` lists alternatives. Any one of them, with all of its parts in view, is
enough to answer the turn. `[]` means the turn needs nothing earlier and `null`
means nobody could tell. A part is its ID, or `{"part_id", "quote"}` when the
validator should check a verbatim quote from it.

The fit reads the canonical form, which `normalize_label` produces: a
`status` and `task_requirements`, each with `historical_context_needed` and
`alternatives`. A row that already carries `task_requirements` is canonical.
"""

from __future__ import annotations

import copy

from .errors import ContractError
from .threads import Thread

__all__ = ["build_packet", "label_index", "normalize_label", "validate_label"]

_STATUSES = ("complete", "partial", "unassessable")
_NEEDED = ("yes", "no", "unknown")


def _status(part: dict) -> str:
    return part.get(
        "content_status", "available" if part.get("text") is not None else "missing"
    )


def _available(part: dict) -> bool:
    return _status(part) == "available" and part.get("text") is not None


def _target(part: dict) -> dict:
    """The run's own question or answer as a labeller may see it."""
    shown = {
        k: v for k, v in part.items() if k in ("part_id", "text", "content_status")
    }
    if not _available(part):
        shown["text"] = None
    return shown


def build_packet(thread: Thread | dict, target_run_id: str) -> dict:
    """Everything a labeller may see for one run: its question, its answer and its request.

    Later runs, split names and anything outside the request are left out.
    """
    if isinstance(thread, dict):
        thread = Thread(thread)
    if not isinstance(thread, Thread):
        raise ContractError("thread must be Thread or dict")
    # Read the validated thread in place. Only the packet returned below is copied.
    data, parts = thread._data, {}
    for run in data["runs"]:
        for kind, items in (
            ("input", [run["input"]]),
            ("retrieval", run["retrievals"]),
            ("output", [run["output"]]),
        ):
            for part in items:
                entry = {
                    "part_id": part["part_id"],
                    "origin_run_id": run["run_id"],
                    "origin_run_index": run["run_index"],
                    "kind": kind,
                    "text": part.get("text") if _available(part) else None,
                    "content_status": _status(part),
                }
                if kind == "retrieval":
                    entry.update(
                        {k: part[k] for k in ("document_id", "version", "chunk_id")}
                    )
                parts[part["part_id"]] = entry
        if run["run_id"] == target_run_id:
            visible = [parts[x] for x in run["generation"]["request_part_ids"]]
            notes = [
                "Visible source text unavailable: " + p["part_id"]
                for p in visible
                if p["text"] is None
            ]
            if not _available(run["output"]):
                notes.append("Target output is unavailable.")
            return copy.deepcopy(
                {
                    "schema_version": "compache-label-input-v1",
                    "thread_id": data["thread_id"],
                    "target_run_id": target_run_id,
                    "target_run_index": run["run_index"],
                    "target_question": _target(run["input"]),
                    "target_output": _target(run["output"]),
                    "request_parts": visible,
                    "request_coverage": "partial" if notes else "complete",
                    "coverage_notes": notes,
                }
            )
    raise ContractError("target run not found: " + str(target_run_id))


def _source(value) -> dict:
    if isinstance(value, str) and value:
        return {"part_id": value}
    if (
        isinstance(value, dict)
        and isinstance(value.get("part_id"), str)
        and not set(value) - {"part_id", "quote"}
    ):
        return dict(value)
    raise ContractError("a needed part is a part ID or {part_id, quote}")


def _target_run(row: dict, thread: Thread):
    run_id = row.get("run_id", row.get("target_run_id"))
    index = row.get("run")
    for run in thread.runs:
        if run.run_id == run_id or (run_id is None and run.run_index == index):
            return run
    raise ContractError(
        f"label names no run of thread {thread.thread_id}: {run_id or index!r}"
    )


def normalize_label(row: dict, thread: Thread) -> dict:
    """The canonical form of one label row for a run of `thread`."""
    if not isinstance(row, dict):
        raise ContractError("label must be an object")
    if "task_requirements" in row:
        return row
    if "needs" not in row:
        raise ContractError("a label carries `needs`: alternatives, [] or null")
    run = _target_run(row, thread)
    needs = row["needs"]
    if needs is None:
        status, needed, alternatives = "unassessable", "unknown", []
    elif not isinstance(needs, list):
        raise ContractError("needs must be a list of alternatives, or null")
    else:
        alternatives = []
        for alternative in needs:
            if isinstance(alternative, (str, dict)):
                alternative = [alternative]
            if not isinstance(alternative, list) or not alternative:
                raise ContractError("an alternative is a nonempty list of parts")
            alternatives.append({"sources": [_source(s) for s in alternative]})
        origin = thread.part_origin

        def current(alternative):
            return all(
                origin.get(s["part_id"], (None, 0))[1] == run.run_index
                for s in alternative["sources"]
            )

        # A set of this run's own parts that is enough means no history is needed.
        needed = "no" if not alternatives or any(map(current, alternatives)) else "yes"
        if not alternatives:
            alternatives = [{"sources": [{"part_id": run.input.part_id}]}]
        status = "complete"
    return {
        "schema_version": "compache-label-v2",
        "thread_id": thread.thread_id,
        "target_run_id": run.run_id,
        "status": status,
        "task_requirements": [
            {
                "requirement_id": "needs",
                "historical_context_needed": needed,
                "alternatives": alternatives,
            }
        ],
    }


def _packet_errors(packet) -> list[str]:
    if not isinstance(packet, dict):
        return ["packet: expected object"]
    required = (
        "thread_id",
        "target_run_id",
        "target_run_index",
        "target_question",
        "target_output",
        "request_parts",
    )
    if any(key not in packet for key in required):
        return ["packet: missing required fields"]
    parts = packet["request_parts"]
    if (
        not isinstance(packet["target_question"], dict)
        or not isinstance(packet["target_output"], dict)
        or not isinstance(parts, list)
        or type(packet["target_run_index"]) is not int
    ):
        return ["packet: invalid question, output, index or request_parts"]
    if any(
        not isinstance(p, dict)
        or not isinstance(p.get("part_id"), str)
        or p.get("kind") not in ("input", "retrieval", "output")
        or type(p.get("origin_run_index")) is not int
        or (p.get("text") is not None and not isinstance(p.get("text"), str))
        for p in parts
    ):
        return ["packet: invalid request part"]
    ids = [p["part_id"] for p in parts]
    if len(ids) != len(set(ids)):
        return ["packet: duplicate request part"]
    target = packet["target_run_index"]
    if (
        packet["target_question"].get("part_id") not in ids
        or packet["target_output"].get("part_id") in ids
    ):
        return ["packet: current input/output visibility mismatch"]
    if any(
        p["origin_run_index"] > target
        or (p["origin_run_index"] == target and p["kind"] == "output")
        for p in parts
    ):
        return ["packet: future or foreign source"]
    return []


def validate_label(label, packet) -> list[str]:
    """Problems with one canonical label, checked against what its run could see.

    A source must be a part of the run's request with available text. A quote,
    when given, must be found in it verbatim. `no` needs an alternative made of
    the run's own parts, `yes` needs alternatives that each reach back. This
    checks consistency, not whether the labeller judged the turn correctly.
    """
    errors = _packet_errors(packet)
    if errors:
        return errors
    if not isinstance(label, dict):
        return ["label: expected object"]
    for key in ("thread_id", "target_run_id"):
        if label.get(key) != packet[key]:
            errors.append(f"{key}: does not match packet")
    if label.get("status") not in _STATUSES:
        errors.append("status: expected complete, partial or unassessable")
    requirements = label.get("task_requirements")
    if not isinstance(requirements, list) or not all(
        isinstance(r, dict) for r in requirements
    ):
        return errors + ["task_requirements: expected a list of objects"]
    parts = {p["part_id"]: p for p in packet["request_parts"]}
    target = packet["target_run_index"]
    names = [r.get("requirement_id") for r in requirements]
    if len(names) != len(set(names)):
        errors.append("duplicate task requirement IDs")
    for requirement in requirements:
        name = "task/" + str(requirement.get("requirement_id"))
        needed = requirement.get("historical_context_needed")
        alternatives = requirement.get("alternatives")
        if needed not in _NEEDED:
            errors.append(name + ": historical_context_needed is yes, no or unknown")
        if not isinstance(alternatives, list):
            errors.append(name + ": alternatives must be a list")
            continue
        own_parts_suffice = False
        for alternative in alternatives:
            sources = (
                alternative.get("sources") if isinstance(alternative, dict) else None
            )
            if not isinstance(sources, list) or not sources:
                errors.append(name + ": empty alternative is not sufficient evidence")
                continue
            ids = [s.get("part_id") if isinstance(s, dict) else None for s in sources]
            if len(ids) != len(set(ids)):
                errors.append(name + ": duplicate sources in alternative")
            reaches_back = False
            for source, part_id in zip(sources, ids):
                part = parts.get(part_id)
                if part is None:
                    errors.append(
                        f"{name}: source is absent, future, or not visible: {part_id}"
                    )
                    reaches_back = True
                    continue
                reaches_back = reaches_back or part["origin_run_index"] < target
                if part.get("text") is None:
                    errors.append(name + ": unavailable source content")
                elif "quote" in source and (
                    not isinstance(source["quote"], str)
                    or not source["quote"]
                    or source["quote"] not in part["text"]
                ):
                    errors.append(f"{name}/{part_id}: quote not found verbatim")
            own_parts_suffice = own_parts_suffice or not reaches_back
        if needed == "no" and not own_parts_suffice:
            errors.append(
                name + ": no history requires a sufficient current-source alternative"
            )
        elif needed == "yes" and (not alternatives or own_parts_suffice):
            errors.append(
                name + ": history needed conflicts with sufficient alternatives"
            )
    return errors[:100]


def label_index(threads: list[Thread], labels: list[dict]) -> dict:
    """Validated canonical labels by target run. A packet is built only for a labelled run."""
    # Labels are keyed by run ID, so a collision between threads could leak a
    # held-out observation.
    thread_ids = [t.thread_id for t in threads]
    run_ids = [r.run_id for t in threads for r in t.runs]
    if len(set(thread_ids)) != len(thread_ids) or len(set(run_ids)) != len(run_ids):
        raise ContractError(
            "training/evaluation requires globally unique thread and run IDs"
        )
    by_id = {t.thread_id: t for t in threads}
    owner = {r.run_id: t for t in threads for r in t.runs}
    result = {}
    for row in labels:
        thread = by_id.get(row.get("thread_id")) if isinstance(row, dict) else None
        if thread is None and isinstance(row, dict):
            thread = owner.get(row.get("target_run_id", row.get("run_id")))
        if thread is None:
            raise ContractError(
                "label targets must be unique runs in the supplied traces"
            )
        label = normalize_label(row, thread)
        run_id = label.get("target_run_id")
        if owner.get(run_id) is not thread or run_id in result:
            raise ContractError(
                "label targets must be unique runs in the supplied traces"
            )
        errors = validate_label(label, build_packet(thread, run_id))
        if errors:
            raise ContractError(f"invalid label for {run_id}: " + "; ".join(errors[:3]))
        result[run_id] = label
    return result
