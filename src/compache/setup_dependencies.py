"""Local, opt-in preparation and assessment of dependency evidence for setup."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .dependencies import build_packet, validate_label
from .optimizer.contracts import ModelSpec, digest, number
from .optimizer.retention import fit_recipes
from .optimizer.workload import DEFAULT_RECIPES
from .traces import ContractError, Thread, _unique


def dependency_preview(*, max_threads=3):
    """Describe a prospective local content operation without accessing any files."""
    number(max_threads, "max_threads", 1, integer=True)
    return {
        "max_threads": max_threads,
        "source": "An explicitly supplied, preferably redacted normalized thread JSONL export.",
        "selection": "Scan the export locally. Decode up to the chosen number of whole threads, sampled by seed and row position.",
        "packet_contents": "One packet per selected run with the question, observed answer, model-visible source text, run and source IDs, and retrieval provenance.",
        "excluded_from_packets": "Timestamps, usage receipts, company metadata, unexposed retrievals and future turns.",
        "storage": "Only preparation writes sensitive packets and a hash manifest into a new private directory. Evidence labels can contain exact quotes.",
        "agent": "Preparation runs on this machine. A person or a local model can label packets here. A hosted coding agent requires separate approval for the provider and selected packets before opening them.",
        "network": "No provider requests, telemetry uploads or credential lookup.",
    }


def _sample(threads_path, allow_content, max_threads, seed):
    if allow_content is not True:
        raise ContractError("Explicit content access is required before opening a trace export")
    number(max_threads, "max_threads", 1, integer=True)
    if type(seed) is not int:
        raise ContractError("seed must be an integer")

    # Rank row numbers, not customer content. Only the selected JSON objects
    # are decoded. One descriptor keeps both passes on the same source file.
    with Path(threads_path).open("rb") as export:
        def ranked_rows():
            offset = 0
            for row, raw in enumerate(export, 1):
                if not raw.isspace():
                    rank = hashlib.sha256(f"{seed}/{row}".encode()).digest()
                    yield rank, row, offset, len(raw)
                offset += len(raw)

        before = os.fstat(export.fileno())
        chosen = heapq.nsmallest(max_threads, ranked_rows())
        if not chosen:
            raise ContractError("The supplied content export contains no threads")
        threads = []
        for _, row, offset, length in sorted(chosen, key=lambda entry: entry[1]):
            export.seek(offset)
            try:
                thread = Thread.from_dict(json.loads(export.read(length)))
            except (ValueError, TypeError, KeyError):
                # Trace contract errors can include customer IDs and source keys.
                raise ContractError(f"Invalid normalized thread on selected row {row}") from None
            threads.append(thread)
        after = os.fstat(export.fileno())
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ContractError("Content export changed during sampling")

    try:
        _unique(threads)
    except ContractError:
        raise ContractError("Duplicate identities in the selected content sample") from None
    return threads


def _write_private_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def prepare_dependency_sample(
    threads_path, output_dir, *, allow_content=False, max_threads=3, seed=0
):
    """Create private per-run packets in a new directory. Return counts only."""
    threads = _sample(threads_path, allow_content, max_threads, seed)
    destination = Path(output_dir)
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        packets = destination / "packets"
        packets.mkdir(mode=0o700)
        hashes = {}
        for thread in threads:
            for run in thread.runs:
                name = f"packets/{len(hashes) + 1:06}.json"
                hashes[name] = _write_private_json(
                    destination / name, build_packet(thread, run.run_id)
                )
        result = {"selected_threads": len(threads), "packets": len(hashes)}
        _write_private_json(destination / "manifest.json", {
            "schema_version": "compache-dependency-sample-v1",
            "max_threads": max_threads,
            "seed": seed,
            **result,
            "sha256": hashes,
        })
        return result
    except BaseException:
        shutil.rmtree(destination)
        raise


def _read_scoped_labels(path, threads):
    expected = {(t.thread_id, r.run_id) for t in threads for r in t.runs}
    rows = []
    with Path(path).open("rb") as handle:
        for index, raw in enumerate(handle, 1):
            if raw.isspace():
                continue
            try:
                row = json.loads(raw)
                target = (row["thread_id"], row["target_run_id"])
                expected.remove(target)
            except (ValueError, KeyError, TypeError):
                raise ContractError(
                    f"Label row {index} is invalid, duplicated or outside the selected sample"
                ) from None
            rows.append(row)
    if expected:
        raise ContractError(
            f"Missing {len(expected)} selected run label(s). Record unresolved targets explicitly."
        )
    return rows


def _fit_visible_prefixes(threads, labels, spec, retention_floor):
    """Score the observable part of each cumulative cut trajectory."""
    by_run = {label["target_run_id"]: label for label in labels}
    scorable = []
    scorable_labels = []
    tails = []
    for thread in threads:
        end = len(thread.runs)
        for i, run in enumerate(thread.runs):
            # This generation sees all earlier answers, its own question, and
            # exposed retrievals. Its new answer first enters the next input.
            if ((i and thread.runs[i - 1].output.text is None)
                or run.input.text is None
                or any(p.text is None for p in run.retrievals if p.exposed_to_model)):
                end = i
                break
        if end == len(thread.runs):
            scorable.append(thread)
        else:
            if end:
                data = thread.to_dict()
                data["runs"] = data["runs"][:end]
                scorable.append(Thread.from_dict(data))
            # Still validate every excluded label against the real, possibly
            # partial packet. Missing evidence cannot conceal an invalid label.
            for run in thread.runs[end:]:
                label = by_run[run.run_id]
                if validate_label(label, build_packet(thread, run.run_id)):
                    raise ContractError("Invalid label on a partially visible thread")
                tails.append(label)
        scorable_labels.extend(by_run[r.run_id] for r in thread.runs[:end])

    if scorable:
        scores = fit_recipes(
            scorable, scorable_labels, spec, DEFAULT_RECIPES,
            retention_floor=retention_floor, max_unknown_change_rate=0,
        )
    else:
        # No cost or coverage estimate can be assigned to unreadable requests.
        scores = [
            {"recipe": asdict(recipe), "eligible": False, "individual_eligible": False,
             "historical_required": 0, "historical_retained": 0, "coverage": None,
             "turns": 0, "changed_runs": 0, "unknown_runs": 0, "unknown_changed_runs": 0,
             "unknown_change_rate": None, "max_unknown_change_rate": 0,
             "joint_envelope": None, "retention_floor": retention_floor}
            for recipe in DEFAULT_RECIPES
        ]
    projection = {
        "scored_runs": len(scorable_labels),
        "unscorable_runs": len(tails),
        "affected_threads": len({r["thread_id"] for r in tails}),
        "unscorable_historical_requirements": sum(
            req["historical_context_needed"] == "yes"
            for label in tails for req in label["task_requirements"]
        ),
    }
    return scores, projection


def fit_content_recipes(
    threads_path, labels_path, model_spec, *, allow_content=False,
    max_threads=3, seed=0, retention_floor=0.95,
):
    """Fit retention only from a scoped sample and its validated evidence labels."""
    threads = _sample(threads_path, allow_content, max_threads, seed)
    spec = model_spec if isinstance(model_spec, ModelSpec) else ModelSpec.from_dict(model_spec)
    if any(run.generation.model != spec.model_id for t in threads for run in t.runs):
        raise ContractError("A selected generation uses a different model from the model spec")
    labels = _read_scoped_labels(labels_path, threads)
    number(retention_floor, "retention_floor", 0, 1)
    try:
        scores, projection = _fit_visible_prefixes(threads, labels, spec, retention_floor)
    except ContractError:
        # The shared fitter validates every quote, reference, and visible part.
        # Its detailed errors are for the private label-validation workflow.
        raise ContractError("Invalid dependency evidence. Validate each label against its private packet.") from None

    for score in scores:
        if projection["unscorable_runs"]:
            score["eligible"] = False
            score["setup_admission_reason"] = "incomplete_visible_history"
        elif score["eligible"] and score["changed_runs"] == 0:
            score["eligible"] = False
            score["setup_admission_reason"] = "no_cut_exercised_in_sample"

    turn_index = {r.run_id: r.run_index for t in threads for r in t.runs}
    part_index = {}
    for thread in threads:
        for run in thread.runs:
            part_index[run.input.part_id] = ("input", run.run_index)
            part_index[run.output.part_id] = ("output", run.run_index)
            for part in run.retrievals:
                part_index[part.part_id] = ("retrieval", run.run_index)
    lookbacks = {kind: Counter() for kind in ("input", "retrieval", "output")}
    statuses = Counter(label["status"] for label in labels)
    source_kinds = Counter()
    history = Counter()
    for label in labels:
        for req in label["task_requirements"]:
            history[req["historical_context_needed"]] += 1
            if req["historical_context_needed"] == "yes":
                source_kinds[req["source_kind"]] += 1
                target = turn_index[label["target_run_id"]]
                alternatives = [
                    [part_index[s["part_id"]] for s in alt["sources"]]
                    for alt in req["alternatives"]
                ]
                for kind, counts in lookbacks.items():
                    # A requirement depends specifically on historical sources
                    # of this kind only when every sufficient alternative does.
                    lag = min(
                        max((target - i for k, i in sources if k == kind), default=0)
                        for sources in alternatives
                    )
                    if lag:
                        counts[str(lag)] += 1

    recipe_results = []
    for score in scores:
        reasons = []
        if score["coverage"] is None:
            reasons.append("no_assessed_historical_requirements")
        elif score["coverage"] < retention_floor:
            reasons.append("below_retention_floor")
        if projection["unscorable_runs"]:
            reasons.append("incomplete_visible_history")
        if score["unknown_change_rate"] is not None and score["unknown_change_rate"] > score["max_unknown_change_rate"]:
            reasons.append("changed_unassessable_turns")
        if score["changed_runs"] == 0:
            reasons.append("no_cut_exercised_in_sample")
        if score["individual_eligible"] and not score["eligible"] and not reasons:
            reasons.append("combined_recipes_failed_retention_or_unknown_guard")
        recipe_results.append({
            "recipe": dict(score["recipe"]),
            "historical_retained": score["historical_retained"],
            "historical_required": score["historical_required"],
            "coverage": score["coverage"],
            "changed_runs": score["changed_runs"],
            "unknown_runs": score["unknown_runs"],
            "unknown_changed_runs": score["unknown_changed_runs"],
            "unknown_change_rate": score["unknown_change_rate"],
            "eligible": score["eligible"],
            "reasons": reasons,
        })

    return {
        "recipes": scores,
        "training_threads": sorted(digest(t.thread_id) for t in threads),
        "report": {
            "selected_threads": len(threads),
            "selected_runs": len(labels),
            "projection": projection,
            "label_status": dict(statuses),
            "requirements_needing_history": dict(history),
            "historical_requirement_source_kind": dict(source_kinds),
            "required_lookback_runs_by_kind": {k: dict(v) for k, v in lookbacks.items()},
            "eligible_recipes": sum(r["eligible"] for r in scores),
            "retention_floor": retention_floor,
            "recipe_results": recipe_results,
        },
    }
