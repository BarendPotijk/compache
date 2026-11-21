#!/usr/bin/env python3
"""Convert BEAM conversations into Compache threads, evidence labels and a fixed split.

BEAM supplies long single-thread chats with twenty graded questions each and the
message IDs that hold the evidence. It supplies no clock, so return timing is a
declared scenario. Nothing from BEAM is bundled: download the pinned files first.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

SCHEMA = "compache-beam-workload-v1"
SOURCE = {
    "dataset": "Mohammadta/BEAM",
    "revision": "3205395e897e7318c7b094ef4e6047b9b82dbb03",
    "licence": "CC BY-SA 4.0",
    "citation": "Tavakoli et al., Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs, arXiv:2510.27246",
    "files": {
        "100K": (
            "100K-00000-of-00001.parquet",
            "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace",
        ),
        "500K": (
            "500K-00000-of-00001.parquet",
            "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122",
        ),
        "1M": (
            "1M-00000-of-00001.parquet",
            "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533",
        ),
    },
}

# Declared return scenarios. An idle gap inside a session is lognormal around the
# median. With sigma 1.2 about 97%, 57% and 7% of gaps fall inside five minutes.
CLOCKS = {
    "fast": {"median_idle_seconds": 30, "sigma": 1.2},
    "mixed": {"median_idle_seconds": 240, "sigma": 1.2},
    "slow": {"median_idle_seconds": 1800, "sigma": 1.2},
}
SESSION_START_HOUR_UTC = 9
MIN_SESSION_BREAK_SECONDS = 8 * 3600
FIRST_TOKEN_SECONDS = 1.0
OUTPUT_TOKENS_PER_SECOND = 60.0
PROBE_DELAY_SECONDS = 24 * 3600
PROBE_IDLE_SECONDS = 60
QUOTE_CHARS = 200
ANSWER_FIELDS = (
    "answer",
    "ideal_answer",
    "ideal_response",
    "ideal_summary",
    "expected_compliance",
)
DATE_FORMATS = ("%B-%d-%Y", "%B %d, %Y", "%Y-%m-%d")


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as z,
    ):
        for row in rows:
            z.write((compact(row) + "\n").encode())


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def thread_name(size, conversation_id):
    return f"beam-{size.lower()}-{int(conversation_id):02d}"


def unit(*key):
    """A reproducible number strictly between zero and one for this key."""
    raw = hashlib.sha256(compact(list(key)).encode()).digest()
    return (int.from_bytes(raw[:8], "big") + 0.5) / 2**64


def idle_seconds(clock, *key):
    spec = CLOCKS[clock]
    z = NormalDist().inv_cdf(unit(*key))
    return max(1, round(spec["median_idle_seconds"] * math.exp(spec["sigma"] * z)))


def answer_seconds(text):
    return max(1, round(FIRST_TOKEN_SECONDS + len(text) / 4 / OUTPUT_TOKENS_PER_SECOND))


def session_date(session):
    anchor = next((m["time_anchor"] for m in session if m.get("time_anchor")), None)
    for pattern in DATE_FORMATS:
        try:
            parsed = datetime.strptime(anchor or "", pattern)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)
    raise ValueError(f"unreadable session date: {anchor!r}")


def evidence_ids(source):
    """Every message ID named as evidence, however the question type nests them."""
    if source is None:
        return []
    if isinstance(source, dict):
        return [i for value in source.values() for i in evidence_ids(value)]
    if isinstance(source, (list, tuple)):
        return [i for value in source for i in evidence_ids(value)]
    if isinstance(source, int) and not isinstance(source, bool):
        return [source]
    raise ValueError(f"unexpected evidence reference: {source!r}")


def read_probes(text):
    probes = ast.literal_eval(text)
    if not isinstance(probes, dict) or not probes:
        raise ValueError("probing_questions must be a nonempty mapping")
    rows = []
    for category in sorted(probes):
        for item in probes[category]:
            answer = next((item[k] for k in ANSWER_FIELDS if item.get(k)), None)
            if (
                not isinstance(item.get("question"), str)
                or not item["question"].strip()
            ):
                raise ValueError(f"{category}: question text is missing")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{category}: reference answer is missing")
            rows.append(
                {
                    "category": category,
                    "question": item["question"].strip(),
                    "answer": answer.strip(),
                    "rubric": list(item.get("rubric") or []),
                    "evidence": sorted(set(evidence_ids(item.get("source_chat_ids")))),
                }
            )
    return rows


def convert(row, size, *, clock="mixed", seed=20261120, model="replay"):
    """One BEAM row to a thread, its probe labels and a manifest entry."""
    if clock not in CLOCKS:
        raise ValueError(f"unknown clock scenario: {clock}")
    tid = thread_name(size, row["conversation_id"])
    sessions = [list(s) for s in row["chat"] if s]
    messages = [m for s in sessions for m in s]
    roles = [m["role"] for m in messages]
    if (
        not messages
        or len(messages) % 2
        or roles != ["user", "assistant"] * (len(messages) // 2)
    ):
        raise ValueError(f"{tid}: expected alternating user and assistant messages")
    if any(
        not isinstance(m["content"], str) or not m["content"].strip() for m in messages
    ):
        raise ValueError(f"{tid}: empty message")
    if any(len(s) % 2 for s in sessions):
        raise ValueError(f"{tid}: a session ends on a user message")
    part = [f"{tid}-m{position:04d}" for position in range(len(messages))]
    # Four conversations restart message numbering in a late session. An ID that
    # then names two messages cannot identify evidence.
    count = Counter(m["id"] for m in messages)
    position = {m["id"]: i for i, m in enumerate(messages) if count[m["id"]] == 1}

    runs, history = [], []
    cursor, at, session_of = None, 0, []

    def add(input_id, question, output_id, answer, started):
        ended = started + timedelta(seconds=answer_seconds(answer))
        index = len(runs) + 1
        run_id = f"{tid}-r{index:04d}"
        request = history + [input_id]
        runs.append(
            {
                "run_id": run_id,
                "run_index": index,
                "started_at": iso(started),
                "ended_at": iso(ended),
                "history_policy": "full_canonical",
                "input": {"part_id": input_id, "text": question},
                "retrievals": [],
                "output": {"part_id": output_id, "text": answer},
                "generation": {
                    "observation_id": run_id + "-g",
                    "model": model,
                    "started_at": iso(started),
                    "ended_at": iso(ended),
                    "usage": None,
                    "request_part_ids": request,
                },
            }
        )
        history.extend((input_id, output_id))
        return run_id, ended

    for number, session in enumerate(sessions, 1):
        opens = session_date(session) + timedelta(hours=SESSION_START_HOUR_UTC)
        if cursor is not None:
            opens = max(opens, cursor + timedelta(seconds=MIN_SESSION_BREAK_SECONDS))
        for turn in range(len(session) // 2):
            started = (
                opens
                if turn == 0
                else cursor + timedelta(seconds=idle_seconds(clock, seed, tid, at))
            )
            _, cursor = add(
                part[at],
                messages[at]["content"],
                part[at + 1],
                messages[at + 1]["content"],
                started,
            )
            session_of.append(number)
            at += 2
    chat_runs = len(runs)

    labels, tally = [], Counter()
    for k, probe in enumerate(read_probes(row["probing_questions"]), 1):
        started = cursor + timedelta(
            seconds=PROBE_DELAY_SECONDS if k == 1 else PROBE_IDLE_SECONDS
        )
        run_id, cursor = add(
            f"{tid}-q{k:02d}",
            probe["question"],
            f"{tid}-a{k:02d}",
            probe["answer"],
            started,
        )
        unresolved = [i for i in probe["evidence"] if i not in position]
        # Assessed when every evidence ID names exactly one message. Otherwise the need is unknown.
        status = (
            "unknown"
            if not probe["evidence"]
            else "ambiguous"
            if unresolved
            else "assessed"
        )
        sources = (
            [
                {"part_id": part[p], "quote": messages[p]["content"][:QUOTE_CHARS]}
                for p in sorted(position[i] for i in probe["evidence"])
            ]
            if status == "assessed"
            else []
        )
        tally[status] += 1
        tally["sources"] += len(sources)
        labels.append(
            {
                "thread_id": tid,
                "run_id": run_id,
                "category": probe["category"],
                # Every named message is needed, so they form one alternative.
                "needs": [sources] if sources else None,
            }
        )
    thread = {
        "thread_id": tid,
        "company_id": re.sub(
            r"[^a-z0-9]+", "-", row["conversation_seed"]["category"].lower()
        ).strip("-"),
        "scenario_id": f"beam-{clock}",
        "provenance": "BEAM conversation with reference answers; declared clock; usage unavailable",
        "runs": runs,
    }
    entry = {
        "thread_id": tid,
        "company_id": thread["company_id"],
        "size": size,
        "category": row["conversation_seed"]["category"],
        "sessions": len(sessions),
        "chat_runs": chat_runs,
        "probe_runs": len(labels),
        "turn_count": len(runs),
        "assessed_probes": tally["assessed"],
        "unknown_probes": tally["unknown"],
        "ambiguous_probes": tally["ambiguous"],
        "evidence_sources": tally["sources"],
        "planning_tokens": sum(
            math.ceil(len(m["content"].encode()) / 4) for m in messages
        ),
    }
    return thread, labels, entry


def assign_splits(entries, *, test_fraction=1 / 3, salt="beam-split-v1"):
    """Fix the test set inside each size bucket from thread IDs alone."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    by_size = {}
    for entry in entries:
        by_size.setdefault(entry["size"], []).append(entry)
    for group in by_size.values():
        ordered = sorted(
            group,
            key=lambda e: hashlib.sha256(
                f"{salt}:{e['thread_id']}".encode()
            ).hexdigest(),
        )
        cut = (
            min(len(group) - 1, max(1, math.ceil(len(group) * test_fraction)))
            if len(group) > 1
            else 0
        )
        for i, entry in enumerate(ordered):
            entry["split"] = "test" if i < cut else "train"
    return entries


def read_source(source, sizes):
    paths = []
    for size in sizes:
        name, expected = SOURCE["files"][size]
        path = Path(source) / name
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"{name} is missing or is not the pinned BEAM file")
        paths.append((size, path))
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("Reading BEAM needs pyarrow: pip install pyarrow") from error
    for size, path in paths:
        for row in parquet.read_table(path).to_pylist():
            yield size, row


def build(
    rows,
    output,
    *,
    clock="mixed",
    seed=20261120,
    model="replay",
    test_fraction=1 / 3,
    check=True,
):
    """Write one trace shard and one label shard per conversation, then the manifest."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Choose a new or empty output directory")
    pending = []
    for size, row in rows:
        thread, labels, entry = convert(row, size, clock=clock, seed=seed, model=model)
        if check:
            verify(thread, labels)
        entry["trace_path"] = f"traces/{entry['thread_id']}.jsonl.gz"
        entry["labels_path"] = f"labels/{entry['thread_id']}.jsonl.gz"
        write_gz(output / entry["trace_path"], [thread])
        write_gz(output / entry["labels_path"], labels)
        entry["trace_sha256"] = sha256_file(output / entry["trace_path"])
        entry["labels_sha256"] = sha256_file(output / entry["labels_path"])
        pending.append(entry)
    if len({e["thread_id"] for e in pending}) != len(pending):
        raise SystemExit("Duplicate conversation in the source")
    entries = assign_splits(pending, test_fraction=test_fraction)
    manifest = {
        "schema_version": SCHEMA,
        "source": {**SOURCE, "files": {k: list(v) for k, v in SOURCE["files"].items()}},
        "model": model,
        "clock": {
            "scenario": clock,
            **CLOCKS[clock],
            "seed": seed,
            "session_start_hour_utc": SESSION_START_HOUR_UTC,
            "min_session_break_seconds": MIN_SESSION_BREAK_SECONDS,
            "first_token_seconds": FIRST_TOKEN_SECONDS,
            "output_tokens_per_second": OUTPUT_TOKENS_PER_SECOND,
            "probe_delay_seconds": PROBE_DELAY_SECONDS,
            "probe_idle_seconds": PROBE_IDLE_SECONDS,
            "interpretation": "Declared scenario. BEAM dates each session and times nothing inside it.",
        },
        "split": {
            "rule": "Inside each size bucket, order threads by SHA-256 of 'beam-split-v1:<thread_id>' and assign the first share to test.",
            "test_fraction": test_fraction,
            "counts": dict(
                sorted(Counter(f"{e['size']}/{e['split']}" for e in entries).items())
            ),
        },
        "probes": "The last probe_runs runs of a thread are BEAM's questions, appended in category order with their reference answers. BEAM asks each one independently of the others. Fit demand on the first chat_runs runs only.",
        "evidence": "A question is assessed when every evidence ID names exactly one message. All named messages are required together.",
        "thread_count": len(entries),
        "turn_count": sum(e["turn_count"] for e in entries),
        "chat_runs": sum(e["chat_runs"] for e in entries),
        "assessed_probes": sum(e["assessed_probes"] for e in entries),
        "unknown_probes": sum(e["unknown_probes"] for e in entries),
        "ambiguous_probes": sum(e["ambiguous_probes"] for e in entries),
        "usage": "null; provider usage must be measured",
        "threads": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def iter_threads(output):
    """Each converted conversation of a build: its thread, its labels and its manifest entry."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def read(path):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    for entry in manifest["threads"]:
        yield (
            read(output / entry["trace_path"])[0],
            read(output / entry["labels_path"]),
            entry,
        )


def packet(thread_id, run, visible):
    """What a labeller may cite for this turn: the same view as the package's build_packet."""
    question = {
        "part_id": run["input"]["part_id"],
        "origin_run_id": run["run_id"],
        "origin_run_index": run["run_index"],
        "kind": "input",
        "text": run["input"]["text"],
        "content_status": "available",
    }
    return {
        "schema_version": "compache-label-input-v1",
        "thread_id": thread_id,
        "target_run_id": run["run_id"],
        "target_run_index": run["run_index"],
        "target_question": {k: run["input"][k] for k in ("part_id", "text")},
        "target_output": {k: run["output"][k] for k in ("part_id", "text")},
        "request_parts": [
            visible.get(part_id, question)
            for part_id in run["generation"]["request_part_ids"]
        ],
        "request_coverage": "complete",
        "coverage_notes": [],
    }


def verify(thread, labels):
    """Check the thread format, and every label against what its turn could see.

    The package's own packet builder copies the whole thread for each label, which
    is too slow for a thousand-turn conversation. This walks the thread once.
    """
    from compache.labels import normalize_label, validate_label
    from compache.threads import Thread

    checked = Thread.from_dict(thread)
    wanted = {}
    for row in labels:
        label = normalize_label(row, checked)
        if label["target_run_id"] in wanted:
            raise ValueError("two labels target one run")
        wanted[label["target_run_id"]] = label
    visible = {}
    for run in thread["runs"]:
        label = wanted.pop(run["run_id"], None)
        if label is not None:
            errors = validate_label(label, packet(thread["thread_id"], run, visible))
            if errors:
                raise ValueError(f"{run['run_id']}: {errors[0]}")
        for kind in ("input", "output"):
            visible[run[kind]["part_id"]] = {
                "part_id": run[kind]["part_id"],
                "origin_run_id": run["run_id"],
                "origin_run_index": run["run_index"],
                "kind": kind,
                "text": run[kind]["text"],
                "content_status": "available",
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder holding the three pinned BEAM parquet files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clock", choices=sorted(CLOCKS), default="mixed")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument(
        "--model",
        default="replay",
        help="Model ID written to every generation; the fit's model spec must use the same ID",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=list(SOURCE["files"]),
        default=list(SOURCE["files"]),
    )
    parser.add_argument("--test-fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip the trace-contract and label checks",
    )
    args = parser.parse_args(argv)
    manifest = build(
        read_source(args.source, args.sizes),
        args.output,
        clock=args.clock,
        seed=args.seed,
        model=args.model,
        test_fraction=args.test_fraction,
        check=not args.skip_checks,
    )
    print(
        compact(
            {k: v for k, v in manifest.items() if k != "threads"}
            | {"output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
