"""Streaming access and single-thread export for scaled RAG shards."""

from __future__ import annotations
import gzip
import json
from pathlib import Path


def _manifest(dataset_dir):
    return json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(root, entry):
    threads = _read(root / entry["trace_path"])
    if len(threads) != 1 or threads[0]["thread_id"] != entry["thread_id"]:
        raise ValueError("trace shard must contain its one named thread")
    return threads[0], _read(root / entry["labels_path"]), entry


def iter_threads(dataset_dir):
    root = Path(dataset_dir)
    for entry in _manifest(root)["threads"]:
        yield _load(root, entry)


def load_thread(dataset_dir, thread_id):
    root = Path(dataset_dir)
    for entry in _manifest(root)["threads"]:
        if entry["thread_id"] == thread_id:
            return _load(root, entry)
    raise KeyError(thread_id)


def export_thread(dataset_dir, thread_id, output_dir):
    thread, labels, entry = load_thread(dataset_dir, thread_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "threads.jsonl").write_text(
        json.dumps(thread, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "labels.jsonl").write_text(
        "".join(
            json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in labels
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "compache-scaled-rag-single-thread-export-v1",
        "source_thread": {k: v for k, v in entry.items() if k != "run_design"},
        "train_thread_ids": [thread_id] if entry["split"] == "train" else [],
        "validation_thread_ids": [thread_id] if entry["split"] == "validation" else [],
        "heldout_thread_ids": [thread_id] if entry["split"] == "heldout" else [],
        "answer_status": "fabricated scaffolding; output dependencies must use actual live prior answers",
        "usage": "null; provider usage must be measured",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output
