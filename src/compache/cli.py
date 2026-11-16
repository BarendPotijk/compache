"""Local-only commands for preparing and validating Compache evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dependencies import build_packet, validate_label
from .traces import ContractError, export_langfuse, load_threads, normalize_langfuse, write_threads


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)
    traces = groups.add_parser("traces", help="Validate and convert offline RAG trace files").add_subparsers(dest="action", required=True)
    validate = traces.add_parser("validate")
    validate.add_argument("input", type=Path)
    for name in ("normalize", "export-langfuse"):
        command = traces.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    dependencies = groups.add_parser("dependencies", help="Prepare and validate dependency labels").add_subparsers(dest="action", required=True)
    prepare = dependencies.add_parser("prepare")
    prepare.add_argument("--threads", type=Path, required=True)
    prepare.add_argument("--thread-id", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    labels = dependencies.add_parser("validate")
    labels.add_argument("--input", type=Path, required=True, help="Prepared target-run packet")
    labels.add_argument("--labels", type=Path, required=True, help="Exactly one JSONL label for this packet")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.group == "traces":
            if args.action == "normalize":
                threads = normalize_langfuse(_read_jsonl(args.input))
                write_threads(threads, args.output)
            else:
                threads = load_threads(args.input)
                if args.action == "export-langfuse":
                    _write_jsonl(args.output, export_langfuse(threads))
            counts = [thread.to_dict() for thread in threads]
            print(json.dumps({"valid": True, "threads": len(counts), "runs": sum(len(t["runs"]) for t in counts)}))
        elif args.action == "prepare":
            threads = load_threads(args.threads)
            matches = [t for t in threads if t.to_dict()["thread_id"] == args.thread_id]
            if not matches:
                raise ContractError(f"unknown thread: {args.thread_id}")
            packet = build_packet(matches[0], args.run_id)
            _write_json(args.output, packet)
            print(json.dumps({"output": str(args.output), "coverage": packet["request_coverage"]}))
        else:
            packet = json.loads(args.input.read_text(encoding="utf-8"))
            rows = _read_jsonl(args.labels)
            if len(rows) != 1:
                raise ContractError("a prepared target-run packet requires exactly one label row")
            errors = validate_label(rows[0], packet)
            if errors:
                print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False), file=sys.stderr)
                return 1
            print(json.dumps({"valid": True, "rows": 1, "semantic_accuracy": "not certified"}))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0
