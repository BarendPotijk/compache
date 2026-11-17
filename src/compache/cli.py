"""Local-only RAG evidence, workload fitting, optimization and replay commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dependencies import build_packet, validate_label
from .traces import (
    ContractError,
    export_langfuse,
    load_threads,
    normalize_langfuse,
    write_threads,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values
        ),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)
    traces = groups.add_parser(
        "traces", help="Validate and convert offline RAG trace files"
    ).add_subparsers(dest="action", required=True)
    validate = traces.add_parser("validate")
    validate.add_argument("input", type=Path)
    for name in ("normalize", "export-langfuse"):
        command = traces.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    dependencies = groups.add_parser(
        "dependencies", help="Prepare and validate dependency labels"
    ).add_subparsers(dest="action", required=True)
    prepare = dependencies.add_parser("prepare")
    prepare.add_argument("--threads", type=Path, required=True)
    prepare.add_argument("--thread-id", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    labels = dependencies.add_parser("validate")
    labels.add_argument(
        "--input", type=Path, required=True, help="Prepared target-run packet"
    )
    labels.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Exactly one JSONL label for this packet",
    )
    comparison = groups.add_parser("compare", help="Replay a frozen policy against pinned Headroom lossless library arms")
    for name in ("threads", "labels", "policy", "profile", "output"):
        comparison.add_argument(f"--{name}", type=Path, required=True)
    comparison.add_argument("--observation-end", required=True)
    comparison.add_argument("--runtime-overhead-tokens", type=int, default=0)
    comparison.add_argument("--headroom-tokenizer-model", default="gpt-4o",
                            help="Headroom's local tokenizer selection; no model call is made")
    for name in ("fit", "solve", "replay", "prepare"):
        command = groups.add_parser(
            name,
            help={
                "fit": "Fit a content-free workload profile",
                "solve": "Fit on training threads, optimize, then evaluate held-out threads",
                "replay": "Replay a saved policy with a modeled cache",
                "prepare": "Project a thread through the deterministic request gate",
            }[name],
        )
        command.add_argument("--threads", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in ("fit", "solve", "replay"):
            command.add_argument(
                "--labels",
                type=Path,
                help="Frozen dependency labels; missing labels retain unknown risk",
            )
            command.add_argument(
                "--observation-end",
                required=True,
                help="Explicit observation cutoff, ISO timestamp with timezone",
            )
        if name in ("fit", "solve"):
            command.add_argument(
                "--model-spec",
                type=Path,
                required=True,
                help="Explicit economic/token-ruler scenario",
            )
            command.add_argument("--timing-source", default="supplied_timestamps")
        if name == "solve":
            command.add_argument("--heldout-company")
            command.add_argument("--heldout-fraction", type=float, default=0.25)
            command.add_argument(
                "--max-miss-rate",
                type=float,
                default=1.0,
                help="Optional ceiling on runs with any missing requirement",
            )
            command.add_argument(
                "--min-dependency-coverage",
                type=float,
                default=0.95,
                help="Minimum retained fraction of historical evidence requirements",
            )
            command.add_argument("--max-unknown-change-rate", type=float, default=0.0)
            command.add_argument("--max-pings", type=int, default=6)
        if name in ("prepare", "replay"):
            command.add_argument("--policy", type=Path, required=True)
        if name == "replay":
            command.add_argument("--profile", type=Path, required=True)
            command.add_argument(
                "--recovery-overhead", choices=("priced", "zero"), default="priced"
            )
            command.add_argument("--runtime-overhead-tokens", type=int, default=0)
        if name == "prepare":
            command.add_argument("--thread-id", required=True)
            command.add_argument("--run-id", required=True)
    return parser


def _optimizer_command(args):
    from dataclasses import asdict
    from .optimizer import (
        Constraints,
        ModelSpec,
        Policy,
        Request,
        ThreadState,
        fit_profile,
        optimize,
        prepare,
        replay,
    )
    from .optimizer.contracts import seconds

    threads = load_threads(args.threads)
    labels = _read_jsonl(args.labels) if getattr(args, "labels", None) else []
    if args.group in ("fit", "solve"):
        spec = ModelSpec.from_dict(
            json.loads(args.model_spec.read_text(encoding="utf-8"))
        )
        if args.group == "fit":
            result = fit_profile(
                threads,
                labels,
                spec,
                observation_end=args.observation_end,
                timing_source=args.timing_source,
            )
            _write_json(args.output, result)
            return {
                "output": str(args.output),
                "threads": result["threads"],
                "runs": result["runs"],
                "dependency": result["dependency"],
            }
        result = optimize(
            threads,
            labels,
            spec,
            observation_end=args.observation_end,
            constraints=Constraints(
                max_miss_rate=args.max_miss_rate,
                max_unknown_change_rate=args.max_unknown_change_rate,
                min_historical_dependency_coverage=args.min_dependency_coverage,
            ),
            heldout_company=args.heldout_company,
            heldout_fraction=args.heldout_fraction,
            timing_source=args.timing_source,
            max_pings=args.max_pings,
        )
        _write_json(args.output / "profile.json", result["profile"])
        _write_json(args.output / "candidate-policy.json", result["policy"])
        _write_json(args.output / "policy.json", result["recommended_policy"])
        _write_json(args.output / "report.json", result)
        _write_jsonl(args.output / "heldout-ledger.jsonl", result["heldout"]["ledger"])
        return {
            "output": str(args.output),
            "selection": {
                "candidates": result["selection"]["candidate_count"],
                "feasible": result["selection"]["feasible_count"],
            },
            **result["result"],
        }
    policy = Policy.from_dict(json.loads(args.policy.read_text(encoding="utf-8")))
    if args.group == "replay":
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        result = replay(
            threads,
            labels,
            policy,
            profile,
            observation_end=args.observation_end,
            recovery_overhead=args.recovery_overhead,
            overhead_tokens=args.runtime_overhead_tokens,
        )
        _write_json(args.output, result)
        return result["summary"]
    thread = next((t for t in threads if t.thread_id == args.thread_id), None)
    if thread is None or args.run_id not in {r.run_id for r in thread.runs}:
        raise ContractError("unknown thread/run")
    state = ThreadState(thread.thread_id)
    for run in thread.runs:
        result = prepare(
            Request.from_thread(thread, run.run_id),
            policy,
            state,
            seconds(run.generation.started_at),
        )
        if not result.admitted:
            raise ContractError(result.reason)
        if run.run_id == args.run_id:
            break
    _write_json(
        args.output,
        {
            "parts": [asdict(p) for p in result.parts],
            "full_part_ids": sorted(result.full_part_ids),
            "removed_part_ids": sorted(result.removed_part_ids),
            **result.telemetry(),
        },
    )
    return result.telemetry()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.group == "compare":
            from .baselines.headroom import HeadroomLossless
            from .comparison import compare_headroom
            from .optimizer import Policy

            result = compare_headroom(
                load_threads(args.threads), _read_jsonl(args.labels),
                Policy.from_dict(json.loads(args.policy.read_text())),
                json.loads(args.profile.read_text()),
                HeadroomLossless(args.headroom_tokenizer_model),
                observation_end=args.observation_end,
                overhead_tokens=args.runtime_overhead_tokens,
            )
            _write_json(args.output, result)
            print(json.dumps({"output": str(args.output), "baselines": {
                k: {"comparable_cost": v["comparable_cost"],
                    "compache_saving_fraction": v["compache_saving_fraction"],
                    **v["summary"]} for k, v in result["baselines"].items()
            }}))
        elif args.group in ("fit", "solve", "replay", "prepare"):
            print(json.dumps(_optimizer_command(args), ensure_ascii=False))
        elif args.group == "traces":
            if args.action == "normalize":
                threads = normalize_langfuse(_read_jsonl(args.input))
                write_threads(threads, args.output)
            else:
                threads = load_threads(args.input)
                if args.action == "export-langfuse":
                    _write_jsonl(args.output, export_langfuse(threads))
            counts = [thread.to_dict() for thread in threads]
            print(
                json.dumps(
                    {
                        "valid": True,
                        "threads": len(counts),
                        "runs": sum(len(t["runs"]) for t in counts),
                    }
                )
            )
        elif args.action == "prepare":
            threads = load_threads(args.threads)
            matches = [t for t in threads if t.to_dict()["thread_id"] == args.thread_id]
            if not matches:
                raise ContractError(f"unknown thread: {args.thread_id}")
            packet = build_packet(matches[0], args.run_id)
            _write_json(args.output, packet)
            print(
                json.dumps(
                    {"output": str(args.output), "coverage": packet["request_coverage"]}
                )
            )
        else:
            packet = json.loads(args.input.read_text(encoding="utf-8"))
            rows = _read_jsonl(args.labels)
            if len(rows) != 1:
                raise ContractError(
                    "a prepared target-run packet requires exactly one label row"
                )
            errors = validate_label(rows[0], packet)
            if errors:
                print(
                    json.dumps({"valid": False, "errors": errors}, ensure_ascii=False),
                    file=sys.stderr,
                )
                return 1
            print(
                json.dumps(
                    {"valid": True, "rows": 1, "semantic_accuracy": "not certified"}
                )
            )
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    return 0
