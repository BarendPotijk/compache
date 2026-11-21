"""compache: fit a context and cache policy to recorded threads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .api import Policy, fit, replay
from .errors import ContractError
from .labels import build_packet, label_index
from .models import _sheet, known_models
from .optimizer.contracts import ModelSpec
from .profile import measure, profile
from .threads import load_threads, read_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compache", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, text, *, model=False, labels=True):
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument(
            "threads",
            type=Path,
            help="Threads as JSONL, gzipped JSONL or a folder of either",
        )
        if labels:
            sub.add_argument(
                "--labels", type=Path, help="Dependency labels for these threads"
            )
        if model:
            sub.add_argument(
                "--model",
                required=True,
                help=f"A known model ({', '.join(known_models())}) or a JSON file of model spec fields",
            )
            sub.add_argument(
                "--system-prompt", type=Path, help="Text file with the system prompt"
            )
            sub.add_argument(
                "--tools",
                type=Path,
                help="JSON file with the tool definitions sent on every request",
            )
        sub.add_argument("--json", action="store_true", help="Print JSON")
        return sub

    sub = command(
        "fit", "Fit a policy and say what it saves on these threads", model=True
    )
    sub.add_argument(
        "--observed-until",
        help="When the record was cut, if conversations were still going",
    )
    sub.add_argument("-o", "--output", type=Path, default=Path("policy.json"))

    sub = command("replay", "Price a fitted policy on threads it has not seen")
    sub.add_argument("--policy", type=Path, required=True)
    sub.add_argument(
        "--observed-until",
        help="When the record was cut, if conversations were still going",
    )
    sub.add_argument("-o", "--output", type=Path, help="Write the full report here")

    sub = command(
        "profile",
        "Measure the workload and open the cost calculator with it",
        model=True,
    )
    sub.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the profile here, for the calculator's Import",
    )

    command("check", "Validate threads and labels and summarize them")

    sub = command(
        "packets", "Write what a labeller may see for a sample of runs", labels=False
    )
    sub.add_argument("-o", "--output", type=Path, default=Path("packets.jsonl"))
    sub.add_argument("--sample", type=int, default=200, help="How many runs to label")
    sub.add_argument("--seed", default="0")

    listing = commands.add_parser(
        "models", help="List the models with a built-in price sheet"
    )
    listing.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def _header(args) -> dict:
    options = {}
    if args.system_prompt:
        options["system_prompt"] = args.system_prompt.read_text("utf-8")
    if args.tools:
        options["tools"] = json.loads(args.tools.read_text("utf-8"))
    return options


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def _money(costs, key) -> str:
    return (
        f"${costs[key]:,.2f}"
        if costs["unit"] == "USD"
        else f"{costs[key]:,.0f} input tokens"
    )


def _fit(args):
    policy = fit(
        args.threads,
        args.labels,
        model=args.model,
        observed_until=args.observed_until,
        **_header(args),
    )
    policy.save(args.output)
    summary = policy.summary
    labelled = (
        f" ({summary['labelled_turns']:,} labelled)"
        if summary["labelled_turns"]
        else ""
    )
    text = (
        f"Fitted on {summary['threads']:,} threads, {summary['turns']:,} turns{labelled} "
        f"for {summary['model']}.\n\n{policy.describe()}\n\nWrote {args.output}"
    )
    return {"output": str(args.output), **summary}, text


def _replay(args):
    report = replay(
        Policy.load(args.policy),
        args.threads,
        args.labels,
        observed_until=args.observed_until,
    )
    detail = report.pop("detail")
    if args.output:
        _write(args.output, {**report, "detail": detail})
    lines = [f"{report['threads']:,} threads, {report['turns']:,} turns."]
    costs = report["cost_per_1000_turns"]
    if report["saving"] is None:
        lines.append(
            "Some requests did not fit the model, so there is no comparable cost."
        )
    else:
        lines.append(
            f"{_money(costs, 'policy')} per 1,000 turns with the policy against "
            f"{_money(costs, 'keep_everything')} keeping everything: {report['saving']:.0%} less."
        )
    if report["coverage"] is not None:
        lines.append(
            f"{report['coverage']:.0%} of the {report['needs_history']:,} turns that need earlier "
            f"context have it in view. {report['tokens_kept']:.0%} of the history tokens are kept."
        )
    lines.append(f"{report['pings']:,} keep-warm pings sent.")
    return report, "\n".join(lines)


def _describe(measured) -> list[str]:
    tokens, lengths = measured["tokens_per_turn"], measured["turns_per_thread"]
    lines = [
        f"{measured['threads']:,} threads, {measured['turns']:,} turns.",
        "Tokens per turn, median (95th percentile): "
        + ", ".join(
            f"{name} {tokens[kind]['median']:,} ({tokens[kind]['p95']:,})"
            for kind, name in (
                ("input", "question"),
                ("retrieval", "retrieved"),
                ("output", "answer"),
            )
        )
        + ".",
        f"Turns per thread: {lengths['median']} ({lengths['p95']}).",
    ]
    if measured["idle_minutes"]:
        idle = measured["idle_minutes"]
        lines.append(
            f"Minutes away between turns: {idle['median']:.1f} ({idle['p95']:.1f})."
        )
    if measured["needs"]:
        share = measured["turns_needing_history"] / measured["assessed_turns"]
        lines.append(
            f"Labels: {measured['labelled_turns']:,} turns. Of the {measured['assessed_turns']:,} "
            f"assessed after a thread's first turn, {share:.0%} need earlier context."
        )
        for kind, name in (
            ("input", "questions"),
            ("retrieval", "retrieved text"),
            ("output", "answers"),
        ):
            found = measured["needs"][kind]
            if found["turns_back"]:
                back = found["turns_back"]
                lines.append(
                    f"  earlier {name}: {found['share_of_turns']:.0%} of turns, "
                    f"median {back['median']} back (95th percentile {back['p95']})."
                )
    elif measured["labelled_turns"]:
        lines.append(f"Labels: {measured['labelled_turns']:,} turns, none assessed.")
    return lines


def _profile(args):
    result = profile(args.threads, args.labels, model=args.model, **_header(args))
    if args.output:
        _write(args.output, result)
    lines = _describe(result["measured"]) + result["notes"]
    lines += ["", "Open the calculator with this workload:", result["link"]]
    if args.output:
        lines.append(f"Wrote {args.output}")
    return result, "\n".join(lines)


def _check(args):
    threads = load_threads(args.threads)
    labels = list(read_jsonl(args.labels)) if args.labels else []
    # Building the index validates every label against what its run could see.
    label_index(threads, labels)
    measured = measure(threads, ModelSpec(threads[0].runs[0].generation.model), labels)
    return {"valid": True, **measured}, "\n".join(["Valid."] + _describe(measured))


def _packets(args):
    threads = load_threads(args.threads)
    candidates = [(t, r) for t in threads for r in t.runs if r.run_index > 1]

    def rank(pair):
        thread, run = pair
        return hashlib.sha256(
            f"{args.seed}:{thread.thread_id}:{run.run_id}".encode()
        ).hexdigest()

    chosen = sorted(
        sorted(candidates, key=rank)[: max(0, args.sample)],
        key=lambda p: (p[0].thread_id, p[1].run_index),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for thread, run in chosen:
            stream.write(
                json.dumps(build_packet(thread, run.run_id), ensure_ascii=False) + "\n"
            )
    result = {
        "output": str(args.output),
        "packets": len(chosen),
        "runs": len(candidates),
    }
    return result, f"Wrote {len(chosen):,} of {len(candidates):,} runs to {args.output}"


def _models(args):
    models = _sheet()["models"]
    lines = [f"Prices per million tokens, checked {_sheet()['checked']}."]
    for model_id, entry in models.items():
        lines.append(
            f"{model_id:<20} input ${entry['input_price']:g}, cache read ${entry['cache_read_price']:g}, "
            f"output ${entry['output_price']:g}, cache {entry['cache_ttl_seconds'] // 60} min"
        )
    return models, "\n".join(lines)


_COMMANDS = {
    "fit": _fit,
    "replay": _replay,
    "profile": _profile,
    "check": _check,
    "packets": _packets,
    "models": _models,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, text = _COMMANDS[args.command](args)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(result, ensure_ascii=False) if getattr(args, "json", False) else text
    )
    return 0
