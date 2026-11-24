"""compache: fit a context and cache policy to recorded threads."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from importlib import resources
from pathlib import Path

from . import __version__
from .api import Policy, fit, replay
from .defaults import default_policy
from .errors import ContractError
from .labels import build_packet, label_index
from .models import _sheet, known_models
from .optimizer.contracts import ModelSpec
from .profile import history_resent, measure, profile
from .record import is_record
from .record import load as load_record
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
                help=f"A known model ({', '.join(known_models())}) or a JSON file of model spec fields. "
                "Left out, the model the threads record",
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

    text = "Fit a policy on four sample conversations that come with compache"
    sub = commands.add_parser("demo", help=text, description=text)
    sub.add_argument(
        "--model",
        default="claude-sonnet-5-5",
        help="The model to fit for, by name or a JSON file of model spec fields",
    )
    sub.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Keep the policy here. Left out, it is only shown",
    )
    sub.add_argument("--json", action="store_true", help="Print JSON")

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
    sub.add_argument(
        "--policy",
        type=Path,
        help="A fitted policy for the calculator to show and price on this workload",
    )

    command("check", "Validate threads and labels and summarize them")

    sub = command(
        "packets", "Write what a labeller may see for a sample of runs", labels=False
    )
    sub.add_argument("-o", "--output", type=Path, default=Path("packets.jsonl"))
    sub.add_argument("--sample", type=int, default=200, help="How many runs to label")
    sub.add_argument("--seed", default="0")

    text = "Read the conversations kept in Langfuse, MLflow or an OpenTelemetry export into a threads file"
    sub = commands.add_parser("pull", help=text, description=text)
    sub.add_argument("source", choices=["langfuse", "mlflow", "otel"])
    sub.add_argument(
        "export",
        nargs="?",
        type=Path,
        help="A file already exported, read instead of the API. Needed for otel",
    )
    sub.add_argument("-o", "--output", type=Path, default=Path("threads.jsonl.gz"))
    sub.add_argument("--days", type=float, default=30, help="How far back to read")
    sub.add_argument("--since", help="Read from this time instead")
    sub.add_argument("--until", help="Read up to this time")
    sub.add_argument("--environment", help="Only this Langfuse environment")
    sub.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only traces with this tag. Repeat for any of several",
    )
    sub.add_argument(
        "--filter",
        help="More conditions: a Langfuse filter in JSON, or an MLflow filter string",
    )
    sub.add_argument(
        "--tracking-uri", help="The MLflow server. Left out, MLFLOW_TRACKING_URI"
    )
    sub.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="An MLflow experiment ID. Repeat for several. Left out, experiment 0",
    )
    sub.add_argument(
        "--rows",
        type=Path,
        help="The same as the export file: observations or traces as JSON or JSONL",
    )
    sub.add_argument("--json", action="store_true", help="Print JSON")

    text = "Write the policy for the default workload, to start from before there is a record"
    sub = commands.add_parser("default", help=text, description=text)
    sub.add_argument(
        "--model",
        required=True,
        help=f"A known model ({', '.join(known_models())}) or a JSON file of model spec fields",
    )
    sub.add_argument("-o", "--output", type=Path, default=Path("policy.json"))
    sub.add_argument("--json", action="store_true", help="Print JSON")

    text = "Run compache behind a base URL: apply a policy, keep a record, or both"
    sub = commands.add_parser("proxy", help=text, description=text)
    sub.add_argument("--policy", type=Path, help="A fitted policy to apply")
    sub.add_argument(
        "--record",
        type=Path,
        help="A folder to record conversations in, without their text",
    )
    sub.add_argument(
        "--sample",
        type=float,
        default=0.0,
        help="Share of turns whose reply is asked what it needed. Needs --record",
    )
    sub.add_argument(
        "--notes",
        choices=["apart", "inline"],
        default="apart",
        help="Ask in a short request of its own after the reply, or in the reply itself",
    )
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=8787)
    sub.add_argument(
        "--upstream", help="One base URL to forward everything to, such as a gateway's"
    )
    sub.add_argument(
        "--openai-url", help="Where OpenAI's API goes instead of api.openai.com"
    )
    sub.add_argument(
        "--anthropic-url",
        help="Where Anthropic's API goes instead of api.anthropic.com",
    )
    sub.add_argument("--gemini-url", help="Where Gemini's API goes instead of Google's")

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
    resent = history_resent(load_threads(args.threads), policy.spec)
    text = "\n".join(
        [
            f"Fitted on {summary['threads']:,} threads, {summary['turns']:,} turns{labelled} "
            f"for {summary['model']}.",
            "",
            policy.describe(),
            *_trimmed(resent),
            "",
            f"Wrote {args.output}",
        ]
    )
    return {"output": str(args.output), **summary}, text


def _demo(args):
    with resources.as_file(resources.files(__package__).joinpath("demo")) as sample:
        policy = fit(
            sample / "threads.jsonl.gz", sample / "labels.jsonl", model=args.model
        )
    if args.output is not None:
        policy.save(args.output)
    summary = policy.summary
    text = "\n".join(
        [
            f"Fitted on {summary['threads']:,} sample threads, {summary['turns']:,} turns "
            f"({summary['labelled_turns']:,} labelled) for {summary['model']}.",
            "",
            policy.describe(),
            "",
            f"Wrote {args.output}"
            if args.output is not None
            else "Not kept. Give -o to keep it.",
            "Your own: compache fit threads.jsonl --labels labels.jsonl",
        ]
    )
    return summary, text


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
    lines = [f"{report['threads']:,} threads, {report['planned_turns']:,} turns."]
    costs = report["cost_per_1000_turns"]
    if report["saving"] is None:
        lines.append(
            "Some requests did not fit the model: the policy answers "
            f"{report['policy_turns']:,} turns and keeping everything {report['baseline_turns']:,}."
        )
        common = report["common_prefix"]
        if common["turns"]:
            paired = common["cost_per_1000_turns"]
            lines.append(
                f"On the {common['turns']:,} turns both answer: {_money(paired, 'policy')} per 1,000 turns "
                f"with the policy against {_money(paired, 'keep_everything')} keeping everything."
            )
    else:
        lines.append(
            f"{_money(costs, 'policy')} per 1,000 turns with the policy against "
            f"{_money(costs, 'keep_everything')} keeping everything: {report['saving']:.0%} less."
        )
    if report["coverage"] is not None:
        lines.append(
            f"{report['coverage']:.0%} of the {report['needs_history']:,} labelled needs for earlier "
            f"context are in view. {report['tokens_kept']:.0%} of the history tokens are kept."
        )
    lines.append(f"{report['pings']:,} keep-warm pings sent.")
    return report, "\n".join(lines)


def _trimmed(resent) -> list[str]:
    """A warning for an app that already leaves earlier text out of its requests."""
    if resent is None or resent >= 0.8:
        return []
    return [
        f"Its requests carried {resent:.0%} of the conversation so far: the app already leaves "
        "earlier text out, so keeping everything costs more than it pays today."
    ]


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
    if 0 < measured["turns_that_retrieve"] < 1:
        lines.append(
            f"Turns that retrieve anything: {measured['turns_that_retrieve']:.0%}."
        )
    if lengths["law"] == "continuation":
        another = lengths["another_turn"]
        lines.append(
            f"Chance of another turn: {another['after_first']:.0%} after turn 1, "
            f"{another['in_long_threads']:.0%} in long threads, halfway by turn {1 + another['half_turn']:g}. "
            f"{lengths['one_turn_share']:.0%} of threads are one turn."
        )
    if measured["idle_minutes"]:
        idle = measured["idle_minutes"]
        where = " in long threads" if idle["half_turn"] else ""
        lines.append(
            f"Minutes away between turns{where}: {idle['median']:.1f} ({idle['p95']:.1f})."
        )
        if idle["half_turn"]:
            lines.append(
                f"  after turn 1 the wait is {idle['first_factor']:.1f} times that, "
                f"halfway by turn {1 + idle['half_turn']:g}."
            )
        if idle["break_share"]:
            lines.append(
                f"  {idle['break_share']:.0%} of returns follow a long break, "
                f"median {idle['break_median'] / 60:.1f} hours."
            )
    extras = []
    if measured["answer_seconds"]:
        extras.append(f"an answer takes {measured['answer_seconds']:.0f} s")
    if measured["calls_per_turn"] and measured["calls_per_turn"] > 1:
        extras.append(f"{measured['calls_per_turn']:.1f} model calls a turn")
    if measured["reasoning_tokens"]:
        extras.append(
            f"{measured['reasoning_tokens']:,} output tokens an answer are billed beyond its text"
        )
    if measured["cache_read_when_live"] is not None:
        extras.append(
            f"a request still in the cache was read to {measured['cache_read_when_live']:.0%} by the next turn"
        )
    if measured["prompt_warmed_by_others"]:
        extras.append(
            f"another thread was active within the cache lifetime at "
            f"{measured['prompt_warmed_by_others']:.0%} of cold starts"
        )
    if extras:
        lines.append("Also: " + ", ".join(extras) + ".")
    lines += _trimmed(measured["history_resent"])
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
    result = profile(
        args.threads,
        args.labels,
        model=args.model,
        policy=args.policy,
        **_header(args),
    )
    if args.output:
        _write(args.output, result)
    lines = _describe(result["measured"]) + result["notes"]
    lines += ["", "Open the calculator with this workload:", result["link"]]
    if args.output:
        lines.append(f"Wrote {args.output}")
    return result, "\n".join(lines)


def _check(args):
    threads = load_threads(args.threads)
    if args.labels:
        labels = list(read_jsonl(args.labels))
    else:
        # A record folder carries its own labels.
        labels = load_record(args.threads)[1] if is_record(args.threads) else []
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


def _exported(path: Path) -> list:
    """Observation rows from an export: a JSON list, an API response, or JSONL."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        # An API response: Langfuse keeps its rows under `data`, MLflow its traces under `traces`.
        return data.get("data") or data.get("traces") or [data]
    return data


_LEFT_OUT = (
    (
        "calls_recorded_twice",
        "{:,} model calls were recorded twice and are counted once.",
    ),
    (
        "helper_calls",
        "{:,} model calls outside the conversation, such as a query rewriter, are left out.",
    ),
    (
        "turns_with_pasted_retrieval",
        "In {:,} turns the retrieved text was pasted into the prompt. It is read as retrieved text.",
    ),
    (
        "threads_read_trace_by_trace",
        "{:,} threads were read trace by trace, as their model calls do not carry the conversation.",
    ),
    (
        "threads_with_trimmed_history",
        "In {:,} threads the app did not resend the whole history, so keeping everything is not what it does today.",
    ),
    (
        "overlapping_turns",
        "{:,} turns began before the answer to the one before. They are read one after the other.",
    ),
    (
        "sessions_begun_earlier",
        "{:,} sessions began before the first row read. Read further back to include them.",
    ),
    (
        "sessions_without_text",
        "{:,} sessions have model calls without their text. Record prompts and answers to use them.",
    ),
    ("sessions_unreadable", "{:,} sessions could not be rebuilt into turns."),
    (
        "traces_without_session",
        "{:,} traces have no session ID, and a thread needs one.",
    ),
)


def _pull(args):
    export = args.rows or args.export
    window = dict(since=args.since, until=args.until, days=args.days)
    if args.source == "otel":
        from .otel import pull

        if export is None:
            raise ContractError(
                "pull otel reads an export file: compache pull otel spans.jsonl"
            )
        threads, report = pull(export, until=args.until)
    elif args.source == "mlflow":
        from .mlflow import pull

        if export is not None:
            threads, report = pull(traces=_exported(export), until=args.until)
        else:
            threads, report = pull(
                tracking_uri=args.tracking_uri,
                experiment_ids=args.experiment or ["0"],
                filter=args.filter,
                **window,
            )
    else:
        from .langfuse import pull

        if export is not None:
            threads, report = pull(rows=_exported(export), **window)
        else:
            conditions = json.loads(args.filter) if args.filter else []
            threads, report = pull(
                environment=args.environment,
                tags=args.tag,
                filters=conditions if isinstance(conditions, list) else [conditions],
                **window,
            )
    rows = "observations" if args.source == "langfuse" else "spans"
    lines = [
        f"Read {report.get('observations', 0):,} {rows}: {report.get('model_calls', 0):,} "
        f"model calls in {report.get('sessions', 0):,} sessions."
    ]
    notes = [text.format(report[key]) for key, text in _LEFT_OUT if report.get(key)]
    if not threads:
        raise ContractError(" ".join(lines + notes + ["No thread could be read."]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.name.endswith(".gz") else open
    with opener(args.output, "wt", encoding="utf-8") as stream:
        for thread in threads:
            stream.write(json.dumps(thread, ensure_ascii=False) + "\n")
    lines.append(
        f"Wrote {report['threads']:,} threads, {report['turns']:,} turns to {args.output}."
    )
    lines += notes
    if report["models"]:
        lines.append(
            "Answered by "
            + ", ".join(
                f"{name} ({count:,} calls)" for name, count in report["models"].items()
            )
            + "."
        )
    if len(report["models"]) > 1:
        lines.append(
            "A fit prices every thread as one model: the most common, unless --model names another. "
            "To fit one app of several, narrow the read with --tag or --environment."
        )
    lines += ["", f"Next: compache fit {args.output}"]
    return {"output": str(args.output), **report}, "\n".join(lines)


def _default(args):
    policy = default_policy(args.model)
    policy.save(args.output)
    summary = policy.summary
    lines = [
        f"Fitted to the default workload, not to yours, for {summary['model']}.",
        "",
        policy.describe(),
        "",
        f"Wrote {args.output}",
        "Next: apply it and record your own traffic, then fit on the record.",
    ]
    return summary, "\n".join(lines)


def _models(args):
    models = _sheet()["models"]
    lines = [f"Prices per million tokens, checked {_sheet()['checked']}."]
    for model_id, entry in models.items():
        seconds = [entry["cache_ttl_seconds"]] + [
            other["seconds"] for other in entry.get("cache_lifetimes", ())
        ]
        lasting = [f"{s // 60} min" if s < 3600 else f"{s / 3600:g} h" for s in seconds]
        lines.append(
            f"{model_id:<20} input ${entry['input_price']:g}, cache read ${entry['cache_read_price']:g}, "
            f"output ${entry['output_price']:g}, cache {' or '.join(lasting)}"
        )
    return models, "\n".join(lines)


_COMMANDS = {
    "fit": _fit,
    "demo": _demo,
    "replay": _replay,
    "profile": _profile,
    "check": _check,
    "packets": _packets,
    "pull": _pull,
    "default": _default,
    "models": _models,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "proxy":
        # Runs until interrupted, so it is imported and started apart from the commands that report.
        from .proxy import main as run_proxy

        try:
            return run_proxy(args)
        except (ContractError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        result, text = _COMMANDS[args.command](args)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(result, ensure_ascii=False) if getattr(args, "json", False) else text
    )
    return 0
