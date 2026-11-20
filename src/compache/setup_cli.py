"""Explicit, offline entry point for configuring Compache from a workload sample."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .optimizer.contracts import ModelSpec, digest, number
from .traces import ContractError
from .workflow_presets import WORKFLOW_NAMES, describe_workflow, example_metadata, example_model_spec


def register_setup(groups):
    cmd = groups.add_parser("setup", help="Preview or fit a workload locally; both data inputs are opt-in")
    cmd.add_argument("--workflow", choices=WORKFLOW_NAMES, default="general")
    cmd.add_argument("--demo", action="store_true", help="Fit and compare on built-in synthetic counts")
    cmd.add_argument("--metadata", type=Path, help="Opt in to reading this local content-free run-count JSONL")
    cmd.add_argument("--model-spec", type=Path, help="Local model economics JSON for fitting")
    cmd.add_argument("--observation-end", help="End of observed traffic, including idle time (timezone-aware)")
    cmd.add_argument("--content-sample", type=Path, help="Opt in to processing this selected local RAG content export")
    cmd.add_argument("--packets-dir", type=Path, help="Prepare private labeling packets in this new directory")
    cmd.add_argument("--labels", type=Path, help="Use reviewed local evidence labels for the same content sample")
    cmd.add_argument("--max-threads", type=int, default=3, help="Maximum whole threads to decode for dependency evidence")
    cmd.add_argument("--retention-floor", type=float, default=0.95, help="Minimum retained fraction of assessed dependencies")
    cmd.add_argument("--policy-out", type=Path, help="Save a fitted candidate to a new local file")
    cmd.add_argument("--json", action="store_true", help="Write the structured setup report to stdout")


def _validate(args):
    number(args.retention_floor, "retention_floor", 0, 1)
    number(args.max_threads, "max_threads", 1, integer=True)
    if args.demo and any((args.metadata, args.content_sample, args.labels, args.packets_dir,
                          args.policy_out, args.model_spec, args.observation_end)):
        raise ContractError("--demo uses only the built-in example")
    if (args.labels or args.packets_dir) and not args.content_sample:
        raise ContractError("--labels and --packets-dir require an explicit --content-sample")
    if args.content_sample and bool(args.packets_dir) == bool(args.labels):
        raise ContractError("choose --packets-dir to prepare or --labels to assess the selected sample")
    if args.packets_dir and any((args.metadata, args.model_spec, args.observation_end, args.policy_out)):
        raise ContractError("packet preparation takes only a content sample and a new packet directory")
    if bool(args.metadata) != bool(args.observation_end):
        raise ContractError("supply --metadata and its --observation-end together")
    if bool(args.model_spec) != bool(args.metadata or args.labels):
        raise ContractError("fitting requires --model-spec and either --metadata or reviewed --labels")
    if args.policy_out and not args.metadata:
        raise ContractError("saving a workload policy requires a metadata fit")
    if args.policy_out and os.path.lexists(args.policy_out):
        raise ContractError("--policy-out must name a new file")


def _split(rows, label_training):
    # Grouped retrospective holdout. Label-training threads cannot enter evaluation.
    if not rows:
        raise ContractError("metadata is empty")
    identifiers = set()
    for row in rows:
        if not isinstance(row.get("thread_id"), str) or not row["thread_id"]:
            raise ContractError("metadata requires a pseudonymous thread_id")
        identifiers.add(row["thread_id"])
    if not label_training.issubset({digest(key) for key in identifiers}):
        raise ContractError("combined fitting needs the sampled threads in metadata under the same pseudonyms")
    available = sorted((key for key in identifiers if digest(key) not in label_training), key=digest)
    size = min(len(identifiers) - 1, len(available), max(1, len(identifiers) // 4))
    heldout = set(available[:size])
    return ([r for r in rows if r["thread_id"] not in heldout],
            [r for r in rows if r["thread_id"] in heldout])


def _read_metadata(path):
    from .setup_metadata import validate_metadata_row

    rows = []
    with path.open("rb") as stream:
        for index, line in enumerate(stream, start=1):
            if line.isspace():
                continue
            try:
                row = json.loads(line)
                validate_metadata_row(row)
            except (ValueError, OverflowError):
                raise ContractError(f"invalid count-only metadata on row {index}") from None
            rows.append(row)
    return rows


def _save_new(path, artifact):
    # Serialize before creating the file, then remove a partial write on failure.
    data = json.dumps(artifact, allow_nan=False, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _cache_comparison(train, holdout, train_end, holdout_end, spec, floor):
    from .setup_metadata import evaluate_metadata, fit_metadata

    candidate, stats = fit_metadata(train, spec, observation_end=train_end, retention_floor=floor)
    if not holdout:
        return candidate, stats, {"status": "insufficient_evidence", "reason": "no independent evaluation threads"}
    own = evaluate_metadata(holdout, candidate, observation_end=holdout_end)
    # Price a single mixed-workload prior against exactly the same returns and route.
    # Align the generic artifact's model and namespace; its observations remain synthetic.
    public, public_end = example_metadata("general")
    route = train[0]["route_id"]
    general = [{**r, "model_id": spec.model_id, "route_id": route,
                "thread_id": digest(["public-general-example", r["thread_id"]])} for r in public]
    reference, _ = fit_metadata(general, spec, observation_end=public_end, retention_floor=floor)
    prior = evaluate_metadata(holdout, reference, observation_end=holdout_end)
    saving = None
    if own["status"] == prior["status"] == "modeled" and prior["candidate_cost"]:
        saving = 1 - own["candidate_cost"] / prior["candidate_cost"]
    return candidate, stats, {"status": own["status"], "scope": "cache_timing_only",
                              "own": own, "generic": prior,
                              "savings_vs_generic_fraction": saving,
                              "heldout_threads": len({r["thread_id"] for r in holdout}),
                              "heldout_runs": len(holdout)}


def run_setup(args):
    from .setup_dependencies import dependency_preview, fit_content_recipes, prepare_dependency_sample

    _validate(args)
    report = {
        "mode": "preview", "workflow": describe_workflow(args.workflow),
        "access": {"metadata_file": bool(args.metadata), "content_file": bool(args.content_sample),
                   "network": False},
        "retention_floor": args.retention_floor,
        "active_actions": {"compaction": "host", "keep_warm_pings": False},
    }
    if args.packets_dir:
        report["mode"] = "prepare_packets"
        report["packets"] = prepare_dependency_sample(args.content_sample, args.packets_dir,
                                                      allow_content=True, max_threads=args.max_threads)
        return report
    if not (args.demo or args.metadata or args.labels):
        report["dependency_preview"] = dependency_preview(max_threads=args.max_threads)
        return report
    spec = example_model_spec() if args.demo else ModelSpec.from_dict(json.loads(args.model_spec.read_text()))
    evidence = None
    if args.labels:
        evidence = fit_content_recipes(args.content_sample, args.labels, spec, allow_content=True,
                                       max_threads=args.max_threads, retention_floor=args.retention_floor)
        report["dependencies"] = evidence["report"]
        report["mode"] = "dependency_assessment"
    if args.demo:
        train, train_end = example_metadata(args.workflow)
        holdout, holdout_end = example_metadata(args.workflow, split="heldout")
        report["mode"] = "synthetic_demo"
    elif args.metadata:
        excluded = set(evidence["training_threads"]) if evidence else set()
        train, holdout = _split(_read_metadata(args.metadata), excluded)
        train_end = holdout_end = args.observation_end
        report["mode"] = "combined_fit" if evidence else "metadata_fit"
    else:
        return report
    candidate, stats, comparison = _cache_comparison(train, holdout, train_end, holdout_end, spec, args.retention_floor)
    report["metadata"] = stats
    report["cache_comparison"] = comparison
    if evidence:
        candidate["recipes"] = evidence["recipes"]
        candidate["training_threads"] = sorted(set(candidate["training_threads"]) | set(evidence["training_threads"]))
    if args.policy_out:
        from .optimizer import WorkloadPolicy

        WorkloadPolicy(candidate)
        _save_new(args.policy_out, candidate)
        report["saved_candidate"] = str(args.policy_out)
    return report


def _percent(value):
    return "unavailable" if value is None else f"{value:.2%}"


def format_setup(report):
    workflow = report["workflow"]
    mode = report["mode"]
    lines = [f"Compache | {workflow['display_name']}", workflow["description"], ""]
    if mode == "preview":
        lines.extend([
            workflow["timing"], workflow["context_growth"], workflow["dependency_example"], "",
            "Your host handles compaction by default. Compache reads no customer data and schedules no pings.",
            "Try a local example using invented counts.",
            f"  compache setup --workflow {workflow['name']} --demo", "",
            "To fit cache timing, opt in with --metadata and a count-only JSONL file.",
            "To assess compaction, opt in separately with --content-sample and a selected, redacted RAG export.",
            "Dependency review uses actual user questions, answers and retrieved text. You choose who may see the packets.",
        ])
        return "\n".join(lines)
    if mode == "prepare_packets":
        packets = report["packets"]
        lines += [f"Prepared {packets['packets']} private packets from {packets['selected_threads']} sampled threads.",
                  "Review the packets locally. Opening them with a hosted coding agent sends their contents to its provider.",
                  "Supply the reviewed labels with --labels to score retention. Cache-timing analysis is off."]
        return "\n".join(lines)
    if "dependencies" in report:
        e = report["dependencies"]
        lines += [f"Reviewed {e['selected_runs']} runs from {e['selected_threads']} sampled threads.",
                  f"{e['eligible_recipes']} recipes qualified at a {_percent(report['retention_floor'])} retention floor."]
        projection = e["projection"]
        if projection["unscorable_runs"]:
            lines.append(f"Could project {projection['scored_runs']} runs. "
                         f"{projection['unscorable_runs']} lacked the full visible history.")
        for card in e["recipe_results"]:
            kept = card["recipe"]
            windows = [str(kept[name]) if kept[name] is not None else "all"
                       for name in ("input_runs", "retrieval_runs", "output_runs")]
            lines.append(f"  Keep {' / '.join(windows)} past input / retrieval / output runs. "
                         f"Retained {card['historical_retained']}/{card['historical_required']} "
                         f"assessed sources. Changed {card['changed_runs']} runs.")
        if mode == "dependency_assessment":
            lines.append("Cache timing was not fitted. This assessment does not install a policy.")
            return "\n".join(lines)
    stats = report["metadata"]
    lines += [("Built-in synthetic example." if mode == "synthetic_demo" else "Local count-only cache fit."),
              f"Trained on {stats['runs']} runs in {stats['threads']} threads. "
              f"The median turn added {stats['median_added_tokens']:,.0f} input tokens."]
    if stats["median_return_gap_seconds"] is not None:
        lines.append(f"The median completion-to-next-request gap was {stats['median_return_gap_seconds']:,.0f} seconds. "
                     f"The median run lasted {stats['median_duration_seconds']:,.0f} seconds.")
    comparison = report["cache_comparison"]
    if comparison.get("own", {}).get("status") == "modeled":
        own = comparison["own"]
        lines += [f"Evaluated {comparison['heldout_runs']} turns in {comparison['heldout_threads']} separate threads.",
                  "Modeled cost in ordinary input-token equivalents",
                  f"  Host preserving history, no pings   {own['baseline_cost']:>14,.1f}",
                  f"  Workload-fitted policy             {own['candidate_cost']:>14,.1f}   ({own['ping_calls']} pings)"]
        generic = comparison["generic"]
        if generic["status"] == "modeled":
            lines += [f"  Mixed synthetic policy             {generic['candidate_cost']:>14,.1f}   ({generic['ping_calls']} pings)"]
        lines += [f"Estimated savings versus no pings were {_percent(own['savings_fraction'])}.",
                  f"Estimated savings versus the mixed policy were {_percent(comparison['savings_vs_generic_fraction'])}."]
    else:
        reason = comparison.get("own", comparison).get("reason", "insufficient evidence")
        lines.append(f"Held-out cache cost unavailable ({reason}).")
    lines += ["The cache comparison assumes a stable, append-only thread. It prices maintenance and foreground model calls.",
              "Dependency retention and answer quality require separate evidence."]
    if "saved_candidate" in report:
        lines.append(f"Saved the candidate to {report['saved_candidate']}.")
    lines.append("Live policy activation and timers remain under host control.")
    return "\n".join(lines)
