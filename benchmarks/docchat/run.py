#!/usr/bin/env python3
"""Run DocChat: what a fitted policy costs on held-out threads, and whether answers hold up.

  cost     fit a policy on each build's training threads and price it on its test threads
  answers  a model answers test turns from the full history and from the policy's view; a judge grades both

Builds of one dataset with different --clock values share their text and differ only in timing.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docchat
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.contracts import ContractError
from compache.optimizer.parts import Request, placeholder
from compache.optimizer.recovery import REOPEN_INSTRUCTIONS, reopen
from compache.optimizer.retention import RetentionRecipe, fit_recipes, render_recipe
from compache.optimizer.workload_replay import replay_workload
from compache.optimizer.workload_solve import solve_workload
from compache.traces import Thread

HERE = Path(__file__).resolve().parent
END = "2026-06-01T00:00:00Z"
HEADER = {
    "system_prompt": "Answer from the conversation and the retrieved excerpts.",
    "default_tools": [],
    "share_headers": False,
}
PART = {"input": "user", "retrieval": "retrieved excerpt", "output": "assistant"}
LIVE = (
    "Answer the user's latest message from the conversation and the retrieved excerpts. Do not use outside "
    "knowledge. Earlier passages may be compacted. "
    + REOPEN_INSTRUCTIONS
    + " To call reopen, reply with "
    "only the word REOPEN followed by the reference. Its result is added below your request, and you can "
    "then answer or reopen again."
)
REOPEN_ROUNDS = 3
GRADER = (
    "You grade one answer for a benchmark of chats over documents. You receive the user's message, the key "
    "facts a correct answer must contain, a reference answer and the answer to grade. correct is true only if "
    "the answer conveys every key fact and contradicts none. An answer that says it lacks the information, or "
    "asks what the user means, is not correct. Be strict and literal."
)
GRADE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "missing": {"type": "string"}},
    "required": ["correct", "missing"],
    "additionalProperties": False,
}


def load(root, split):
    """A build's threads, labels and grading notes for one split."""
    manifest = docchat.read_json(root / "manifest.json")
    rows = [e for e in manifest["threads"] if e["split"] == split]
    lines = lambda key: [
        [json.loads(line) for line in gzip.open(root / e[key], "rt")] for e in rows
    ]  # noqa: E731
    return (
        manifest,
        rows,
        [t[0] for t in lines("trace_path")],
        lines("labels_path"),
        lines("grading_path"),
    )


def cost(args):
    spec = ModelSpec.from_dict(docchat.read_json(args.model_spec))
    results = []
    for root in map(Path, args.data):
        manifest, _, train, train_labels, _ = load(root, "train")
        _, _, test, test_labels, _ = load(root, "test")
        as_threads = lambda rows: [Thread.from_dict(copy.deepcopy(t)) for t in rows]  # noqa: E731
        flat = lambda groups: [label for group in groups for label in group]  # noqa: E731
        policy = solve_workload(
            as_threads(train), flat(train_labels), spec, observation_end=END, **HEADER
        )
        solved, chosen = policy["solve"], policy["solve"]["chosen"]
        policy["planner_options"] = {"max_pings": chosen["max_pings"]}
        for row in policy["recipes"]:
            row["eligible"] = (
                chosen["recipe"] is not None and row["recipe"] == chosen["recipe"]
            )
        priced = replay_workload(
            as_threads(test), [], policy, observation_end=END, **HEADER
        )
        keep, fitted = (
            priced["baseline"]["totals"]["cost_input_equivalents"],
            priced["candidate"]["totals"]["cost_input_equivalents"],
        )
        held = (
            fit_recipes(
                as_threads(test),
                flat(test_labels),
                spec,
                [RetentionRecipe(**chosen["recipe"])],
                retention_floor=0.0,
            )[0]
            if chosen["recipe"]
            else None
        )
        # What the cut loses on the test threads is priced the way the solver prices it on the training ones.
        recovery = (
            spec.read_multiplier * held["recovery_view_tokens"]
            + spec.write_multiplier * held["recovery_restored_tokens"]
            + spec.output_multiplier
            * spec.expansion_output_tokens
            * (
                held["changed_historical_required"]
                - held["changed_historical_retained"]
            )
            if held
            else 0
        )
        turns = priced["baseline"]["answers"]
        results.append(
            {
                "data": str(root),
                "clock": manifest["clock"],
                "model": spec.model_id,
                "train": {
                    "threads": len(train),
                    "turns": solved["turns"],
                    "saving": solved["saving_fraction"],
                    "coverage": chosen["coverage"],
                },
                "policy": {
                    "recipe": chosen["recipe"],
                    "max_pings": chosen["max_pings"],
                },
                "test": {
                    "threads": len(test),
                    "turns": turns,
                    "keep_everything_per_turn": keep / turns,
                    "policy_per_turn": fitted / turns,
                    "recovery_per_turn": recovery / turns,
                    "saving": 1 - (fitted + recovery) / keep,
                    "pings": priced["candidate"]["pings"],
                    "needs_history": held["historical_required"] if held else None,
                    "kept_in_view": held["historical_retained"]
                    / held["historical_required"]
                    if held and held["historical_required"]
                    else None,
                    "tokens_kept": held["projected_tokens"] / held["canonical_tokens"]
                    if held
                    else 1.0,
                },
            }
        )
        print(docchat.compact(results[-1]), flush=True)
    docchat.write_json(args.output, results)


def transcript(run, shown):
    """The request as text. `shown` pairs each part with its text or its compaction notice."""
    lines = ["CONVERSATION SO FAR:"]
    lines += [
        f"[turn {part.run_index}] {PART[part.kind]}: {text}"
        for part, text in shown
        if part.run_index < run["run_index"]
    ]
    if len(lines) == 1:
        lines.append("(this is the first message)")
    lines.append("\nRETRIEVED FOR THE LATEST MESSAGE:")
    lines += [f"({n}) {r['text']}" for n, r in enumerate(run["retrievals"], 1)]
    lines.append("\nLATEST USER MESSAGE:\n" + run["input"]["text"])
    return "\n".join(lines)


def answer(models, model, system, thread, run, recipe, spec):
    """One arm's answer to one turn: everything in view when `recipe` is None, else the policy's view.

    The model may call the package's reopen tool, written as text, up to REOPEN_ROUNDS times.
    """
    request = Request.from_thread(
        Thread.from_dict(copy.deepcopy(thread)), run["run_id"]
    )
    archived = (
        set(render_recipe(request, spec, recipe).archived_ids) if recipe else set()
    )
    shown = [
        (p, placeholder(request.thread_id, p) if p.part_id in archived else p.text)
        for p in request.parts
    ]
    prompt = transcript(run, shown)
    reply, sent, calls, found = models.ask(model, system, prompt), len(prompt), 0, 0
    while reply.strip().upper().startswith("REOPEN") and calls < REOPEN_ROUNDS:
        reference = (
            re.sub(r"^REOPEN[:\s]*", "", reply.strip(), flags=re.I)
            .split("\n")[0]
            .strip(" \"'`")[:256]
        )
        try:
            result = (
                reopen(request, {"reference": reference}, archived_ids=archived)
                if reference
                else {"status": "not_found"}
            )
        except ContractError as problem:
            result = {"status": "error", "message": str(problem)}
        calls, found = calls + 1, found + (result["status"] == "ok")
        prompt += f"\n\nYOU CALLED: REOPEN {reference}\nRESULT:\n{json.dumps(result, ensure_ascii=False)}"
        if calls == REOPEN_ROUNDS:
            prompt += "\n\nAnswer now. Do not reopen again."
        reply, sent = models.ask(model, system, prompt), sent + len(prompt)
    return {
        "answer": reply,
        "reopen_calls": calls,
        "reopened": found,
        "archived": len(archived),
        "prompt_tokens": len(prompt) // 4,
        "sent_tokens": sent // 4,
    }


def answers(args):
    root = Path(args.data)
    spec = ModelSpec.from_dict(docchat.read_json(args.model_spec))
    recipe = RetentionRecipe(**json.loads(args.recipe))
    domains = docchat.read_json(HERE / "domains.json")["domains"]
    _, rows, threads, _, grading = load(root, "test")
    # Graded turns: those that need history first, then as many that do not, in a fixed order.
    pool = [
        (entry, thread, run, note)
        for entry, thread, notes in zip(rows, threads, grading)
        for run, note in zip(thread["runs"], notes)
        if note["reference_supported"] and note["key_facts"] and run["run_index"] > 1
    ]
    pool.sort(key=lambda item: docchat.unit(args.seed, item[2]["run_id"]))
    need = [item for item in pool if item[3]["history_needed"] == "yes"][
        : args.turns * 2 // 3
    ]
    rest = [item for item in pool if item[3]["history_needed"] == "no"][
        : args.turns - len(need)
    ]
    models = docchat.Models(Path(args.work))

    def one(item):
        entry, thread, run, note = item
        system = domains[entry["domain"]]["assistant"] + " " + LIVE
        row = {
            "run_id": run["run_id"],
            "domain": entry["domain"],
            "tier": entry["tier"],
            "history_needed": note["history_needed"],
            "intent": note["intent"],
        }
        for arm, use in (("full", None), ("policy", recipe)):
            got = answer(models, args.model, system, thread, run, use, spec)
            grade = models.ask(
                args.judge,
                GRADER,
                f"USER MESSAGE:\n{run['input']['text']}\n\nKEY FACTS:\n"
                + "\n".join(f"- {fact}" for fact in note["key_facts"])
                + f"\n\nREFERENCE ANSWER:\n{run['output']['text']}\n\nANSWER TO GRADE:\n{got['answer']}",
                GRADE_SCHEMA,
            )
            row[arm] = got | {"correct": grade["correct"], "missing": grade["missing"]}
        return row

    with ThreadPoolExecutor(args.workers) as pool_:
        graded = list(pool_.map(one, need + rest))
    summary = {}
    for key in ("all", "yes", "no"):
        part = [r for r in graded if key == "all" or r["history_needed"] == key]
        summary[key] = {"turns": len(part)} | {
            arm: {
                "correct": sum(r[arm]["correct"] for r in part) / len(part),
                "prompt_tokens": sum(r[arm]["prompt_tokens"] for r in part) / len(part),
                "sent_tokens": sum(r[arm]["sent_tokens"] for r in part) / len(part),
                "reopened_turns": sum(r[arm]["reopen_calls"] > 0 for r in part)
                / len(part),
            }
            for arm in ("full", "policy")
            if part
        }
    docchat.write_json(
        args.output,
        {
            "model": args.model,
            "judge": args.judge,
            "recipe": json.loads(args.recipe),
            "summary": summary,
            "new_calls": dict(models.new),
            "turns": graded,
        },
    )
    print(
        docchat.compact(
            {
                "summary": summary,
                "new_calls": dict(models.new),
                "by_domain": dict(Counter(r["domain"] for r in graded)),
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pricing = commands.add_parser("cost")
    pricing.add_argument("--data", nargs="+", required=True, help="One build per clock")
    live = commands.add_parser("answers")
    live.add_argument("--data", required=True)
    live.add_argument(
        "--recipe",
        required=True,
        help="The policy's recipe as JSON, from the cost command",
    )
    live.add_argument("--work", required=True, help="Folder with stored model answers")
    live.add_argument("--model", default="claude-sonnet-5-5")
    live.add_argument("--judge", default=docchat.JUDGE)
    live.add_argument("--turns", type=int, default=60)
    live.add_argument("--seed", type=int, default=20261120)
    live.add_argument("--workers", type=int, default=6)
    for command in (pricing, live):
        command.add_argument("--model-spec", default=str(HERE / "sonnet-5-5.json"))
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    (cost if args.command == "cost" else answers)(args)


if __name__ == "__main__":
    main()
