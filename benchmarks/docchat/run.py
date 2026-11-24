#!/usr/bin/env python3
"""Run DocChat: what a fitted policy costs on held-out threads, and whether answers hold up.

  cost     fit a policy on each build's training threads and price it on its test threads
  answers  a model answers test turns from the full history and from the policy's view; a judge grades both
  grid     price every candidate the fit considered on the test threads (the policy map figure)
  thread   one thread's turns, what each needed and what the policy kept (the thread figure)

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

import compache
import docchat
from compache.errors import ContractError
from compache.models import cache_lifetimes, model_spec
from compache.optimizer.parts import Request, placeholder
from compache.optimizer.recovery import REOPEN_INSTRUCTIONS, reopen
from compache.optimizer.retention import RetentionRecipe, render_recipe
from compache.threads import Thread

HERE = Path(__file__).resolve().parent
END = "2026-06-01T00:00:00Z"
HEADER = {
    "system_prompt": "Answer from the conversation and the retrieved excerpts.",
    "tools": [],
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

    def lines(key):
        return [
            [json.loads(line) for line in gzip.open(root / e[key], "rt")] for e in rows
        ]

    return (
        manifest,
        rows,
        [t[0] for t in lines("trace_path")],
        lines("labels_path"),
        lines("grading_path"),
    )


def cost(args):
    results = []
    for root in map(Path, args.data):
        manifest, _, train, train_labels, _ = load(root, "train")
        _, _, test, test_labels, _ = load(root, "test")
        policy = compache.fit(
            train,
            sum(train_labels, []),
            model=args.prices,
            observed_until=END,
            **HEADER,
        )
        fitted = policy.summary
        held = compache.replay(
            policy,
            test,
            sum(test_labels, []),
            observed_until=END,
            share_headers=HEADER["share_headers"],
        )
        turns = held["turns"]
        # The whole history on the provider's long cache: the same replay under that cache's rules.
        other = copy.deepcopy(policy.artifact)
        other.pop("standard_spec", None)
        other["model_spec"].update(
            cache_ttl_seconds=args.long_cache_seconds,
            write_multiplier=args.long_cache_write,
        )
        long_cache = compache.replay(
            compache.Policy(other),
            test,
            sum(test_labels, []),
            observed_until=END,
            share_headers=HEADER["share_headers"],
        )["keep_everything_cost"]
        results.append(
            {
                "data": str(root),
                "clock": manifest["clock"],
                "model": fitted["model"],
                "train": {
                    "threads": fitted["threads"],
                    "turns": fitted["turns"],
                    "saving": fitted["saving"],
                    "coverage": fitted["coverage"],
                },
                "policy": {
                    "recipe": policy.artifact["solve"]["chosen"]["recipe"],
                    "max_pings": policy.max_pings,
                    "cache_ttl_seconds": policy.spec.cache_ttl_seconds,
                },
                "test": {
                    "threads": held["threads"],
                    "turns": turns,
                    "keep_everything_per_turn": held["keep_everything_cost"] / turns,
                    "keep_everything_long_cache_per_turn": long_cache / turns,
                    "saving_against_long_cache": 1 - held["policy_cost"] / long_cache,
                    "policy_per_turn": held["replayed_cost"] / turns,
                    "recovery_per_turn": held["recovery_cost"] / turns,
                    "saving": held["saving"],
                    "pings": held["pings"],
                    "needs_history": held["needs_history"],
                    "kept_in_view": held["coverage"],
                    "tokens_kept": held["tokens_kept"],
                    "cost_per_1000_turns": held["cost_per_1000_turns"],
                },
            }
        )
        print(docchat.compact(results[-1]), flush=True)
    docchat.write_json(args.output, results)


def grid(args):
    """Every candidate cut and ping limit of the fit, priced on the threads it did not see."""
    clocks = {}
    for root in map(Path, args.data):
        manifest, _, train, train_labels, _ = load(root, "train")
        _, _, test, test_labels, _ = load(root, "test")
        policy = compache.fit(
            train,
            sum(train_labels, []),
            model=args.prices,
            observed_until=END,
            **HEADER,
        )
        chosen, rows = policy.artifact["solve"]["chosen"], []
        standard = model_spec(args.prices)
        specs = {
            s.cache_ttl_seconds: s.to_dict()
            for s in (standard, *cache_lifetimes(args.prices))
        }
        for candidate in policy.artifact["solve"]["candidates"]:
            recipe, pings = candidate["recipe"], candidate["max_pings"]
            lifetime = candidate["cache_ttl_seconds"]
            held = compache.replay(
                compache.Policy(
                    policy.artifact
                    | {
                        "model_spec": specs[lifetime],
                        "standard_spec": standard.to_dict(),
                        "recipes": [{"recipe": recipe, "eligible": True}]
                        if recipe
                        else [],
                        "planner_options": {"max_pings": pings},
                    }
                ),
                test,
                sum(test_labels, []),
                observed_until=END,
                share_headers=HEADER["share_headers"],
            )
            rows.append(
                {
                    "recipe": recipe,
                    "max_pings": pings,
                    "cache_ttl_seconds": lifetime,
                    "fitted": recipe == chosen["recipe"]
                    and pings == chosen["max_pings"]
                    and lifetime == chosen["cache_ttl_seconds"],
                    "tokens_kept": held["tokens_kept"],
                    "in_view": held["coverage"] if recipe else 1.0,
                    "dollars_per_1000_turns": held["cost_per_1000_turns"]["policy"],
                }
            )
            print(docchat.compact(rows[-1]), flush=True)
        clocks[manifest["clock"]["scenario"]] = {
            "median_idle_seconds": manifest["clock"]["median_idle_seconds"],
            "test_threads": len(test),
            "test_turns": sum(len(t["runs"]) for t in test),
            "candidates": rows,
        }
        priced = policy.artifact["pricing"] or {}
        docchat.write_json(
            args.output,
            {
                "model": policy.spec.model_id,
                "model_name": priced.get("name", policy.spec.model_id),
                "clocks": clocks,
            },
        )


def thread_figure(args):
    """One thread without its text: sizes per turn, what each turn needed, and what stayed in view."""
    from compache.labels import normalize_label
    from compache.optimizer.retention import _key, _retrieval_key, _thread_windows

    root, spec = Path(args.data), model_spec(args.prices)
    recipe = RetentionRecipe(**json.loads(args.recipe))
    manifest = docchat.read_json(root / "manifest.json")
    entry = next(e for e in manifest["threads"] if e["thread_id"] == args.thread)
    data = json.loads(gzip.open(root / entry["trace_path"], "rt").readline())
    thread = compache.load_threads([data], model=spec.model_id)[0]
    labels = [json.loads(line) for line in gzip.open(root / entry["labels_path"], "rt")]
    table, _, _, _ = _thread_windows(thread, spec, [recipe])
    archived_at = {
        "input": table["input"][_key(recipe.input_runs, recipe)].turn,
        "output": table["output"][_key(recipe.output_runs, recipe)].turn,
        "retrieval": table["retrieval"][_retrieval_key(recipe)].turn,
    }
    origin = thread.part_origin
    index = {run.run_id: run.run_index for run in thread.runs}
    turns = [
        {
            "turn": run.run_index,
            "tokens": {
                "input": spec.part_tokens(run.input.text),
                "retrieval": sum(spec.part_tokens(p.text) for p in run.retrievals),
                "output": spec.part_tokens(run.output.text),
            },
            "needs": "nothing earlier",
        }
        for run in thread.runs
    ]
    for row in labels:
        label = normalize_label(row, thread)
        turn = index[label["target_run_id"]]
        requirement = label["task_requirements"][0]
        if (
            label["status"] != "complete"
            or requirement["historical_context_needed"] == "unknown"
        ):
            turns[turn - 1]["needs"] = "unknown"
            continue
        if requirement["historical_context_needed"] != "yes":
            continue
        # The alternative that is still in view, else the one whose evidence is nearest.
        best = None
        for alternative in requirement["alternatives"]:
            parts = [
                (origin[s["part_id"]], s["part_id"])
                for s in alternative["sources"]
                if origin[s["part_id"]][1] < turn
            ]
            kept = all(
                archived_at[kind].get(pid, turn + 1) > turn for (kind, _), pid in parts
            )
            key = (not kept, max(turn - at for (_, at), _ in parts), len(parts))
            if best is None or key < best[0]:
                best = (key, parts, kept)
        turns[turn - 1]["needs"] = {
            "in_view": best[2],
            "parts": sorted(
                {(kind, at) for (kind, at), _ in best[1]}, key=lambda x: (x[1], x[0])
            ),
        }
    docchat.write_json(
        args.output,
        {
            "thread_id": thread.thread_id,
            "domain": entry["domain"],
            "document": entry["document"],
            "recipe": json.loads(args.recipe),
            "turns": turns,
        },
    )
    print(docchat.compact({"thread": thread.thread_id, "turns": len(turns)}))


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
    spec = model_spec(args.prices)
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
    pricing.add_argument(
        "--long-cache-seconds",
        type=int,
        default=3600,
        help="Lifetime of the provider's long cache. Default: Anthropic's",
    )
    pricing.add_argument(
        "--long-cache-write",
        type=float,
        default=2.0,
        help="Its write price, as a multiple of the input price",
    )
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
    priced = commands.add_parser("grid")
    priced.add_argument("--data", nargs="+", required=True, help="One build per clock")
    one = commands.add_parser("thread")
    one.add_argument("--data", required=True)
    one.add_argument("--thread", required=True, help="A thread ID of the build")
    one.add_argument("--recipe", required=True, help="The policy's recipe as JSON")
    for command in (pricing, live, priced, one):
        command.add_argument(
            "--prices",
            default="claude-sonnet-5-5",
            help="The model whose prices and cache rules apply: a known name or a JSON spec",
        )
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"cost": cost, "answers": answers, "grid": grid, "thread": thread_figure}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
