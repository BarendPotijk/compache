#!/usr/bin/env python3
"""compache on live model calls: every way in, and what it saves.

  check    each way of putting compache in, on a few turns: the provider accepts what is sent,
           the cache is read, a note comes back and never reaches the application, an
           archived part is fetched back
  learn    play the training conversations with no policy: a record without text, with the
           model's own notes of what each reply needed, and beside it the chat log
  fit      three policies: for the default workload, fitted on the chat log with the
           benchmark's labels (looking back), and fitted on the record (looking forward)
  apply    play held-out conversations at their own pace under each policy and under none,
           through each way in, and grade the answers: what the provider billed, and
           whether the answers hold
  grade    grade the answers the judge gave no verdict on
  report   the tables of all of the above

Conversations come from a DocChat build (`benchmarks/docchat`). An application meets them
in one of two shapes: it lets the model call a search tool, or it pastes what it retrieved
into the question. Each shape is learned and fitted on its own. Model calls go to
OpenRouter, or to any base URL that speaks the same APIs. The key is read from
OPENROUTER_API_KEY or from a file of `NAME=value` lines given with --env.
Every run stops before it would spend more than --budget.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "docchat"))

import apps  # noqa: E402

import compache  # noqa: E402
from compache import notes  # noqa: E402
from compache.live import Live  # noqa: E402
from compache.proxy import serve  # noqa: E402
from compache.record import load, load_calls, thread_name  # noqa: E402

# The shapes a conversation has on the wire, and the application that produces each while learning.
SHAPES = {"tool": "tool-loop", "stuffed": "stuffed"}
# Every way compache is put in for `apply`: the application, the shape of its conversations,
# how compache goes in, and whose model answers.
WAYS = {
    "wrap-chat": ("tool-loop", "tool", "wrap", "claude"),
    "wrap-stuffed": ("stuffed", "stuffed", "wrap", "claude"),
    "proxy-messages": ("claude-loop", "tool", "proxy", "claude"),
    "langchain-agent": ("agent", "tool", "langchain", "claude"),
    "wrap-responses": ("responses", "tool", "wrap", "openai"),
}
ARMS = ("keep", "default", "backward", "forward")
# How the figure names a way in and a policy.
SHOWN = {
    "wrap-chat": (
        "OpenAI SDK, wrapped",
        "Chat Completions, a search tool",
        "Claude Haiku 5.5",
    ),
    "wrap-stuffed": (
        "OpenAI SDK, wrapped",
        "Chat Completions, passages pasted in",
        "Claude Haiku 5.5",
    ),
    "proxy-messages": (
        "Anthropic SDK, through the proxy",
        "Messages, a search tool",
        "Claude Haiku 5.5",
    ),
    "langchain-agent": (
        "LangChain agent, its clients wrapped",
        "Chat Completions, a search tool",
        "Claude Haiku 5.5",
    ),
    "wrap-responses": ("OpenAI SDK, wrapped", "Responses, a search tool", "GPT-6 Luna"),
}
NAMED = {
    "keep": "No policy",
    "default": "Default policy",
    "backward": "Fitted on past traces",
    "forward": "Fitted on the model's notes",
}
GRADER = (
    "You grade one answer for a benchmark of chats over documents. You receive the user's message, the key "
    "facts a correct answer must contain, a reference answer and the answer to grade. correct is true only if "
    "the answer conveys every key fact and contradicts none. An answer that says it lacks the information, or "
    "asks what the user means, is not correct. Be strict and literal. Reply with JSON only: "
    '{"correct": true or false, "missing": "what is missing, or an empty string"}'
)


# -- setting up --------------------------------------------------------------


def secret(args) -> str:
    """The API key: from the environment, or from a file of NAME=value lines."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and args.env and Path(args.env).is_file():
        for line in Path(args.env).read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise SystemExit(
            "no OPENROUTER_API_KEY in the environment or in the file given with --env"
        )
    return key


def conversations(root: Path, split: str, turns, tiers=("s", "m", "l")) -> list[dict]:
    """Conversations of a DocChat build, cut to their first turns.

    `turns` is one length for all, or several that the conversations take in turn, so
    that a fit sees conversations end at different points.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    domains = json.loads((HERE.parent / "docchat" / "domains.json").read_text())[
        "domains"
    ]
    lengths = [turns] if isinstance(turns, int) else list(turns)
    found = []
    for entry in manifest["threads"]:
        if entry["split"] != split or entry["tier"][0] not in tiers:
            continue
        cut = lengths[len(found) % len(lengths)]

        def rows(key):
            with gzip.open(root / entry[key], "rt") as stream:
                return [json.loads(line) for line in stream if line.strip()]

        thread = rows("trace_path")[0]
        found.append(
            {
                "id": thread["thread_id"],
                "system": domains[entry["domain"]]["assistant"],
                "runs": thread["runs"][:cut],
                "labels": {row["target_run_id"]: row for row in rows("labels_path")},
                "grading": {row["run_id"]: row for row in rows("grading_path")},
            }
        )
    return found


def gap(earlier: dict, later: dict, cap: float | None) -> float:
    """Seconds a user was away between two turns, as recorded, at most `cap`."""
    away = (
        datetime.fromisoformat(later["started_at"].replace("Z", "+00:00"))
        - datetime.fromisoformat(earlier["ended_at"].replace("Z", "+00:00"))
    ).total_seconds()
    return max(0.0, away if cap is None else min(away, cap))


class Clock:
    """Time that is moved by hand, for playing hours of traffic in minutes."""

    at = 1_795_000_000.0

    def __call__(self) -> float:
        return self.at


class OwnClock(Clock, threading.local):
    """The same, kept apart for each conversation that is played side by side."""


class Spend:
    """What the provider has billed so far, read from the records, and the limit it may not pass."""

    def __init__(self, work: Path, budget: float):
        self.work, self.budget = work, budget
        self.extra = 0.0
        self._lock = threading.Lock()

    def add(self, cost) -> None:
        with self._lock:
            self.extra += cost or 0.0

    def keep(self) -> None:
        """Write what was added by hand, such as the judge's calls, where `total` finds it again."""
        with self._lock:
            if self.extra:
                folder = self.work / "judge"
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / "calls.jsonl", "a") as ledger:
                    ledger.write(
                        json.dumps({"kind": "judge", "cost": self.extra}) + "\n"
                    )
                self.extra = 0.0

    def total(self) -> float:
        billed = self.extra
        for calls in self.work.glob("**/calls.jsonl"):
            for line in calls.read_text().splitlines():
                billed += json.loads(line).get("cost") or 0.0
        return billed

    def check(self) -> None:
        billed = self.total()
        if billed > self.budget:
            raise SystemExit(
                f"stopped: ${billed:.2f} billed, over the budget of ${self.budget:.2f}"
            )


def clients(
    args, key: str, base: str | None = None, headers: dict | None = None
) -> dict:
    """SDK clients pointed at the provider, or at `base` where a proxy stands in for it."""
    import anthropic
    import openai

    root = (base or args.base).rstrip("/")
    return {
        "openai": openai.OpenAI(
            api_key=key, base_url=root + "/v1", max_retries=4, default_headers=headers
        ),
        "anthropic": anthropic.Anthropic(
            api_key=key, base_url=root, max_retries=4, default_headers=headers
        ),
    }


def extra(args, model: str, cache: bool = True) -> dict:
    """What every request carries besides the conversation: the provider to stay with, and its own caching."""
    body: dict = {}
    if args.pin and "/" in model:
        vendor = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google-ai-studio",
        }.get(model.split("/")[0])
        if vendor:
            # Cache entries live at one provider: stay with it.
            body["provider"] = {"order": [vendor], "allow_fallbacks": False}
    if cache and "claude" in model:
        # The application's own caching, as Anthropic's guide sets it up: one mark for the whole request.
        body["cache_control"] = {"type": "ephemeral"}
    return body


def play(app, conversation: dict, name: str, pace, on_turn=None) -> list[dict]:
    """One conversation through an application: what was asked, what was answered, and when."""
    turns, runs = [], conversation["runs"]
    for n, run in enumerate(runs, 1):
        if n > 1:
            pace(runs[n - 2], run)
        if on_turn is not None:
            on_turn()
        started = time.time()
        with compache.thread(name):
            try:
                answer = app.turn(
                    run["input"]["text"], [p["text"] for p in run["retrievals"]]
                )
            except (
                Exception
            ) as error:  # A refused or failed call ends this conversation, not the run.
                turns.append({"run_id": run["run_id"], "error": repr(error)[:300]})
                break
        turns.append(
            {
                "run_id": run["run_id"],
                "answer": answer,
                "seconds": round(time.time() - started, 2),
            }
        )
    return turns


# -- check -------------------------------------------------------------------

PASSAGES = [
    "Maintenance schedule for the Orrin K-40 pump. The impeller is inspected every 600 running hours. "
    "The mechanical seal is replaced every 2,400 running hours or after any dry run. "
    + "The service log must record the date, the running hours and the technician's initials. "
    * 34,
    "Warranty terms for the Orrin K-40 pump. Parts and labour are covered for 26 months from delivery. "
    "A claim must reach the service desk within 21 days of the defect being found. "
    + "Wear parts such as seals and bearings are covered for the first 6 months only. "
    * 34,
    "Storage of the Orrin K-40 pump. A pump kept in store for more than 9 months must be turned by hand "
    "every 4 weeks, and its bearings regreased before first use. "
    + "The storage room must stay between 5 and 35 degrees Celsius and below 70 percent humidity. "
    * 34,
]
# A question, the passage its search finds, and the ways a right answer may write its fact.
QUESTIONS = [
    ("How often is the impeller of the K-40 inspected?", 0, ("600",)),
    ("And for how long are parts and labour under warranty?", 1, ("26",)),
    (
        "How often must a stored pump be turned by hand?",
        2,
        ("4 weeks", "four weeks", "4-week"),
    ),
    (
        "Going back to my first question: after how many running hours is the mechanical seal replaced?",
        None,
        ("2,400", "2400"),
    ),
]
SYSTEM = "You are an assistant that answers questions about the service manual of an industrial pump."


def tiny_policy(prices: str, shape: str = "tool"):
    """A policy that keeps what was retrieved for the current turn only, fitted on a small record.

    In the `stuffed` shape what was retrieved sits in the question, and the record says so.
    """
    threads, labels = [], []
    for i in range(4):
        runs = []
        for n in range(1, 9):
            at = 1_780_000_000 + n * 3600
            pasted = shape == "stuffed"
            runs.append(
                {
                    "started_at": at,
                    "ended_at": at + 5,
                    "input": {"bytes": 3860 if pasted else 60},
                    "retrievals": [] if pasted else [{"bytes": 3800}],
                    "output": {"bytes": 200},
                }
            )
            labels.append({"thread_id": f"tiny-{i}", "run": n, "needs": []})
        threads.append(
            {"thread_id": f"tiny-{i}", "provenance": "sizes only", "runs": runs}
        )
    # The default lifetime: the provider's cache sees seconds where the policy sees an hour.
    return compache.fit(
        threads, labels, model=prices, share_headers=False, lifetimes=()
    )


def check(args):
    """Every way in, on four turns each, with a policy that archives as soon as it may."""
    key, work = secret(args), Path(args.work) / "check"
    work.mkdir(parents=True, exist_ok=True)
    spend = Spend(Path(args.work), args.budget)
    policies = {shape: tiny_policy(args.prices, shape) for shape in SHAPES}
    wanted = set(args.only or [])
    file = work / "results.json"
    # A way that is not tried again keeps the result it had.
    results = (
        {row["way"]: row for row in json.loads(file.read_text())}
        if file.is_file()
        else {}
    )

    def one(way: str, kind: str, model: str, *, proxy: bool = False):
        if wanted and way not in wanted:
            return
        # A model does not answer the same way twice. A way that fails is tried once more,
        # and the result says how many tries it took.
        failed = []
        for tries in range(1, args.tries + 1):
            # Each try is a conversation of its own.
            row = attempt(way, f"{way}-{tries}", kind, model, proxy)
            row["tries"] = tries
            if row["ok"]:
                break
            failed.append(
                {k: v for k, v in row.items() if k not in ("way", "model", "tries")}
            )
        if failed and row["ok"]:
            row["failed_before"] = failed
        results[way] = row
        print(json.dumps(row), flush=True)

    def attempt(way: str, name: str, kind: str, model: str, proxy: bool) -> dict:
        spend.check()
        folder = work / way
        shutil.rmtree(folder, ignore_errors=True)
        clock = Clock()
        policy = policies["stuffed" if kind == "stuffed" else "tool"]
        live = Live(policy, record=folder, sample=1.0, notes=args.notes, clock=clock)
        server = None
        if proxy:
            server = serve(live, port=0, upstream=args.base)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            made = clients(
                args,
                key,
                f"http://127.0.0.1:{server.server_address[1]}",
                {"x-compache-thread": name},
            )
        else:
            made = clients(args, key)
        row = {"way": way, "model": model, "answers": [], "ok": False}
        try:
            if kind == "agent":
                from langchain_openai import ChatOpenAI

                chat = ChatOpenAI(
                    model=model,
                    api_key=key,
                    base_url=args.base.rstrip("/") + "/v1",
                    max_tokens=1200,
                    extra_body=extra(args, model, cache=False),
                    **compache.langchain.clients(live, pings=False),
                )
                app = apps.Agent(
                    chat,
                    model,
                    SYSTEM,
                    thread_id=name,
                    middleware=[compache.langchain.middleware()],
                )
            else:
                client = made["anthropic" if kind == "claude-loop" else "openai"]
                if not proxy:
                    client = compache.wrap(client, live=live, pings=False)
                app = apps.APPS[kind](
                    client,
                    model,
                    SYSTEM,
                    limit=1200,
                    extra=extra(args, model, cache=False),
                )
            for question, index, facts in QUESTIONS:
                with compache.thread(name):
                    answer = app.turn(
                        question, [PASSAGES[index]] if index is not None else []
                    )
                found = any(fact in answer for fact in facts)
                row["answers"].append(
                    {"fact": facts[0], "found": found, "answer": answer[:300]}
                )
                if proxy:
                    for _ in range(300):
                        if live.stats["turns"] >= len(row["answers"]):
                            break
                        time.sleep(0.02)
                # A note asked after the reply belongs to this turn: wait for it.
                live.drain()
                # An hour passes for the policy between turns, so that it archives. The provider's
                # cache sees the seconds that really passed.
                clock.at += 3600
            calls = load_calls(folder)
            threads, labels = load(folder)
            sent = [c for c in calls if c["kind"] in ("answer", "reopen")]
            row.update(
                turns=len(threads[0]["runs"]) if threads else 0,
                calls=len(calls),
                cached_tokens=sum(
                    (c.get("usage") or {}).get("cache_read", 0) for c in sent
                ),
                archived=max((len(c.get("archived") or ()) for c in calls), default=0),
                reopened=live.stats["reopened"],
                asked=live.stats["asked"],
                noted=live.stats["noted"],
                leaked=live.stats["leaked"]
                + sum("refs:" in a["answer"] for a in row["answers"]),
                retried=live.stats["retried"],
                refused=live.stats["policy_refused"],
                stood_down=live.stats["stood_down"],
                cost=round(sum(c.get("cost") or 0 for c in calls), 6),
                labels=[
                    (r["run"], r["needs"] is not None and len(r["needs"]))
                    for r in labels
                ],
            )
            row["ok"] = (
                row["turns"] == len(QUESTIONS)
                and all(a["found"] for a in row["answers"])
                and row["archived"] > 0
                and row["reopened"] > 0
                and row["noted"] > 0
                and row["leaked"] == 0
                and row["refused"] == 0
                and row["stood_down"] == 0
                # Where compache itself marks the conversation cacheable, the cache has to be read.
                and ("claude" not in model or row["cached_tokens"] > 0)
            )
        except Exception as error:
            row["error"] = repr(error)[:400]
        finally:
            if server is not None:
                server.shutdown()
        return row

    claude, gpt, gemini = args.model, args.openai_model, args.gemini_model
    one("wrap-openai-sdk-chat-claude", "tool-loop", claude)
    one("wrap-openai-sdk-chat-claude-streamed", "tool-loop-streamed", claude)
    one("wrap-openai-sdk-chat-gpt", "tool-loop", gpt)
    one("wrap-openai-sdk-chat-gemini", "tool-loop", gemini)
    one("wrap-openai-sdk-stuffed-claude", "stuffed", claude)
    one("wrap-anthropic-sdk-messages", "claude-loop", claude)
    one("wrap-openai-sdk-responses-gpt", "responses", gpt)
    one("proxy-chat-claude-streamed", "tool-loop-streamed", claude, proxy=True)
    one("proxy-anthropic-messages", "claude-loop", claude, proxy=True)
    one("langchain-agent-claude", "agent", claude)
    file.write_text(json.dumps(list(results.values()), indent=1))
    print(f"${spend.total():.4f} billed so far")


# -- learn -------------------------------------------------------------------


def chat_log(name: str, app, times: list[list]) -> dict:
    """The conversation as an application stores it: its messages, with the times of its turns."""
    messages, n = [], -1
    for message in app.messages:
        entry = {"role": message["role"], "content": message.get("content") or ""}
        if isinstance(entry["content"], list):
            entry["content"] = [
                {"type": "text", "text": p["text"]} for p in entry["content"]
            ]
        if message["role"] == "user":
            n += 1
            entry["at"] = times[n][0]
        messages.append(entry)
        if (
            message["role"] == "assistant"
            and not message.get("tool_calls")
            and n < len(times)
        ):
            entry["at"] = times[n][1]
    return {"thread_id": name, "messages": messages}


def learn(args):
    """The training conversations with no policy, once in each shape: a record, and the chat log beside it."""
    key, work = secret(args), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    spend = Spend(work, args.budget)
    made = clients(args, key)
    train = conversations(Path(args.data), "train", args.train_turns)
    for shape in args.shapes:
        record = work / f"record-{shape}"
        shutil.rmtree(record, ignore_errors=True)
        clock = OwnClock()
        live = Live(record=record, sample=args.sample, notes=args.notes, clock=clock)
        client = compache.wrap(made["openai"], live=live, pings=False)
        logs, lock = [], threading.Lock()

        def one(conversation):
            name = f"learn-{shape}-{conversation['id']}"
            # The clock is the conversation's own: the waits of its users pass in no time.
            clock.at = 1_795_000_000.0
            app = apps.APPS[SHAPES[shape]](
                client,
                args.model,
                conversation["system"],
                extra=extra(args, args.model),
            )
            runs, times, turns = conversation["runs"], [], []
            for n, run in enumerate(runs, 1):
                if n > 1:
                    clock.at += gap(runs[n - 2], run, args.cap)
                spend.check()
                times.append([clock.at, None])
                with compache.thread(name):
                    try:
                        answer = app.turn(
                            run["input"]["text"], [p["text"] for p in run["retrievals"]]
                        )
                    except Exception as error:
                        turns.append(
                            {"run_id": run["run_id"], "error": repr(error)[:300]}
                        )
                        times.pop()
                        break
                clock.at += 6
                times[-1][1] = clock.at
                turns.append({"run_id": run["run_id"], "answer": answer})
            with lock:
                logs.append(
                    {
                        "conversation": conversation["id"],
                        "log": chat_log(name, app, times),
                        "turns": turns,
                    }
                )
            print(shape, conversation["id"], len(turns), "turns", flush=True)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, train))
        # Notes asked after their replies may still be on their way.
        live.drain()
        (work / f"learn-{shape}.json").write_text(json.dumps(logs, indent=1))
        print(
            shape,
            json.dumps(live.stats),
            f"${spend.total():.4f} billed so far",
            flush=True,
        )


# -- fit ---------------------------------------------------------------------


def holders(conversation: dict, name: str, searched: dict, shape: str) -> dict:
    """For every part of the benchmark: the turns of a played conversation that hold its text, and as which part.

    A question and an answer sit in their own turn. A retrieved chunk sits in every turn
    whose search returned it, and a search returns a turn's chunks as one result, which is
    one part. Where the application pastes them into the question, that question holds them.
    """
    where, chunks = {}, {}
    for n, run in enumerate(conversation["runs"], 1):
        where[run["input"]["part_id"]] = [(n, f"{name}:{n}:input")]
        where[run["output"]["part_id"]] = [(n, f"{name}:{n}:output")]
        if shape != "stuffed" and not searched.get(n):
            continue
        held = f"{name}:{n}:input" if shape == "stuffed" else f"{name}:{n}:retrieval:1"
        for part in run["retrievals"]:
            chunks.setdefault((part["document_id"], part["chunk_id"]), []).append(
                (n, held)
            )
    for run in conversation["runs"]:
        for part in run["retrievals"]:
            where[part["part_id"]] = chunks.get(
                (part["document_id"], part["chunk_id"]), []
            )
    return where


def needs_of(label: dict, where: dict, n: int, limit: int = 24):
    """A benchmark label as the alternatives of earlier parts turn `n` needs.

    [] for none, None where it cannot be said. What the turn itself holds is in view
    anyway, and a chunk that was retrieved more than once is needed from any one of those turns.
    """
    needed = [[]]
    for requirement in label["task_requirements"]:
        if requirement["historical_context_needed"] == "unknown":
            return None
        if requirement["historical_context_needed"] != "yes":
            continue
        ways = []
        for alternative in requirement["alternatives"]:
            options = []
            for source in alternative["sources"]:
                held = where.get(source["part_id"], [])
                if any(turn == n for turn, _ in held):
                    continue
                earlier = sorted({part for turn, part in held if turn < n})
                if not earlier:
                    # Never retrieved in this play of the conversation.
                    options = None
                    break
                options.append(earlier)
            if options is None:
                continue
            for combination in itertools.product(*options):
                if sorted(set(combination)) not in ways:
                    ways.append(sorted(set(combination)))
        if not ways:
            return None
        if [] in ways:
            # This turn's own text is enough for it after all.
            continue
        needed = [
            sorted(set(a) | set(b)) for a, b in itertools.product(needed, ways[:limit])
        ][:limit]
    return [alternative for alternative in needed if alternative]


def reference_labels(
    conversation: dict, name: str, thread: dict, shape: str
) -> list[dict]:
    """The benchmark's labels for a played conversation, by the part IDs it has there."""
    searched = {n: bool(run["retrievals"]) for n, run in enumerate(thread["runs"], 1)}
    where = holders(conversation, name, searched, shape)
    labels = []
    for n, run in enumerate(conversation["runs"][: len(thread["runs"])], 1):
        label = conversation["labels"].get(run["run_id"])
        if label is not None:
            labels.append(
                {"thread_id": name, "run": n, "needs": needs_of(label, where, n)}
            )
    return labels


def agreement(reference: list[dict], noted: list[dict]) -> dict:
    """How the replies' own notes compare with the benchmark's labels, turn by turn."""
    said = {(row["thread_id"], row["run"]): row["needs"] for row in noted}
    both = [
        (row["needs"], set(sum(said[row["thread_id"], row["run"]], [])))
        for row in reference
        if row["run"] > 1
        and row["needs"] is not None
        and said.get((row["thread_id"], row["run"])) is not None
    ]
    need = [(want, got) for want, got in both if want]
    free = [(want, got) for want, got in both if not want]
    return {
        "turns_compared": len(both),
        "need_history": len(need),
        # The note names every part of one sufficient set, some part of one, or anything at all.
        "named_enough": sum(any(set(alt) <= got for alt in want) for want, got in need),
        "named_part_of_it": sum(
            any(set(alt) & got for alt in want) for want, got in need
        ),
        "named_something": sum(bool(got) for _, got in need),
        "need_nothing": len(free),
        "named_nothing": sum(not got for _, got in free),
    }


def brief(policy) -> dict:
    got = policy.summary
    return {
        "describe": policy.describe(),
        "keep": got["keep"],
        "max_pings": got["max_pings"],
        "labelled_turns": got["labelled_turns"],
        "coverage": got["coverage"],
        "saving": got["saving"],
    }


def served(args, family: str) -> tuple[str, str]:
    """The model a way calls and the price sheet its policies are fitted for."""
    return (
        (args.openai_model, args.openai_prices)
        if family == "openai"
        else (args.model, args.prices)
    )


def fit(args):
    """For each price sheet: the default policy, and per shape one fitted looking back and one looking forward."""
    work = Path(args.work)
    train = {
        c["id"]: c for c in conversations(Path(args.data), "train", args.train_turns)
    }
    summary = {"labels": {}, "policies": {}}
    learned = {}
    for shape in args.shapes:
        if not (work / f"learn-{shape}.json").is_file():
            continue
        logs = json.loads((work / f"learn-{shape}.json").read_text())
        threads = [entry["log"] for entry in logs]
        loaded = {
            t.thread_id: t for t in compache.load_threads(threads, model=args.prices)
        }
        back = []
        for entry in logs:
            name = entry["log"]["thread_id"]
            back += reference_labels(
                train[entry["conversation"]], name, loaded[name].to_dict(), shape
            )
        recorded, forward = load(work / f"record-{shape}")
        # A note also says which retrieved text its reply used, and a fit can keep such text
        # longer. `apply` asks no notes, so no answer there is seen using anything: the
        # policies are fitted on what the notes say was needed, and not on what they cite.
        for thread in recorded:
            for run in thread["runs"]:
                run.pop("cited_part_ids", None)
        hashed = {thread_name(name): name for name in loaded}
        named = [
            {
                **row,
                "thread_id": hashed[row["thread_id"]],
                "needs": row["needs"]
                and [
                    [
                        part.replace(row["thread_id"], hashed[row["thread_id"]], 1)
                        for part in alt
                    ]
                    for alt in row["needs"]
                ],
            }
            for row in forward
            if row["thread_id"] in hashed
        ]
        learned[shape] = (threads, back, recorded, forward)
        summary["labels"][shape] = {
            "agreement": agreement(back, named),
            "backward": {
                "turns": len(back),
                "need_history": sum(bool(r["needs"]) for r in back),
            },
            "forward": {
                "turns": len(forward),
                "need_history": sum(bool(r["needs"]) for r in forward),
            },
        }
        print(shape, json.dumps(summary["labels"][shape]), flush=True)
    sheets = dict.fromkeys(served(args, family)[1] for *_, family in WAYS.values())
    for sheet in sheets:
        default = compache.default_policy(sheet)
        default.save(work / f"policy-{sheet}-default.json")
        entry = summary["policies"][sheet] = {"default": brief(default)}
        print(f"--- {sheet}, default\n{default.describe()}")
        for shape, (threads, back, recorded, forward) in learned.items():
            if not any(
                w[1] == shape and served(args, w[3])[1] == sheet for w in WAYS.values()
            ):
                continue
            policies = {
                "backward": compache.fit(
                    threads, back, model=sheet, share_headers=False
                ),
                "forward": compache.fit(
                    recorded, forward, model=sheet, share_headers=False
                ),
            }
            entry[shape] = {"sampled": {}}
            for name, policy in policies.items():
                policy.save(work / f"policy-{sheet}-{shape}-{name}.json")
                entry[shape][name] = brief(policy)
                print(f"--- {sheet}, {shape}, {name}\n{policy.describe()}", flush=True)
            # What a lower sampling rate would have fitted: the same record with a share of its notes.
            for rate in args.rates if sheet == args.prices else ():
                kept = [
                    row
                    for row in forward
                    if row["run"] == 1
                    or notes.sampled(row["thread_id"], row["run"], rate, "rate")
                ]
                try:
                    less = compache.fit(
                        recorded, kept, model=sheet, share_headers=False
                    )
                    entry[shape]["sampled"][str(rate)] = {
                        **brief(less),
                        "labels": len(kept),
                    }
                except Exception as error:
                    entry[shape]["sampled"][str(rate)] = {
                        "error": str(error)[:200],
                        "labels": len(kept),
                    }
    (work / "fit.json").write_text(json.dumps(summary, indent=1))


# -- apply -------------------------------------------------------------------


def grade(
    client, args, question: str, note: dict, reference: str, answer: str, spend: Spend
):
    """Whether an answer conveys the key facts, as the same small model judges it."""
    asked = (
        f"USER MESSAGE:\n{question}\n\nKEY FACTS:\n"
        + "\n".join(f"- {fact}" for fact in note["key_facts"])
        + f"\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}"
    )
    reply = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": GRADER},
            {"role": "user", "content": asked},
        ],
        # Room to think before the verdict: with less, one verdict in five never came.
        max_tokens=1500,
        extra_body=extra(args, args.model, cache=False),
    )
    spend.add(getattr(reply.usage, "cost", None) if reply.usage else None)
    text = reply.choices[0].message.content or ""
    try:
        return bool(json.loads(text[text.index("{") : text.rindex("}") + 1])["correct"])
    except (ValueError, KeyError):
        return None


def graded(played: list, by_id: dict, client, args, spend: Spend) -> int:
    """Grade every answer that has no verdict yet. Returns how many were tried."""

    def judge(item):
        entry, turn = item
        conversation = by_id[entry["conversation"]]
        run = next(r for r in conversation["runs"] if r["run_id"] == turn["run_id"])
        note = conversation["grading"].get(run["run_id"]) or {}
        turn["needs_history"] = note.get("history_needed")
        try:
            turn["correct"] = grade(
                client,
                args,
                run["input"]["text"],
                note,
                run["output"]["text"],
                turn["answer"],
                spend,
            )
        except Exception as error:
            turn["grade_error"] = repr(error)[:200]

    def gradable(entry, turn):
        run_id = turn["run_id"]
        note = by_id[entry["conversation"]]["grading"].get(run_id) or {}
        return (
            "answer" in turn
            and note.get("key_facts")
            and note.get("reference_supported")
        )

    todo = [
        (entry, turn)
        for entry in played
        for turn in entry["turns"]
        if turn.get("correct") is None and gradable(entry, turn)
    ]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(judge, todo))
    spend.keep()
    return len(todo)


def regrade(args):
    """Grade the answers of `apply` that the judge gave no verdict on."""
    key, work = secret(args), Path(args.work)
    spend = Spend(work, args.budget)
    test = conversations(Path(args.data), "test", args.turns, tiers=tuple(args.tiers))
    file = work / "apply.json"
    got = json.loads(file.read_text())
    tried = graded(
        got["played"],
        {c["id"]: c for c in test},
        clients(args, key)["openai"],
        args,
        spend,
    )
    file.write_text(json.dumps(got, indent=1))
    left = sum(
        t.get("correct") is None and "needs_history" in t
        for e in got["played"]
        for t in e["turns"]
    )
    print(
        f"{tried} answers graded, {left} still without a verdict. ${spend.total():.4f} billed so far"
    )


def application(
    args,
    key: str,
    made: dict,
    kind: str,
    how: str,
    model: str,
    live: Live,
    servers: list,
):
    """How one way in builds an application for a conversation: `make(name, system)`."""
    if how == "wrap":
        client = compache.wrap(made["openai"], live=live)
        return lambda name, system: apps.APPS[kind](
            client, model, system, extra=extra(args, model)
        )
    if how == "proxy":
        server = serve(live, port=0, upstream=args.base)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def through_proxy(name, system):
            # Nothing but the base URL is the application's own. The header names its conversation.
            mine = clients(args, key, base, {"x-compache-thread": name})
            client = mine["anthropic" if kind == "claude-loop" else "openai"]
            return apps.APPS[kind](client, model, system, extra=extra(args, model))

        return through_proxy
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=model,
        api_key=key,
        base_url=args.base.rstrip("/") + "/v1",
        max_tokens=1500,
        max_retries=4,
        extra_body=extra(args, model),
        **compache.langchain.clients(live),
    )
    return lambda name, system: apps.Agent(
        chat,
        model,
        system,
        thread_id=name,
        middleware=[compache.langchain.middleware()],
    )


def apply(args):
    """Held-out conversations at their own pace, under every policy and through every way in."""
    key, work = secret(args), Path(args.work)
    spend = Spend(work, args.budget)
    test = conversations(Path(args.data), "test", args.turns, tiers=tuple(args.tiers))
    made = clients(args, key)
    lives, makers, servers = {}, {}, []
    for way in args.ways or WAYS:
        kind, shape, how, family = WAYS[way]
        model, sheet = served(args, family)
        for arm in args.arms or ARMS:
            policy = None
            if arm != "keep":
                name = "default" if arm == "default" else f"{shape}-{arm}"
                file = work / f"policy-{sheet}-{name}.json"
                if not file.is_file():
                    continue
                policy = compache.Policy.load(file)
            folder = work / "apply" / way / arm
            shutil.rmtree(folder, ignore_errors=True)
            lives[way, arm] = Live(policy, record=folder)
            makers[way, arm] = application(
                args, key, made, kind, how, model, lives[way, arm], servers
            )
    played, lock = [], threading.Lock()

    def one(job):
        index, (way, arm), conversation = job
        # Conversations start a moment apart, so that the provider is not met by all of them at once.
        time.sleep(index * args.stagger)
        name = f"{way}-{arm}-{conversation['id']}"
        # Its own line in the system prompt keeps each conversation's cache entries its own:
        # no arm reads what another arm wrote.
        app = makers[way, arm](name, f"{conversation['system']} (Session {name}.)")
        turns = play(
            app,
            conversation,
            name,
            lambda earlier, later: time.sleep(
                gap(earlier, later, args.cap) * args.pace
            ),
            spend.check,
        )
        with lock:
            played.append(
                {
                    "way": way,
                    "arm": arm,
                    "conversation": conversation["id"],
                    "name": name,
                    "turns": turns,
                }
            )
        print(way, arm, conversation["id"], len(turns), "turns", flush=True)

    jobs = [(pair, conversation) for conversation in test for pair in lives]
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        list(pool.map(one, [(index, *job) for index, job in enumerate(jobs)]))
    # A policy cannot know that a conversation is over: the pings it sends after the last
    # answer are part of what it costs. Wait for them.
    waited = time.time()
    while time.time() - waited < args.tail * args.pace:
        if all(live.next_ping_at() is None for live in lives.values()):
            break
        time.sleep(2)
    for live in lives.values():
        if live.pinger is not None:
            live.pinger.stop()
    for server in servers:
        server.shutdown()
    graded(played, {c["id"]: c for c in test}, made["openai"], args, spend)
    # Ways and arms that were not played this time keep what they had.
    file = work / "apply.json"
    before = (
        json.loads(file.read_text()) if file.is_file() else {"played": [], "stats": {}}
    )
    kept = [e for e in before["played"] if (e["way"], e["arm"]) not in lives]
    stats = before["stats"]
    for (way, arm), live in lives.items():
        stats.setdefault(way, {})[arm] = live.stats
    file.write_text(json.dumps({"played": kept + played, "stats": stats}, indent=1))
    print(
        json.dumps({f"{way}/{arm}": live.stats for (way, arm), live in lives.items()})
    )
    print(f"${spend.total():.4f} billed so far")


# -- report ------------------------------------------------------------------


def report(args):
    work = Path(args.work)
    out = {}
    if (work / "check" / "results.json").is_file():
        out["check"] = json.loads((work / "check" / "results.json").read_text())
    if (work / "fit.json").is_file():
        out["fit"] = json.loads((work / "fit.json").read_text())
    if (work / "apply.json").is_file():
        got = json.loads((work / "apply.json").read_text())
        ways = {}
        for entry in got["played"]:
            arm = ways.setdefault(entry["way"], {}).setdefault(
                entry["arm"],
                {
                    "conversations": 0,
                    "turns": 0,
                    "errors": 0,
                    "graded": 0,
                    "correct": 0,
                    "graded_needing": 0,
                    "correct_needing": 0,
                },
            )
            arm["conversations"] += 1
            for turn in entry["turns"]:
                arm["turns"] += "answer" in turn
                arm["errors"] += "error" in turn
                if turn.get("correct") is not None:
                    arm["graded"] += 1
                    arm["correct"] += turn["correct"]
                    if turn.get("needs_history") == "yes":
                        arm["graded_needing"] += 1
                        arm["correct_needing"] += turn["correct"]
        for way, arms in ways.items():
            for name, arm in arms.items():
                calls = load_calls(work / "apply" / way / name)
                sums = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
                kinds, costs = {}, {}
                for call in calls:
                    for key in sums:
                        sums[key] += (call.get("usage") or {}).get(key, 0)
                    kinds[call["kind"]] = kinds.get(call["kind"], 0) + 1
                    costs[call["kind"]] = costs.get(call["kind"], 0) + (
                        call.get("cost") or 0
                    )
                cost = sum(costs.values())
                arm.update(
                    cost=round(cost, 5),
                    cost_by_kind={k: round(v, 5) for k, v in costs.items()},
                    calls=kinds,
                    tokens=sums,
                    per_1000_turns=round(1000 * cost / max(1, arm["turns"]), 3),
                    stats=(got["stats"].get(way) or {}).get(name),
                )
            base = arms.get("keep", {}).get("per_1000_turns")
            for arm in arms.values():
                arm["saving"] = (
                    None if not base else round(1 - arm["per_1000_turns"] / base, 4)
                )
        out["apply"] = {way: ways[way] for way in WAYS if way in ways}
        print(
            f"{'way':17}{'policy':16}{'turns':>6}{'$/1000 turns':>14}{'saving':>8}{'correct':>10}{'needing':>10}"
            f"{'reopen':>8}{'pings':>7}{'retry':>7}"
        )
        for way, arms in out["apply"].items():
            for name in [*ARMS, *sorted(set(arms) - set(ARMS))]:
                arm = arms.get(name)
                if arm is None:
                    continue
                saving = (
                    ""
                    if arm["saving"] is None or name == "keep"
                    else f"{arm['saving']:.0%}"
                )
                print(
                    f"{way:17}{name:16}{arm['turns']:>6}{arm['per_1000_turns']:>14.2f}{saving:>8}"
                    f"{arm['correct']:>6}/{arm['graded']:<3}{arm['correct_needing']:>6}/{arm['graded_needing']:<3}"
                    f"{arm['calls'].get('reopen', 0):>8}{arm['calls'].get('ping', 0):>7}{arm['calls'].get('retry', 0):>7}"
                )
        if args.figure:
            # What `scripts/figures.py live` draws.
            shown = {
                "subtitle": (
                    f"DocChat, {len({e['conversation'] for e in got['played']})} held-out conversations "
                    f"of 10 to 12 turns, played at their users' pace. Billed by the providers through OpenRouter."
                ),
                "ways": [
                    {
                        "name": SHOWN[way][0],
                        "detail": list(SHOWN[way][1:]),
                        "arms": [
                            {"name": label, "cost": arms[name]["per_1000_turns"]}
                            for name, label in NAMED.items()
                            if name in arms
                        ],
                    }
                    for way, arms in out["apply"].items()
                ],
            }
            Path(args.figure).write_text(json.dumps(shown, indent=1) + "\n")
    out["billed"] = round(Spend(work, 1e9).total(), 4)
    (work / "report.json").write_text(json.dumps(out, indent=1))
    if args.verbose:
        print(json.dumps(out, indent=1))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["check", "learn", "fit", "apply", "grade", "report"]
    )
    parser.add_argument(
        "--work", required=True, help="Folder for records, policies and results"
    )
    parser.add_argument("--data", help="A DocChat build, for learn, fit and apply")
    parser.add_argument(
        "--env", help="A file of NAME=value lines holding OPENROUTER_API_KEY"
    )
    parser.add_argument(
        "--base", default="https://openrouter.ai/api", help="Where the model calls go"
    )
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5")
    parser.add_argument("--openai-model", default="openai/gpt-6-luna")
    parser.add_argument("--gemini-model", default="google/gemini-3.5-flash-lite")
    parser.add_argument(
        "--prices",
        default="claude-haiku-5-5",
        help="The price sheet the policies are fitted for",
    )
    parser.add_argument(
        "--openai-prices",
        default="gpt-6-luna",
        help="The same for the ways on --openai-model",
    )
    parser.add_argument(
        "--no-pin",
        dest="pin",
        action="store_false",
        help="Let the gateway choose the provider",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=3.0,
        help="Stop before billing more than this, in dollars",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=12,
        help="Turns played of each held-out conversation",
    )
    parser.add_argument(
        "--train-turns",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12, 14],
        help="Turns played of the training conversations, taken in turn",
    )
    parser.add_argument(
        "--tiers", default="s", help="Which conversations apply plays: s, m, l"
    )
    parser.add_argument(
        "--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES)
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Share of turns asked for a note in learn",
    )
    parser.add_argument(
        "--notes",
        default="apart",
        choices=["apart", "inline"],
        help="How check and learn ask for notes: after the reply, or in it",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="*",
        default=[0.5, 0.25, 0.1],
        help="Lower sampling rates that fit also tries",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=420.0,
        help="Longest wait between turns, in seconds",
    )
    parser.add_argument(
        "--pace", type=float, default=1.0, help="Scale of the waits in apply"
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.4,
        help="Seconds between the starts of conversations",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=1200.0,
        help="Longest wait for pings after the last answer",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ways", nargs="*", choices=list(WAYS), help="Which ways in apply plays"
    )
    parser.add_argument(
        "--arms", nargs="*", choices=list(ARMS), help="Which policies apply plays"
    )
    parser.add_argument("--only", nargs="*", help="Which ways in check tries")
    parser.add_argument(
        "--tries", type=int, default=2, help="How often check tries a way that fails"
    )
    parser.add_argument("--verbose", action="store_true", help="Print the whole report")
    parser.add_argument("--figure", help="Where report writes the data of the figure")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    commands = {
        "check": check,
        "learn": learn,
        "fit": fit,
        "apply": apply,
        "grade": regrade,
        "report": report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
