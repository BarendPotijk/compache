"""Threads rebuilt from Langfuse observations, in the shapes its integrations record."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compache
from compache import langfuse
from compache.cli import main
from compache.errors import ContractError
from compache.langfuse import _usage, fetch, pull, threads_from_observations
from compache.threads import load_threads, recorded

SYSTEM = "Answer from the document."
TOOLS = [{"type": "function", "function": {"name": "search_docs"}}]
PASSAGE_1 = "The warranty covers parts and labour for two years from delivery."
PASSAGE_2 = "A claim must be filed within thirty days of finding the defect."


def at(seconds):
    return f"2026-11-10T09:{seconds // 60:02d}:{seconds % 60:06.3f}Z"


def row(kind, start, end, *, trace, session="s1", name="call", **fields):
    """One observation as the API returns it: inputs and outputs are text."""
    for key in ("input", "output"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    return {
        "id": fields.pop("id", f"{trace}-{kind}-{start}"),
        "traceId": trace,
        "parentObservationId": fields.pop("parent", None),
        "type": kind,
        "name": name,
        "startTime": at(start),
        "endTime": at(end),
        "sessionId": session,
        **fields,
    }


def openai_usage(prompt, completion, cached=0, reasoning=0):
    """Langfuse's exclusive buckets for an OpenAI call."""
    return {
        "input": prompt - cached,
        "input_cached_tokens": cached,
        "output": completion - reasoning,
        "output_reasoning_tokens": reasoning,
        "total": prompt + completion,
    }


def tool_loop(session="s1", model="claude-sonnet-5-5"):
    """Two turns of an OpenAI tool loop, every model call its own trace."""
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "search_docs", "arguments": "{}"},
    }
    asked = {"role": "assistant", "tool_calls": [call], "content": None}
    first = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "How long is the warranty?"},
    ]
    second = first + [
        asked,
        {"role": "tool", "tool_call_id": "c1", "content": PASSAGE_1},
    ]
    third = second + [
        {"role": "assistant", "content": "Two years."},
        {"role": "user", "content": "And claims?"},
    ]
    fourth = third + [
        asked,
        {"role": "tool", "tool_call_id": "c1", "content": PASSAGE_2},
    ]
    answers = [
        asked,
        {"role": "assistant", "content": "Two years."},
        asked,
        {"role": "assistant", "content": "Within thirty days."},
    ]
    times = [(0, 2), (3, 6), (300, 302), (303, 307)]
    usages = [
        openai_usage(60, 20),
        openai_usage(1200, 30, reasoning=10),
        openai_usage(1300, 20, cached=1152),
        openai_usage(2500, 40, cached=1280),
    ]
    return [
        row(
            "GENERATION",
            *times[i],
            trace=f"{session}-t{i}",
            session=session,
            name="OpenAI-generation",
            input={"tools": TOOLS, "messages": sent},
            output=answers[i],
            model=model,
            usageDetails=usages[i],
        )
        for i, sent in enumerate([first, second, third, fourth])
    ]


class RebuildTests(unittest.TestCase):
    def test_a_tool_loop_traced_call_by_call_becomes_one_run_per_question(self):
        rows = tool_loop()
        # Rows arrive newest first.
        threads, report = threads_from_observations(
            reversed(rows), observed_until=at(900)
        )
        self.assertEqual(
            (
                report["sessions"],
                report["threads"],
                report["turns"],
                report["model_calls"],
            ),
            (1, 1, 2, 4),
        )
        self.assertEqual(report["models"], {"claude-sonnet-5-5": 4})
        thread = threads[0]
        self.assertEqual(
            (thread["thread_id"], thread["tools"], thread["observed_until"]),
            ("s1", TOOLS, at(900)),
        )
        first, second = load_threads(threads)[0].runs
        self.assertEqual(
            (first.input.text, first.output.text),
            ("How long is the warranty?", "Two years."),
        )
        self.assertEqual([p.text for p in first.retrievals], [PASSAGE_1])
        self.assertEqual(
            (first.started_at, first.ended_at, first.calls), (at(0), at(6), 2)
        )
        self.assertEqual((second.started_at, second.ended_at), (at(300), at(307)))
        self.assertEqual([p.text for p in second.retrievals], [PASSAGE_2])
        # The last call's input, all the output with reasoning, and what the turn found cached.
        self.assertEqual(
            dict(first.generation.usage),
            {"input": 1200, "output": 50, "cache_read": 0, "cache_write": 0},
        )
        self.assertEqual(
            dict(second.generation.usage),
            {"input": 2500, "output": 60, "cache_read": 1152, "cache_write": 0},
        )
        found = recorded(threads)
        self.assertEqual(
            (
                found["system_prompt"],
                found["tools"],
                found["model"],
                found["observed_until"],
            ),
            (SYSTEM, TOOLS, "claude-sonnet-5-5", at(900)),
        )

    def test_a_decorated_trace_starts_with_its_root_and_pasted_passages_are_retrieved_text(
        self,
    ):
        documents = [
            {"id": "d1", "page_content": PASSAGE_1},
            {"id": "d2", "page_content": PASSAGE_2},
        ]
        prompt = f"Context:\n{PASSAGE_1}\n\n{PASSAGE_2}\n\nQuestion: "
        history = [
            {"role": "user", "content": "How long is the warranty?"},
            {"role": "assistant", "content": "Two years."},
        ]
        rows = []
        for n, (start, sent, answer) in enumerate(
            [
                (
                    0,
                    [{"role": "user", "content": prompt + "How long is the warranty?"}],
                    "Two years.",
                ),
                (
                    600,
                    history + [{"role": "user", "content": prompt + "And claims?"}],
                    "Within thirty days.",
                ),
            ]
        ):
            trace = f"t{n}"
            rows += [
                row(
                    "SPAN",
                    start,
                    start + 5,
                    trace=trace,
                    id=f"root{n}",
                    name="rag-turn",
                    input={"args": ["q"], "kwargs": {}},
                    output=answer,
                ),
                row(
                    "RETRIEVER",
                    start,
                    start + 1,
                    trace=trace,
                    parent=f"root{n}",
                    name="retrieve",
                    output=documents,
                ),
                row(
                    "GENERATION",
                    start + 1,
                    start + 5,
                    trace=trace,
                    parent=f"root{n}",
                    input=[{"role": "system", "content": SYSTEM}, *sent],
                    output={"role": "assistant", "content": answer},
                    model="gpt-6-luna-2026-10-01",
                    usageDetails=openai_usage(900, 12),
                ),
            ]
        threads, report = threads_from_observations(rows)
        self.assertEqual(report["turns_with_pasted_retrieval"], 2)
        self.assertEqual(report["threads_with_trimmed_history"], 1)
        first, second = load_threads(threads)[0].runs
        # The turn began when the trace did, a second before its model call.
        self.assertEqual((first.started_at, first.ended_at), (at(0), at(5)))
        self.assertEqual(
            first.input.text, "Context:\n\nQuestion: How long is the warranty?"
        )
        self.assertEqual([p.text for p in second.retrievals], [PASSAGE_1, PASSAGE_2])
        self.assertEqual((second.calls, second.output.text), (1, "Within thirty days."))

    def test_calls_that_do_not_carry_the_conversation_are_read_trace_by_trace(self):
        documents = [{"id": "d1", "page_content": PASSAGE_1}]
        turns = [
            (0, "How long is the warranty?", "Two years."),
            (600, "And claims?", "Within thirty days."),
            (1200, "Who pays shipping?", "The buyer."),
        ]
        rows = []
        for n, (start, question, answer) in enumerate(turns):
            trace, root = f"t{n}", f"root{n}"
            rows.append(
                row(
                    "SPAN",
                    start,
                    start + 9,
                    trace=trace,
                    id=root,
                    name="rag-turn",
                    input={"args": [question], "kwargs": {}},
                    output=answer,
                )
            )
            if n:
                # The follow-up is rewritten to stand alone: a prompt of its own, no messages of the history.
                rows.append(
                    row(
                        "GENERATION",
                        start,
                        start + 2,
                        trace=trace,
                        parent=root,
                        input=[
                            {
                                "role": "user",
                                "content": f"Rephrase to stand alone. History: ... Follow up: {question}",
                            }
                        ],
                        output={"role": "assistant", "content": question},
                        model="gpt-6-luna",
                        usageDetails=openai_usage(300, 12),
                    )
                )
            rows.append(
                row(
                    "RETRIEVER",
                    start + 2,
                    start + 3,
                    trace=trace,
                    parent=root,
                    output=documents,
                )
            )
            rows.append(
                row(
                    "GENERATION",
                    start + 3,
                    start + 8,
                    trace=trace,
                    parent=root,
                    input=[
                        {
                            "role": "system",
                            "content": f"Answer from this context.\n\n{PASSAGE_1}",
                        },
                        {"role": "user", "content": question},
                    ],
                    output={"role": "assistant", "content": answer},
                    model="gpt-6-luna",
                    usageDetails=openai_usage(900, 20),
                )
            )
        threads, report = threads_from_observations(rows)
        self.assertEqual(
            (report["threads"], report["turns"], report["threads_read_trace_by_trace"]),
            (1, 3, 1),
        )
        self.assertEqual(
            (report["helper_calls"], report["threads_with_trimmed_history"]), (0, 1)
        )
        runs = load_threads(threads)[0].runs
        self.assertEqual([r.input.text for r in runs], [q for _, q, _ in turns])
        self.assertEqual([r.output.text for r in runs], [a for _, _, a in turns])
        self.assertEqual([r.calls for r in runs], [1, 2, 2])
        self.assertEqual(
            [[p.text for p in r.retrievals] for r in runs], [[PASSAGE_1]] * 3
        )
        # The passage pasted into the system prompt is retrieved text, not part of the prompt.
        self.assertEqual(
            recorded(threads)["system_prompt"], "Answer from this context."
        )
        # A turn runs from the start of its trace to the end of its last model call.
        self.assertEqual((runs[1].started_at, runs[1].ended_at), (at(600), at(608)))
        self.assertEqual(
            dict(runs[1].generation.usage),
            {"input": 900, "output": 32, "cache_read": 0, "cache_write": 0},
        )

    def test_a_call_recorded_by_two_instrumentations_counts_once(self):
        usage = {
            "input": 20,
            "input_cached_tokens": 600,
            "input_cache_creation": 400,
            "output": 30,
            "total": 1050,
        }
        sent = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": "How long is the warranty?"}],
                },
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "content": "I'll look."},
                        {
                            "type": "tool_call",
                            "id": "u1",
                            "name": "search_docs",
                            "arguments": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "u1",
                            "response": PASSAGE_1,
                        }
                    ],
                },
            ],
            "tools": [{"name": "search_docs"}],
        }
        said = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Two years."}],
                "finish_reason": "stop",
            }
        ]
        rows = [
            row("SPAN", 0, 6, trace="t", id="root", name="chat-turn"),
            row(
                "GENERATION",
                0,
                2,
                trace="t",
                parent="root",
                id="a1",
                input={**sent, "messages": sent["messages"][:2]},
                output=[{"role": "assistant", "parts": sent["messages"][2]["parts"]}],
                model="claude-opus-5-5",
                usageDetails={"input": 80, "output": 20, "total": 100},
            ),
            row(
                "GENERATION",
                0,
                2,
                trace="t",
                parent="root",
                id="b1",
                model="claude-opus-5-5",
                usageDetails={"input": 80, "output": 20, "total": 100},
            ),
            row(
                "GENERATION",
                3,
                6,
                trace="t",
                parent="root",
                id="b2",
                model="claude-opus-5-5",
                usageDetails=usage,
            ),
            row(
                "GENERATION",
                3,
                6,
                trace="t",
                parent="root",
                id="a2",
                input=sent,
                output=said,
                model="claude-opus-5-5",
                usageDetails=usage,
            ),
        ]
        threads, report = threads_from_observations(rows)
        self.assertEqual(
            (report["calls_recorded_twice"], report["model_calls"]), (2, 2)
        )
        (run,) = load_threads(threads)[0].runs
        self.assertEqual((run.calls, run.output.text), (2, "I'll look.\n\nTwo years."))
        self.assertEqual([p.text for p in run.retrievals], [PASSAGE_1])
        self.assertEqual(
            dict(run.generation.usage),
            {"input": 1020, "output": 50, "cache_read": 0, "cache_write": 400},
        )
        self.assertEqual(threads[0]["tools"], [{"name": "search_docs"}])

    def test_what_is_not_a_conversation_is_left_out_and_counted(self):
        rows = tool_loop()
        # A helper call inside the session: its own prompt, none of the history.
        rows.append(
            row(
                "GENERATION",
                302,
                303,
                trace="helper",
                input=[
                    {
                        "role": "user",
                        "content": "Rewrite as a search query: And claims?",
                    }
                ],
                output={"role": "assistant", "content": "warranty claim deadline"},
                model="claude-sonnet-5-5",
            )
        )
        # A session whose first call already carries an answer: it began before the record.
        rows.append(
            row(
                "GENERATION",
                0,
                4,
                trace="late",
                session="s2",
                input=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello."},
                    {"role": "user", "content": "Still there?"},
                ],
                output={"role": "assistant", "content": "Yes."},
            )
        )
        # Model calls recorded without their text, and a trace with no session.
        rows.append(
            row(
                "GENERATION",
                0,
                4,
                trace="bare",
                session="s3",
                usageDetails={"input": 5, "output": 5, "total": 10},
            )
        )
        rows.append(
            row(
                "GENERATION",
                0,
                4,
                trace="loose",
                session=None,
                input=[{"role": "user", "content": "Hi"}],
                output="Hello.",
            )
        )
        threads, report = threads_from_observations(rows)
        self.assertEqual([t["thread_id"] for t in threads], ["s1"])
        self.assertEqual(len(load_threads(threads)[0].runs), 2)
        self.assertEqual(
            {
                k: report[k]
                for k in (
                    "helper_calls",
                    "sessions_begun_earlier",
                    "sessions_without_text",
                    "traces_without_session",
                )
            },
            {
                "helper_calls": 1,
                "sessions_begun_earlier": 1,
                "sessions_without_text": 1,
                "traces_without_session": 1,
            },
        )

    def test_usage_buckets_add_up_to_the_canonical_counters(self):
        self.assertEqual(
            _usage(openai_usage(9927, 547, cached=5120, reasoning=408)),
            {"input": 9927, "output": 547, "cache_read": 5120, "cache_write": 0},
        )
        self.assertEqual(
            _usage(
                {
                    "input": 28,
                    "output": 139,
                    "input_cached_tokens": 5182,
                    "input_cache_creation": 4721,
                    "total": 10070,
                }
            ),
            {"input": 9931, "output": 139, "cache_read": 5182, "cache_write": 4721},
        )
        # Counters passed through as Anthropic names them, and a provider's own usage object.
        self.assertEqual(
            _usage(
                {
                    "input": 10,
                    "output": 5,
                    "cache_read_input_tokens": 70,
                    "cache_creation_input_tokens": 20,
                }
            ),
            {"input": 100, "output": 5, "cache_read": 70, "cache_write": 20},
        )
        self.assertEqual(
            _usage(
                {
                    "prompt_tokens": 50,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 30},
                }
            ),
            {"input": 50, "output": 7, "cache_read": 30, "cache_write": 0},
        )
        # Buckets that do not add up to the total are a shape this does not know.
        self.assertIsNone(_usage({"input": 10, "output": 5, "total": 99}))
        self.assertIsNone(_usage(None))


class FetchTests(unittest.TestCase):
    def reply(self, status, body, headers=None):
        return status, headers or {}, json.dumps(body).encode()

    def test_pages_follow_the_cursor_and_a_rate_limit_is_waited_out(self):
        replies = [
            self.reply(429, {"message": "slow down"}, {"Retry-After": "7"}),
            self.reply(200, {"data": [{"id": "1"}], "meta": {"cursor": "next"}}),
            self.reply(200, {"data": [{"id": "2"}], "meta": {}}),
        ]
        seen, waits = [], []

        def get(url, headers, timeout):
            seen.append((url, headers))
            return replies.pop(0)

        with mock.patch.object(langfuse, "_get", get):
            rows = list(
                fetch(
                    since="2026-11-01T00:00:00Z",
                    environment="production",
                    tags=["chat"],
                    public_key="pk",
                    secret_key="sk",
                    base_url="https://langfuse.example/",
                    pause=waits.append,
                )
            )
        self.assertEqual([r["id"] for r in rows], ["1", "2"])
        self.assertEqual(waits, [7.0])
        url, headers = seen[0]
        self.assertTrue(
            url.startswith("https://langfuse.example/api/public/v2/observations?")
        )
        for part in (
            "fields=core%2Cbasic%2Cio%2Cmodel%2Cusage",
            "fromStartTime=2026-11-01T00%3A00%3A00Z",
            "environment=production",
            "limit=100",
            "%22any+of%22",
        ):
            self.assertIn(part, url)
        self.assertEqual(headers["Authorization"], "Basic cGs6c2s=")
        self.assertIn("cursor=next", seen[2][0])

    def test_a_page_too_large_is_asked_for_in_smaller_pieces(self):
        replies = [self.reply(502, {}), self.reply(200, {"data": [], "meta": {}})]
        seen = []

        def get(url, headers, timeout):
            seen.append(url)
            return replies.pop(0)

        with mock.patch.object(langfuse, "_get", get):
            list(fetch(public_key="pk", secret_key="sk", pause=lambda _: None))
        self.assertIn("limit=100", seen[0])
        self.assertIn("limit=50", seen[1])

    def test_refusals_say_what_is_wrong(self):
        for status, words in (
            (401, "rejected the Langfuse keys"),
            (410, "needs Langfuse 4"),
            (400, "refused the query"),
        ):
            with mock.patch.object(
                langfuse,
                "_get",
                lambda *a, status=status: self.reply(status, {"message": "no"}),
            ):
                with self.assertRaisesRegex(ContractError, words):
                    list(fetch(public_key="pk", secret_key="sk"))
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ContractError, "LANGFUSE_PUBLIC_KEY"):
                list(fetch())


class PullTests(unittest.TestCase):
    def test_pulled_threads_are_fitted_as_they_are(self):
        rows = [r for n in range(6) for r in tool_loop(session=f"s{n}")]
        threads, report = pull(rows=rows, until="2026-11-20T00:00:00Z")
        self.assertEqual(
            (report["threads"], report["until"]), (6, "2026-11-20T00:00:00Z")
        )
        # The model, the system prompt, the tools and the time of the read all come from the threads.
        policy = compache.fit(threads)
        self.assertEqual(policy.spec.model_id, "claude-sonnet-5-5")
        self.assertEqual(policy.artifact["header"]["system_prompt"], SYSTEM)
        self.assertEqual(policy.artifact["header"]["default_tools"], TOOLS)
        self.assertEqual(
            compache.profile(threads)["source"]["model"], "claude-sonnet-5-5"
        )

    def test_the_command_writes_a_threads_file_from_an_export(self):
        rows = [r for n in range(6) for r in tool_loop(session=f"s{n}")]
        with tempfile.TemporaryDirectory() as folder:
            export, out = Path(folder, "export.json"), Path(folder, "threads.jsonl.gz")
            export.write_text(json.dumps({"data": rows, "meta": {}}))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main(["pull", "langfuse", "--rows", str(export), "-o", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("Wrote 6 threads, 12 turns", printed.getvalue())
            self.assertIn(
                "Answered by claude-sonnet-5-5 (24 calls).", printed.getvalue()
            )
            self.assertEqual(len(load_threads(out)), 6)
            with contextlib.redirect_stdout(io.StringIO()) as fitted:
                self.assertEqual(
                    main(["fit", str(out), "-o", str(Path(folder, "policy.json"))]), 0
                )
            self.assertIn("for claude-sonnet-5-5", fitted.getvalue())
            # Rows with nothing to rebuild are an error that says why.
            export.write_text(
                json.dumps(
                    [
                        row(
                            "GENERATION",
                            0,
                            1,
                            trace="t",
                            session=None,
                            input="Hi",
                            output="Hello",
                        )
                    ]
                )
            )
            failed = io.StringIO()
            with contextlib.redirect_stderr(failed):
                self.assertEqual(
                    main(["pull", "langfuse", "--rows", str(export), "-o", str(out)]), 1
                )
            self.assertIn("1 traces have no session ID", failed.getvalue())


if __name__ == "__main__":
    unittest.main()
