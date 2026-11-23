"""Threads rebuilt from OpenTelemetry spans, in the GenAI conventions and in OpenInference."""

import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import compache
from compache.errors import ContractError
from compache.otel import pull, rows_from_spans, threads_from_spans
from compache.threads import load_threads, recorded

# 2026-11-10T09:00:00Z
START = 1_794_301_200
SYSTEM = "Answer from the visitor guide."
MODEL = "claude-sonnet-5-5"
TOOLS = [{"type": "function", "name": "search_guide", "parameters": {"type": "object"}}]
QUESTIONS = ["When does the museum close?", "Can I return a ticket?"]
ANSWERS = ["At six, and at nine on Thursdays.", "Yes, until the day before your visit."]
PASSAGE_1 = "The museum is open from ten to six, and until nine on Thursdays."
PASSAGE_2 = "Tickets bought online can be returned until the day before the visit."

# Two turns of a tool loop: what each model call was sent, what it answered, and its
# tokens as input, output, read from the cache and written to it.
CALL_1, CALL_2 = ("call", "c1"), ("call", "c2")
FIRST = [("user", QUESTIONS[0])]
SECOND = FIRST + [CALL_1, ("result", "c1", PASSAGE_1)]
THIRD = SECOND + [("assistant", ANSWERS[0]), ("user", QUESTIONS[1])]
FOURTH = THIRD + [CALL_2, ("result", "c2", PASSAGE_2)]
CALLS = [
    (1, 1, 2, FIRST, CALL_1, (1200, 20, 0, 1100)),
    (1, 3, 6, SECOND, ("assistant", ANSWERS[0]), (1300, 30, 1152, 100)),
    (2, 301, 302, THIRD, CALL_2, (1400, 20, 1280, 64)),
    (2, 303, 307, FOURTH, ("assistant", ANSWERS[1]), (1500, 40, 1344, 128)),
]
TURNS = [(1, 0, 7, PASSAGE_1), (2, 300, 308, PASSAGE_2)]


def at(seconds):
    """A time as a thread holds it."""
    return f"2026-11-10T09:{seconds // 60:02d}:{seconds % 60:02d}.000000Z"


def typed(value):
    """A value as OTLP JSON writes it."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [typed(v) for v in value]}}
    if isinstance(value, dict):
        pairs = [{"key": k, "value": typed(v)} for k, v in value.items()]
        return {"kvlistValue": {"values": pairs}}
    return {"stringValue": value}


def span(span_id, start, end, attributes, *, trace, parent=""):
    return {
        "traceId": trace,
        "spanId": span_id,
        "parentSpanId": parent,
        "name": span_id,
        "kind": 1,
        "startTimeUnixNano": str((START + start) * 10**9),
        "endTimeUnixNano": str((START + end) * 10**9),
        "attributes": [{"key": k, "value": typed(v)} for k, v in attributes.items()],
        "status": {"code": 1},
    }


def export(spans):
    resource = {"attributes": [{"key": "service.name", "value": typed("desk")}]}
    scopes = [{"scope": {"name": "app"}, "spans": list(spans)}]
    return {"resourceSpans": [{"resource": resource, "scopeSpans": scopes}]}


def parts(message, answer="response"):
    """A message in the parts of the GenAI conventions."""
    kind, *rest = message
    if kind == "call":
        call = {"type": "tool_call", "id": rest[0], "name": "search_guide"}
        return {"role": "assistant", "parts": [{**call, "arguments": {}}]}
    if kind == "result":
        told = {"type": "tool_call_response", "id": rest[0], answer: rest[1]}
        return {"role": "tool", "parts": [told]}
    return {"role": kind, "parts": [{"type": "text", "content": rest[0]}]}


def genai_spans(session="visit-1", older=False):
    """The tool loop as the GenAI conventions record it, with today's keys or older ones."""
    spans = []
    for turn, start, end, passage in TURNS:
        trace = f"{session}-t{turn}"
        # Only the agent's span names the conversation.
        agent = {"gen_ai.operation.name": "invoke_agent"}
        agent["gen_ai.conversation.id"] = session
        tool = {"gen_ai.operation.name": "execute_tool"}
        tool["gen_ai.tool.call.result"] = passage
        # A root's parent is empty, or all zeros.
        root = {"trace": trace, "parent": "0000000000000000" if turn == 2 else ""}
        under = {"trace": trace, "parent": f"{trace}-agent"}
        spans.append(span(f"{trace}-agent", start, end, agent, **root))
        spans.append(span(f"{trace}-tool", start + 2, start + 3, tool, **under))
    for n, (turn, start, end, sent, answered, counts) in enumerate(CALLS):
        given, output, read, written = counts
        if older:
            # Content as JSON text, the system prompt among the messages, no name for the operation.
            said = [("system", SYSTEM), *sent]
            attributes = {
                "gen_ai.request.model": MODEL,
                "gen_ai.input.messages": json.dumps([parts(m, "result") for m in said]),
                "gen_ai.output.messages": json.dumps([parts(answered)]),
                "gen_ai.tool.definitions": json.dumps(TOOLS),
                "gen_ai.usage.prompt_tokens": given,
                "gen_ai.usage.completion_tokens": output,
                "gen_ai.usage.cache_read.input_tokens": read,
                "gen_ai.usage.cache_creation.input_tokens": written,
            }
        else:
            attributes = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "sonnet",
                "gen_ai.response.model": MODEL,
                "gen_ai.system_instructions": [{"type": "text", "content": SYSTEM}],
                "gen_ai.input.messages": [parts(m) for m in sent],
                "gen_ai.output.messages": [
                    {**parts(answered), "finish_reason": "stop"}
                ],
                "gen_ai.tool.definitions": TOOLS,
                "gen_ai.usage.input_tokens": given,
                "gen_ai.usage.output_tokens": output,
                "gen_ai.usage.cache_read.input_tokens": read,
                "gen_ai.usage.cache_write.input_tokens": written,
            }
        under = {"trace": f"{session}-t{turn}", "parent": f"{session}-t{turn}-agent"}
        spans.append(span(f"{under['trace']}-c{n}", start, end, attributes, **under))
    return spans


def flat(prefix, messages):
    """Messages spread over numbered attributes, as OpenInference records them."""
    found = {}
    for n, (kind, *rest) in enumerate(messages):
        key = f"{prefix}.{n}.message"
        if kind == "call":
            call = f"{key}.tool_calls.0.tool_call"
            found[f"{key}.role"] = "assistant"
            found[f"{call}.id"] = rest[0]
            found[f"{call}.function.name"] = "search_guide"
            found[f"{call}.function.arguments"] = "{}"
        elif kind == "result":
            found[f"{key}.role"] = "tool"
            found[f"{key}.tool_call_id"] = rest[0]
            found[f"{key}.content"] = rest[1]
        else:
            found[f"{key}.role"] = kind
            found[f"{key}.content"] = rest[0]
    return found


def openinference_spans(session="visit-1"):
    """The same tool loop as OpenInference records it, the session on every span."""
    spans = []
    for turn, start, end, passage in TURNS:
        trace = f"{session}-t{turn}"
        chain = {"openinference.span.kind": "CHAIN", "session.id": session}
        chain["input.value"] = QUESTIONS[turn - 1]
        chain["output.value"] = ANSWERS[turn - 1]
        tool = {"openinference.span.kind": "TOOL", "session.id": session}
        tool.update({"tool.name": "search_guide", "output.value": passage})
        under = {"trace": trace, "parent": f"{trace}-chain"}
        spans.append(span(f"{trace}-chain", start, end, chain, trace=trace))
        spans.append(span(f"{trace}-tool", start + 2, start + 3, tool, **under))
    for n, (turn, start, end, sent, answered, counts) in enumerate(CALLS):
        given, output, read, written = counts
        attributes = {
            "openinference.span.kind": "LLM",
            "session.id": session,
            "llm.model_name": MODEL,
            "llm.tools.0.tool.json_schema": json.dumps(TOOLS[0]),
            # What the instrumentation was handed, which the numbered messages make redundant.
            "input.value": json.dumps({"model": "sonnet", "max_tokens": 1024}),
            "output.value": json.dumps({"id": "msg_1", "type": "message"}),
            **flat("llm.input_messages", [("system", SYSTEM), *sent]),
            **flat("llm.output_messages", [answered]),
            "llm.token_count.prompt": given,
            "llm.token_count.completion": output,
            "llm.token_count.total": given + output,
            "llm.token_count.prompt_details.cache_write": written,
        }
        if read:
            # A count of zero is left out by some instrumentations.
            attributes["llm.token_count.prompt_details.cache_read"] = read
        under = {"trace": f"{session}-t{turn}", "parent": f"{session}-t{turn}-chain"}
        spans.append(span(f"{under['trace']}-c{n}", start, end, attributes, **under))
    return spans


class RebuildTests(unittest.TestCase):
    def tool_loop(self, threads, report, tools, calls=4):
        """The two turns of the tool loop, however it was recorded."""
        counted = (
            report["sessions"],
            report["threads"],
            report["turns"],
            report["model_calls"],
        )
        self.assertEqual(counted, (1, 1, 2, calls))
        self.assertEqual(report["models"], {MODEL: 4})
        # Read from the model calls, not from what the span around a turn took and returned.
        self.assertEqual(report["threads_read_trace_by_trace"], 0)
        self.assertEqual(
            (threads[0]["thread_id"], threads[0]["tools"]), ("visit-1", tools)
        )
        first, second = load_threads(threads)[0].runs
        self.assertEqual([r.input.text for r in (first, second)], QUESTIONS)
        self.assertEqual([r.output.text for r in (first, second)], ANSWERS)
        self.assertEqual([p.text for p in first.retrievals], [PASSAGE_1])
        self.assertEqual([p.text for p in second.retrievals], [PASSAGE_2])
        # A turn runs from the start of its trace, a second before its first call, to its last answer.
        self.assertEqual(
            (first.started_at, first.ended_at, first.calls), (at(0), at(6), 2)
        )
        self.assertEqual(
            (second.started_at, second.ended_at, second.calls), (at(300), at(307), 2)
        )
        self.assertEqual(
            dict(first.generation.usage),
            {"input": 1300, "output": 50, "cache_read": 0, "cache_write": 100},
        )
        self.assertEqual(
            dict(second.generation.usage),
            {"input": 1500, "output": 60, "cache_read": 1280, "cache_write": 128},
        )
        self.assertTrue(first.generation.cache_read_reported)
        found = recorded(threads)
        self.assertEqual((found["system_prompt"], found["model"]), (SYSTEM, MODEL))

    def test_a_genai_tool_loop_becomes_one_run_per_question(self):
        threads, report = threads_from_spans(
            export(genai_spans()), observed_until=at(900)
        )
        self.tool_loop(threads, report, TOOLS)
        self.assertEqual(threads[0]["observed_until"], at(900))
        self.assertEqual(report["observations"], 8)

    def test_older_genai_keys_and_content_as_json_text_read_the_same(self):
        threads, report = threads_from_spans(export(genai_spans(older=True)))
        self.tool_loop(threads, report, TOOLS)
        self.assertEqual(threads, threads_from_spans(export(genai_spans()))[0])

    def test_openinference_messages_tools_and_counts_read_the_same(self):
        threads, report = threads_from_spans(export(openinference_spans()))
        self.tool_loop(threads, report, [TOOLS[0]])
        self.assertEqual(threads, threads_from_spans(export(genai_spans()))[0])

    def test_openinference_rows_hold_the_calls_as_chat_messages(self):
        rows = rows_from_spans(openinference_spans())
        kinds = [r["type"] for r in rows if r["traceId"] == "visit-1-t1"]
        self.assertEqual(sorted(kinds), ["GENERATION", "GENERATION", "SPAN", "TOOL"])
        call = next(r for r in rows if r["id"] == "visit-1-t1-c1")
        self.assertEqual(call["input"]["tools"], TOOLS)
        asked = {"name": "search_guide", "arguments": "{}"}
        self.assertEqual(
            call["input"]["messages"][2:],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": asked}],
                },
                {"role": "tool", "content": PASSAGE_1, "tool_call_id": "c1"},
            ],
        )
        self.assertEqual(call["output"], [{"role": "assistant", "content": ANSWERS[0]}])
        tool = next(r for r in rows if r["type"] == "TOOL")
        self.assertEqual((tool["output"], tool["sessionId"]), (PASSAGE_1, "visit-1"))

    def test_openinference_documents_pasted_into_the_prompt_are_retrieved_text(self):
        turns = [
            (0, QUESTIONS[0], ANSWERS[0], [PASSAGE_1, PASSAGE_2], []),
            (
                600,
                QUESTIONS[1],
                ANSWERS[1],
                [PASSAGE_2],
                [("user", QUESTIONS[0]), ("assistant", ANSWERS[0])],
            ),
        ]
        spans = []
        for n, (start, question, answer, found, history) in enumerate(turns):
            # A session named by a number is the thread of that name.
            trace, session = f"t{n}", {"session.id": 2026}
            chain = {
                "openinference.span.kind": "CHAIN",
                "input.value": question,
                "output.value": answer,
            }
            retriever = {
                "openinference.span.kind": "RETRIEVER",
                "input.value": question,
            }
            for k, text in enumerate(found):
                retriever[f"retrieval.documents.{k}.document.id"] = f"d{k}"
                retriever[f"retrieval.documents.{k}.document.content"] = text
            prompt = "\n\n".join([SYSTEM, *found])
            sent = [("system", prompt), *history, ("user", question)]
            llm = {
                "openinference.span.kind": "LLM",
                "llm.model_name": MODEL,
                **flat("llm.input_messages", sent),
                # An answer in content blocks, as the Anthropic instrumentation records it.
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.contents.0.message_content.type": "text",
                "llm.output_messages.0.message.contents.0.message_content.text": answer,
                "llm.token_count.prompt": 900,
                "llm.token_count.completion": 12,
            }
            under = {"trace": trace, "parent": f"{trace}-chain"}
            spans += [
                span(f"{trace}-chain", start, start + 5, chain | session, trace=trace),
                span(f"{trace}-find", start, start + 1, retriever | session, **under),
                span(f"{trace}-llm", start + 1, start + 5, llm | session, **under),
            ]
        threads, report = threads_from_spans(spans)
        self.assertEqual(
            (report["threads"], report["turns_with_pasted_retrieval"]), (1, 2)
        )
        self.assertEqual(threads[0]["thread_id"], "2026")
        self.assertEqual(report["threads_with_trimmed_history"], 1)
        first, second = load_threads(threads)[0].runs
        self.assertEqual(
            (first.input.text, first.output.text), (QUESTIONS[0], ANSWERS[0])
        )
        self.assertEqual([p.text for p in first.retrievals], [PASSAGE_1, PASSAGE_2])
        self.assertEqual([p.text for p in second.retrievals], [PASSAGE_2])
        self.assertEqual(
            (second.started_at, second.ended_at, second.calls), (at(600), at(605), 1)
        )
        self.assertEqual(recorded(threads)["system_prompt"], SYSTEM)
        # Counts without a cache counter say nothing about the cache.
        self.assertEqual(
            dict(first.generation.usage),
            {"input": 900, "output": 12, "cache_read": 0, "cache_write": 0},
        )
        self.assertFalse(first.generation.cache_read_reported)

    def test_calls_that_do_not_carry_the_conversation_are_read_agent_by_agent(self):
        questions = [*QUESTIONS, "Is there a cloakroom?"]
        answers = [*ANSWERS, "Yes, beside the entrance."]
        spans = []
        for n, (question, answer) in enumerate(zip(questions, answers)):
            trace, start = f"t{n}", 600 * n
            agent = {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.conversation.id": "visit-3",
                "gen_ai.input.messages": [parts(("user", question))],
                "gen_ai.output.messages": [parts(("assistant", answer))],
            }
            # Every question is rewritten to stand alone and answered without the history.
            alone = [("system", SYSTEM), ("user", f"Standing alone: {question}")]
            chat = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": MODEL,
                "gen_ai.input.messages": [parts(m) for m in alone],
                "gen_ai.output.messages": [parts(("assistant", answer))],
                "gen_ai.usage.input_tokens": 900,
                "gen_ai.usage.output_tokens": 20,
            }
            under = {"trace": trace, "parent": f"{trace}-agent"}
            spans.append(span(f"{trace}-agent", start, start + 9, agent, trace=trace))
            spans.append(span(f"{trace}-chat", start + 3, start + 8, chat, **under))
        threads, report = threads_from_spans(spans)
        counted = (
            report["threads"],
            report["turns"],
            report["threads_read_trace_by_trace"],
        )
        self.assertEqual(counted, (1, 3, 1))
        runs = load_threads(threads)[0].runs
        self.assertEqual([r.input.text for r in runs], questions)
        self.assertEqual([r.output.text for r in runs], answers)
        self.assertEqual(
            (runs[1].started_at, runs[1].ended_at, runs[1].calls), (at(600), at(608), 1)
        )

    def test_an_input_count_that_leaves_cached_tokens_out_is_added_up(self):
        spans = genai_spans()
        for item in spans:
            for pair in item["attributes"]:
                if pair["key"] == "gen_ai.usage.input_tokens":
                    # As Anthropic counts: what was neither read from the cache nor written to it.
                    pair["value"] = typed(12)
        (thread,) = load_threads(threads_from_spans(spans)[0])
        self.assertEqual(
            [dict(r.generation.usage)["input"] for r in thread.runs],
            [12 + 1152 + 100, 12 + 1344 + 128],
        )

    def test_what_is_not_a_model_call_or_has_no_conversation_is_left_out(self):
        spans = genai_spans()
        trace = "visit-1-t1"
        under = {"trace": trace, "parent": f"{trace}-agent"}
        counted = {"gen_ai.usage.input_tokens": 9000, "gen_ai.usage.output_tokens": 900}
        embedding = {"gen_ai.operation.name": "embeddings", **counted}
        embedding["gen_ai.request.model"] = "embed-1"
        spans.append(span("embed", 2, 3, embedding, **under))
        # An agent's span may repeat the messages and add up the tokens of the calls inside it.
        for item in spans:
            if item["spanId"].endswith("-agent"):
                item["attributes"] += [
                    {"key": k, "value": typed(v)} for k, v in counted.items()
                ]
        loose = {"gen_ai.operation.name": "chat", "gen_ai.request.model": MODEL}
        loose["gen_ai.input.messages"] = [parts(("user", "Hello?"))]
        loose["gen_ai.output.messages"] = [parts(("assistant", "Hello."))]
        spans.append(span("alone", 0, 1, loose, trace="no-conversation"))
        # A call recorded with its instructions but not its messages is no turn.
        bare = {"gen_ai.operation.name": "chat", "gen_ai.request.model": MODEL}
        bare["gen_ai.system_instructions"] = [{"type": "text", "content": SYSTEM}]
        spans.append(span("bare", 4, 5, bare, **under))
        unstarted, unset = (
            span(name, 0, 1, loose, trace=trace) for name in ("unstarted", "unset")
        )
        del unstarted["startTimeUnixNano"]
        unset["startTimeUnixNano"] = "0"
        threads, report = threads_from_spans(spans + [unstarted, unset])
        self.tool_loop(threads, report, TOOLS, calls=5)
        counted = (
            report["observations"],
            report["helper_calls"],
            report["traces_without_session"],
        )
        self.assertEqual(counted, (11, 1, 1))

    def test_every_container_and_spelling_holds_the_same_rows(self):
        spans = genai_spans()
        rows = rows_from_spans(export(spans))
        self.assertEqual(len(rows), 8)
        call = next(r for r in rows if r["id"] == "visit-1-t1-c0")
        self.assertEqual(
            {
                k: call[k]
                for k in (
                    "traceId",
                    "parentObservationId",
                    "type",
                    "sessionId",
                    "model",
                )
            },
            {
                "traceId": "visit-1-t1",
                "parentObservationId": "visit-1-t1-agent",
                "type": "GENERATION",
                "sessionId": "visit-1",
                "model": MODEL,
            },
        )
        self.assertEqual((call["startTime"], call["endTime"]), (at(1), at(2)))
        self.assertEqual(
            call["usageDetails"],
            {
                "input": 100,
                "output": 20,
                "input_cached_tokens": 0,
                "input_cache_creation": 1100,
            },
        )
        kinds = [r["type"] for r in rows if r["traceId"] == "visit-1-t1"]
        self.assertEqual(sorted(kinds), ["GENERATION", "GENERATION", "SPAN", "TOOL"])
        self.assertEqual(
            [r["parentObservationId"] for r in rows if r["type"] == "SPAN"],
            [None, None],
        )
        self.assertEqual(
            next(r for r in rows if r["type"] == "TOOL")["output"], PASSAGE_1
        )
        # A JSON Lines export holds several requests, and a plain list of spans is read too.
        self.assertEqual(rows_from_spans([export(spans[:3]), export(spans[3:])]), rows)
        self.assertEqual(rows_from_spans(iter(spans)), rows)
        # What is no span is passed over.
        junk = export([*spans, None, "span"])
        self.assertEqual(rows_from_spans([junk, None, 7, "export"]), rows)
        # A retrieval of the conventions holds IDs and scores, no text to read.
        found = {"gen_ai.operation.name": "retrieval"}
        found["gen_ai.retrieval.documents"] = [{"id": "d1", "score": 0.9}]
        (row,) = rows_from_spans([span("find", 0, 1, found, trace="t")])
        self.assertEqual(row["type"], "RETRIEVER")
        self.assertEqual(row["output"], [{"id": "d1", "score": 0.9}])

        def snake(item):
            """A span as protobuf's own JSON spells it, with numbers as numbers."""
            if isinstance(item, list):
                return [snake(v) for v in item]
            if not isinstance(item, dict):
                return item
            named = {
                "".join("_" + c.lower() if c.isupper() else c for c in k): snake(v)
                for k, v in item.items()
            }
            for key in ("start_time_unix_nano", "end_time_unix_nano", "int_value"):
                if key in named:
                    named[key] = int(named[key])
            return named

        self.assertEqual(rows_from_spans(snake(export(spans))), rows)


class PullTests(unittest.TestCase):
    def test_an_export_file_is_json_or_json_lines_gzipped_or_not(self):
        spans = genai_spans()
        expected = threads_from_spans(spans, observed_until="2026-11-20T00:00:00Z")[0]
        lines = "\n".join(json.dumps(export(part)) for part in (spans[:3], spans[3:]))
        with tempfile.TemporaryDirectory() as folder:
            whole, split = Path(folder, "spans.json"), Path(folder, "spans.jsonl")
            packed = Path(folder, "spans.jsonl.gz")
            whole.write_text(json.dumps(export(spans), indent=1))
            split.write_text(lines.replace("\n", "\n\n") + "\n")
            with gzip.open(packed, "wt", encoding="utf-8") as stream:
                stream.write(lines)
            for path in (whole, split, str(packed)):
                threads, report = pull(path, until="2026-11-20T00:00:00Z")
                self.assertEqual(threads, expected)
                self.assertEqual(
                    (report["threads"], report["until"]), (1, "2026-11-20T00:00:00Z")
                )
            # Without a time of export the file is taken to have been read just now.
            threads, report = pull(whole)
            self.assertEqual(threads[0]["observed_until"], report["until"])
            read = datetime.fromisoformat(report["until"].replace("Z", "+00:00"))
            self.assertLess((datetime.now(timezone.utc) - read).total_seconds(), 60)
            split.write_text(lines + "\n{broken")
            with self.assertRaisesRegex(ContractError, r"spans\.jsonl:3: invalid JSON"):
                pull(split)

    def test_pulled_threads_are_fitted_as_they_are(self):
        spans = [s for n in range(6) for s in genai_spans(session=f"visit-{n}")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "spans.jsonl")
            path.write_text(json.dumps(export(spans)))
            threads, report = pull(path, until="2026-11-20T00:00:00Z")
        self.assertEqual((report["threads"], report["turns"]), (6, 12))
        # The model, the system prompt, the tools and the time of the read all come from the threads.
        policy = compache.fit(threads)
        self.assertEqual(policy.spec.model_id, MODEL)
        self.assertEqual(policy.artifact["header"]["system_prompt"], SYSTEM)
        self.assertEqual(policy.artifact["header"]["default_tools"], TOOLS)


if __name__ == "__main__":
    unittest.main()
