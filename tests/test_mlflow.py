"""Threads rebuilt from MLflow traces, in the shapes its autologging records."""

import base64
import json
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import compache
from compache import mlflow
from compache.errors import ContractError
from compache.mlflow import (
    _model,
    _spent,
    fetch,
    pull,
    rows_from_traces,
    threads_from_traces,
)
from compache.threads import load_threads, recorded

# 2026-11-10T09:00:00Z
START = 1_794_301_200
SYSTEM = "Answer from the house rules."
LEAD_IN = "I'll look."
QUESTIONS = ["When is the laundry room open?", "Can I book it for a whole day?"]
ANSWERS = ["From seven to ten, every day.", "No, a slot is three hours at most."]
PASSAGE_1 = "The laundry room is open every day from 07:00 to 22:00."
PASSAGE_2 = "A booking covers one slot of three hours, two slots a week per household."
CHAT_TOOLS = [{"type": "function", "function": {"name": "search_rules"}}]
ANTHROPIC_TOOLS = [{"name": "search_rules", "input_schema": {"type": "object"}}]
RESPONSES_TOOLS = [{"type": "function", "name": "search_rules"}]

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


# -- the two encodings of a span -------------------------------------------


def span(span_id, start, end, kind, sent=None, said=None, more=None, parent=None):
    """A span before it is encoded: times in seconds, attributes as plain values."""
    held = {"mlflow.spanType": kind, "mlflow.spanInputs": sent}
    held.update({"mlflow.spanOutputs": said, **(more or {})})
    held = {name: value for name, value in held.items() if value is not None}
    times = {"start": start, "end": end}
    return {"id": span_id, "parent": parent, **times, "attributes": held}


def key(name):
    """An ID as MLflow writes it: its bytes in base64."""
    return base64.b64encode(name.encode()).decode()


def stored(item):
    """A span as a trace's artifact and `Trace.to_dict()` hold it: every attribute as JSON text."""
    return {
        "trace_id": key("trace"),
        "span_id": key(item["id"]),
        "parent_span_id": key(item["parent"]) if item["parent"] else None,
        "name": item["id"],
        "kind": "SPAN_KIND_INTERNAL",
        "start_time_unix_nano": (START + item["start"]) * 10**9,
        "end_time_unix_nano": (START + item["end"]) * 10**9,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {k: json.dumps(v) for k, v in item["attributes"].items()},
        "links": [],
    }


def pair(name, value):
    """A key and its typed value as `traces/get` writes them. None has no value at all."""
    if value is None:
        return {"key": name}
    if isinstance(value, bool):
        return {"key": name, "value": {"bool_value": value}}
    if isinstance(value, int):
        return {"key": name, "value": {"int_value": value}}
    if isinstance(value, float):
        return {"key": name, "value": {"double_value": value}}
    if isinstance(value, list):
        values = [pair(None, v).get("value", {}) for v in value]
        held = {"values": values} if values else {}
        return {"key": name, "value": {"array_value": held}}
    if isinstance(value, dict):
        pairs = [pair(k, v) for k, v in value.items()]
        return {"key": name, "value": {"kvlist_value": {"values": pairs}}}
    return {"key": name, "value": {"string_value": value}}


def served(item):
    """A span as `traces/get` serves it: OTLP in protobuf's JSON."""
    encoded = {
        "trace_id": key("trace"),
        "span_id": key(item["id"]),
        "name": item["id"],
        "kind": "SPAN_KIND_INTERNAL",
        "start_time_unix_nano": (START + item["start"]) * 10**9,
        "end_time_unix_nano": (START + item["end"]) * 10**9,
        "status": {"code": "STATUS_CODE_OK"},
        "attributes": [pair(k, v) for k, v in item["attributes"].items()],
    }
    if item["parent"]:
        encoded["parent_span_id"] = key(item["parent"])
    return encoded


def trace(trace_id, spans, session="s1", form=stored, location="TRACKING_STORE"):
    """A trace as `fetch` yields it: its info from the search, and its spans."""
    metadata = {"mlflow.trace_schema.version": "3"}
    if session:
        metadata["mlflow.trace.session"] = session
    info = {
        "trace_id": trace_id,
        "request_time": at(0),
        "state": "OK",
        "trace_metadata": metadata,
        "tags": {"mlflow.trace.spansLocation": location} if location else {},
    }
    return {"trace_info": info, "spans": [form(item) for item in spans]}


# -- the tool loop as the autologging of each SDK records it ----------------


def chat_message(message):
    kind, *rest = message
    if kind == "call":
        call = {"id": rest[0], "type": "function"}
        call["function"] = {"name": "search_rules", "arguments": "{}"}
        return {"role": "assistant", "content": None, "tool_calls": [call]}
    if kind == "result":
        return {"role": "tool", "tool_call_id": rest[0], "content": rest[1]}
    return {"role": kind, "content": rest[0]}


def openai_call(sent, answered, counts):
    """The arguments and the response of a chat completion, and the attributes beside them."""
    given, output, read, written = counts
    asked = {"model": "gpt-6-luna", "tools": CHAT_TOOLS}
    asked["messages"] = [chat_message(m) for m in [("system", SYSTEM), *sent]]
    cached = {"cached_tokens": read, "cache_write_tokens": written}
    usage = {"completion_tokens": output, "prompt_tokens": given}
    usage.update(total_tokens=given + output, prompt_tokens_details=cached)
    usage["completion_tokens_details"] = {"reasoning_tokens": 10, "text_tokens": None}
    choice = {"finish_reason": "stop", "index": 0, "logprobs": None}
    choice["message"] = {**chat_message(answered), "refusal": None, "audio": None}
    said = {"id": "chatcmpl-1", "choices": [choice], "object": "chat.completion"}
    said.update(model="gpt-6-luna-2026-10-01", usage=usage)
    # MLflow's own count leaves the cache writes out.
    counted = {"input_tokens": given, "output_tokens": output}
    counted.update(total_tokens=given + output, cache_read_input_tokens=read)
    more = {"mlflow.chat.tokenUsage": counted, "mlflow.message.format": "openai"}
    return asked, said, {**more, "mlflow.llm.model": "gpt-6-luna-2026-10-01"}


def blocks(message):
    kind, *rest = message
    if kind == "call":
        use = {"type": "tool_use", "id": rest[0], "name": "search_rules", "input": {}}
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": LEAD_IN}, use],
        }
    if kind == "result":
        found = [{"type": "text", "text": rest[1]}]
        result = {"type": "tool_result", "tool_use_id": rest[0], "content": found}
        return {"role": "user", "content": [result]}
    if kind == "assistant":
        text = {"type": "text", "text": rest[0], "citations": None}
        return {"role": kind, "content": [text]}
    return {"role": kind, "content": rest[0]}


def anthropic_call(sent, answered, counts, usage=True, inclusive=True):
    """The arguments and the response of a Messages call, and the attributes beside them."""
    given, output, read, written = counts
    fresh = given - read - written
    system = SYSTEM
    if len(sent) > 3:
        # The system prompt as blocks, as a call that marks it for caching sends it.
        system = [{"type": "text", "text": SYSTEM}]
        system[0]["cache_control"] = {"type": "ephemeral"}
    asked = {"max_tokens": 1024, "messages": [blocks(m) for m in sent]}
    asked.update(model="claude-sonnet-5-5", system=system, tools=ANTHROPIC_TOOLS)
    said = {"id": "msg_1", "content": blocks(answered)["content"], "role": "assistant"}
    said.update(model="claude-sonnet-5-5", stop_reason="end_turn", type="message")
    if usage:
        said["usage"] = {
            "cache_creation": {"ephemeral_5m_input_tokens": written},
            "cache_creation_input_tokens": written,
            "cache_read_input_tokens": read,
            "input_tokens": fresh,
            "output_tokens": output,
            "server_tool_use": None,
        }
    # Since MLflow 3.12 the cached tokens are counted into the input. Before, they were not.
    counted = {"input_tokens": given if inclusive else fresh, "output_tokens": output}
    counted["total_tokens"] = counted["input_tokens"] + output
    counted.update(cache_read_input_tokens=read, cache_creation_input_tokens=written)
    more = {"mlflow.chat.tokenUsage": counted, "mlflow.chat.tools": CHAT_TOOLS}
    return asked, said, {**more, "mlflow.llm.model": "claude-sonnet-5-5"}


def item(message):
    """A message as an item of OpenAI's Responses API."""
    kind, *rest = message
    if kind == "call":
        call = {"type": "function_call", "call_id": rest[0], "name": "search_rules"}
        return {**call, "arguments": "{}", "id": "fc_1", "status": "completed"}
    if kind == "result":
        return {"type": "function_call_output", "call_id": rest[0], "output": rest[1]}
    if kind == "assistant":
        text = {"type": "output_text", "text": rest[0], "annotations": []}
        return {"type": "message", "role": kind, "content": [text], "id": "msg_1"}
    return {"role": kind, "content": rest[0]}


def responses_call(sent, answered, counts):
    """The arguments and the response of a Responses call."""
    given, output, read, written = counts
    asked = {"model": "gpt-6-luna", "instructions": SYSTEM, "tools": RESPONSES_TOOLS}
    # A first question may be sent as text alone.
    asked["input"] = sent[0][1] if len(sent) == 1 else [item(m) for m in sent]
    cached = {"cached_tokens": read, "cache_write_tokens": written}
    usage = {"input_tokens": given, "input_tokens_details": cached}
    usage.update(output_tokens=output, total_tokens=given + output)
    said = {"id": "resp_1", "object": "response", "output": [item(answered)]}
    said.update(model="gpt-6-luna-2026-10-01", usage=usage, error=None)
    return asked, said, {}


def tool_loop(call=openai_call, session="s1", form=stored, **options):
    """The two turns as traces: a traced function around two model calls and a tool run."""
    traces = []
    for turn, start, end, passage in TURNS:
        name = f"{session}-t{turn}"
        root = f"{name}-root"
        question = {"question": QUESTIONS[turn - 1]}
        spans = [
            span(root, start, end, "AGENT", question, ANSWERS[turn - 1]),
            span(
                f"{name}-tool", start + 2, start + 3, "TOOL", {}, passage, parent=root
            ),
        ]
        for n, (of, began, ended, sent, answered, counts) in enumerate(CALLS):
            if of == turn:
                held = call(sent, answered, counts, **options)
                spans.append(
                    span(f"{name}-c{n}", began, ended, "CHAT_MODEL", *held, parent=root)
                )
        traces.append(trace(f"tr-{name}", spans, session, form))
    return traces


class RebuildTests(unittest.TestCase):
    def loop(self, threads, report, model, tools, lead_in=""):
        """The two turns of the tool loop, whichever SDK made the calls."""
        counted = (report["sessions"], report["threads"], report["turns"])
        self.assertEqual(counted, (1, 1, 2))
        self.assertEqual((report["observations"], report["model_calls"]), (8, 4))
        self.assertEqual(report["models"], {model: 4})
        # Read from the model calls, not from what the traced function took and returned.
        self.assertEqual(report["threads_read_trace_by_trace"], 0)
        self.assertEqual((threads[0]["thread_id"], threads[0]["tools"]), ("s1", tools))
        first, second = load_threads(threads)[0].runs
        self.assertEqual([r.input.text for r in (first, second)], QUESTIONS)
        answers = [lead_in + answer for answer in ANSWERS]
        self.assertEqual([r.output.text for r in (first, second)], answers)
        self.assertEqual([p.text for p in first.retrievals], [PASSAGE_1])
        self.assertEqual([p.text for p in second.retrievals], [PASSAGE_2])
        # A turn runs from the start of its trace, a second before its first call, to its last answer.
        ran = (first.started_at, first.ended_at, first.calls)
        self.assertEqual(ran, (at(0), at(6), 2))
        ran = (second.started_at, second.ended_at, second.calls)
        self.assertEqual(ran, (at(300), at(307), 2))
        # The cache writes are in the provider's own usage only.
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
        self.assertEqual((found["system_prompt"], found["model"]), (SYSTEM, model))

    def test_an_openai_tool_loop_becomes_one_run_per_question_in_both_encodings(self):
        for form in (stored, served):
            with self.subTest(form=form.__name__):
                traces = tool_loop(form=form)
                threads, report = threads_from_traces(traces, observed_until=at(900))
                self.loop(threads, report, "gpt-6-luna-2026-10-01", CHAT_TOOLS)
                self.assertEqual(threads[0]["observed_until"], at(900))
        rows = rows_from_traces(tool_loop())
        self.assertEqual(rows_from_traces(tool_loop(form=served)), rows)

    def test_an_anthropic_tool_loop_reads_the_system_prompt_and_the_blocks(self):
        for form in (stored, served):
            with self.subTest(form=form.__name__):
                traces = tool_loop(anthropic_call, form=form)
                threads, report = threads_from_traces(traces)
                said = LEAD_IN + "\n\n"
                self.loop(threads, report, "claude-sonnet-5-5", ANTHROPIC_TOOLS, said)

    def test_the_responses_api_reads_as_instructions_and_items(self):
        threads, report = threads_from_traces(tool_loop(responses_call, form=served))
        self.loop(threads, report, "gpt-6-luna-2026-10-01", RESPONSES_TOOLS)

    def test_without_the_response_usage_mlflow_counts_are_read_of_either_version(self):
        for inclusive in (True, False):
            with self.subTest(inclusive=inclusive):
                traces = tool_loop(anthropic_call, usage=False, inclusive=inclusive)
                threads, report = threads_from_traces(traces)
                said = LEAD_IN + "\n\n"
                self.loop(threads, report, "claude-sonnet-5-5", ANTHROPIC_TOOLS, said)

    def rag(self, model_call, session="s1", form=stored):
        """Two turns of a LangChain app: a retriever, then a chat model with the passages in its system prompt."""
        earlier = [("user", QUESTIONS[0]), ("assistant", ANSWERS[0])]
        turns = [
            (0, QUESTIONS[0], ANSWERS[0], [PASSAGE_1, PASSAGE_2], []),
            (600, QUESTIONS[1], ANSWERS[1], [PASSAGE_2], earlier),
        ]
        traces = []
        for n, (start, question, answer, found, history) in enumerate(turns):
            name, root = f"{session}-t{n}", f"{session}-t{n}-root"
            documents = [
                {"page_content": text, "metadata": {}, "id": None} for text in found
            ]
            prompt = "\n\n".join([SYSTEM, *found])
            sent = [("system", prompt), *history, ("user", question)]
            times = (start, start + 1, "RETRIEVER")
            spans = [
                span(root, start, start + 5, "CHAIN", {"question": question}, answer),
                span(f"{name}-find", *times, question, documents, parent=root),
                *model_call(f"{name}-llm", start + 1, start + 5, sent, answer, root),
            ]
            traces.append(trace(f"tr-{name}", spans, session, form))
        return traces

    def test_langchain_with_a_retriever_reads_pasted_passages_and_mlflow_counts(self):
        def chat_model(span_id, start, end, sent, answer, root):
            empty = {"tool_calls": None, "refusal": None, "tool_call_id": None}
            asked = {"messages": [{**chat_message(m), **empty} for m in sent]}
            reply = {"role": "assistant", "content": answer, **empty}
            said = {"choices": [{"message": reply, "finish_reason": None}]}
            counted = {"input_tokens": 1227, "output_tokens": 54, "total_tokens": 1281}
            counted["cache_read_input_tokens"] = 1024
            counted["cache_creation_input_tokens"] = 128
            more = {
                "invocation_params": {"model": "gpt-6-luna", "_type": "openai-chat"},
                "mlflow.message.format": "langchain",
                "mlflow.llm.model": "gpt-6-luna",
                "mlflow.chat.tokenUsage": counted,
            }
            times = (start, end, "CHAT_MODEL")
            return [span(span_id, *times, asked, said, more, parent=root)]

        for form in (stored, served):
            with self.subTest(form=form.__name__):
                threads, report = threads_from_traces(self.rag(chat_model, form=form))
                counted = (report["threads"], report["turns_with_pasted_retrieval"])
                self.assertEqual(counted, (1, 2))
                self.assertEqual(report["models"], {"gpt-6-luna": 2})
                self.assertNotIn("tools", threads[0])
                first, second = load_threads(threads)[0].runs
                asked = (first.input.text, first.output.text)
                self.assertEqual(asked, (QUESTIONS[0], ANSWERS[0]))
                found = [p.text for p in first.retrievals]
                self.assertEqual(found, [PASSAGE_1, PASSAGE_2])
                self.assertEqual([p.text for p in second.retrievals], [PASSAGE_2])
                ran = (second.started_at, second.ended_at, second.calls)
                self.assertEqual(ran, (at(600), at(605), 1))
                # MLflow's own count holds the cached tokens inside the input.
                self.assertEqual(
                    dict(second.generation.usage),
                    {
                        "input": 1227,
                        "output": 54,
                        "cache_read": 1024,
                        "cache_write": 128,
                    },
                )
                self.assertTrue(second.generation.cache_read_reported)
                self.assertEqual(recorded(threads)["system_prompt"], SYSTEM)

    def test_a_call_traced_by_langchain_and_by_the_sdk_inside_it_counts_once(self):
        def twice(span_id, start, end, sent, answer, root):
            asked = {"messages": [chat_message(m) for m in sent]}
            said = {"choices": [{"message": {"role": "assistant", "content": answer}}]}
            named = {"mlflow.llm.model": "gpt-6-luna"}
            times = (start, end, "CHAT_MODEL")
            outer = span(span_id, *times, asked, said, named, parent=root)
            held = openai_call(sent[1:], ("assistant", answer), (900, 12, 0, 0))
            return [outer, span(f"{span_id}-sdk", *times, *held, parent=span_id)]

        threads, report = threads_from_traces(self.rag(twice))
        counted = (report["calls_recorded_twice"], report["model_calls"])
        self.assertEqual(counted, (2, 2))
        self.assertEqual([r.calls for r in load_threads(threads)[0].runs], [1, 1])
        # LangChain's span is the one kept, with the model MLflow names on it.
        self.assertEqual(report["models"], {"gpt-6-luna": 2})

    def test_langchain_messages_and_results_as_mlflow_stored_them_up_to_3_10(self):
        def older(span_id, start, end, sent, answer, root):
            # LangChain's own objects: a list of messages for each prompt of a batch, named by type.
            kinds = {"system": "system", "user": "human", "assistant": "ai"}
            asked = [
                [
                    {"content": text, "additional_kwargs": {}, "type": kinds[role]}
                    for role, text in sent
                ]
            ]
            cached = {"cached_tokens": 640, "cache_write_tokens": 128}
            usage = {"completion_tokens": 12, "prompt_tokens": 900}
            usage["prompt_tokens_details"] = cached
            reply = {"content": answer, "type": "ai", "additional_kwargs": {}}
            generation = {"text": answer, "type": "ChatGeneration", "message": reply}
            said = {"generations": [[generation]], "run": None, "type": "LLMResult"}
            said["llm_output"] = {"token_usage": usage}
            said["llm_output"]["model_name"] = "gpt-6-luna-2026-10-01"
            # Before MLflow 3.10 its own count had no cache counters.
            counted = {"input_tokens": 900, "output_tokens": 12, "total_tokens": 912}
            more = {"mlflow.chat.tokenUsage": counted, "mlflow.llm.model": "gpt-6-luna"}
            times = (start, end, "CHAT_MODEL")
            return [span(span_id, *times, asked, said, more, parent=root)]

        for form in (stored, served):
            with self.subTest(form=form.__name__):
                threads, report = threads_from_traces(self.rag(older, form=form))
                self.assertEqual(report["turns"], 2)
                self.assertEqual(report["threads_read_trace_by_trace"], 0)
                # The result names the model that answered and holds the provider's usage.
                self.assertEqual(report["models"], {"gpt-6-luna-2026-10-01": 2})
                first, second = load_threads(threads)[0].runs
                asked = (second.input.text, second.output.text)
                self.assertEqual(asked, (QUESTIONS[1], ANSWERS[1]))
                found = [p.text for p in first.retrievals]
                self.assertEqual(found, [PASSAGE_1, PASSAGE_2])
                self.assertEqual(
                    dict(first.generation.usage),
                    {"input": 900, "output": 12, "cache_read": 640, "cache_write": 128},
                )

    def test_a_completion_of_a_prompt_is_a_model_call_too(self):
        # OpenAI's legacy completions, and a LangChain model that is not a chat model.
        counts = {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
        said = {"choices": [{"text": ANSWERS[0], "index": 0}], "usage": counts}
        asked = {"model": "gpt-6-luna", "prompt": QUESTIONS[0]}
        legacy = span("c1", 0, 2, "LLM", asked, said)
        said = {"choices": [{"message": {"role": "assistant", "content": ANSWERS[1]}}]}
        named = {"invocation_params": {"model_name": "gpt-6-luna"}}
        plain = span("c2", 0, 2, "LLM", [QUESTIONS[1]], said, named)
        for form in (stored, served):
            with self.subTest(form=form.__name__):
                traces = [trace("tr-1", [legacy], "s1", form)]
                traces.append(trace("tr-2", [plain], "s2", form))
                threads, report = threads_from_traces(traces)
                self.assertEqual((report["threads"], report["model_calls"]), (2, 2))
                self.assertEqual(report["models"], {"gpt-6-luna": 2})
                runs = [t.runs[0] for t in load_threads(threads)]
                self.assertEqual([r.input.text for r in runs], QUESTIONS)
                self.assertEqual([r.output.text for r in runs], ANSWERS)
                self.assertEqual(dict(runs[0].generation.usage)["input"], 9)

    def test_the_model_is_the_one_that_answered_and_usage_the_providers_own(self):
        asked, said = {"model": "sonnet"}, {"model": "claude-sonnet-5-5"}
        named = {"invocation_params": {"model_name": "sonnet-as-asked"}}
        both = {**named, "mlflow.llm.model": "claude-sonnet-5-5-v2"}
        self.assertEqual(_model(asked, said, both), "claude-sonnet-5-5")
        self.assertEqual(_model(asked, {"model": None}, both), "claude-sonnet-5-5-v2")
        self.assertEqual(_model(asked, None, {}), "sonnet")
        self.assertEqual(_model([["Hi"]], None, named), "sonnet-as-asked")
        self.assertIsNone(_model(None, None, {}))
        counted = {"input_tokens": 1300, "output_tokens": 30}
        raw = {"prompt_tokens": 1300, "completion_tokens": 30}
        self.assertIs(_spent({"usage": raw}, counted), raw)
        # A usage in a shape of its own is no use: MLflow's count is read instead.
        own = {"usage": {"tokens": 1330}}
        self.assertEqual(_spent(own, counted), {"input": 1300, "output": 30})
        counted["cache_read_input_tokens"] = 1152
        self.assertEqual(
            _spent(own, counted),
            {"input": 148, "output": 30, "input_cached_tokens": 1152},
        )
        self.assertIsNone(_spent({"usage": None}, None))

    def test_spans_that_came_in_over_opentelemetry_are_read_by_their_own_attributes(
        self,
    ):
        asked = [{"role": "user", "parts": [{"type": "text", "content": QUESTIONS[0]}]}]
        said = [
            {"role": "assistant", "parts": [{"type": "text", "content": ANSWERS[0]}]}
        ]
        # MLflow adds its own attributes on the way in. Its inputs leave the system prompt out.
        own = {
            "gen_ai.operation.name": "chat",
            "gen_ai.response.model": "gpt-6-luna-2026-10-01",
            "gen_ai.conversation.id": "s1",
            "gen_ai.system_instructions": [{"type": "text", "content": SYSTEM}],
            "gen_ai.input.messages": json.dumps(asked),
            "gen_ai.output.messages": json.dumps(said),
            "gen_ai.usage.input_tokens": 1200,
            "gen_ai.usage.output_tokens": 20,
            "gen_ai.usage.cache_read.input_tokens": 1024,
            "mlflow.llm.model": "gpt-6-luna",
        }
        chat = span("chat", 1, 4, "CHAT_MODEL", asked, said, own)
        # OpenInference's numbered messages, where MLflow took its inputs from `input.value`.
        earlier = [SYSTEM, QUESTIONS[0], ANSWERS[0], QUESTIONS[1]]
        own = {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-6-luna-2026-10-01",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": ANSWERS[1],
            "llm.token_count.prompt": 1300,
            "llm.token_count.completion": 30,
            "llm.token_count.prompt_details.cache_read": 1152,
        }
        for n, (role, text) in enumerate(
            zip(("system", "user", "assistant", "user"), earlier)
        ):
            own[f"llm.input_messages.{n}.message.role"] = role
            own[f"llm.input_messages.{n}.message.content"] = text
        numbered = span("llm", 301, 304, "LLM", {"messages": []}, ANSWERS[1], own)
        for form in (stored, served):
            with self.subTest(form=form.__name__):
                traces = [trace("tr-1", [chat], form=form)]
                traces.append(trace("tr-2", [numbered], form=form))
                threads, report = threads_from_traces(traces)
                self.assertEqual((report["threads"], report["turns"]), (1, 2))
                first, second = load_threads(threads)[0].runs
                self.assertEqual([r.input.text for r in (first, second)], QUESTIONS)
                self.assertEqual([r.output.text for r in (first, second)], ANSWERS)
                self.assertEqual(
                    dict(second.generation.usage),
                    {"input": 1300, "output": 30, "cache_read": 1152, "cache_write": 0},
                )
                self.assertEqual(first.generation.model, "gpt-6-luna-2026-10-01")
                self.assertEqual(recorded(threads)["system_prompt"], SYSTEM)

    def test_traces_are_taken_in_every_form_mlflow_hands_them_out(self):
        traces = tool_loop()
        rows = rows_from_traces(traces)
        call = next(r for r in rows if r["id"] == key("s1-t1-c1"))
        held = ("traceId", "parentObservationId", "type", "sessionId", "model")
        self.assertEqual(
            [call[name] for name in held],
            [
                "tr-s1-t1",
                key("s1-t1-root"),
                "GENERATION",
                "s1",
                "gpt-6-luna-2026-10-01",
            ],
        )
        self.assertEqual((call["startTime"], call["endTime"]), (at(3), at(6)))
        first = call["input"]["messages"][0]
        self.assertEqual(first, {"role": "system", "content": SYSTEM})
        self.assertEqual(call["output"]["content"], ANSWERS[0])
        kinds = [r["type"] for r in rows if r["traceId"] == "tr-s1-t1"]
        self.assertEqual(sorted(kinds), ["GENERATION", "GENERATION", "SPAN", "TOOL"])
        # One reply of `traces/get`, a reply that holds several, and `Trace.to_dict()`.
        self.assertEqual(rows_from_traces([{"trace": t} for t in traces]), rows)
        reply = {"traces": traces, "next_page_token": "x"}
        self.assertEqual(rows_from_traces(reply), rows)
        exported = [
            {"info": t["trace_info"], "data": {"spans": t["spans"]}} for t in traces
        ]
        self.assertEqual(rows_from_traces(iter(exported)), rows)
        # What is no trace or no span is passed over.
        traces[0]["spans"] += [None, "span"]
        self.assertEqual(rows_from_traces([*traces, None, "trace"]), rows)

    def test_what_has_no_session_or_is_no_model_call_is_left_out(self):
        traces = tool_loop()
        root = "s1-t1-root"
        embedding = span("embed", 2, 3, "EMBEDDING", {"input": "laundry"}, parent=root)
        unstarted = stored(span("unstarted", 0, 1, "CHAT_MODEL", parent=root))
        del unstarted["start_time_unix_nano"]
        traces[0]["spans"] += [stored(embedding), unstarted]
        traces += tool_loop(session=None)
        threads, report = threads_from_traces(traces)
        self.assertEqual([t["thread_id"] for t in threads], ["s1"])
        counted = (report["observations"], report["model_calls"])
        self.assertEqual(counted, (17, 4))
        self.assertEqual(report["traces_without_session"], 2)


class FetchTests(unittest.TestCase):
    def reply(self, status, body, headers=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return status, headers or {}, raw

    def served_by(self, replies, seen):
        """A stand-in for the HTTP call that answers each address from a list of replies."""

        def send(url, headers, timeout, body=None):
            seen.append((url, headers, json.loads(body) if body else None))
            route = url.split("?")[0].rsplit("/", 1)[-1]
            return replies[route].pop(0)

        return send

    def info(self, trace_id, location="TRACKING_STORE"):
        return trace(trace_id, [], location=location)["trace_info"]

    def test_pages_follow_the_token_and_a_rate_limit_is_waited_out(self):
        spans = [served(span("root", 0, 1, "CHAIN"))]
        page = {"traces": [self.info("tr-1"), self.info("tr 2")]}
        replies = {
            "search": [
                self.reply(429, {"message": "slow down"}, {"Retry-After": "7"}),
                self.reply(200, {**page, "next_page_token": "next"}),
                self.reply(200, {"traces": [self.info("tr-3")]}),
            ],
            "get": [self.reply(200, {"trace": {"trace_info": {}, "spans": spans}})] * 3,
        }
        seen, waits = [], []
        with mock.patch.object(mlflow, "_send", self.served_by(replies, seen)):
            traces = list(
                fetch(
                    tracking_uri="https://mlflow.example/",
                    experiment_ids=["3", 4],
                    # A time without a zone is read as UTC.
                    since="2026-11-01T00:00:00",
                    until="2026-11-20T01:00:00+01:00",
                    filter="tags.app = 'desk'",
                    token="secret",
                    pause=waits.append,
                )
            )
        found = [t["trace_info"]["trace_id"] for t in traces]
        self.assertEqual(found, ["tr-1", "tr 2", "tr-3"])
        self.assertEqual([t["spans"] for t in traces], [spans] * 3)
        self.assertEqual(waits, [7.0])
        url, headers, body = seen[0]
        self.assertEqual(url, "https://mlflow.example/api/3.0/mlflow/traces/search")
        self.assertEqual(headers["Authorization"], "Bearer secret")
        since = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
        until = int(datetime(2026, 11, 20, tzinfo=timezone.utc).timestamp() * 1000)
        places = [
            {"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": n}}
            for n in ("3", "4")
        ]
        self.assertEqual(
            body,
            {
                "locations": places,
                "order_by": ["timestamp_ms ASC"],
                "max_results": 100,
                "filter": f"tags.app = 'desk' AND trace.timestamp_ms >= {since} "
                f"AND trace.timestamp_ms < {until}",
            },
        )
        self.assertEqual(
            [url.split("/mlflow/")[1] for url, _, _ in seen[2:]],
            [
                "traces/get?trace_id=tr-1&allow_partial=true",
                "traces/get?trace_id=tr%202&allow_partial=true",
                "traces/search",
                "traces/get?trace_id=tr-3&allow_partial=true",
            ],
        )
        self.assertEqual(seen[4][2]["page_token"], "next")

    def test_an_experiment_is_named_by_its_name_too(self):
        found = {"experiment": {"experiment_id": "12", "name": "chat app"}}
        replies = {
            "get-by-name": [self.reply(200, found)],
            "search": [self.reply(200, {"traces": []})],
        }
        seen = []
        options = {"tracking_uri": "http://localhost:5000", "pause": lambda s: None}
        with mock.patch.object(mlflow, "_send", self.served_by(replies, seen)):
            self.assertEqual(
                list(fetch(experiment_ids=["chat app", "7"], **options)), []
            )
        self.assertIn("get-by-name?experiment_name=chat%20app", seen[0][0])
        places = [place["mlflow_experiment"] for place in seen[1][2]["locations"]]
        self.assertEqual(places, [{"experiment_id": "12"}, {"experiment_id": "7"}])
        missing = {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No such thing"}
        replies = {"get-by-name": [self.reply(404, missing)]}
        with mock.patch.object(mlflow, "_send", self.served_by(replies, [])):
            with self.assertRaisesRegex(ContractError, "no experiment named 'chat'"):
                list(fetch(experiment_ids="chat", **options))

    def test_spans_the_store_does_not_hold_come_from_the_trace_artifact(self):
        spans = [stored(span("root", 0, 1, "CHAIN"))]
        elsewhere = {"error_code": "INTERNAL_ERROR"}
        elsewhere["message"] = "Trace data not stored in tracking store"
        missing = {"error_code": "RESOURCE_DOES_NOT_EXIST"}
        missing["message"] = "Trace data not found"
        found = [self.info("tr-1"), self.info("tr-2", None)]
        found.append(self.info("tr-3", "ARTIFACT_REPO"))
        replies = {
            "search": [
                self.reply(200, {"traces": found, "next_page_token": "next"}),
                # A trace logged during the read can push one already read onto the next page.
                self.reply(200, {"traces": [self.info("tr-3")]}),
            ],
            "get": [self.reply(500, elsewhere)],
            "get-trace-artifact": [
                self.reply(200, {"spans": spans}),
                self.reply(200, {"spans": spans}),
                self.reply(404, missing),
            ],
        }
        seen, waits = [], []
        options = {"tracking_uri": "http://localhost:5000", "pause": waits.append}
        with mock.patch.object(mlflow, "_send", self.served_by(replies, seen)):
            # One experiment may be named by its ID alone.
            traces = list(fetch(experiment_ids="17", **options))
        found = [t["trace_info"]["trace_id"] for t in traces]
        self.assertEqual(found, ["tr-1", "tr-2", "tr-3"])
        self.assertEqual([t["spans"] for t in traces], [spans, spans, []])
        self.assertEqual(waits, [])
        self.assertEqual(
            [url.split("/mlflow/")[1] for url, _, _ in seen],
            [
                "traces/search",
                "traces/get?trace_id=tr-1&allow_partial=true",
                "get-trace-artifact?request_id=tr-1",
                "get-trace-artifact?request_id=tr-2",
                "get-trace-artifact?request_id=tr-3",
                "traces/search",
            ],
        )
        artifact = "http://localhost:5000/ajax-api/3.0/mlflow/get-trace-artifact"
        self.assertTrue(seen[2][0].startswith(artifact))
        places = [place["mlflow_experiment"] for place in seen[0][2]["locations"]]
        self.assertEqual(places, [{"experiment_id": "17"}])
        self.assertNotIn("Authorization", seen[0][1])

    def test_refusals_say_what_is_wrong(self):
        missing = {"error_code": "RESOURCE_DOES_NOT_EXIST"}
        missing["message"] = "Experiment 9 does not exist."
        refused = {"error_code": "INVALID_PARAMETER_VALUE"}
        refused["message"] = "Invalid clause(s) in filter string"
        page = b"<!doctype html><title>404 Not Found</title>"
        for reply, words in (
            (self.reply(401, {}), r"rejected the MLflow token \(401\)"),
            (self.reply(404, page), "needs MLflow 3"),
            (
                self.reply(404, missing),
                "refused the query: Experiment 9 does not exist",
            ),
            (
                self.reply(400, refused),
                r"refused the query: Invalid clause\(s\) in filter",
            ),
        ):
            with mock.patch.object(mlflow, "_send", lambda *a, reply=reply, **k: reply):
                with self.assertRaisesRegex(ContractError, words):
                    list(fetch(tracking_uri="http://localhost:5000"))
        waits = []
        busy = self.reply(503, b"busy")
        with mock.patch.object(mlflow, "_send", lambda *a, **k: busy):
            with self.assertRaisesRegex(ContractError, r"keeps failing \(503\): busy"):
                list(fetch(tracking_uri="http://localhost:5000", pause=waits.append))
        self.assertEqual(waits, [1.0] * 8)
        with self.assertRaisesRegex(ContractError, "expected an ISO timestamp"):
            list(fetch(tracking_uri="http://localhost:5000", since="yesterday"))
        with self.assertRaisesRegex(ContractError, "http address of a tracking server"):
            list(fetch(tracking_uri="sqlite:///mlflow.db"))
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ContractError, "MLFLOW_TRACKING_URI"):
                list(fetch())

    def test_the_server_and_the_token_default_to_mlflows_own_variables(self):
        seen = []
        replies = {"search": [self.reply(200, {})]}
        variables = {"MLFLOW_TRACKING_URI": "http://tracking:5000"}
        variables["MLFLOW_TRACKING_TOKEN"] = "t0ken"
        with mock.patch.dict("os.environ", variables, clear=True):
            with mock.patch.object(mlflow, "_send", self.served_by(replies, seen)):
                self.assertEqual(list(fetch(page=9000)), [])
        url, headers, body = seen[0]
        self.assertEqual(url, "http://tracking:5000/api/3.0/mlflow/traces/search")
        self.assertEqual(headers["Authorization"], "Bearer t0ken")
        # The default experiment, and no more traces a page than the server allows.
        place = body["locations"][0]["mlflow_experiment"]
        self.assertEqual(place, {"experiment_id": "0"})
        self.assertEqual((body["max_results"], "filter" in body), (500, False))

    def test_a_request_with_a_body_is_posted_and_a_refusal_is_its_status(self):
        class Handler(BaseHTTPRequestHandler):
            def answer(self, status, text):
                self.send_response(status)
                self.send_header("Content-Length", str(len(text)))
                self.send_header("X-Method", self.command)
                self.end_headers()
                self.wfile.write(text)

            def do_GET(self):
                self.answer(404 if self.path == "/missing" else 200, self.path.encode())

            def do_POST(self):
                sent = self.rfile.read(int(self.headers["Content-Length"]))
                self.answer(200, sent + self.headers["Authorization"].encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host = f"http://127.0.0.1:{server.server_port}"
        token = {"Authorization": "Bearer t"}
        status, headers, raw = mlflow._send(host, token, 5, b'{"a": 1}')
        self.assertEqual((status, raw), (200, b'{"a": 1}Bearer t'))
        self.assertEqual(headers["X-Method"], "POST")
        status, headers, raw = mlflow._send(host + "/get?trace_id=tr-1", {}, 5)
        self.assertEqual((status, raw), (200, b"/get?trace_id=tr-1"))
        self.assertEqual(headers["X-Method"], "GET")
        self.assertEqual(mlflow._send(host + "/missing", {}, 5)[0], 404)


class PullTests(unittest.TestCase):
    def test_pulled_threads_are_fitted_as_they_are(self):
        traces = []
        for n in range(6):
            traces += tool_loop(anthropic_call, session=f"s{n}")
        threads, report = pull(traces=traces, until="2026-11-20T00:00:00Z")
        self.assertEqual((report["threads"], report["turns"]), (6, 12))
        bounds = (report["since"], report["until"])
        self.assertEqual(bounds, (None, "2026-11-20T00:00:00Z"))
        # The model, the system prompt, the tools and the time of the read all come from the threads.
        policy = compache.fit(threads)
        self.assertEqual(policy.spec.model_id, "claude-sonnet-5-5")
        self.assertEqual(policy.artifact["header"]["system_prompt"], SYSTEM)
        self.assertEqual(policy.artifact["header"]["default_tools"], ANTHROPIC_TOOLS)

    def test_a_pull_reads_the_last_days_of_the_named_experiments(self):
        traces = tool_loop(form=served)
        infos = [t["trace_info"] for t in traces]
        replies = [(200, {}, json.dumps({"traces": infos}).encode())]
        replies += [(200, {}, json.dumps({"trace": t}).encode()) for t in traces]
        bodies = []

        def send(url, headers, timeout, body=None):
            bodies.append(json.loads(body) if body else None)
            return replies.pop(0)

        server = {"tracking_uri": "http://localhost:5000", "experiment_ids": ["12"]}
        with mock.patch.object(mlflow, "_send", send):
            threads, report = pull(days=2, **server)
        self.assertEqual((report["threads"], report["turns"]), (1, 2))
        self.assertEqual(threads[0]["observed_until"], report["until"])
        began = datetime.fromisoformat(report["since"].replace("Z", "+00:00"))
        ended = datetime.fromisoformat(report["until"].replace("Z", "+00:00"))
        self.assertEqual((ended - began).days, 2)
        since = f"trace.timestamp_ms >= {int(began.timestamp() * 1000)}"
        self.assertEqual(bodies[0]["filter"], since)
        place = bodies[0]["locations"][0]["mlflow_experiment"]
        self.assertEqual(place, {"experiment_id": "12"})


if __name__ == "__main__":
    unittest.main()
