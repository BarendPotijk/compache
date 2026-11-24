"""Provider request and response shapes, as compache reads and rewrites them."""

import copy
import json
import unittest

from compache import wire
from compache.live import _segments
from compache.optimizer.recovery import REOPEN_INSTRUCTIONS, REOPEN_TOOL

PASSAGE = "The warranty covers parts and labour for two years."

CHAT = {
    "model": "gpt-6-luna",
    "tools": [
        {"type": "function", "function": {"name": "search_docs", "parameters": {}}}
    ],
    "messages": [
        {"role": "system", "content": "Answer from the documents."},
        {"role": "user", "content": "How long is the warranty?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": PASSAGE},
        {"role": "assistant", "content": "Two years."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "And claims?"},
                {"type": "image_url", "image_url": {}},
            ],
        },
    ],
}
CLAUDE = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 512,
    "system": [{"type": "text", "text": "Answer from the documents."}],
    "tools": [{"name": "search_docs", "input_schema": {}}],
    "messages": [
        {"role": "user", "content": "How long is the warranty?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Search first.", "signature": "s"},
                {"type": "text", "text": "I will search."},
                {"type": "tool_use", "id": "t1", "name": "search_docs", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": PASSAGE}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Two years."}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": "And claims?"}],
        },
    ],
}
RESPONSES = {
    "model": "gpt-6-luna",
    "instructions": "Answer from the documents.",
    "tools": [{"type": "function", "name": "search_docs", "parameters": {}}],
    "input": [
        {"role": "user", "content": "How long is the warranty?"},
        {"type": "reasoning", "id": "rs1", "summary": []},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "search_docs",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "c1", "output": PASSAGE},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Two years."}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "And claims?"}]},
    ],
}
GEMINI = {
    "systemInstruction": {"parts": [{"text": "Answer from the documents."}]},
    "tools": [{"functionDeclarations": [{"name": "search_docs"}]}],
    "contents": [
        {"role": "user", "parts": [{"text": "How long is the warranty?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {"name": "search_docs", "args": {}},
                    "thoughtSignature": "s",
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search_docs",
                        "response": {"result": PASSAGE},
                    }
                }
            ],
        },
        {"role": "model", "parts": [{"text": "Two years."}]},
        {"role": "user", "parts": [{"text": "And claims?"}]},
    ],
}
BODIES = {
    "/v1/chat/completions": CHAT,
    "/v1/messages": CLAUDE,
    "/v1/responses": RESPONSES,
    "/v1beta/models/gemini-2.5-flash-lite:generateContent": GEMINI,
}


class ScanTests(unittest.TestCase):
    def test_every_api_is_read_as_the_same_conversation(self):
        for path, body in BODIES.items():
            with self.subTest(path=path):
                dialect = wire.dialect_for(path)
                scan = dialect.scan(body)
                self.assertEqual(scan.system, "Answer from the documents.")
                self.assertEqual(len(scan.tools), 1)
                self.assertTrue(scan.awaiting)
                first, second = _segments(scan.slots)
                self.assertEqual(first.question, "How long is the warranty?")
                self.assertIn(PASSAGE, first.retrievals[0].text)
                self.assertTrue(first.answer.endswith("Two years."))
                self.assertEqual(second.question, "And claims?")
                self.assertEqual((second.retrievals, second.outputs), ([], []))
                self.assertIn("search_docs", dialect.tool_names(body))

    def test_an_answer_that_says_nothing_still_ends_its_turn(self):
        def emptied(body, key, index, value):
            changed = copy.deepcopy(body)
            changed[key][index] = value
            return changed

        silent = {
            "/v1/chat/completions": [
                emptied(CHAT, "messages", 4, {"role": "assistant", "content": content})
                for content in ("", None, [])
            ],
            "/v1/messages": [
                emptied(
                    CLAUDE, "messages", 3, {"role": "assistant", "content": content}
                )
                for content in (
                    "",
                    [],
                    [{"type": "thinking", "thinking": "Hm.", "signature": "s"}],
                )
            ],
            "/v1/responses": [
                emptied(
                    RESPONSES,
                    "input",
                    4,
                    {"type": "message", "role": "assistant", "content": content},
                )
                for content in ([], [{"type": "output_text", "text": ""}])
            ],
            "/v1beta/models/m:generateContent": [
                emptied(GEMINI, "contents", 3, {"role": "model", "parts": parts})
                for parts in ([], [{"text": ""}], [{"text": "Hm.", "thought": True}])
            ],
        }
        for path, bodies in silent.items():
            for body in bodies:
                with self.subTest(path=path, body=body):
                    first, second = _segments(wire.dialect_for(path).scan(body).slots)
                    self.assertNotIn("Two years.", first.answer)
                    self.assertEqual(
                        (first.outputs[-1].text, first.outputs[-1].closes), ("", True)
                    )
                    self.assertEqual(second.question, "And claims?")

    def test_the_path_says_which_api_it_is(self):
        self.assertIs(wire.dialect_for("/api/v1/chat/completions?x=1"), wire.OpenAIChat)
        self.assertIs(wire.dialect_for("/v1/messages/"), wire.Anthropic)
        self.assertIs(
            wire.dialect_for("/v1beta/models/m:streamGenerateContent?alt=sse"),
            wire.Gemini,
        )
        for other in ("/v1/embeddings", "/v1/messages/count_tokens", "/v1/models", "/"):
            self.assertIsNone(wire.dialect_for(other))

    def test_a_request_that_does_not_end_on_the_user_is_not_awaiting_an_answer(self):
        prefilled = {
            **CLAUDE,
            "messages": [*CLAUDE["messages"], {"role": "assistant", "content": "The"}],
        }
        self.assertFalse(wire.Anthropic.scan(prefilled).awaiting)
        stored = [
            {"role": "user", "content": "First?"},
            {"type": "item_reference", "id": "msg_1"},
            {"role": "user", "content": "Next?"},
        ]
        for kept in (
            {"previous_response_id": "r1", "input": "next"},
            {"input": stored},
        ):
            with self.assertRaises(ValueError):
                wire.OpenAIResponses.scan(kept)

    def test_an_instruction_inside_a_conversation_is_nobody_s_turn(self):
        # As Claude Code sends it: after the user's message, with the cache mark of the request.
        told = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Today is Tuesday.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        body = {**CLAUDE, "messages": [*CLAUDE["messages"], told]}
        before, after = wire.Anthropic.scan(CLAUDE), wire.Anthropic.scan(body)
        self.assertEqual(before.slots, after.slots)
        self.assertEqual(before.awaiting, after.awaiting)
        self.assertIn("Today is Tuesday.", after.system)
        if not after.awaiting:
            body = {
                **body,
                "messages": [
                    *body["messages"],
                    {"role": "user", "content": "Go on."},
                    told,
                ],
            }
        noted = wire.Anthropic.with_note(body, "[ask]")
        self.assertEqual(noted["messages"][-1], told)
        self.assertIn("[ask]", json.dumps(noted["messages"][-2]))

    def test_replacing_text_copies_only_what_changes(self):
        for path, body in BODIES.items():
            with self.subTest(path=path):
                dialect, before = wire.dialect_for(path), copy.deepcopy(body)
                slot = next(
                    s for s in dialect.scan(body).slots if s.kind == "retrieval"
                )
                changed = wire.set_text(dialect, body, slot.path, "[notice]")
                self.assertEqual(body, before)
                self.assertNotIn(PASSAGE, json.dumps(changed))
                self.assertIn("[notice]", json.dumps(changed))
                again = next(
                    s for s in dialect.scan(changed).slots if s.kind == "retrieval"
                )
                self.assertIn("[notice]", again.text)
                # Everything but the changed message is the very same object.
                key = (
                    "messages"
                    if "messages" in body
                    else "input"
                    if "input" in body
                    else "contents"
                )
                same = sum(a is b for a, b in zip(body[key], changed[key]))
                self.assertEqual(same, len(body[key]) - 1)
                self.assertIs(changed["tools"], body["tools"])


class RewriteTests(unittest.TestCase):
    def test_a_note_goes_last_and_the_cache_mark_before_it(self):
        marked = wire.Anthropic.with_cache_mark(CLAUDE)
        noted = wire.Anthropic.with_note(marked, "NOTE")
        last = noted["messages"][-1]["content"]
        self.assertEqual([b.get("text") for b in last], ["And claims?", "NOTE"])
        self.assertEqual(["cache_control" in b for b in last], [True, False])
        self.assertNotIn("cache_control", json.dumps(CLAUDE))
        # An application's own marks are left as they are.
        own = {
            **CLAUDE,
            "system": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ],
        }
        self.assertIs(wire.Anthropic.with_cache_mark(own), own)
        # Chat Completions: only a model that has to be told is told.
        self.assertIs(wire.OpenAIChat.with_cache_mark(CHAT), CHAT)
        claude = wire.OpenAIChat.with_cache_mark(
            {**CHAT, "model": "anthropic/claude-haiku-5.5"}
        )
        self.assertEqual(
            claude["messages"][-1]["content"][0],
            {"type": "text", "text": "And claims?"},
        )
        self.assertIn("cache_control", claude["messages"][-1]["content"][-1])
        text = {
            **CHAT,
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        self.assertEqual(
            wire.OpenAIChat.with_cache_mark(text)["messages"][0]["content"],
            [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}],
        )
        self.assertEqual(
            wire.OpenAIChat.with_note(CHAT, "NOTE")["messages"][-1],
            {"role": "user", "content": "NOTE"},
        )
        self.assertEqual(
            wire.OpenAIResponses.with_note(RESPONSES, "NOTE")["input"][-1]["content"],
            "NOTE",
        )
        self.assertEqual(
            wire.Gemini.with_note(GEMINI, "NOTE")["contents"][-1]["parts"][-1],
            {"text": "NOTE"},
        )

    def test_a_note_is_never_inside_what_a_request_asks_to_cache(self):
        # A mark for the whole request would cache the note with it. The next request has no
        # note, so nothing it wrote could be read: the mark is put before the note instead.
        whole = {**CLAUDE, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        noted = wire.Anthropic.with_note(whole, "NOTE")
        self.assertNotIn("cache_control", noted)
        last = noted["messages"][-1]["content"]
        self.assertEqual(last[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(last[1], {"type": "text", "text": "NOTE"})
        self.assertIn("cache_control", whole)
        chat = wire.OpenAIChat.with_note(
            {**CHAT, "cache_control": {"type": "ephemeral"}}, "NOTE"
        )
        self.assertNotIn("cache_control", chat)
        self.assertIn("cache_control", chat["messages"][-2]["content"][-1])
        self.assertEqual(chat["messages"][-1], {"role": "user", "content": "NOTE"})
        # The same holds for a ping: the answer is marked, the full stop after it is not.
        ping = wire.Anthropic.follow_up(whole, "Thirty days.", ".", 1)
        self.assertNotIn("cache_control", ping)
        self.assertEqual(
            ping["messages"][-2]["content"][0]["cache_control"]["ttl"], "1h"
        )

    def test_the_tool_to_fetch_a_part_back_is_added_in_each_api_s_own_form(self):
        for path, body in BODIES.items():
            with self.subTest(path=path):
                dialect = wire.dialect_for(path)
                out = dialect.with_tool(body, REOPEN_TOOL, REOPEN_INSTRUCTIONS)
                self.assertIn("reopen", dialect.tool_names(out))
                self.assertIn("search_docs", dialect.tool_names(out))
                scan = dialect.scan(out)
                self.assertTrue(scan.system.startswith("Answer from the documents."))
                self.assertTrue(scan.system.endswith(REOPEN_INSTRUCTIONS))
                self.assertEqual(len(body["tools"]), 1)
        bare = wire.OpenAIChat.with_tool(
            {"messages": [{"role": "user", "content": "Hi"}]}, REOPEN_TOOL, "HOW"
        )
        self.assertEqual(bare["messages"][0], {"role": "system", "content": "HOW"})
        gemini = wire.Gemini.with_tool(GEMINI, REOPEN_TOOL, "HOW")
        self.assertNotIn("additionalProperties", json.dumps(gemini["tools"]))

    def test_a_ping_is_the_conversation_plus_its_answer_and_one_token(self):
        ping = wire.Anthropic.follow_up(CLAUDE, "Thirty days.", ".", 1)
        self.assertEqual(ping["max_tokens"], 1)
        self.assertEqual(ping["messages"][-1], {"role": "user", "content": "."})
        self.assertEqual(
            ping["messages"][-2]["content"],
            [
                {
                    "type": "text",
                    "text": "Thirty days.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(len(CLAUDE["messages"]), 5)
        # A thinking budget has to fit under the output limit, so there is no short request to send.
        self.assertIsNone(
            wire.Anthropic.follow_up(
                {**CLAUDE, "thinking": {"type": "enabled", "budget_tokens": 2000}},
                "x",
                ".",
                1,
            )
        )
        chat = wire.OpenAIChat.follow_up(
            {**CHAT, "stream": True, "max_completion_tokens": 900}, "x", ".", 1
        )
        self.assertEqual(
            (chat["max_completion_tokens"], "stream" in chat, "max_tokens" in chat),
            (1, False, False),
        )
        self.assertEqual(
            wire.OpenAIResponses.follow_up(RESPONSES, "x", ".", 1)["max_output_tokens"],
            16,
        )
        gemini = wire.Gemini.follow_up(GEMINI, "x", ".", 1)
        self.assertEqual(gemini["generationConfig"], {"maxOutputTokens": 1})
        self.assertEqual(
            gemini["contents"][-2], {"role": "model", "parts": [{"text": "x"}]}
        )

    def test_a_schema_is_found_and_replaced_where_each_api_keeps_it(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        wider = {"type": "object", "properties": {"answer": {}, "more": {}}}
        cases = {
            wire.OpenAIChat: {
                **CHAT,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": schema},
                },
            },
            wire.Anthropic: {
                **CLAUDE,
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            },
            wire.OpenAIResponses: {
                **RESPONSES,
                "text": {
                    "format": {"type": "json_schema", "name": "a", "schema": schema}
                },
            },
            wire.Gemini: {
                **GEMINI,
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            },
        }
        for dialect, body in cases.items():
            with self.subTest(dialect=dialect.name):
                self.assertEqual(dialect.schema(body), schema)
                self.assertEqual(
                    dialect.schema(dialect.with_schema(body, wider)), wider
                )
                self.assertEqual(dialect.schema(body), schema)
        self.assertIsNone(
            wire.OpenAIChat.schema({**CHAT, "response_format": {"type": "json_object"}})
        )
        self.assertIsNone(wire.Anthropic.schema(CLAUDE))


class ReplyTests(unittest.TestCase):
    def test_a_reply_is_read_the_same_from_each_api(self):
        replies = {
            wire.OpenAIChat: {
                "model": "m",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Thirty days.",
                            "tool_calls": [
                                {
                                    "id": "c9",
                                    "function": {
                                        "name": "reopen",
                                        "arguments": '{"reference": "ctx:1"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.002},
            },
            wire.Anthropic: {
                "model": "m",
                "content": [
                    {"type": "text", "text": "Thirty days."},
                    {
                        "type": "tool_use",
                        "id": "c9",
                        "name": "reopen",
                        "input": {"reference": "ctx:1"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
            wire.OpenAIResponses: {
                "model": "m",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Thirty days."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c9",
                        "name": "reopen",
                        "arguments": '{"reference": "ctx:1"}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
            wire.Gemini: {
                "modelVersion": "m",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Thirty days."},
                                {
                                    "functionCall": {
                                        "id": "c9",
                                        "name": "reopen",
                                        "args": {"reference": "ctx:1"},
                                    }
                                },
                            ],
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
            },
        }
        for dialect, body in replies.items():
            with self.subTest(dialect=dialect.name):
                reply = dialect.reply(body)
                self.assertEqual((reply.text, reply.model), ("Thirty days.", "m"))
                self.assertEqual(
                    reply.calls,
                    [
                        {
                            "id": "c9",
                            "name": "reopen",
                            "arguments": {"reference": "ctx:1"},
                        }
                    ],
                )
                self.assertEqual(wire.get(body, reply.paths[0]), "Thirty days.")
                self.assertIsNotNone(reply.usage)
                without = dialect.without_tool(body, "reopen")
                self.assertEqual(dialect.reply(without).calls, [])
                self.assertEqual(dialect.reply(without).text, "Thirty days.")
                self.assertEqual(len(dialect.reply(body).calls), 1)
        self.assertEqual(wire.OpenAIChat.reply(replies[wire.OpenAIChat]).cost, 0.002)
        chat = wire.OpenAIChat.without_tool(replies[wire.OpenAIChat], "reopen")
        self.assertEqual(chat["choices"][0]["finish_reason"], "stop")
        claude = wire.Anthropic.without_tool(replies[wire.Anthropic], "reopen")
        self.assertEqual(claude["stop_reason"], "end_turn")

    def test_a_reply_the_output_limit_ended_says_so(self):
        cut = {
            "/v1/chat/completions": {
                "choices": [{"message": {"content": "A"}, "finish_reason": "length"}]
            },
            "/v1/messages": {
                "content": [{"type": "text", "text": "A"}],
                "stop_reason": "max_tokens",
            },
            "/v1/responses": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
            "/v1beta/models/m:generateContent": {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "A"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
        }
        whole = {
            "/v1/chat/completions": {
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]
            },
            "/v1/messages": {
                "content": [{"type": "text", "text": "A"}],
                "stop_reason": "end_turn",
            },
            "/v1/responses": {"status": "completed", "output": []},
            "/v1beta/models/m:generateContent": {
                "candidates": [
                    {"content": {"parts": [{"text": "A"}]}, "finishReason": "STOP"}
                ]
            },
        }
        for path in cut:
            with self.subTest(path=path):
                dialect = wire.dialect_for(path)
                self.assertTrue(dialect.reply(cut[path]).cut)
                self.assertFalse(dialect.reply(whole[path]).cut)

    def test_a_text_block_left_empty_is_dropped(self):
        body = {
            "content": [
                {"type": "text", "text": "Thirty days."},
                {"type": "text", "text": ""},
            ]
        }
        self.assertEqual(
            wire.Anthropic.prune(body)["content"],
            [{"type": "text", "text": "Thirty days."}],
        )
        only = {"content": [{"type": "text", "text": ""}]}
        self.assertIs(wire.Anthropic.prune(only), only)

    def test_server_sent_events_are_split_wherever_the_bytes_were_cut(self):
        raw = (
            b'event: a\ndata: {"n": 1}\n\n'
            b": keep-alive\n\n"
            b'data: {"n": 2}\r\n\r\n'
            b"data: [DONE]\n\n"
        )
        whole = list(wire.sse_events([raw]))
        self.assertEqual(
            [(name, data) for _, name, data in whole],
            [("a", {"n": 1}), (None, None), (None, {"n": 2}), (None, None)],
        )
        self.assertEqual(b"".join(r for r, _, _ in whole), raw)
        for size in (1, 3, 7, 50):
            pieces = [raw[i : i + size] for i in range(0, len(raw), size)]
            self.assertEqual(
                [(n, d) for _, n, d in wire.sse_events(pieces)],
                [(n, d) for _, n, d in whole],
            )
        self.assertEqual(
            list(wire.sse_events([wire.sse_bytes("x", {"k": 1})]))[0][1:],
            ("x", {"k": 1}),
        )


if __name__ == "__main__":
    unittest.main()
