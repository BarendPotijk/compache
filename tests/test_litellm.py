"""compache inside LiteLLM: the HTTP path and the recording callback, against a local stand-in.

Needs LiteLLM: `pip install litellm`.
"""

import asyncio
import importlib.util
import json
import tempfile
import time
import unittest

import compache
from compache.live import Live
from compache.record import load, load_calls
from compache.wrap import thread
from fakes import NOTE_ASKED, FakeModel, noticed, serve
from test_live import SEARCH, SYSTEM, Clock, claude_looping, fitted, looping, passage

HAS = importlib.util.find_spec("litellm") is not None


@unittest.skipUnless(HAS, "needs LiteLLM")
class LiteLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = fitted()
        cls.started = 0

    def setUp(self):
        import litellm

        litellm.suppress_debug_info = True
        self.litellm = litellm
        self.saved = (
            litellm.client_session,
            litellm.aclient_session,
            list(litellm.callbacks),
        )

        def restore():
            compache.litellm.uninstall()
            litellm.client_session, litellm.aclient_session, callbacks = (
                self.saved[0],
                self.saved[1],
                self.saved[2],
            )
            litellm.callbacks = callbacks
            for name in ("flush_cache", "clear"):
                cache = getattr(litellm, "in_memory_llm_clients_cache", None)
                if callable(getattr(cache, name, None)):
                    getattr(cache, name)()
                    break

        self.addCleanup(restore)
        type(self).started += 1
        self.clock = Clock(1_870_000_000.0 + self.started * 1e7)

    def provider(self, script):
        self.model = FakeModel(script)
        url, stop = serve(self.model)
        self.addCleanup(stop)
        return url + "/v1"

    def folder(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def chat(
        self,
        url,
        name,
        turns,
        *,
        asynchronous=False,
        stream=False,
        until=None,
        model="openai/claude-sonnet-5-5",
        **options,
    ):
        """A conversation whose model searches and then answers: the tool loop LiteLLM users write.

        `until` waits, inside the event loop, for LiteLLM's callbacks, which run after the call.
        """
        messages = [{"role": "system", "content": SYSTEM}]

        def ask():
            return dict(
                model=model,
                api_base=url,
                api_key="key",
                messages=list(messages),
                tools=[SEARCH],
                max_tokens=256,
                stream=stream,
                **options,
            )

        def kept(message):
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls or ():
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": passage(self.turn),
                    }
                )
            return not message.tool_calls

        def joined(chunks):
            return self.litellm.stream_chunk_builder(chunks, messages=messages)

        def next_turn():
            self.clock.at += 3600
            live = getattr(self, "live", None)
            if live is not None:
                # Notes asked after the reply belong to the turn before the next one.
                live.drain()

        if not asynchronous:
            for self.turn in range(1, turns + 1):
                messages.append({"role": "user", "content": f"Question {self.turn}?"})
                while True:
                    with thread(name):
                        reply = self.litellm.completion(**ask())
                        reply = joined(list(reply)) if stream else reply
                    if kept(reply.choices[0].message):
                        break
                next_turn()
            return messages

        async def play():
            for self.turn in range(1, turns + 1):
                messages.append({"role": "user", "content": f"Question {self.turn}?"})
                while True:
                    with thread(name):
                        reply = await self.litellm.acompletion(**ask())
                        if stream:
                            reply = joined([chunk async for chunk in reply])
                    if kept(reply.choices[0].message):
                        break
                next_turn()
            for _ in range(100):
                if until is None or until():
                    break
                await asyncio.sleep(0.05)
            return messages

        return asyncio.run(play())

    def test_install_puts_the_policy_in_every_call_sync_async_and_streamed(self):
        for asynchronous, stream in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(asynchronous=asynchronous, stream=stream):
                url = self.provider(looping)
                folder = self.folder()
                self.live = Live(self.policy, record=folder, clock=self.clock)
                compache.litellm.install(self.live)
                name = f"install-{asynchronous}-{stream}"
                messages = self.chat(
                    url, name, 4, asynchronous=asynchronous, stream=stream
                )
                self.assertEqual(
                    [
                        m["content"]
                        for m in messages
                        if m["role"] == "assistant" and m.get("content")
                    ],
                    [f"Answer {n}." for n in range(1, 5)],
                )
                self.assertGreaterEqual(
                    sum(map(noticed, (json.dumps(b) for b in self.model.bodies))), 3
                )
                self.assertFalse(noticed(messages))
                self.assertEqual(self.live.stats["turns"], 4)
                self.assertEqual(self.live.stats["stood_down"], 0)
                threads, _ = load(folder)
                self.assertEqual([len(t["runs"]) for t in threads], [4])
                # The conversation went under its own name, hashed.
                self.assertEqual(threads[0]["thread_id"], self.live.recorder.name(name))

    def test_install_reaches_the_providers_litellm_calls_itself(self):
        """Most providers LiteLLM calls through HTTP clients of its own, not the OpenAI SDK."""
        for model, path, script in (
            ("openrouter/anthropic/claude-sonnet-5-5", "/api/v1", looping),
            ("anthropic/claude-sonnet-5-5", "", claude_looping),
            ("hosted_vllm/claude-sonnet-5-5", "/v1", looping),
        ):
            for asynchronous in (False, True):
                with self.subTest(model=model, asynchronous=asynchronous):
                    url = self.provider(script)[: -len("/v1")] + path
                    self.live = Live(self.policy, clock=self.clock)
                    compache.litellm.install(self.live)
                    name = f"{model}-{asynchronous}"
                    messages = self.chat(
                        url, name, 4, asynchronous=asynchronous, model=model
                    )
                    said = [
                        m["content"]
                        for m in messages
                        if m["role"] == "assistant" and not m.get("tool_calls")
                    ]
                    self.assertEqual(said, [f"Answer {n}." for n in range(1, 5)])
                    sent = [json.dumps(b) for b in self.model.bodies]
                    self.assertGreaterEqual(sum(map(noticed, sent)), 3)
                    self.assertFalse(noticed(messages))
                    self.assertEqual(self.live.stats["turns"], 4)

    def test_installing_again_hands_over_wraps_once_and_comes_out(self):
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

        first, second = Live(self.policy), Live(self.policy)
        compache.litellm.install(first)
        compache.litellm.install(second)
        made = HTTPHandler().client
        self.assertIs(made._compache, second)
        self.assertIsNone(getattr(made._compache_inner, "_compache", None))
        kept = self.litellm.module_level_client.client
        self.assertIs(kept._compache, second)
        compache.litellm.uninstall()
        self.assertIsNone(getattr(HTTPHandler().client, "_compache", None))
        self.assertIsNone(getattr(AsyncHTTPHandler().client, "_compache", None))
        self.assertIsNone(getattr(self.litellm.module_level_client.client, "_compache", None))
        self.assertIsNone(self.litellm.client_session)

    def test_the_callback_records_without_keeping_any_text(self):
        for asynchronous, stream in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(asynchronous=asynchronous, stream=stream):
                url = self.provider(looping)
                folder = self.folder()
                self.live = None
                callback = compache.litellm.Callback(record=folder, sample=0.0)
                self.litellm.callbacks = [callback]
                name = f"callback-{asynchronous}-{stream}"

                def recorded(callback=callback):
                    return callback.compache.stats["turns"] >= 3

                self.chat(
                    url,
                    name,
                    3,
                    asynchronous=asynchronous,
                    stream=stream,
                    until=recorded,
                    metadata={"session_id": name},
                )
                # LiteLLM runs its callbacks after the call, in its own time.
                for _ in range(100):
                    if recorded():
                        break
                    time.sleep(0.05)
                self.assertEqual(callback.compache.stats["turns"], 3)
                threads, labels = load(folder)
                self.assertEqual([len(t["runs"]) for t in threads], [3])
                self.assertEqual(
                    threads[0]["thread_id"], callback.compache.recorder.name(name)
                )
                # Sizes and usage, and not one word of it.
                run = threads[0]["runs"][0]
                self.assertEqual(run["retrievals"], [{"bytes": len(passage(1))}])
                self.assertGreater(run["usage"]["input"], 0)
                text = "".join(
                    (folder_file.read_text())
                    for folder_file in __import__("pathlib").Path(folder).iterdir()
                )
                self.assertNotIn("Clause", text)
                calls = load_calls(folder)
                self.assertTrue(all(c["kind"] == "answer" for c in calls))
                # The stand-in names a price per call, which goes into the record.
                self.assertGreater(sum(c.get("cost") or 0 for c in calls), 0)
                self.assertEqual([row["run"] for row in labels], [1])

    def test_the_callback_asks_notes_through_litellm_itself(self):
        def scripted(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {R1.1} {A1} -->"}
            return looping(request)

        url = self.provider(scripted)
        folder = self.folder()
        callback = compache.litellm.Callback(record=folder, sample=1.0)
        self.litellm.callbacks = [callback]
        messages = [{"role": "system", "content": SYSTEM}]
        for n in range(1, 4):
            messages.append({"role": "user", "content": f"Question {n}?"})
            while True:
                reply = self.litellm.completion(
                    model="openai/claude-sonnet-5-5",
                    api_base=url,
                    api_key="key",
                    messages=list(messages),
                    tools=[SEARCH],
                    metadata={"session_id": "noted"},
                )
                message = reply.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_calls[0].id,
                        "content": passage(n),
                    }
                )
        for _ in range(100):
            if callback.compache.stats["noted"] >= 2:
                break
            time.sleep(0.05)
        callback.compache.drain()
        # Turns 2 and 3 are asked. The notes' own calls are not turns of the conversation.
        self.assertEqual(callback.compache.stats["noted"], 2)
        self.assertEqual(callback.compache.stats["turns"], 3)
        threads, labels = load(folder)
        name = callback.compache.recorder.name("noted")
        self.assertEqual(threads[0]["thread_id"], name)
        self.assertEqual(
            labels[1]["needs"], [[f"{name}:1:retrieval:1", f"{name}:1:output"]]
        )

    def test_every_event_loop_has_connections_of_its_own(self):
        url = self.provider(lambda request: {"text": "Answer."})
        compache.litellm.install(Live(), pings=False)

        async def one():
            reply = await self.litellm.acompletion(
                model="openai/claude-sonnet-5-5",
                api_base=url,
                api_key="key",
                messages=[{"role": "user", "content": "Question?"}],
            )
            return reply.choices[0].message.content

        # One `asyncio.run` a request, as a script, a worker or a function does it.
        self.assertEqual([asyncio.run(one()) for _ in range(3)], ["Answer."] * 3)

    def finished(self, **more):
        """What LiteLLM hands its callbacks for a finished call."""
        call = {
            "model": "claude-haiku-5-5",
            "custom_llm_provider": "azure",
            "call_type": "acompletion",
            "messages": [{"role": "user", "content": "Question?"}],
            "litellm_params": {
                "metadata": {"session_id": "s"},
                "api_base": "https://example.invalid",
                "api_version": "2026-01-01",
                "vertex_project": None,
            },
            "optional_params": {"aws_region_name": "eu-west-1", "temperature": 0.2},
        }
        reply = {
            "model": "claude-haiku-5-5",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Answer."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        return {**call, **more}, reply

    def test_an_answer_from_litellm_s_own_cache_is_no_model_call(self):
        folder = self.folder()
        callback = compache.litellm.Callback(record=folder)
        call, reply = self.finished(cache_hit=True)
        callback.log_success_event(call, reply, 1_870_000_000.0, 1_870_000_000.1)
        self.assertEqual(callback.compache.stats["calls"], 0)
        call, reply = self.finished(cache_hit=False)
        callback.log_success_event(call, reply, 1_870_000_000.0, 1_870_000_001.5)
        (row,) = load_calls(folder)
        # The provider reported no reads, which is not the same as none.
        self.assertFalse(row.get("cache_read_reported"))

    def test_a_note_goes_the_way_its_call_went_and_a_failure_counts_as_refused(self):
        from unittest import mock

        call, _ = self.finished()
        ask = compache.litellm._asker(call)
        note = {
            "model": "claude-haiku-5-5",
            "messages": [{"role": "user", "content": "?"}],
        }
        with mock.patch("litellm.completion", return_value={"choices": []}) as made:
            self.assertEqual(ask({**note, "max_tokens": 64}).status, 200)
        sent = made.call_args.kwargs
        self.assertEqual(sent["model"], "azure/claude-haiku-5-5")
        self.assertEqual(
            (sent["api_base"], sent["api_version"], sent["aws_region_name"]),
            ("https://example.invalid", "2026-01-01", "eu-west-1"),
        )
        # The call's own sampling and limits are not the note's.
        self.assertNotIn("temperature", sent)
        self.assertNotIn("vertex_project", sent)
        with mock.patch(
            "litellm.completion", side_effect=RuntimeError("no such deployment")
        ):
            self.assertGreaterEqual(ask(note).status, 400)
        # After three of them no more notes are asked.
        live = Live(record=self.folder(), sample=1.0, clock=self.clock)
        for n in range(1, 8):
            messages = []
            for k in range(1, n + 1):
                messages.append({"role": "user", "content": f"Question {k}?"})
                if k < n:
                    messages.append({"role": "assistant", "content": f"Answer {k}."})
            reply = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": f"Answer {n}."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
            at = self.clock.at + 60 * n
            with mock.patch("litellm.completion", side_effect=RuntimeError("down")):
                live.observe(
                    "/v1/chat/completions",
                    {"model": "claude-haiku-5-5", "messages": messages},
                    reply,
                    thread="t",
                    at=(at, at + 1),
                    ask=ask,
                )
                live.drain()
        self.assertTrue(live._notes_off)

    def test_a_proxy_s_alias_and_an_unknown_price_are_not_recorded(self):
        folder = self.folder()
        callback = compache.litellm.Callback(record=folder)
        # What a LiteLLM proxy hands its callbacks: the reply under the proxy's own name for the
        # model, the call under the real one, and no price for it.
        reply = {
            "model": "haiku",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Answer."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        call = {
            "model": "claude-haiku-5-5",
            "call_type": "acompletion",
            "messages": [{"role": "user", "content": "Question?"}],
            "litellm_params": {"metadata": {"session_id": "aliased"}},
            "response_cost": 0.0,
        }
        callback.log_success_event(call, reply, 1_870_000_000.0, 1_870_000_001.5)
        (row,) = load_calls(folder)
        self.assertEqual(row["model"], "claude-haiku-5-5")
        self.assertNotIn("cost", {k for k, v in row.items() if v is not None})
        self.assertEqual(load(folder)[0][0]["runs"][0]["started_at"], 1_870_000_000.0)

    def test_the_proxy_callback_comes_from_the_environment(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMPACHE_RECORD", None)
            with self.assertRaisesRegex(RuntimeError, "COMPACHE_RECORD"):
                getattr(compache.litellm, "callback")
            os.environ["COMPACHE_RECORD"] = self.folder()
            os.environ["COMPACHE_SAMPLE"] = "0.5"
            made = getattr(compache.litellm, "callback")
            self.assertEqual(made.compache.sample, 0.5)
            self.assertIs(compache.litellm.callback, made)
            # A fresh import state for the next test.
            vars(compache.litellm).pop("callback", None)


if __name__ == "__main__":
    unittest.main()
