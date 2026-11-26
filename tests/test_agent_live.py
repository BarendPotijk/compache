"""The agent's policy in the request path: loops of calls on every API compache speaks."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from unittest import mock

import compache
from compache import agent, wire
from compache import live as live_module
from compache.api import Session
from compache.learn import Learner
from compache.models import binds_reasoning
from compache.record import load, load_calls
from fakes import NOTICE, FakeModel, noticed, tokens

import test_live

Clock = test_live.Clock

SPEC = compache.ModelSpec(
    "agent-model",
    context_tokens=200_000,
    cache_ttl_seconds=300,
    cache_min_tokens=0,
    read_multiplier=0.1,
    write_multiplier=1.25,
    output_multiplier=5.0,
)
REOPEN = "reopen"


def policy(rounds=2, ratio=0.0, tool_pings=0, user_pings=0, spec=SPEC):
    return compache.Policy(
        {
            "schema_version": agent.SCHEMA,
            "model_spec": spec.to_dict(),
            "recipe": {
                "rounds": rounds,
                "ratio": ratio,
                "tool_pings": tool_pings,
                "user_pings": user_pings,
            },
            "reserve_tokens": 0,
            "training_threads": [],
        }
    )


def result(k):
    return f"result {k}: " + f"a line of file number {k} that only it holds\n" * 60


class Chat:
    path, model = "/v1/chat/completions", "agent-model"

    def __init__(self):
        self.said = [{"role": "system", "content": "You are an agent."}]

    def ask(self, text):
        self.said.append({"role": "user", "content": text})

    def body(self, stream):
        tool = {"type": "function", "function": {"name": "read", "parameters": {}}}
        body = {"model": self.model, "messages": list(self.said), "tools": [tool]}
        return {**body, "stream": True} if stream else body

    def take(self, reply):
        message = reply["choices"][0]["message"]
        self.said.append(message)
        return message.get("content") or "", [
            (c["id"], c["function"]["name"]) for c in message.get("tool_calls") or ()
        ]

    def answer(self, results):
        self.said += [
            {"role": "tool", "tool_call_id": call, "content": text}
            for call, text in results
        ]

    @staticmethod
    def gathered(events):
        text, calls = "", {}
        for data in events:
            for choice in data.get("choices") or ():
                delta = choice["delta"]
                text += delta.get("content") or ""
                for call in delta.get("tool_calls") or ():
                    held = calls.setdefault(
                        call["index"],
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    held["id"] = call.get("id") or held["id"]
                    for key, value in (call.get("function") or {}).items():
                        held["function"][key] += value
        message = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [calls[k] for k in sorted(calls)]
        return {"choices": [{"message": message}]}

    @staticmethod
    def results(body):
        return [m["content"] for m in body["messages"] if m["role"] == "tool"]


class Claude(Chat):
    path, model = "/v1/messages", "claude-agent"

    def __init__(self):
        self.said = []

    def body(self, stream):
        body = {
            "model": self.model,
            "max_tokens": 500,
            "system": "You are an agent.",
            "messages": list(self.said),
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        }
        return {**body, "stream": True} if stream else body

    def take(self, reply):
        self.said.append({"role": "assistant", "content": reply["content"]})
        text = "".join(b["text"] for b in reply["content"] if b["type"] == "text")
        return text, [
            (b["id"], b["name"]) for b in reply["content"] if b["type"] == "tool_use"
        ]

    def answer(self, results, remark=None):
        content = [
            {"type": "tool_result", "tool_use_id": call, "content": text}
            for call, text in results
        ]
        if remark:
            content.append({"type": "text", "text": remark})
        self.said.append({"role": "user", "content": content})

    @staticmethod
    def gathered(events):
        blocks = {}
        for data in events:
            if data["type"] == "content_block_start":
                blocks[data["index"]] = {**data["content_block"], "json": ""}
            elif data["type"] == "content_block_delta":
                block, delta = blocks[data["index"]], data["delta"]
                if "text" in delta:
                    block["text"] += delta["text"]
                if "signature" in delta:
                    block["signature"] = delta["signature"]
                block["json"] += delta.get("partial_json") or ""
        content = []
        for k in sorted(blocks):
            block = blocks[k]
            raw = block.pop("json")
            if block["type"] == "tool_use":
                block["input"] = json.loads(raw or "{}")
            content.append(block)
        return {"content": content}

    @staticmethod
    def results(body):
        return [
            b["content"]
            for m in body["messages"]
            if isinstance(m["content"], list)
            for b in m["content"]
            if b["type"] == "tool_result"
        ]


class Codex(Chat):
    path, model = "/v1/responses", "agent-model"

    def __init__(self):
        self.said = []

    def ask(self, text):
        self.said.append({"role": "user", "content": text})

    def body(self, stream):
        body = {
            "model": self.model,
            "instructions": "You are an agent.",
            "store": False,
            "input": list(self.said),
            "tools": [{"type": "function", "name": "read", "parameters": {}}],
        }
        return {**body, "stream": True} if stream else body

    def take(self, reply):
        self.said += reply["output"]
        text = "".join(
            part["text"]
            for item in reply["output"]
            if item["type"] == "message"
            for part in item["content"]
        )
        return text, [
            (item["call_id"], item["name"])
            for item in reply["output"]
            if item["type"] == "function_call"
        ]

    def answer(self, results):
        self.said += [
            {"type": "function_call_output", "call_id": call, "output": text}
            for call, text in results
        ]

    @staticmethod
    def gathered(events):
        numbers = [e["sequence_number"] for e in events]
        assert numbers == list(range(len(events))), numbers
        added = [
            e["output_index"]
            for e in events
            if e["type"] == "response.output_item.added"
        ]
        assert added == list(range(len(added))), added
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        assert events[-1]["type"] == "response.completed", events[-1]["type"]
        assert events[-1]["response"]["output"] == done
        return {"output": done}

    @staticmethod
    def results(body):
        return [
            i["output"]
            for i in body["input"]
            if i.get("type") == "function_call_output"
        ]


class Gemini(Chat):
    path, model = "/v1beta/models/agent-model:generateContent", "agent-model"

    def __init__(self):
        self.said = []

    def ask(self, text):
        self.said.append({"role": "user", "parts": [{"text": text}]})

    def body(self, stream):
        return {
            "systemInstruction": {"parts": [{"text": "You are an agent."}]},
            "contents": list(self.said),
            "tools": [{"functionDeclarations": [{"name": "read"}]}],
        }

    def take(self, reply):
        content = reply["candidates"][0]["content"]
        self.said.append(content)
        text = "".join(p.get("text", "") for p in content["parts"])
        return text, [
            (None, p["functionCall"]["name"])
            for p in content["parts"]
            if "functionCall" in p
        ]

    def answer(self, results):
        self.said.append(
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "read", "response": {"output": text}}}
                    for _, text in results
                ],
            }
        )

    @staticmethod
    def gathered(events):
        parts = [
            part
            for data in events
            for part in data["candidates"][0]["content"].get("parts") or ()
        ]
        return {"candidates": [{"content": {"role": "model", "parts": parts}}]}

    @staticmethod
    def results(body):
        return [
            p["functionResponse"]["response"]["output"]
            for c in body["contents"]
            for p in c["parts"]
            if "functionResponse" in p
        ]


class Agent:
    """A harness: it sends the whole conversation, runs the tools it is told to, and comes back."""

    def __init__(self, live, shape, model, *, stream=False, thread=None, clock=None):
        self.live, self.shape, self.model = live, shape, model
        self.stream, self.thread, self.clock = stream, thread, clock
        self.made = 0

    def call(self):
        path = self.shape.path
        if self.stream and path.startswith("/v1beta"):
            path = path.replace("generateContent", "streamGenerateContent")
        answered = self.live.exchange(
            path,
            self.shape.body(self.stream),
            self.model.send(path),
            thread=self.thread,
        )
        if answered.events is None:
            return self.shape.take(answered.json)
        data = [d for _, _, d in wire.sse_events(answered.events) if d is not None]
        return self.shape.take(self.shape.gathered(data))

    def run(self, question, most=60):
        """Ask, and work until the model is done. Returns its last words."""
        self.shape.ask(question)
        for _ in range(most):
            text, calls = self.call()
            if not calls:
                return text
            for _, name in calls:
                assert name != REOPEN, "the application sees our tool"
            if self.clock:
                self.clock.at += 5
            self.made += 1
            self.shape.answer([(call, result(self.made)) for call, _ in calls])
        raise AssertionError("the loop does not end")


def worker(count, extra=None):
    """A model that reads `count` files one by one and then is done. `extra(request, n)` may say otherwise."""

    def script(request):
        seen = sum(
            1
            for t in request["texts"]
            if (
                "a line of file number" in t
                and t.lstrip('{"output": ').startswith("result")
            )
            or NOTICE.search(t)
        )
        made = extra(request, seen) if extra else None
        if made is not None:
            return made
        if seen < count:
            return {"text": f"Reading {seen + 1}.", "tool": ("read", {"file": seen})}
        return {"text": "All done."}

    return script


SHAPES = (Chat, Claude, Codex, Gemini)


class LoopTests(unittest.TestCase):
    def test_old_results_go_out_as_notices_on_every_api(self):
        for shape in SHAPES:
            for stream in (False, True):
                with self.subTest(shape=shape.__name__, stream=stream):
                    model = FakeModel(worker(6))
                    live = compache.Live(policy(rounds=2))
                    app = Agent(live, shape(), model, stream=stream)
                    self.assertEqual(app.run("Read six files"), "All done.")
                    sent = shape.results(model.bodies[-1])
                    self.assertEqual(len(sent), 6)
                    self.assertTrue(all(noticed(t) for t in sent[:4]))
                    self.assertFalse(any(noticed(t) for t in sent[4:]))
                    # The application's own conversation is whole.
                    self.assertFalse(noticed(app.shape.said))
                    # One loop, found again at every call without being named.
                    self.assertEqual(live.stats["turns"], 1)
                    self.assertEqual(live.stats["stood_down"], 0)
                    self.assertEqual(live.stats["passed_through"], 0)

    def test_the_first_notice_says_what_they_all_are_and_stays_as_it_is(self):
        # Live, a model took a notice for what the tool had returned, and what it had written from
        # the result for made up.
        for shape in SHAPES:
            with self.subTest(shape=shape.__name__):
                model = FakeModel(worker(9))
                live = compache.Live(policy(rounds=2))
                Agent(live, shape(), model).run("Read nine files")
                firsts = set()
                for body in model.bodies:
                    gone = [t for t in shape.results(body) if noticed(t)]
                    long = ["what you wrote next comes from it" in t for t in gone]
                    self.assertEqual(long, [k == 0 for k in range(len(gone))])
                    firsts.update(map(json.dumps, gone[:1]))
                self.assertEqual(len(gone), 7)
                # Nothing that is cached ahead of a later cut is written again.
                self.assertEqual(len(firsts), 1)
                self.assertTrue(all(str(t).count("You read this") == 1 for t in gone))

    def test_an_agent_has_the_tool_from_its_first_call(self):
        model = FakeModel(worker(1))
        live = compache.Live(policy())
        Agent(live, Chat(), model).run("Read a file")
        for body in model.bodies:
            names = [t["function"]["name"] for t in body["tools"]]
            self.assertEqual(names, ["read", REOPEN])
        # A request that offers no tools, such as one for a title, gets none.
        live.exchange(
            "/v1/chat/completions",
            {"model": "agent-model", "messages": [{"role": "user", "content": "hi"}]},
            model.send(),
        )
        self.assertNotIn("tools", model.bodies[-1])

    def test_a_warm_cache_is_left_alone_until_a_cut_pays(self):
        clock = Clock()
        model = FakeModel(worker(12), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=2, ratio=2.0), clock=clock)
        Agent(live, Claude(), model, clock=clock).run("Read twelve files")
        cuts, before = [], 0
        for n, body in enumerate(model.bodies):
            gone = sum(noticed(t) for t in Claude.results(body))
            self.assertGreaterEqual(gone, before)
            if gone > before:
                cuts.append(n)
            before = gone
        self.assertGreaterEqual(len(cuts), 1)
        self.assertLess(len(cuts), 4, "not at every call")

    def test_the_start_of_a_cut_conversation_is_read_from_the_cache(self):
        clock = Clock()
        model = FakeModel(worker(40), clock=lambda: clock.at)
        replies = []
        send = model.send("/v1/messages")

        def keeping(body):
            answered = send(body)
            replies.append(answered.json["usage"])
            return answered

        model.send = lambda path: keeping
        live = compache.Live(policy(rounds=10, ratio=1.0), clock=clock)
        Agent(live, Claude(), model, clock=clock).run("Read forty files")
        cuts = 0
        for n in range(1, len(model.bodies)):
            before = [noticed(t) for t in Claude.results(model.bodies[n - 1])]
            now = [noticed(t) for t in Claude.results(model.bodies[n])]
            marks = len(wire._marks(model.bodies[n]))
            self.assertLessEqual(marks, 4)
            if sum(now) > sum(before):
                cuts += 1
                kept = before.count(True)
                if kept:
                    self.assertGreater(
                        replies[n]["cache_read_input_tokens"],
                        kept * 20,
                        "what comes ahead of the cut was cached and is read",
                    )
            self.assertGreaterEqual(marks, 2)
        self.assertGreaterEqual(cuts, 2)

    def test_the_policy_s_flow_agrees_with_what_the_provider_counts(self):
        clock = Clock()
        model = FakeModel(worker(15), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=3, ratio=1.0), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.run("Read fifteen files")
        (conversation,) = live.conversations()
        flow = conversation.session._engine.flow
        self.assertEqual(flow.size, sum(flow.shown))
        counted = tokens([model.bodies[-1]["messages"]])
        self.assertLess(abs(flow.size - counted) / counted, 0.25)

    def test_a_conversation_met_in_the_middle_is_sized_like_any(self):
        """A proxy is restarted while an agent works."""
        sizes = []
        for restarted in (False, True):
            clock = Clock()
            model = FakeModel(worker(40), clock=lambda: clock.at)
            live = compache.Live(policy(rounds=8, ratio=1.5), clock=clock)
            app = Agent(live, Claude(), model, clock=clock)
            app.shape.ask("Read forty files")
            for k in range(34):
                if restarted and k == 30:
                    app.live = compache.Live(policy(rounds=8, ratio=1.5), clock=clock)
                _, calls = app.call()
                clock.at += 5
                app.made += 1
                app.shape.answer([(c, result(app.made)) for c, _ in calls])
            (conversation,) = app.live.conversations()
            engine = conversation.session._engine
            flow = engine.flow
            self.assertEqual(flow.size, sum(flow.shown))
            self.assertEqual(
                flow._before(flow.live[0]), sum(flow.shown[: flow.live[0]])
            )
            sizes.append(engine._off)
        self.assertLess(abs(sizes[1] / sizes[0] - 1), 0.2)

    def test_what_a_turn_of_one_call_fetched_is_counted(self):
        clock = Clock()
        model = FakeModel(worker(12), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=2), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.run("Read twelve files")
        asked = []

        def script(request):
            if not asked:
                asked.append(NOTICE.search(json.dumps(request["body"])).group())
                return {"text": "", "tools": [(REOPEN, {"reference": asked[0]})]}
            return {"text": "It said so."}

        model.script = script
        clock.at += 20
        app.run("What did the first file say?")
        (conversation,) = live.conversations()
        flow = conversation.session._engine.flow
        counted = tokens([model.bodies[-1]["messages"]])
        self.assertLess(abs(flow.size - counted) / counted, 0.2)

    def test_another_model_has_a_cache_of_its_own(self):
        clock = Clock()
        model = FakeModel(worker(12), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=2, ratio=None), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.run("Read twelve files")
        self.assertFalse(any(noticed(b["messages"]) for b in model.bodies))
        app.shape.model = "agent-model-2"
        model.script = worker(13)
        app.run("And one more")
        self.assertTrue(noticed(model.bodies[-1]["messages"]))

    def test_a_second_task_goes_on_where_the_first_ended(self):
        model = FakeModel(worker(4))
        live = compache.Live(policy(rounds=2))
        app = Agent(live, Claude(), model)
        app.run("Read four files")
        model.script = worker(8)
        app.run("Read four more")
        self.assertEqual(live.stats["turns"], 2)
        sent = Claude.results(model.bodies[-1])
        self.assertEqual([noticed(t) for t in sent], [True] * 6 + [False] * 2)

    def test_a_reminder_beside_the_results_does_not_end_the_turn(self):
        model = FakeModel(worker(5))
        live = compache.Live(policy(rounds=2))
        app = Agent(live, Claude(), model)
        answer = app.shape.answer
        app.shape.answer = lambda results: answer(results, "<system-reminder>go on")
        app.run("Read five files")
        self.assertEqual(live.stats["turns"], 1)
        self.assertEqual(
            [noticed(t) for t in Claude.results(model.bodies[-1])],
            [True] * 3 + [False] * 2,
        )


class FetchTests(unittest.TestCase):
    """The model fetches an archived result back in the middle of its loop."""

    @staticmethod
    def script(count, at=5, mixed=False, reasoning=False):
        fetched = []

        def extra(request, seen):
            if seen == at and not fetched:
                handle = NOTICE.search(json.dumps(request["body"])).group()
                fetched.append(handle)
                call = (REOPEN, {"reference": handle})
                made = {
                    "text": "" if reasoning else "Let me look back.",
                    "tools": [call],
                }
                if mixed:
                    made["tools"] = [("read", {"file": seen}), call]
                if reasoning:
                    made["reasoning"] = "I need that again."
                return made
            return None

        return worker(count, extra), fetched

    def test_what_it_fetched_stays_until_it_is_old_and_the_application_never_sees_it(
        self,
    ):
        for shape in SHAPES:
            for stream in (False, True):
                with self.subTest(shape=shape.__name__, stream=stream):
                    script, fetched = self.script(9, reasoning=shape is Codex)
                    model = FakeModel(script)
                    live = compache.Live(policy(rounds=2))
                    app = Agent(live, shape(), model, stream=stream)
                    self.assertEqual(app.run("Read nine files"), "All done.")
                    self.assertEqual(len(fetched), 1)
                    self.assertEqual(live.stats["reopened"], 1)
                    self.assertNotIn(REOPEN, json.dumps(app.shape.said))
                    holds = [
                        "a line of file number 1 that" in json.dumps(body)
                        for body in model.bodies
                    ]
                    # Calls 0 to 5 make the first five reads. Call 5 fetches the first file back, and
                    # the same exchange goes on with call 6, which has it.
                    self.assertEqual(holds[1:3], [True, True])
                    self.assertFalse(any(holds[3:6]))
                    self.assertTrue(holds[6], "the request that answers the fetch")
                    self.assertTrue(holds[7], "it stays in the next one")
                    self.assertFalse(any(holds[9:]), "and goes once it is old")

    def test_a_fetch_beside_the_application_s_own_tools_is_answered_too(self):
        for shape in SHAPES:
            for stream in (False, True):
                with self.subTest(shape=shape.__name__, stream=stream):
                    script, fetched = self.script(7, mixed=True)
                    model = FakeModel(script)
                    live = compache.Live(policy(rounds=2))
                    app = Agent(live, shape(), model, stream=stream)
                    self.assertEqual(app.run("Read seven files"), "All done.")
                    self.assertNotIn(REOPEN, json.dumps(app.shape.said))
                    self.assertEqual(live.stats["reopened"], 1)
                    self.assertIn(
                        "a line of file number 1 that", json.dumps(model.bodies[6])
                    )

    def test_every_call_of_a_request_is_answered_in_it(self):
        """A provider refuses a tool call without its result, and a result without its call."""
        for mixed in (False, True):
            script, _ = self.script(9, mixed=mixed)
            model = FakeModel(script)
            live = compache.Live(policy(rounds=2))
            Agent(live, Claude(), model).run("Read nine files")
            for body in model.bodies:
                roles = [m["role"] for m in body["messages"]]
                self.assertEqual(
                    roles, ["user", "assistant"] * (len(roles) // 2) + ["user"]
                )
                for before, after in zip(body["messages"], body["messages"][1:]):
                    if before["role"] != "assistant":
                        continue
                    made = {
                        b["id"] for b in before["content"] if b["type"] == "tool_use"
                    }
                    given = {
                        b["tool_use_id"]
                        for b in after["content"]
                        if b["type"] == "tool_result"
                    }
                    self.assertEqual(made, given)

    def test_codex_gives_its_tools_among_what_is_said(self):
        shape = Codex()
        shape.said.append(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{"type": "custom", "name": "exec"}],
            }
        )
        shape.body, body = lambda s: {**body(s), "tools": []}, shape.body
        model = FakeModel(worker(6))
        live = compache.Live(policy(rounds=2))
        Agent(live, shape, model, stream=True).run("Read six files")
        self.assertEqual([t["name"] for t in model.bodies[0]["tools"]], [REOPEN])
        self.assertTrue(noticed(model.bodies[-1]["input"]))

    def test_the_results_of_custom_tools_count_like_any(self):
        shape = Codex()

        def answer(results):
            for call, text in results:
                for item in shape.said:
                    if item.get("call_id") == call:
                        item["type"] = "custom_tool_call"
                        item["input"] = item.pop("arguments")
                parts = [
                    {"type": "input_text", "text": "Script completed"},
                    {"type": "input_text", "text": text},
                ]
                shape.said.append(
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call,
                        "output": parts,
                    }
                )

        shape.answer = answer
        model = FakeModel(worker(6))
        live = compache.Live(policy(rounds=2))
        Agent(live, shape, model, stream=True).run("Read six files")
        self.assertEqual(live.stats["turns"], 1)
        sent = [
            i["output"]
            for i in model.bodies[-1]["input"]
            if i.get("type") == "custom_tool_call_output"
        ]
        self.assertEqual([noticed(o) for o in sent], [True] * 4 + [False] * 2)

    def test_reasoning_that_led_to_a_fetch_stays_out_of_the_application_s_hands(self):
        script, _ = self.script(6, reasoning=True)
        model = FakeModel(script)
        live = compache.Live(policy(rounds=2))
        app = Agent(live, Codex(), model, stream=True)
        app.run("Read six files")
        self.assertNotIn("enclosed-", json.dumps(app.shape.said))
        # The model itself has it back while it answers with what it fetched.
        self.assertIn("enclosed-", json.dumps(model.bodies[6]))

    def test_a_model_that_goes_on_fetching_is_asked_as_the_application_wrote_it(self):
        def extra(request, seen):
            if seen >= 4:
                return {"tool": (REOPEN, {"reference": "result number one please"})}

        model = FakeModel(worker(5, extra))
        live = compache.Live(policy(rounds=1))
        app = Agent(live, Chat(), model)
        app.shape.ask("Read")
        for _ in range(4):
            _, calls = app.call()
            app.made += 1
            app.shape.answer([(c, result(app.made)) for c, _ in calls])
        app.call()
        self.assertFalse(noticed(model.bodies[-1]["messages"]))
        self.assertEqual(live.stats["fell_back"], 1)


def reasoning(count, tokens=200):
    """A model that reasons before everything it says."""
    script = worker(count)

    def made(request):
        return {**script(request), "thinking": tokens, "reasoning": "So many words."}

    return made


class OldClaude(Claude):
    """One that takes its reasoning back whatever has changed ahead of it."""

    model = "claude-opus-5"


class ReasoningTests(unittest.TestCase):
    """What the model reasoned stays in later requests, and goes with the results of its time."""

    def held(self, body):
        """Which replies of a request still hold their reasoning, in order."""
        if "input" in body:
            found, calls = [], 0
            for item in body["input"]:
                if item.get("type") == "reasoning":
                    found.append(calls)
                calls += item.get("type") == "function_call"
            return found
        replies = [m for m in body["messages"] if m["role"] == "assistant"]
        return [
            k
            for k, m in enumerate(replies)
            if any(b["type"] == "thinking" for b in m["content"])
        ]

    def test_that_of_a_finished_turn_goes_with_the_cut(self):
        for shape in (OldClaude, Codex):
            for stream in (False, True):
                with self.subTest(shape=shape.__name__, stream=stream):
                    model = FakeModel(reasoning(5))
                    live = compache.Live(policy(rounds=2))
                    app = Agent(live, shape(), model, stream=stream)
                    app.run("Read five files")
                    # While its turn went on, all of it stayed.
                    self.assertEqual(self.held(model.bodies[-1]), list(range(5)))
                    model.script = reasoning(9)
                    app.run("Read four more")
                    kept = self.held(model.bodies[-1])
                    self.assertFalse(set(range(5)) & set(kept), kept)
                    self.assertEqual(len(kept), 4, "that of the turn that goes on")
                    self.assertEqual(live.stats["stood_down"], 0)
                    (conversation,) = live.conversations()
                    flow = conversation.session._engine.flow
                    self.assertEqual(flow.size, sum(flow.shown))

    def test_what_the_policy_counts_is_what_the_provider_counts(self):
        clock = Clock()
        model = FakeModel(reasoning(8), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=3, ratio=1.0), clock=clock)
        app = Agent(live, OldClaude(), model, clock=clock)
        app.run("Read eight files")
        model.script = reasoning(16)
        app.run("Read eight more")
        (conversation,) = live.conversations()
        flow = conversation.session._engine.flow
        counted = tokens([model.bodies[-1]["messages"]])
        self.assertLess(abs(flow.size - counted) / counted, 0.25)

    def test_an_application_that_drops_it_itself(self):
        model = FakeModel(reasoning(4))
        live = compache.Live(policy(rounds=2))
        app = Agent(live, OldClaude(), model)
        app.run("Read four files")
        for message in app.shape.said:
            if message["role"] == "assistant":
                message["content"] = [
                    b for b in message["content"] if b["type"] != "thinking"
                ]
        model.script = reasoning(8)
        self.assertEqual(app.run("Read four more"), "All done.")
        self.assertEqual(live.stats["stood_down"], 0)
        (conversation,) = live.conversations()
        flow = conversation.session._engine.flow
        self.assertEqual(flow.size, sum(flow.shown))

    def test_a_conversation_met_in_the_middle(self):
        model = FakeModel(reasoning(5))
        app = Agent(compache.Live(), OldClaude(), model)
        app.run("Read five files")
        app.live = compache.Live(policy(rounds=2))
        model.script = reasoning(9)
        app.run("Read four more")
        kept = self.held(model.bodies[-1])
        self.assertFalse(set(range(5)) & set(kept), kept)

    def test_what_was_fetched_back_keeps_its_place_where_reasoning_is_gone(self):
        fetched = []

        def script(request):
            made = reasoning(9)(request)
            seen = sum("ctx:" in t for t in request["texts"])
            if seen >= 6 and not fetched and "tool" in made:
                handle = NOTICE.search(json.dumps(request["body"])).group()
                fetched.append(handle)
                made = {**made, "text": "", "tool": (REOPEN, {"reference": handle})}
            return made

        for shape in (OldClaude, Codex):
            with self.subTest(shape=shape.__name__):
                fetched.clear()
                model = FakeModel(reasoning(5))
                live = compache.Live(policy(rounds=2))
                app = Agent(live, shape(), model)
                app.run("Read five files")
                model.script = script
                self.assertEqual(app.run("Read four more"), "All done.")
                self.assertEqual(len(fetched), 1)
                self.assertEqual(live.stats["reopened"], 1)
                for body in model.bodies:
                    if shape is not Codex:
                        continue
                    # Every call is followed by its result, whatever was left out around them.
                    open_calls = set()
                    for item in body["input"]:
                        if item.get("type") == "function_call":
                            open_calls.add(item["call_id"])
                        elif item.get("type") == "function_call_output":
                            self.assertIn(item["call_id"], open_calls)
                            open_calls.discard(item["call_id"])
                    self.assertFalse(open_calls)

    def test_it_is_on_record(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel(reasoning(4))
            live = compache.Live(record=folder)
            Agent(live, OldClaude(), model).run("Read four files")
            steps = load(folder)[0][0]["runs"][0]["steps"]
            self.assertEqual([s["thought_tokens"] for s in steps], [200] * 5)
            thread = compache.load_threads(load(folder)[0], model="agent-model")[0]
            calls = agent.calls_of(thread, SPEC)
            self.assertEqual([c.thought for c in calls], [0] + [200] * 4)


class BoundReasoningTests(unittest.TestCase):
    """Claude from 5.1 on takes its reasoning back only behind what stood ahead of it when it was made."""

    SPEC = replace(SPEC, model_id="claude-haiku-5-5")

    def play(self, *runs, live=None, rounds=2, ratio=0.0, away=0, **options):
        clock = options.get("clock") or Clock()
        model = options.get("model") or FakeModel(clock=lambda: clock.at, binds="error")
        live = live or compache.Live(policy(rounds, ratio, spec=self.SPEC), clock=clock)
        shape = options.get("shape") or Claude()
        app = Agent(
            live, shape, model, stream=options.get("stream", False), clock=clock
        )
        for count in runs:
            self.more(app, count)
            clock.at += away
        return app, model

    def more(self, app, count):
        """One more prompt, worked on in so many rounds."""
        app.model.script = reasoning(app.made + count)
        self.assertEqual(app.run(f"Work through {count} files"), "All done.")

    def check(self, app, model):
        """Nothing was refused, and what the policy counts is what goes out."""
        self.assertEqual(app.live.stats["retried"], 0)
        self.assertEqual(app.live.stats["stood_down"], 0)
        self.assertEqual(model.dropped, 0)
        for conversation in app.live.conversations():
            flow = conversation.session._engine.flow
            self.assertEqual(flow.size, sum(flow.shown))
            counted = tokens([model.bodies[-1]["messages"]])
            self.assertLess(abs(flow.size - counted) / counted, 0.3)

    def held(self, body):
        replies = [m for m in body["messages"] if m["role"] == "assistant"]
        return [
            k
            for k, m in enumerate(replies)
            if any(b["type"] == "thinking" for b in m["content"])
        ]

    def test_a_cut_takes_what_was_made_behind_it(self):
        for stream in (False, True):
            for away in (0, 400):
                with self.subTest(stream=stream, away=away):
                    app, model = self.play(5, 9, 7, stream=stream, away=away)
                    self.check(app, model)
                    last = model.bodies[-1]
                    self.assertTrue(noticed(last))
                    replies = sum(m["role"] == "assistant" for m in last["messages"])
                    kept = self.held(last)
                    # What was reasoned since the latest cut is there, and it is the newest.
                    self.assertEqual(kept, list(range(replies - len(kept), replies)))

    def test_it_stays_until_something_is_cut(self):
        app, model = self.play(6, 6, rounds=30)
        self.assertEqual(self.held(model.bodies[-1]), list(range(13)))
        self.check(app, model)

    def test_what_is_made_after_a_cut_stays(self):
        app, model = self.play(6, rounds=4, ratio=0.0)
        before = [len(self.held(b)) for b in model.bodies]
        # Four rounds stay whole, so from the fifth result on every call cuts one more.
        self.assertEqual(before[:5], [0, 1, 2, 3, 4])
        app, model = self.play(6, 1, rounds=4, ratio=None, away=400)
        # Cut once, when the cache had expired. What came after it stayed.
        self.assertEqual([len(self.held(b)) for b in model.bodies[-2:]], [0, 1])
        self.check(app, model)

    def test_reasoning_of_the_turn_that_goes_on_is_worth_keeping(self):
        """Little that is old does not cost the model what it has in mind, though the cache is gone."""
        clock = Clock()
        model = FakeModel(clock=lambda: clock.at, binds="error")
        live = compache.Live(policy(2, None, spec=self.SPEC), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.shape.ask("Work")
        small = [60, 60, 60, 60]
        for k, size in enumerate(small, 1):
            model.script = lambda request: {"tool": ("read", {}), "thinking": 3000}
            _, calls = app.call()
            clock.at += 400
            app.shape.answer([(c, f"line {k} " * size) for c, _ in calls])
        app.call()
        self.assertEqual(len(self.held(model.bodies[-1])), 4)
        self.assertFalse(noticed(model.bodies[-1]))

    def test_a_conversation_met_in_the_middle(self):
        clock = Clock()
        model = FakeModel(clock=lambda: clock.at, binds="error")
        app, _ = self.play(5, live=compache.Live(clock=clock), clock=clock, model=model)
        self.assertEqual(self.held(model.bodies[-1]), list(range(5)))
        # Its reasoning was made without our tool. It is left out, and what comes after it stays.
        app.live = compache.Live(policy(30, spec=self.SPEC), clock=clock)
        self.more(app, 4)
        self.check(app, model)
        self.assertEqual(self.held(model.bodies[-1]), [6, 7, 8, 9])
        # And met once more, by a process that knows nothing of the first.
        app.live = compache.Live(policy(30, spec=self.SPEC), clock=clock)
        self.more(app, 4)
        self.check(app, model)

    def test_a_policy_that_is_refused(self):
        """What was reasoned under the policy does not go back as the application wrote it."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                app, model = self.play(6, stream=stream)
                refusing = reasoning(11)

                def script(request, refusing=refusing):
                    if noticed(request["body"]):
                        return {"status": 400, "error": "no notices here"}
                    return refusing(request)

                model.script = script
                self.assertEqual(app.run("Work through five more"), "All done.")
                self.assertEqual(app.live.stats["policy_refused"], 1)
                self.assertFalse(noticed(model.bodies[-1]))
                # From there on its reasoning is made behind what the application writes.
                self.assertEqual(self.held(model.bodies[-1]), [7, 8, 9, 10, 11])
                self.more(app, 3)

    def working(self, size=100):
        """An agent that goes on reading, and a model that `what` can make do something else once."""
        what: list = []

        def script(request):
            made = {"tool": ("read", {"file": request["n"]}), "thinking": size}
            return what.pop(0)(request) if what else made

        clock = Clock()
        model = FakeModel(script, clock=lambda: clock.at, binds="error")
        live = compache.Live(policy(3, None, spec=self.SPEC), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.shape.ask("Work")
        return app, model, what, clock

    def read(self, app, count, away=0):
        for _ in range(count):
            _, calls = app.call()
            app.shape.answer([(call, result(app.made)) for call, _ in calls])
            app.made += 1
        app.clock.at += away

    def test_a_request_that_is_not_taken_costs_no_reasoning(self):
        """One with a tool that is refused, or with other instructions while the provider is overloaded."""
        for odd, status in (
            ({"tools": [{"name": "broken", "input_schema": {}}]}, 400),
            ({"system": "Another mode."}, 529),
        ):
            with self.subTest(status=status):
                app, model, what, _ = self.working()
                self.read(app, 5)
                plain = app.shape.body
                app.shape.body = lambda stream, plain=plain, odd=odd: {
                    **plain(stream),
                    **odd,
                }
                for _ in range(3):
                    what.append(lambda request, s=status: {"status": s, "error": "no"})
                with self.assertRaises(KeyError):
                    app.call()
                what.clear()
                app.shape.body = plain
                before = app.live.tokens["cache_read"]
                self.read(app, 1)
                self.assertEqual(self.held(model.bodies[-1]), [0, 1, 2, 3, 4])
                self.assertGreater(app.live.tokens["cache_read"] - before, 1000)

    def test_a_chat_whose_first_cut_is_refused(self):
        """What the application wrote goes out with all that was reasoned before."""

        def script(request):
            if any(tool["name"] == REOPEN for tool in request["body"]["tools"]):
                return {"status": 400, "error": "tool 'reopen' is not allowed here"}
            return {"text": f"Answer {request['n']}.", "thinking": 300}

        clock = Clock(1_970_000_000.0)
        model = FakeModel(script, clock=clock, binds="error")
        live = compache.Live(test_live.fitted("claude-sonnet-5-5"), clock=clock)
        shape = Claude()
        shape.model = "claude-sonnet-5-5"
        app = Agent(live, shape, model, thread="t")
        for n in range(1, 7):
            shape.ask(f"What does section {n} say?")
            call = {"type": "tool_use", "id": f"toolu_{n:020d}aB", "name": "read"}
            shape.said.append({"role": "assistant", "content": [{**call, "input": {}}]})
            shape.answer([(call["id"], test_live.passage(n))])
            app.call()
            clock.at += 3600
        self.assertEqual(live.stats["policy_refused"], 1)
        # A turn's call was made by nobody here. Its answer is every second reply.
        self.assertEqual(len(self.held(model.bodies[-1])), 5)

    def test_a_refusal_costs_no_reasoning(self):
        """What is wrong with the request itself says nothing about the reasoning in it."""
        app, model, what, _ = self.working()
        self.read(app, 5)
        for _ in range(2):
            what.append(lambda request: {"status": 400, "error": "image too large"})
        with self.assertRaises(KeyError):
            app.call()
        before = app.live.tokens["cache_read"]
        self.read(app, 1)
        self.assertEqual(self.held(model.bodies[-1]), [0, 1, 2, 3, 4])
        self.assertGreater(app.live.tokens["cache_read"] - before, 1000)

    def test_a_provider_that_is_overloaded_costs_no_reasoning(self):
        app, model, what, _ = self.working()
        self.read(app, 5, away=400)
        self.read(app, 5)
        before = len(self.held(model.bodies[-1]))

        def fetch(request):
            handle = NOTICE.findall(json.dumps(request["body"]))[0]
            return {"tool": (REOPEN, {"reference": handle}), "thinking": 100}

        what += [fetch, *[lambda request: {"status": 529, "error": "overloaded"}] * 2]
        with self.assertRaises(KeyError):
            app.call()
        read = app.live.tokens["cache_read"]
        app.call()
        # And what the model reasoned when it looked back.
        self.assertEqual(len(self.held(model.bodies[-1])), before + 2)
        self.assertGreater(app.live.tokens["cache_read"] - read, 1000)

    def test_a_reply_that_is_thrown_away(self):
        """The turn goes on in the request that was asked anew, with what that had fetched."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                app, model, what, _ = self.working()
                app.stream = stream
                self.read(app, 6, away=400)
                self.read(app, 3)

                def fetch(request):
                    handle = NOTICE.findall(json.dumps(request["body"]))[-1]
                    return {"tool": (REOPEN, {"reference": handle})}

                app.call()
                said = app.shape.said.pop()
                self.assertEqual(said["role"], "assistant")
                what += [fetch, lambda request: {"text": "So it is.", "thinking": 90}]
                self.assertEqual(app.call()[0], "So it is.")
                before = len(model.bodies)
                app.shape.ask("And now?")
                app.call()
                self.assertEqual(len(model.bodies), before + 1)
                fetched = json.dumps(model.bodies[-1]["messages"]).count('"reopen"')
                self.assertEqual(fetched, 1)
                self.assertEqual(app.live.stats["stood_down"], 0)
                self.assertEqual(app.live.stats["retried"], 0)

    def test_one_that_was_begun_elsewhere_and_is_left_as_it_is(self):
        """Its reasoning was made behind what another process had sent. It is asked again without."""
        for policies in ({"record": True}, {}):
            with self.subTest(**policies):
                app, model, _, clock = self.working()
                self.read(app, 4)
                with tempfile.TemporaryDirectory() as folder:
                    record = folder if policies else None
                    app.live = compache.Live(record=record, clock=clock)
                    self.read(app, 1)
                    self.assertEqual(self.held(model.bodies[-1]), [])
                    self.assertEqual(app.live.stats["retried"], 1)
                    # What is made from there on goes back.
                    self.read(app, 2)
                    self.assertEqual(self.held(model.bodies[-1]), [4, 5])
                    self.assertEqual(app.live.stats["retried"], 1)

    def test_another_model_that_is_left_as_it_is(self):
        clock = Clock(1.9e9)
        model = FakeModel(clock=lambda: clock.at, binds="error")
        chosen = policy(3, None, spec=self.SPEC)

        def choose(learner, name, tools):
            return (chosen, False) if "haiku" in name else (None, True)

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(Learner, "choose", choose):
                live = compache.Live(learn=folder, clock=clock)
                app = Agent(live, Claude(), model, clock=clock)
                app.shape.model = "claude-haiku-5-5"
                self.more(app, 8)
                clock.at += 4000
                self.more(app, 2)
                self.assertTrue(noticed(model.bodies[-1]))
                app.shape.model = "claude-sonnet-5-5"
                self.more(app, 2)
                self.assertFalse(noticed(model.bodies[-1]))
                # Which does not make it the first one's way.
                app.shape.model = "claude-haiku-5-5"
                self.more(app, 1)
                self.assertTrue(noticed(model.bodies[-1]))

    def test_a_gateway_s_own_name_for_the_model(self):
        """What it is, is known once it has replied. What it reasoned in that reply stays."""

        class Named(FakeModel):
            def messages(self, body, *more):
                status, said = super().messages(body, *more)
                name = "claude-sonnet-5-5-20260901"
                return status, {**said, "model": name} if status == 200 else said

        app, model, what, clock = self.working()
        model = app.model = Named(model.script, binds="error")
        app.shape.model = "smart"
        self.read(app, 4)
        self.assertEqual(self.held(model.bodies[-1]), [0, 1, 2])
        self.assertEqual(app.live.stats["retried"], 0)

    def test_after_a_fetch(self):
        for stream in (False, True):
            for beside in (False, True):
                with self.subTest(stream=stream, beside=beside):
                    fetched = []

                    def script(request, fetched=fetched, beside=beside):
                        made = reasoning(12)(request)
                        body = json.dumps(request["body"])
                        if "tool" in made and len(fetched) < 3 and noticed(body):
                            handle = NOTICE.findall(body)[0]
                            fetched.append(handle)
                            ask = (REOPEN, {"reference": handle})
                            made.pop("tool")
                            made["tools"] = [ask, ("read", {})] if beside else [ask]
                            made["text"] = "Let me look."
                        return made

                    clock = Clock()
                    model = FakeModel(script, clock=lambda: clock.at, binds="error")
                    live = compache.Live(
                        policy(3, spec=self.SPEC, ratio=1.0), clock=clock
                    )
                    app = Agent(live, Claude(), model, stream=stream, clock=clock)
                    said = app.run("Work through twelve files")
                    self.assertTrue(said.endswith("All done."), said)
                    self.assertEqual(live.stats["reopened"], 3)
                    self.assertEqual(live.stats["retried"], 0)
                    self.assertEqual(live.stats["stood_down"], 0)

    def test_reasoning_after_a_fetch_goes_back(self):
        """A reply that was read in full goes back as it was made: behind what it had fetched."""
        clock = Clock()
        model = FakeModel(clock=lambda: clock.at, binds="error")
        live = compache.Live(policy(3, None, spec=self.SPEC), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        model.script = reasoning(6)
        app.run("Work through six files")
        clock.at += 400
        fetched = []

        def script(request):
            made = reasoning(11)(request)
            body = json.dumps(request["body"])
            if not fetched and noticed(body):
                fetched.append(NOTICE.findall(body)[0])
                made["tool"] = (REOPEN, {"reference": fetched[0]})
                made["text"] = "Let me look."
            return made

        model.script = script
        app.run("Work through five more")
        self.assertEqual(live.stats["reopened"], 1)
        # That of the reply which fetched, and of the four that read on.
        self.assertEqual(len(self.held(model.bodies[-1])), 5)
        self.check(app, model)

    def test_a_chat(self):
        """Not only an agent's policy changes what stands ahead of reasoning."""
        for stream in (False, True):
            for notes in ("apart", "inline"):
                with self.subTest(stream=stream, notes=notes):
                    clock = Clock(1_990_000_000.0)

                    def script(request):
                        last = request["body"]["messages"][-1]["content"]
                        if isinstance(last, list) and last[0]["type"] == "tool_result":
                            return {"text": f"Answer {request['n']}.", "thinking": 90}
                        return {"tool": ("search_docs", {"query": "x"}), "thinking": 60}

                    model = FakeModel(script, clock=clock, binds="error")
                    folder = tempfile.TemporaryDirectory()
                    self.addCleanup(folder.cleanup)
                    live = compache.Live(
                        test_live.fitted(),
                        record=folder.name,
                        sample=1.0,
                        notes=notes,
                        clock=clock,
                    )
                    shape = Claude()
                    shape.model = "claude-sonnet-5-5"
                    app = Agent(live, shape, model, stream=stream)
                    for n in range(1, 7):
                        shape.ask(f"What does section {n} say?")
                        _, calls = app.call()
                        self.assertTrue(calls)
                        found = test_live.passage(n)
                        shape.answer([(call, found) for call, _ in calls])
                        text, calls = app.call()
                        self.assertFalse(calls)
                        self.assertTrue(text.startswith("Answer"), text)
                        live.drain(5)
                        clock.at += 4000
                    self.assertTrue(noticed(model.bodies[-1]))
                    self.assertEqual(live.stats["policy_refused"], 0)
                    self.assertEqual(live.stats["stood_down"], 0)

    def test_two_that_go_on_from_the_same_start(self):
        app, model = self.play(8)
        other = Agent(app.live, Claude(), model, clock=app.clock)
        other.shape.said = list(app.shape.said)
        for k in range(3):
            for one in (app, other):
                one.made = app.made
                self.more(one, 2 + k)
        self.assertEqual(app.live.stats["retried"], 0)
        self.assertEqual(app.live.stats["stood_down"], 0)

    def test_an_application_that_rewrites_what_it_had_said(self):
        app, model = self.play(8)
        for message in app.shape.said[3:6]:
            if message["role"] == "user" and isinstance(message["content"], list):
                for block in message["content"]:
                    if block["type"] == "tool_result":
                        block["content"] = "[cleared]"
        self.more(app, 3)
        self.assertEqual(app.live.stats["retried"], 0)

    def test_cache_marks_that_move_change_nothing(self):
        """Wherever an application puts them: on a result, or on the text within one."""

        class Marking(Claude):
            within = False

            def body(self, stream):
                body = json.loads(json.dumps(super().body(stream)))
                for message in body["messages"]:
                    content = message["content"]
                    for block in content if isinstance(content, list) else ():
                        if block["type"] == "tool_result":
                            text = {"type": "text", "text": block["content"]}
                            block["content"] = [text]
                last = body["messages"][-1]["content"]
                if isinstance(last, list):
                    marked = last[-1]["content"][-1] if self.within else last[-1]
                    marked["cache_control"] = {"type": "ephemeral"}
                return body

        for within in (False, True):
            with self.subTest(within=within):
                app, model, _, _ = self.working()
                app.shape = Marking()
                app.shape.within = within
                app.shape.ask("Work")
                app.live = compache.Live(policy(30, spec=self.SPEC), clock=app.clock)
                self.read(app, 7)
                self.assertEqual(self.held(model.bodies[-1]), list(range(6)))

    def test_older_models_keep_theirs(self):
        app, model = self.play(5, 4, shape=OldClaude(), model=FakeModel())
        self.assertEqual(len(self.held(model.bodies[-1])), 4)
        shape = Claude()
        plain = shape.body
        manual = {"type": "enabled", "budget_tokens": 2000}
        shape.body = lambda stream: {**plain(stream), "thinking": manual}
        app, model = self.play(5, shape=shape, model=FakeModel())
        self.assertEqual(len(self.held(model.bodies[-1])), 5)

    def test_names_of_models(self):
        for name, binds in (
            ("claude-haiku-5-5", True),
            ("anthropic/claude-haiku-5.5", True),
            ("claude-sonnet-5-5-20260928", True),
            ("us.anthropic.claude-opus-5-5-v1:0", True),
            ("claude-fable-5-1", True),
            ("claude-opus-6", True),
            ("claude-sonnet-5-20260601", False),
            ("claude-opus-5", False),
            ("claude-opus-4-8", False),
            ("claude-3-5-sonnet-20241022", False),
            ("gpt-6-luna", False),
            ("kimi-k3", False),
            ("haiku", True),
            (None, False),
        ):
            self.assertEqual(binds_reasoning(name), binds, name)
        # A budget for it says that a name without a version is an older model's.
        self.assertFalse(binds_reasoning("haiku", True))
        self.assertTrue(binds_reasoning("anthropic/claude-haiku-5.5", True))

    def test_a_budget_through_a_gateway(self):
        """An agent may think that it speaks to an older model, and asks as that one wants it."""

        class Asking(Claude):
            model = "claude-haiku-4-5"

            def body(self, stream):
                budget = {"type": "enabled", "budget_tokens": 2048}
                return {**super().body(stream), "thinking": budget}

        class Served(FakeModel):
            def messages(self, body, *more):
                status, said = super().messages(body, *more)
                name = "anthropic/claude-haiku-5.5"
                return status, {**said, "model": name} if status == 200 else said

        app, model, _, clock = self.working()
        served = app.model = Served(model.script, binds="error")
        app.shape, app.live = Asking(), compache.Live(policy(3, None), clock=clock)
        app.shape.ask("Work")
        self.read(app, 6, away=400)
        self.read(app, 3)
        self.assertTrue(noticed(served.bodies[-1]))
        self.assertEqual(app.live.stats["retried"], 0)
        # What stands ahead of the cut, and what came after it.
        self.assertEqual(self.held(served.bodies[-1]), [0, 6, 7])


class ParallelTests(unittest.TestCase):
    def test_agents_that_share_a_name_are_kept_apart(self):
        model = FakeModel(worker(6))
        live = compache.Live(policy(rounds=2))
        first = Agent(live, Claude(), model, thread="session-1")
        second = Agent(live, Claude(), model, thread="session-1")
        first.shape.ask("The main task")
        second.shape.ask("A subtask for another agent")
        done = []
        for k in range(1, 12):
            for app in (first, second):
                if app in done:
                    continue
                text, calls = app.call()
                if not calls:
                    done.append(app)
                    continue
                app.made += 1
                app.shape.answer([(c, result(app.made)) for c, _ in calls])
        self.assertEqual(len(done), 2)
        self.assertEqual(live.stats["turns"], 2)
        for body in model.bodies[-2:]:
            self.assertEqual(
                [noticed(t) for t in Claude.results(body)], [True] * 4 + [False] * 2
            )

    def test_two_agents_at_the_same_moment(self):
        model = FakeModel(worker(8))
        live = compache.Live(policy(rounds=2))
        answers = []

        def work(k):
            app = Agent(live, Claude(), model, stream=k % 2 == 0)
            answers.append(app.run("The same words to open with"))

        threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(answers, ["All done."] * 6)
        self.assertEqual(live.stats["turns"], 6)
        self.assertEqual(live.stats["stood_down"], 0)

    def test_a_side_request_of_the_host_does_not_lose_the_main_line(self):
        """A host asks for a suggestion behind the conversation, and then the user goes on."""
        clock = Clock()
        model = FakeModel(worker(4), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=2, ratio=5.0), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.run("Read four files")
        kept = list(app.shape.said)
        app.shape.ask("Suggest what the user may say next")
        app.call()
        app.shape.said = kept
        model.script = worker(6)
        clock.at += 20
        app.run("Read two more")
        # The cache was alive all along, so nothing was cut for free.
        self.assertFalse(any(noticed(b["messages"]) for b in model.bodies))

    def test_a_history_the_application_rewrote_is_read_anew(self):
        model = FakeModel(worker(9))
        live = compache.Live(policy(rounds=2))
        app = Agent(live, Claude(), model)
        app.shape.ask("Read nine files")
        for k in range(1, 10):
            _, calls = app.call()
            app.shape.answer([(c, result(k)) for c, _ in calls])
            if k == 5:
                # It shortens what its tools returned, as some harnesses do.
                for message in app.shape.said:
                    for block in message["content"]:
                        if isinstance(block, dict) and block["type"] == "tool_result":
                            block["content"] = block["content"][:90]
        text, calls = app.call()
        self.assertEqual((text, calls), ("All done.", []))
        sent = Claude.results(model.bodies[-1])
        self.assertFalse(any(noticed(t) for t in sent[-2:]))
        self.assertEqual(live.stats["stood_down"], 0)
        self.assertEqual(live.stats["turns"], 1)


class AgainTests(unittest.TestCase):
    """A model that works with more than stays in view."""

    def test_what_is_fetched_twice_makes_results_stay_longer(self):
        for shape in SHAPES:
            with self.subTest(shape.__name__):
                live, fetches = compache.Live(policy(2)), []
                app = Agent(live, shape(), FakeModel(None))

                def script(request, app=app, fetches=fetches):
                    body = json.dumps(request["body"])
                    for k in range(1, min(3, app.made) + 1):
                        if f"line of file number {k} that" not in body:
                            fetches.append(k)
                            looks = (REOPEN, {"reference": f"file number {k} that"})
                            return {"text": "", "tool": looks}
                    return {"text": "Reading on.", "tool": ("read", {})}

                app.model.script = script
                app.shape.ask("Compare the first three with all others")
                for _ in range(60):
                    _, calls = app.call()
                    app.made += 1
                    app.shape.answer([(call, result(app.made)) for call, _ in calls])
                # Each time they stay for twice as long. Otherwise it would be one call in two.
                self.assertGreaterEqual(len(fetches), 6)
                self.assertLess(len(fetches), 20)
                (conversation,) = live.conversations()
                self.assertGreaterEqual(conversation.session._engine.flow.longer, 14)
                self.assertEqual(live.stats["stood_down"], 0)

    def test_what_its_own_tools_return_again_makes_results_stay_longer(self):
        """It needs six results at once, writes nothing down and reads them over and over."""
        for shape in SHAPES:
            with self.subTest(shape.__name__):
                live, reads = compache.Live(policy(2)), []
                app = Agent(live, shape(), FakeModel(None))

                def script(request, reads=reads):
                    body = json.dumps(request["body"])
                    if all(
                        f"line of file number {k} that" in body for k in range(1, 7)
                    ):
                        return {"text": "All done."}
                    reads.append(len(reads) % 6 + 1)
                    return {"text": "", "tool": ("read", {"file": reads[-1]})}

                app.model.script = script
                app.shape.ask("Compare six files")
                for _ in range(80):
                    _, calls = app.call()
                    if not calls:
                        break
                    app.shape.answer([(call, result(reads[-1])) for call, _ in calls])
                self.assertFalse(calls, "it goes round for ever")
                self.assertLess(len(reads), 40)
                self.assertEqual(live.stats["stood_down"], 0)

    def test_a_test_that_is_run_again_and_again_is_no_sign_of_that(self):
        live, made = compache.Live(policy(2)), [0]

        def script(request):
            made[0] += 1
            return {"text": "", "tool": ("test" if made[0] % 3 else "read", {})}

        app = Agent(live, Claude(), FakeModel(script))
        app.shape.ask("Mend it")
        for k in range(60):
            _, calls = app.call()
            said = "all tests pass\n" * 40 if made[0] % 3 else result(k)
            app.shape.answer([(call, said) for call, _ in calls])
        (conversation,) = live.conversations()
        self.assertEqual(conversation.session._engine.flow.longer, 0)

    def test_there_has_to_be_room_for_what_stays_longer(self):
        """And a request that is billed dearer for its size is none to grow into."""
        dearer = {"long_input_factor": 5.0, "long_output_factor": 5.0}
        dearer["schema_version"] = "compache-model-spec-v2"
        for spec in (
            replace(SPEC, context_tokens=9_000),
            replace(SPEC, long_input_threshold=9_000, **dearer),
        ):
            flow, largest = agent.Flow(spec, agent.AgentRecipe(2, 0.0)), 0
            for k in range(60):
                flow.rounds += 1
                flow.add(700, f"result {k}")
                if k in (5, 10, 15, 30):
                    flow.stay_longer()
                flow.cut(5.0 * k)
                flow.sent(5.0 * k, flow.size)
                largest = max(largest, flow.size)
            self.assertGreater(largest, 5_000)
            self.assertLess(largest, 9_800)

    def test_what_is_no_repetition(self):
        """The same twice in one reply, and what comes after the application has rewritten its history."""
        fetch: list = []

        def script(request):
            handles = NOTICE.findall(json.dumps(request["body"]["messages"]))
            made = {"tool": ("read", {"file": len(request["texts"])})}
            if fetch:
                looks = (REOPEN, {"reference": handles[0]})
                made = {"tools": [looks] * fetch.pop()}
            return made

        clock = Clock()
        live = compache.Live(policy(2), clock=clock)
        app = Agent(live, Claude(), FakeModel(script), clock=clock)
        app.shape.ask("Go")

        def read(*files):
            for k in files:
                _, calls = app.call()
                app.shape.answer([(call, result(k)) for call, _ in calls])

        def longer():
            (conversation,) = live.conversations()
            return conversation.session._engine.flow.longer

        read(*range(6))
        fetch.append(2)
        read(6)
        self.assertEqual((live.stats["reopened"], longer()), (2, 0))
        for message in app.shape.said:
            if message["role"] == "user" and isinstance(message["content"], list):
                for block in message["content"]:
                    block["content"] = block["content"].replace("a line of", "pruned")
        read(*range(7, 12))
        fetch.append(1)
        read(12)
        self.assertEqual((live.stats["reopened"], longer()), (3, 0))

    def test_pages_of_one_result_are_no_repetition(self):
        live, asked = compache.Live(policy(2)), []

        def script(request):
            handles = NOTICE.findall(json.dumps(request["body"]))
            if handles and len(asked) < 3:
                asked.append(handles[0])
                where = {"reference": handles[0], "offset": 400 * len(asked)}
                return {"text": "", "tool": (REOPEN, where)}
            return {"text": "Reading on.", "tool": ("read", {})}

        app = Agent(live, Claude(), FakeModel(script))
        app.shape.ask("Read on")
        for k in range(1, 12):
            _, calls = app.call()
            app.shape.answer([(call, result(k)) for call, _ in calls])
        self.assertEqual(live.stats["reopened"], 3)
        (conversation,) = live.conversations()
        self.assertEqual(conversation.session._engine.flow.longer, 0)


class SwitchTests(unittest.TestCase):
    """One model plans, another works, and the first is asked again while its cache lives."""

    def play(self, stream=False, rounds=3, ratio=None, record=None):
        class PerModel(FakeModel):
            def messages(self, body, *more):
                self.asked = body["model"]
                return super().messages(body, *more)

            def _meter(self, head, *more):
                return super()._meter([self.asked, head], *more)

        clock = Clock()
        model = PerModel(clock=lambda: clock.at, binds="error")
        live = compache.Live(policy(rounds, ratio), clock=clock, record=record)
        app = Agent(live, Claude(), model, thread="s", clock=clock, stream=stream)

        def turn(name, count):
            """What the turn read of the cache with its first request, and how many notices that held."""
            app.shape.model = name
            script, first = reasoning(app.made + count), []
            before = live.tokens["cache_read"]

            def watched(request):
                if len(first) == 1:
                    first.append(live.tokens["cache_read"] - before)
                if not first:
                    first.append(len(NOTICE.findall(json.dumps(request["body"]))))
                return script(request)

            model.script = watched
            self.assertEqual(app.run(f"Work through {count} files"), "All done.")
            clock.at += 20
            return first[1], first[0]

        return app, model, turn

    def test_the_first_goes_on_where_it_left_off(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                app, model, turn = self.play(stream, rounds=12)
                turn("claude-opus-5-5", 20)
                app.clock.at += 400
                turn("claude-opus-5-5", 2)
                whole, notices = turn("claude-opus-5-5", 2)
                self.assertGreater(notices, 6)
                turn("claude-sonnet-5-5", 2)
                read, shown = turn("claude-opus-5-5", 2)
                # What it had in its cache it reads, and what is out of view is so as before.
                self.assertGreater(read, 0.9 * whole)
                self.assertEqual(shown, notices)
                self.assertEqual(app.live.stats["retried"], 0)
                self.assertEqual(app.live.stats["stood_down"], 0)
                # And so it does after several turns of the other.
                for _ in range(3):
                    turn("claude-sonnet-5-5", 1)
                read, shown = turn("claude-opus-5-5", 1)
                self.assertGreater(read, 0.9 * whole)
                self.assertEqual(shown, notices)
                self.assertEqual(app.live.stats["retried"], 0)

    def test_it_cuts_as_the_policy_says(self):
        app, model, turn = self.play(ratio=0.0)
        for name in ("opus", "sonnet", "opus", "opus", "sonnet", "opus"):
            turn(f"claude-{name}-5-5", 4)
            held = Claude.results(model.bodies[-1])
            self.assertEqual(sum(not NOTICE.search(text) for text in held), 3)
        self.assertEqual(app.live.stats["retried"], 0)

    def test_what_went_to_the_other_can_be_fetched_back(self):
        app, model, turn = self.play(ratio=0.0)
        turn("claude-opus-5-5", 8)
        turn("claude-sonnet-5-5", 4)
        turn("claude-opus-5-5", 6)
        for k in (2, 10):
            fetched = []

            def script(request, k=k, fetched=fetched):
                if not fetched:
                    fetched.append(k)
                    words = f"a line of file number {k} that"
                    return {"tool": (REOPEN, {"reference": words})}
                return {"text": "Found."}

            model.script = script
            app.shape.ask(f"What was in file {k}?")
            self.assertEqual(app.call()[0], "Found.")
            sent = json.dumps(model.bodies[-1]["messages"])
            self.assertIn(f"result {k}: a line of file number {k}", sent)

    def test_every_turn_is_on_record_once(self):
        with tempfile.TemporaryDirectory() as folder:
            app, model, turn = self.play(record=folder)
            for name in ("opus", "sonnet", "opus", "sonnet", "opus"):
                turn(f"claude-{name}-5-5", 3)
            threads, _ = load(folder)
            runs = [run for thread in threads for run in thread["runs"]]
            self.assertEqual([len(threads), len(runs)], [1, 5])
            self.assertEqual([run["calls"] for run in runs], [4] * 5)

    def test_another_agent_that_goes_on_from_the_same_start_is_left_alone(self):
        """The one that a copy was made of goes on as it was."""
        app, model, turn = self.play()
        turn("claude-opus-5-5", 8)
        other = Agent(app.live, Claude(), model, thread="s", clock=app.clock)
        other.shape.said = [*app.shape.said]
        other.shape.ask("Said elsewhere")
        other.shape.said.append(
            {"role": "assistant", "content": [{"type": "text", "text": "Answered."}]}
        )
        other.shape.model, other.made = "claude-opus-5-5", app.made
        model.script = reasoning(app.made + 3)
        self.assertEqual(other.run("Work through three files"), "All done.")
        read, _ = turn("claude-opus-5-5", 2)
        self.assertGreater(read, 1000)
        self.assertEqual(app.live.stats["retried"], 0)


class BesideTests(unittest.TestCase):
    def test_another_model_beside_is_one_conversation_more(self):
        for thread in (None, "t"):
            with self.subTest(thread=thread):
                model = FakeModel(worker(9))
                live = compache.Live(policy(2))
                app = Agent(live, Claude(), model, thread=thread)
                app.run("Work through nine files")
                model.script = worker(99)
                app.shape.ask("Go on")
                _, calls = app.call()
                app.shape.answer([(call, result(1)) for call, _ in calls])
                app.call()
                before = len(live.conversations())
                other = Agent(live, Claude(), model, thread=thread)
                other.shape.model = "claude-other"
                other.shape.said = [*app.shape.said[:-3]]
                other.shape.ask("A title, please")
                model.script = lambda request: {"text": "Nine files"}
                with mock.patch.object(
                    Session, "copy", autospec=True, side_effect=Session.copy
                ) as copied:
                    other.call()
                self.assertLessEqual(copied.call_count, 4)
                self.assertLessEqual(len(live.conversations()), before + 1)

    def test_another_model_beside_one_that_is_left_alone(self):
        model = FakeModel(worker(3))
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(Learner, "choose", lambda *of: (None, True)):
                live = compache.Live(learn=folder)
                app = Agent(live, Claude(), model, thread="t")
                app.run("Work through three files")
                app.shape.ask("Go on")
                app.call()
                other = Agent(live, Claude(), model, thread="t")
                other.shape.model = "claude-other"
                other.shape.said = [*app.shape.said[:-2]]
                other.shape.ask("A title, please")
                model.script = lambda request: {"text": "Three files"}
                made = []
                whole = live_module._Conversation.__init__

                def init(self, *of, **named):
                    made.append(self)
                    whole(self, *of, **named)

                with mock.patch.object(live_module._Conversation, "__init__", init):
                    other.call()
                self.assertLessEqual(len(made), 1)


class ToolPingTests(unittest.TestCase):
    def test_the_cache_is_kept_warm_while_a_tool_runs(self):
        for shape in (Claude, Chat, Codex):
            with self.subTest(shape=shape.__name__):
                clock = Clock()
                model = FakeModel(worker(3), clock=lambda: clock.at)
                live = compache.Live(
                    policy(rounds=None, ratio=None, tool_pings=2), clock=clock
                )
                app = Agent(live, shape(), model, stream=True)
                app.shape.ask("Run the tests")
                _, calls = app.call()
                due = live.next_ping_at()
                self.assertAlmostEqual(due - clock.at, 255, delta=1)
                clock.at += 256
                self.assertEqual(live.tick(), 1)
                ping = model.bodies[-1]
                self.assertNotIn("stream", ping)
                before = {k: v for k, v in model.bodies[-2].items() if k != "stream"}
                limits = {"max_tokens", "max_output_tokens"}
                self.assertEqual(
                    {k: v for k, v in ping.items() if k not in limits},
                    {k: v for k, v in before.items() if k not in limits},
                )
                self.assertLessEqual(
                    ping.get("max_tokens") or ping.get("max_output_tokens"), 16
                )
                clock.at += 200
                app.shape.answer([(c, result(1)) for c, _ in calls])
                app.call()
                self.assertIsNotNone(live.next_ping_at(), "planned anew")

    def test_after_an_answer_the_same_request_goes_out_once_more(self):
        """A further message would have the provider leave reasoning out, and miss its own cache."""
        clock = Clock()
        model = FakeModel(reasoning(3), clock=lambda: clock.at)
        live = compache.Live(policy(None, None, user_pings=2), clock=clock)
        app = Agent(live, Claude(), model, clock=clock)
        app.run("Read three files")
        before = model.bodies[-1]
        clock.at += 256
        self.assertEqual(live.tick(), 1)
        self.assertEqual(model.bodies[-1]["messages"], before["messages"])
        self.assertIsNotNone(live.next_ping_at())

    def test_a_ping_that_read_only_what_opens_the_request_is_the_last(self):
        session = policy(None, None, tool_pings=3).session("t")
        turn = session.prepare("Run the tests " * 4_000, at=0.0)
        usage = {"input_tokens": 100, "cache_creation_input_tokens": 20_000}
        session.replied(turn, "", [{}], at=4.0, usage={**usage, "output_tokens": 9})
        read = {"input_tokens": 10, "cache_read_input_tokens": 4_000}
        read["cache_creation_input_tokens"] = 16_000
        self.assertFalse(session.ping_sent(at=256.0, usage=read))
        self.assertIsNone(session.next_ping_at)

    def test_no_ping_goes_out_while_a_call_is_under_way(self):
        clock = Clock()
        model = FakeModel(worker(3), clock=lambda: clock.at)
        live = compache.Live(policy(None, None, tool_pings=2), clock=clock)
        app = Agent(live, Claude(), model)
        app.shape.ask("Run the tests")
        _, calls = app.call()
        app.shape.answer([(c, result(1)) for c, _ in calls])
        inside = []

        def slow(body):
            clock.at += 300
            inside.append(live.tick())
            return FakeModel.send(model, "/v1/messages")(body)

        live.exchange("/v1/messages", app.shape.body(False), slow)
        self.assertEqual(inside, [0])

    def test_a_thinking_budget_is_not_pinged(self):
        self.assertIsNone(
            wire.Anthropic.again({"thinking": {"budget_tokens": 1024}}, 1)
        )
        self.assertEqual(
            wire.Anthropic.again({"thinking": {"type": "adaptive"}}, 1)["max_tokens"], 1
        )


class HeaderTests(unittest.TestCase):
    """Instructions and tools cached for an hour, the conversation behind them for five minutes."""

    def policy(self):
        made = policy(rounds=2, spec=SPEC)
        hour = {**SPEC.to_dict(), "cache_ttl_seconds": 3600, "write_multiplier": 2.0}
        return compache.Policy({**made.artifact, "header_spec": hour})

    def marks(self, body):
        found = []
        for path in wire._marks(body):
            found.append((path[0], wire.get(body, path)["cache_control"].get("ttl")))
        return found

    def test_the_marks_ask_for_it(self):
        model = FakeModel(worker(3))
        live = compache.Live(self.policy())
        Agent(live, Claude(), model).run("Read three files")
        for body in model.bodies:
            marks = self.marks(body)
            self.assertEqual(marks[0], ("system", "1h"))
            self.assertTrue(all(ttl is None for _, ttl in marks[1:]), marks)

    def test_the_application_s_own_marks_ahead_of_the_conversation_ask_for_it(self):
        shape = Claude()
        body = shape.body

        def marked(stream):
            made = body(stream)
            mark = {"type": "ephemeral"}
            made["system"] = [
                {"type": "text", "text": "You are", "cache_control": mark},
                {"type": "text", "text": "an agent.", "cache_control": mark},
            ]
            return made

        shape.body = marked
        model = FakeModel(worker(3))
        Agent(compache.Live(self.policy()), shape, model).run("Read three files")
        marks = self.marks(model.bodies[-1])
        self.assertEqual(marks[:2], [("system", "1h"), ("system", "1h")])
        self.assertLessEqual(len(marks), 4)

    def test_a_chat_completions_request_for_claude(self):
        shape = Chat()
        shape.model = "anthropic/claude-agent"
        model = FakeModel(worker(3))
        Agent(compache.Live(self.policy()), shape, model).run("Read three files")
        first = model.bodies[-1]["messages"][0]
        self.assertEqual(first["content"][-1]["cache_control"]["ttl"], "1h")
        last = model.bodies[-1]["messages"][-1]
        self.assertNotIn("ttl", last["content"][-1]["cache_control"])

    def test_a_provider_that_refuses_it_is_asked_for_its_default(self):
        def refusing(request):
            if '"1h"' in json.dumps(request["body"]):
                return {"status": 400, "error": "ttl"}
            return worker(3)(request)

        model = FakeModel(refusing)
        live = compache.Live(self.policy())
        app = Agent(live, Claude(), model)
        self.assertEqual(app.run("Read three files"), "All done.")
        self.assertEqual(live.stats["retried"], 1)
        self.assertIsNone(live.policy.header_lifetime)

    def test_pings_and_fetches_ask_for_it_too(self):
        clock = Clock()
        model = FakeModel(worker(3), clock=lambda: clock.at)
        made = self.policy().artifact
        made["recipe"] = {**made["recipe"], "tool_pings": 1}
        live = compache.Live(compache.Policy(made), clock=clock)
        app = Agent(live, Claude(), model)
        app.shape.ask("Run")
        app.call()
        clock.at += 256
        self.assertEqual(live.tick(), 1)
        self.assertEqual(self.marks(model.bodies[-1])[0], ("system", "1h"))

    def test_the_record_says_what_was_written_for_an_hour(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel(worker(1))
            send = model.send("/v1/messages")

            def counting(body):
                answered = send(body)
                answered.json["usage"]["cache_creation"] = {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 30,
                }
                return answered

            model.send = lambda path: counting
            live = compache.Live(self.policy(), record=folder)
            Agent(live, Claude(), model).run("Read a file")
            rows = load_calls(folder)
            self.assertEqual({r["header_ttl"] for r in rows}, {3600})
            self.assertEqual({r["long_write"] for r in rows}, {30})


class RecordTests(unittest.TestCase):
    def test_a_record_of_loops_says_how_each_call_went_and_fits_an_agent_s_policy(self):
        def extra(request, seen):
            # Every third call repeats a line of the file it read two calls ago.
            if seen >= 3 and seen % 3 == 0 and seen < 24:
                line = f"a line of file number {seen - 2} that only it holds"
                return {"text": line, "tool": ("read", {"file": seen})}

        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel(worker(24, extra), clock=lambda: clock.at)
            live = compache.Live(record=folder, clock=clock)
            for k in range(8):
                clock.at += 1_000
                app = Agent(live, Claude(), model, clock=clock)
                app.model.script = worker(24, extra)
                app.run(f"Task {k}")
            threads, _ = load(folder)
            self.assertEqual(len(threads), 8)
            steps = threads[0]["runs"][0]["steps"]
            self.assertEqual(len(steps), 25)
            self.assertEqual([s["results"] for s in steps], list(range(25)))
            quoted = [s["needs"] for s in steps if s["needs"]]
            self.assertEqual(len(quoted), 7)
            self.assertTrue(all(":retrieval:" in n for q in quoted for n in q))
            policy_ = compache.fit(folder, model="claude-haiku-5-5")
            self.assertTrue(policy_.agentic)
            self.assertEqual(policy_.summary["calls"], 8 * 25)
            # Under a policy that kept things out of view, what was relied on is not known.
            app = Agent(compache.Live(policy(), record=folder), Claude(), model)
            app.run("Another task")
            threads, _ = load(folder)
            steps = threads[-1]["runs"][0]["steps"]
            self.assertEqual({s["needs"] is None for s in steps}, {True})
            self.assertIsNotNone(threads[0]["runs"][0]["steps"][0]["needs"])
            self.assertTrue(compache.fit(folder, model="claude-haiku-5-5").agentic)
            # No word of what was said or read is in it.
            with open(f"{folder}/turns.jsonl", encoding="utf-8") as stream:
                self.assertNotIn("a line of file", stream.read())

    def test_a_cut_inside_a_warm_turn_leaves_a_record_to_fit_on(self):
        class Haiku(Claude):
            model = "claude-haiku-5-5"

        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            live = compache.Live(
                policy(rounds=8, ratio=2.0), record=folder, clock=clock
            )
            model = FakeModel(worker(25), clock=lambda: clock.at)
            app = Agent(live, Haiku(), model, clock=clock)
            app.run("Read files")
            clock.at += 20
            model.script = worker(60)
            app.run("Read more")
            self.assertTrue(noticed(model.bodies[-1]["messages"]))
            rows, _ = load(folder)
            for run in rows[0]["runs"]:
                usage = run["usage"]
                cached = usage["cache_read"] + usage["cache_write"]
                self.assertLessEqual(cached, usage["input"])
            compache.load_threads(rows)

    def test_a_request_of_the_host_s_that_takes_long_leaves_a_record_to_fit_on(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            live = compache.Live(policy(rounds=8), record=folder, clock=clock)
            model = FakeModel(worker(5), clock=lambda: clock.at)
            app = Agent(live, Claude(), model, clock=clock)
            app.run("Read files")
            # The host asks for a suggestion, and it streams while the user goes on.
            side = Claude()
            side.said = list(app.shape.said)
            side.ask("Suggest what the user may say next")
            model.script = lambda request: {"text": "Perhaps: run the tests."}
            slow = live.exchange(side.path, side.body(True), model.send(side.path))
            events = iter(slow.events)
            next(events)
            for k, count in enumerate((6, 7)):
                clock.at += 5
                model.script = worker(count)
                if k:
                    app.shape.ask("And then?")
                    _, calls = app.call()
                    clock.at += 2
                    list(events)
                    app.made += 1
                    app.shape.answer([(c, result(app.made)) for c, _ in calls])
                    app.call()
                else:
                    app.run("Next, please")
            rows, _ = load(folder)
            compache.load_threads(rows)
            self.assertEqual([len(row["runs"]) for row in rows], [3])

    def test_what_was_left_out_is_on_record_with_every_call(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel(worker(6))
            live = compache.Live(policy(rounds=2), record=folder)
            Agent(live, Claude(), model).run("Read six files")
            rows = load_calls(folder)
            left = [row.get("left_bytes", 0) for row in rows]
            # What it adds counts against what it leaves out: the tool, from the first call on.
            self.assertEqual(len(set(left[:3])), 1)
            self.assertTrue(-900 < left[0] < -300)
            self.assertEqual(left, sorted(left))
            self.assertGreater(left[-1], 4 * 2_000)
            self.assertGreater(rows[-1]["sent_bytes"], 0)


class StreamTests(unittest.TestCase):
    def test_a_responses_stream_without_our_tool_passes_byte_for_byte(self):
        model = FakeModel(worker(1))
        live = compache.Live(policy())
        shape = Codex()
        shape.ask("Read a file")
        body = shape.body(True)
        answered = live.exchange(shape.path, body, model.send(shape.path))
        passed = b"".join(answered.events)
        again = FakeModel(worker(1))
        again.bodies.append({})
        again.bodies.clear()
        _, events = again.responses(body)
        # But for the tools, which the provider repeats as it was given them.
        ours = [dict(d, response=None) for _, _, d in wire.sse_events([passed])]
        theirs = [dict(d, response=None) for _, _, d in wire.sse_events(events)]
        self.assertEqual(ours, theirs)


if __name__ == "__main__":
    unittest.main()


class ThreadTests(unittest.TestCase):
    """Anthropic's message threads keep the conversation at the provider: a request is passed on as it is."""

    def test_create_and_continue_go_out_untouched(self):
        live = compache.Live(policy(rounds=1))
        tools = [{"name": "read", "input_schema": {"type": "object"}}]
        create = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 64,
            "system": "You read files.",
            "tools": tools,
            "thread": {"type": "create"},
            "messages": [
                {"role": "user", "content": "Read them"},
                *[
                    m
                    for k in range(1, 6)
                    for m in (
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": f"toolu_{k:020d}ab", "name": "read", "input": {}}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": f"toolu_{k:020d}ab", "content": result(k)}
                            ],
                        },
                    )
                ],
            ],
        }
        cont = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 64,
            "thread": {"type": "continue", "previous_message_id": "msg_1"},
            "messages": [{"role": "user", "content": "And now?"}],
        }
        for body in (create, cont):
            seen = []

            def send(payload):
                seen.append(payload)
                return live_module.Result(
                    200,
                    json={
                        "id": "msg_2",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-5-5",
                        "content": [{"type": "text", "text": "Done."}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    },
                )

            live.exchange("/v1/messages", body, send, thread="threaded")
            self.assertEqual(len(seen), 1)
            self.assertIs(seen[0], body)
        self.assertEqual(live.stats["passed_through"], 2)


class WarmTests(unittest.TestCase):
    """Requests that only warm the cache, as Anthropic documents and agents send between prompts."""

    def play(self, warm):
        clock = Clock(1_900_000_000.0)
        live = compache.Live(policy(rounds=2, ratio=0.0), clock=clock)
        sent, messages = [], []
        tools = [{"name": "read", "input_schema": {"type": "object"}}]

        def send(payload):
            sent.append(payload)
            limit = payload.get("max_tokens")
            if limit is not None and limit <= 1:
                content, stop = [], "max_tokens"
            elif state["k"] < 5:
                state["n"] += 1
                content = [{"type": "tool_use", "id": f"toolu_{state['n']:020d}ab", "name": "read", "input": {}}]
                stop = "tool_use"
            else:
                content, stop = [{"type": "text", "text": "Done."}], "end_turn"
            return live_module.Result(
                200,
                json={
                    "id": f"msg_{len(sent)}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": content,
                    "stop_reason": stop,
                    "usage": {"input_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0 if not content else 5},
                },
            )

        state = {"k": 0, "n": 0}
        base = {"model": "claude-sonnet-5-5", "max_tokens": 64, "system": "You read files.", "tools": tools}
        mark = 0
        for prompt in range(3):
            messages.append({"role": "user", "content": f"Prompt {prompt}"})
            for k in range(6):
                state["k"] = k
                body = {**base, "messages": json.loads(json.dumps(messages))}
                reply = live.exchange("/v1/messages", body, send, thread="warmed").json
                messages.append({"role": "assistant", "content": reply["content"]})
                for block in reply["content"]:
                    if block["type"] == "tool_use":
                        messages.append(
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": result(len(messages))}]}
                        )
                clock.at += 5
            clock.at += 200
            if warm and prompt < 2:
                for limit in (0, 1):
                    mark = len(sent)
                    again = {**base, "max_tokens": limit, "messages": json.loads(json.dumps(messages[:-1]))}
                    live.exchange("/v1/messages", again, send, thread="warmed")
                    # One request each, never asked again as written.
                    self.assertEqual(len(sent), mark + 1)
                    clock.at += 30
        return live, [b for b in sent if b.get("max_tokens", 64) > 1]

    def test_a_warm_request_goes_out_once_and_leaves_no_trace(self):
        plain, unwarmed = self.play(warm=False)
        live, warmed = self.play(warm=True)
        self.assertEqual(live.stats["warmed"], 4)
        self.assertEqual(live.stats["turns"], plain.stats["turns"])
        self.assertEqual(live.stats["fell_back"], 0)
        # What the application's own requests became is the same with and without them.
        self.assertEqual(
            [json.dumps(b, sort_keys=True) for b in warmed],
            [json.dumps(b, sort_keys=True) for b in unwarmed],
        )
