"""The agent's policy in the request path: loops of calls on every API compache speaks."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest

import compache
from compache import agent, wire
from compache.live import Result
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
        (conversation,) = live._by_chain.values()
        flow = conversation.session._engine.flow
        self.assertEqual(flow.size, sum(flow.shown))
        counted = tokens([model.bodies[-1]["messages"]])
        self.assertLess(abs(flow.size - counted) / counted, 0.25)

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

    def test_a_stream_that_ends_on_our_tool_alone_is_closed(self):
        """The follow-up fails: the reply ends as it stands, and as a stream should."""
        for shape in SHAPES:
            with self.subTest(shape=shape.__name__):
                model = FakeModel(
                    lambda r: {"text": "Hm.", "tool": (REOPEN, {"reference": "x y z"})}
                )
                app, sent = shape(), []
                app.ask("Read a file")
                path = shape.path.replace(":generate", ":streamGenerate")

                def send(body, model=model, path=path, sent=sent):
                    sent.append(body)
                    if len(sent) > 1:
                        return Result(500, json={"error": "no"})
                    return model.send(path)(body)

                answered = compache.Live(policy()).exchange(path, app.body(True), send)
                events = [d for _, _, d in wire.sse_events(answered.events) if d]
                text, calls = app.take(app.gathered(events))
                self.assertEqual((text, calls), ("Hm.", []))


if __name__ == "__main__":
    unittest.main()
