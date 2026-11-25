"""Where requests of agents go astray: several under one name, replies that break off, odd shapes on the wire."""

from __future__ import annotations

import gc
import json
import tempfile
import threading
import time
import unittest

import compache
from compache import stats, wire
from compache.live import Result
from compache.record import load, load_calls
from compache.wrap import _let_go
from fakes import NOTICE, FakeModel, noticed, tokens
from test_agent_live import (
    REOPEN,
    SHAPES,
    SPEC,
    Agent,
    Chat,
    Claude,
    Clock,
    Codex,
    Gemini,
    policy,
    result,
    worker,
)

MARK = {"type": "ephemeral"}


def for_an_hour(**recipe):
    """A policy whose instructions and tools are cached for an hour."""
    made = policy(rounds=2, **recipe)
    hour = {**SPEC.to_dict(), "cache_ttl_seconds": 3600, "write_multiplier": 2.0}
    return compache.Policy({**made.artifact, "header_spec": hour})


def fetching(at, count, text="Let me look back.", beside=()):
    """A model that reads `count` files and looks back once, after `at` of them."""
    done: list = []

    def extra(request, seen):
        if seen == at and not done:
            done.append(NOTICE.search(json.dumps(request["body"])).group())
            return {"text": text, "tools": [*beside, (REOPEN, {"reference": done[0]})]}
        return None

    return worker(count, extra)


def sse(data) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def files(k, j):
    line = f"a line of file number {k}-{j} that only it holds\n"
    return f"result {k}.{j}: " + line * 60


class NameTests(unittest.TestCase):
    """Several agents under one session name, at the same time."""

    def play(self, thread, fork=False, same=True, unique=True, shape=Claude):
        """What three agents sent that was not read from the cache, and how many sessions were begun."""
        clock = Clock()
        model = FakeModel(worker(30), clock=lambda: clock.at)
        usages: list = []
        send = model.send(shape.path)

        def keeping(body):
            answered = send(body)
            usages.append(answered.json["usage"])
            if not unique:
                made = sum(len(m.get("tool_calls") or ()) for m in body["messages"])
                calls = answered.json["choices"][0]["message"].get("tool_calls") or ()
                for k, call in enumerate(calls):
                    call["id"] = f"functions.read:{made + k}"
            return answered

        model.send = lambda path: keeping
        live = compache.Live(policy(rounds=12, ratio=1.5), clock=clock)
        apps = [Agent(live, shape(), model, thread=thread) for _ in range(3)]
        for k, app in enumerate(apps):
            if not fork:
                app.shape.ask("Review the diff" if same else f"Review part {k}")
        if fork:
            apps[0].shape.ask("Do the whole task")
            for _ in range(3):
                _, calls = apps[0].call()
                apps[0].made += 1
                done = [(call, result(apps[0].made)) for call, _ in calls]
                apps[0].shape.answer(done)
            for app in apps[1:]:
                app.shape.said = json.loads(json.dumps(apps[0].shape.said))
                app.made = 3
        begun = list(apps)
        while begun:
            for app in list(begun):
                _, calls = app.call()
                if not calls:
                    begun.remove(app)
                    continue
                app.made += 1
                clock.at += 1
                found = files(apps.index(app), app.made)
                app.shape.answer([(call, found) for call, _ in calls])
        fresh = 0
        for usage in usages:
            counted = compache.threads.canonical_usage(usage)
            fresh += counted["input"] - counted["cache_read"]
        self.assertEqual(live.stats["turns"], 3)
        return fresh, len(live.conversations())

    def test_agents_with_one_task_are_kept_apart(self):
        alone, _ = self.play(None)
        for same in (True, False):
            named, conversations = self.play("session", same=same)
            self.assertEqual(conversations, 3)
            self.assertLess(abs(named / alone - 1), 0.02)

    def test_agents_that_go_on_from_one_conversation_are_kept_apart(self):
        alone, _ = self.play(None, fork=True)
        named, conversations = self.play("session", fork=True)
        self.assertEqual((named, conversations), (alone, 3))

    def test_call_ids_need_not_be_unique(self):
        alone, _ = self.play(None, shape=Chat, same=False)
        for thread in (None, "session"):
            fresh, conversations = self.play(
                thread, shape=Chat, same=False, unique=False
            )
            self.assertEqual(conversations, 3)
            self.assertLess(abs(fresh / alone - 1), 0.02)

    def test_another_agent_s_past_is_not_taken_for_this_one_s(self):
        model = FakeModel(worker(8))
        live = compache.Live(policy(rounds=1))
        main = Agent(live, Claude(), model, thread="session")
        main.shape.ask("The code word is PINEAPPLE. Fix the parser " + "x" * 300)
        body, path = main.shape.body(True), main.shape.path
        # Its first call is under way.
        events = iter(
            live.exchange(path, body, model.send(path), thread="session").events
        )
        next(events)
        other = Agent(live, Claude(), model, thread="session")
        other.shape.said = [
            {"role": "user", "content": "Summary so far"},
            {"role": "assistant", "content": [{"type": "text", "text": "Understood."}]},
            {"role": "user", "content": "Go on"},
        ]
        other.call()
        self.assertEqual(live.stats["turns"], 0)
        found = live._latest["session"].session.reopen(
            "code word", search=("retrieval", "input", "output")
        )
        self.assertNotIn("PINEAPPLE", json.dumps(found))

    def test_first_requests_at_the_same_moment(self):
        for _ in range(10):
            model = FakeModel(worker(3))
            live = compache.Live(policy(rounds=2))
            met: dict = {}
            prepare = live._prepare

            def keeping(conversation, chains, dialect, body, scan, turns, *more, **kw):
                met[turns[-1].question] = conversation
                return prepare(
                    conversation, chains, dialect, body, scan, turns, *more, **kw
                )

            live._prepare = keeping
            start = threading.Barrier(3)

            def go(k):
                app = Agent(live, Claude(), model, thread="session", stream=True)
                app.shape.ask(f"Subtask {k} " + "context " * 2_000)
                start.wait()
                app.call()

            threads = [threading.Thread(target=go, args=(k,)) for k in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len({id(c) for c in met.values()}), 3)

    def test_the_main_agent_s_cache_is_kept_while_another_works(self):
        clock = Clock()
        model = FakeModel(worker(6), clock=lambda: clock.at)
        kept = policy(rounds=None, ratio=None, tool_pings=3)
        live = compache.Live(kept, clock=clock)
        main = Agent(live, Claude(), model, thread="session", clock=clock)
        main.shape.ask("The main task")
        _, calls = main.call()
        (conversation,) = live.conversations()
        sub = Agent(live, Claude(), model, thread="session", clock=clock)
        sub.shape.ask("A subtask")
        for _ in range(3):
            _, made = sub.call()
            sub.made += 1
            sub.shape.answer([(call, result(sub.made)) for call, _ in made])
            clock.at += 5
        for _ in range(2):
            clock.at += 256
            live.tick()
            flow = conversation.session._engine.flow
            self.assertGreater(flow.at, clock.at - 5)
            main.shape.answer([(call, result(1)) for call, _ in calls])
            _, calls = main.call()

    def test_a_side_request_on_another_model_leaves_the_main_line_s_cache(self):
        for other in (None, "claude-small"):
            clock = Clock()
            model = FakeModel(worker(8), clock=lambda: clock.at)
            live = compache.Live(policy(rounds=2, ratio=5.0), clock=clock)
            app = Agent(live, Claude(), model, clock=clock, thread="session")
            app.run("Read eight files")
            kept = list(app.shape.said)
            app.shape.ask("Suggest what the user may say next")
            app.shape.model = other or Claude.model
            app.call()
            app.shape.said, app.shape.model = kept, Claude.model
            model.script, before = worker(10), len(model.bodies)
            clock.at += 20
            app.run("Read two more")
            later = model.bodies[before:]
            self.assertFalse(any(noticed(b["messages"]) for b in later))

    def test_a_side_request_on_another_model_leaves_the_main_line_s_pings(self):
        clock = Clock()
        model = FakeModel(worker(8), clock=lambda: clock.at)
        live = compache.Live(policy(rounds=2, ratio=5.0, user_pings=3), clock=clock)
        app = Agent(live, Claude(), model, clock=clock, thread="s")
        app.run("Read eight files")
        before = list(app.shape.said)
        app.shape.ask("Suggest what the user may say next")
        app.shape.model = "claude-small"
        app.call()
        app.shape.said, app.shape.model = before, "claude-agent"
        sent = len(model.bodies)
        for _ in range(170):
            clock.at += 5
            live.tick()
        pinged = [b["model"] for b in model.bodies[sent:]]
        self.assertEqual(pinged.count("claude-agent"), 3)

    def test_an_application_s_own_tool_of_that_name_is_its_own(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                model = FakeModel(lambda request: {"tool": (REOPEN, {"ticket": 7})})
                live = compache.Live(policy(rounds=1))
                shape = Chat()
                plain = shape.body
                own = {
                    "type": "function",
                    "function": {"name": REOPEN, "parameters": {}},
                }
                shape.body = lambda stream, plain=plain: {
                    **plain(stream),
                    "tools": [own],
                }
                app = Agent(live, shape, model, stream=stream)
                shape.ask("Open ticket seven again")
                _, calls = app.call()
                self.assertEqual([name for _, name in calls], [REOPEN])
                self.assertEqual(model.bodies[-1]["tools"], [own])
                self.assertEqual(live.stats["reopened"], 0)

    def test_a_request_without_tools_does_not_keep_the_agent_from_its_policy(self):
        """A host asks for a title under the same name, with the user's own words, before the agent begins."""
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            model = FakeModel(worker(40))
            body = {"model": Claude.model, "max_tokens": 50, "stream": True}
            body["messages"] = [{"role": "user", "content": "Read forty files"}]
            send = model.send(Claude.path)
            title = live.exchange(Claude.path, body, send, thread="session")
            next(title.events)
            Agent(live, Claude(), model, thread="session").run("Read forty files")
            self.assertTrue(noticed(model.bodies[-1]["messages"]))
            title.events.close()

    def test_the_host_s_own_calls_are_told_to_the_agent_that_made_them(self):
        """LangChain's way in: the host sends the requests, and two agents of one thread take turns."""
        tools = [
            {"type": "function", "function": {"name": n}} for n in ("read", REOPEN)
        ]
        live = compache.Live(policy(rounds=1))

        class Sub:
            def __init__(self, name, size):
                self.name, self.size, self.k = name, size, 0
                self.said = [{"role": "user", "content": f"Task of {name}"}]

            def call(self):
                body = {"model": "agent-model", "messages": list(self.said)}
                body["tools"] = tools
                self.shown = live.show("/v1/chat/completions", body, thread="t")

            def replied(self):
                self.k += 1
                call = {"id": f"{self.name}-{self.k}", "arguments": {}}
                usage = {"prompt_tokens": self.size + 600 * self.k}
                usage["completion_tokens"] = 10
                self.shown.heard(usage, said="", calls=[call])
                made = {"name": "read", "arguments": "{}"}
                made = {"id": call["id"], "type": "function", "function": made}
                text = f"SECRET-OF-{self.name} number {self.k}\n" * 80
                self.said += [
                    {"role": "assistant", "content": None, "tool_calls": [made]},
                    {"role": "tool", "tool_call_id": call["id"], "content": text},
                ]

        agents = [Sub("A", 1_000), Sub("B", 60_000)]
        for _ in range(4):
            for agent in agents:
                agent.call()
            for agent in agents:
                agent.replied()
        for agent in agents:
            agent.call()
        self.assertEqual(len(live.conversations()), 2)
        for agent in agents:
            handles = NOTICE.findall(json.dumps(agent.shown.body))
            self.assertEqual(len(handles), 3)
            for words in (handles[0], f"SECRET-OF-{agent.name} number 1"):
                call = f"{agent.name}-{agent.k}"
                found = live.reopen("t", words, call=call)
                self.assertIn(f"SECRET-OF-{agent.name}", json.dumps(found))
            size = agent.shown._conversation.session._engine.flow.size
            self.assertEqual(size > 30_000, agent.name == "B")

    def test_the_next_call_may_come_before_the_body_of_the_last_one_ends(self):
        for shape in (Chat, Claude, Codex):
            model = FakeModel(worker(8))
            live = compache.Live(policy(rounds=2))
            app = Agent(live, shape(), model, stream=True)
            app.shape.ask("Read eight files")
            left, most = [], 12
            for k in range(most):
                path = shape.path
                answered = live.exchange(path, app.shape.body(True), model.send(path))
                events, data = iter(answered.events), []
                ends = ("message_stop", "response.completed")
                for chunk in events:
                    data += [d for _, _, d in wire.sse_events([chunk]) if d]
                    if b"[DONE]" in chunk or (data and data[-1].get("type") in ends):
                        # As an SDK that stops reading at the last event.
                        break
                left.append(events)
                text, calls = app.shape.take(app.shape.gathered(data))
                if not calls:
                    break
                app.shape.answer([(call, result(k + 1)) for call, _ in calls])
            self.assertEqual(text, "All done.")
            self.assertEqual(live.stats["turns"], 1, shape.__name__)
            for events in left:
                events.close()
            self.assertEqual(len(live.conversations()), 1)


class NobodyElsesTests(unittest.TestCase):
    """Unnamed conversations that open alike, or whose tool calls go by the same IDs."""

    SECRET = "My name is Carol and my account is 4711-CAROL: look at my contract."

    def setUp(self):
        self.clock, self.down = Clock(), 0
        self.model = FakeModel(self.script, clock=self.clock)
        self.live = self.made()

    def made(self):
        from test_live import fitted

        chat = fitted("gpt-6-luna") if self.chat else policy(rounds=1, ratio=0.0)
        return compache.Live(chat, clock=self.clock)

    chat = False
    counted = False

    @staticmethod
    def script(request):
        said = str(request["body"]["messages"][-1]["content"])
        if said == "Hi":
            return {"text": "Hello! How can I help?"}
        if request["body"]["messages"][-1]["role"] == "user":
            return {"text": "", "tool": ("search_docs", {"query": "clause"})}
        return {"text": f"Answer {request['n']}."}

    def send(self, body):
        if self.down:
            self.down -= 1
            return Result(529, json={"error": {"message": "overloaded"}})
        answered = self.model.send()(body)
        if self.counted:
            before = sum(len(m.get("tool_calls") or ()) for m in body["messages"])
            message = answered.json["choices"][0]["message"]
            for k, call in enumerate(message.get("tool_calls") or ()):
                call["id"] = f"functions.search_docs:{before + k}"
        return answered

    def call(self, said):
        from test_live import SEARCH

        body = {"model": "gpt-6-luna", "messages": list(said), "tools": [SEARCH]}
        answered = self.live.exchange("/v1/chat/completions", body, self.send)
        if answered.status >= 400:
            return None
        message = answered.json["choices"][0]["message"]
        said.append(message)
        return message

    def results(self, said, message, found):
        said += [
            {"role": "tool", "tool_call_id": c["id"], "content": found}
            for c in message["tool_calls"]
        ]

    def turn(self, said, question, found="nothing"):
        said.append({"role": "user", "content": question})
        while True:
            message = self.call(said)
            if message is None or not message.get("tool_calls"):
                return message
            self.results(said, message, found)

    def holds(self):
        """The texts each conversation's session holds."""
        for conversation in self.live.conversations():
            session = conversation.session
            parts = [*session._history, *((session._pending or ((), ()))[1])]
            yield " ".join(p.text for p in parts)

    def check(self):
        for held in self.holds():
            self.assertFalse("4711-CAROL" in held and "clause" in held.lower(), held)

    def test_after_a_request_that_failed(self):
        from test_live import passage

        for self.chat in (False, True):
            with self.subTest(chat=self.chat):
                self.live = self.made()
                carol, bob, erin = [], [], []
                self.turn(carol, "Hi")
                self.turn(bob, "Hi")
                self.down = 1
                self.assertIsNone(self.turn(carol, self.SECRET))
                # Another says hello, and Carol's application asks again.
                self.turn(erin, "Hi")
                self.call(carol)
                for k in range(1, 5):
                    self.turn(bob, f"What does clause {k} say?", passage(k))
                    self.turn(erin, f"And what of clause {k}?", passage(k))
                    self.clock.at += 4000
                self.check()

    def test_ids_that_count(self):
        from test_live import passage

        self.counted = True
        for self.chat in (False, True):
            for restart in (False, True):
                with self.subTest(chat=self.chat, restart=restart):
                    self.live = self.made()
                    carol, bob = [], []
                    bob.append({"role": "user", "content": "What does clause 1 say?"})
                    waits = self.call(bob)
                    if restart:
                        # A process that knows nothing of Bob, whose tool still runs.
                        self.live = self.made()
                    carol.append({"role": "user", "content": self.SECRET})
                    self.call(carol)
                    self.results(bob, waits, passage(1))
                    self.call(bob)
                    for k in range(2, 5):
                        self.turn(bob, f"And clause {k}?", passage(k))
                        self.clock.at += 4000
                    self.check()

    def test_while_another_s_turn_is_under_way(self):
        from test_live import passage

        self.chat = True
        self.live = self.made()
        carol, bob = [], []
        self.turn(carol, "Hi")
        self.turn(bob, "Hi")
        self.turn(bob, "What does clause 1 say?", passage(1))
        carol.append({"role": "user", "content": self.SECRET})
        self.call(carol)
        # Bob's application shows his answer otherwise, so it is not found by what was said.
        bob[-1] = {"role": "assistant", "content": "Shown otherwise."}
        for k in range(2, 5):
            self.turn(bob, f"And clause {k}?", passage(k))
            self.clock.at += 4000
        self.check()

    def test_a_question_that_was_refused_and_another_in_its_place(self):
        from test_live import SEARCH, passage

        self.chat = True
        self.live = self.made()
        said: list = []

        def named(said):
            body = {"model": "gpt-6-luna", "messages": list(said), "tools": [SEARCH]}
            path = "/v1/chat/completions"
            answered = self.live.exchange(path, body, self.send, thread="t")
            if answered.status >= 400:
                return None
            said.append(answered.json["choices"][0]["message"])
            return said[-1]

        self.call = named
        self.down = 1
        self.assertIsNone(self.turn(said, "THE FIRST QUESTION, refused"))
        said.pop()
        self.turn(said, "THE SECOND QUESTION", passage(1))
        self.turn(said, "And clause 2?", passage(2))
        for held in self.holds():
            if "clause 2" in held:
                self.assertIn("THE SECOND QUESTION", held)
                self.assertNotIn("THE FIRST QUESTION", held)

    def test_a_tool_s_answer_comes_from_its_own_thread(self):
        """As the LangChain middleware asks for it, by the ID of the model's call."""
        live = compache.Live(policy(rounds=1, ratio=0.0), clock=self.clock)
        tools = [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in ("search", "reopen")
        ]
        for name in ("alice", "bob"):
            said = [{"role": "user", "content": "Check my ledger"}]
            for k in range(6):
                body = {"model": "m", "messages": list(said), "tools": tools}
                shown = live.show("/v1/chat/completions", body, thread=name)
                call = {"id": f"functions.search:{k}", "arguments": {"q": k}}
                usage = {"prompt_tokens": 900, "completion_tokens": 9}
                shown.heard(usage, model="m", said="", calls=[call])
                function = {"name": "search", "arguments": "{}"}
                made = {"id": call["id"], "type": "function", "function": function}
                said += [
                    {"role": "assistant", "content": None, "tool_calls": [made]},
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"{name}'s private ledger, page {k} " * 60,
                    },
                ]
                self.clock.at += 1
        for name, other in (("alice", "bob"), ("bob", "alice")):
            found = live.reopen(
                name, "private ledger page", 0, call="functions.search:5"
            )
            self.assertIn(f"{name}'s private", json.dumps(found))
            self.assertNotIn(f"{other}'s private", json.dumps(found))

    others = 0

    def test_calls_of_two_agents_that_count_alike_on_a_busy_server(self):
        self.others = 16
        self.test_calls_of_two_agents_that_count_alike()

    def test_calls_of_two_agents_that_count_alike(self):
        """Both call it `functions.reopen:4`: the notice says whose text is meant, and words do not."""
        live = compache.Live(policy(rounds=1, ratio=0.0), clock=self.clock)
        tools = [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in ("read", "reopen")
        ]
        said: dict = {}

        def call(name, tool, k):
            body = {"model": "m", "messages": list(said[name]), "tools": tools}
            thread = "t-1" if name in names else name
            shown = live.show("/v1/chat/completions", body, thread=thread)
            made = {"id": f"functions.{tool}:{k}", "arguments": {"k": k}}
            usage = {"prompt_tokens": 900, "completion_tokens": 9}
            shown.heard(usage, model="m", said="", calls=[made])
            function = {"name": tool, "arguments": "{}"}
            made = {"id": made["id"], "type": "function", "function": function}
            message = {"role": "assistant", "content": None, "tool_calls": [made]}
            said[name].append(message)
            return shown, made["id"]

        names = ("alice", "bob")
        # On a busy server, others make calls by the same IDs.
        others = [f"other {k}" for k in range(self.others)]
        for name in (*names, *others):
            said[name] = [{"role": "user", "content": f"Work on the files of {name}"}]
        for k in range(4):
            for name in (*names, *others):
                _, made = call(name, "read", k)
                text = f"the secret plan of {name}, line {k} " * 60
                said[name].append(
                    {"role": "tool", "tool_call_id": made, "content": text}
                )
        last = {name: call(name, "reopen", 4) for name in (*names, *others)}
        for name, other in (names, names[::-1]):
            shown, made = last[name]
            for reference in ("secret plan", f"the secret plan of {other}"):
                found = json.dumps(live.reopen("t-1", reference, call=made))
                self.assertNotIn("the secret plan of", found)
            handle = NOTICE.search(json.dumps(shown.body)).group()
            found = live.reopen("t-1", handle, call=made)
            self.assertEqual(found["status"], "ok")
            self.assertIn(f"the secret plan of {name}", found["text"])


class HostTests(unittest.TestCase):
    """A host that runs our tool itself, as LangChain's agents do: what it returns is a result like others."""

    def host(self, live):
        said = [{"role": "user", "content": "Go"}]
        tools = [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in ("read", REOPEN)
        ]

        def call(name, k):
            body = {"model": "m", "messages": list(said), "tools": tools}
            shown = live.show(Chat.path, body, thread="t")
            made = {"id": f"call_{k}0f{name}a1b2c3d4e5f6", "arguments": {"k": k}}
            usage = {"prompt_tokens": 900, "completion_tokens": 9}
            shown.heard(usage, model="m", said="", calls=[made])
            function = {"name": name, "arguments": "{}"}
            made = {"id": made["id"], "type": "function", "function": function}
            message = {"role": "assistant", "content": None, "tool_calls": [made]}
            said.append(message)
            return shown.body["messages"], made["id"]

        def answer(made, text):
            said.append({"role": "tool", "tool_call_id": made, "content": text})

        return said, call, answer

    def test_a_copy_whose_text_the_host_has_dropped_names_itself(self):
        """As where LangGraph puts a summary in place of the first rounds."""
        live = compache.Live(policy(2))
        said, call, answer = self.host(live)
        for k in range(5):
            answer(call("read", k)[1], result(k))
        out, made = call(REOPEN, 5)
        handle = NOTICE.search(json.dumps(out)).group()
        answer(made, json.dumps(live.reopen("t", handle, call=made)))
        said[:-2] = [{"role": "user", "content": "Summary: five files read. Go on."}]
        for k in range(6, 11):
            answer(call("read", k)[1], result(k))
        out, made = call(REOPEN, 11)
        handles = NOTICE.findall(
            json.dumps([m["content"] for m in out if m["role"] == "tool"])
        )
        self.assertGreater(len(handles), 3)
        for handle in handles:
            found = live.reopen("t", handle, call=made)
            self.assertEqual(found["status"], "ok", handle)

    def test_a_copy_that_is_in_view_is_no_repetition(self):
        live = compache.Live(policy(2))
        said, call, answer = self.host(live)
        for k in range(5):
            answer(call("read", k)[1], result(k))
        for k in (5, 6):
            out, made = call(REOPEN, k)
            handle = NOTICE.search(json.dumps(out)).group()
            answer(made, json.dumps(live.reopen("t", handle, call=made)))
        (conversation,) = live.conversations()
        self.assertEqual(conversation.session._engine.flow.longer, 0)

    def test_a_copy_is_no_text_of_its_own(self):
        live = compache.Live(policy(2))
        said, call, answer = self.host(live)
        for k in range(1, 5):
            answer(call("read", k)[1], result(k))
        out, made = call(REOPEN, 5)
        first = NOTICE.search(json.dumps(out)).group()
        found = live.reopen("t", first, call=made)
        self.assertIn("file number 1 ", found["text"])
        answer(made, json.dumps(found))
        for k in range(6, 12):
            out, made = call("read", k)
            answer(made, result(k))
        # It went out of view as well, and what stands in its place names what it was a copy of.
        handles = NOTICE.findall(json.dumps(out))
        self.assertEqual(handles.count(first), 2)
        out, made = call(REOPEN, 12)
        again = live.reopen("t", first, call=made)
        self.assertEqual(again["text"], found["text"])
        # For the second time: from here on results stay longer.
        (conversation,) = live.conversations()
        self.assertEqual(conversation.session._engine.flow.longer, 2)


class ResultTests(unittest.TestCase):
    """Results that hold something else than text, or nothing at all."""

    IMAGE = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "x" * 6000},
    }

    def play(self, content, steps=24, rounds=4):
        clock = Clock()

        def script(request):
            if len(model.bodies) <= steps:
                return {"text": "", "tool": ("look", {"at": len(model.bodies)})}
            return {"text": "All done."}

        model = FakeModel(script, clock=lambda: clock.at)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        live = compache.Live(
            policy(rounds=rounds, ratio=0.0, tool_pings=2),
            record=folder.name,
            clock=clock,
        )
        shape = Claude()
        shape.ask("Click through the site.")
        for k in range(1, steps + 2):
            path = shape.path
            answered = live.exchange(path, shape.body(False), model.send(path))
            _, calls = shape.take(answered.json)
            if not calls:
                break
            clock.at += 5
            shape.answer([(c, content(k)) for c, _ in calls])
        return live, model, folder.name

    def test_one_without_text_goes_on_with_the_same_conversation(self):
        for content in (
            "",
            [self.IMAGE],
            [{"type": "tool_reference", "tool_name": "x"}],
        ):
            live, model, folder = self.play(
                lambda k, content=content: content if k % 3 == 0 else result(k)
            )
            self.assertEqual(len(live.conversations()), 1)
            self.assertEqual(live.stats["turns"], 1)
            threads, _ = load(folder)
            self.assertEqual([r["calls"] for t in threads for r in t["runs"]], [25])
            self.assertTrue(noticed(model.bodies[-1]))

    @staticmethod
    def pictured(shape, k):
        """A result of some text and a screenshot, as each API has it."""
        data = "x" * 6000
        if shape is Claude:
            image = {"type": "base64", "media_type": "image/png", "data": data}
            return [
                {"type": "text", "text": result(k)},
                {"type": "image", "source": image},
            ]
        url = "data:image/png;base64," + data
        if shape is Codex:
            return [
                {"type": "input_text", "text": result(k)},
                {"type": "input_image", "image_url": url},
            ]
        return [
            {"type": "text", "text": result(k)},
            {"type": "image_url", "image_url": {"url": url}},
        ]

    def test_one_with_an_image_goes_and_comes_back_as_a_whole(self):
        for shape in (Claude, Chat, Codex):
            for stream in (False, True):
                with self.subTest(shape=shape.__name__, stream=stream):
                    fetched: list = []

                    def script(request, fetched=fetched):
                        body = json.dumps(request["body"])
                        if len(model.bodies) <= 24:
                            if len(model.bodies) == 20 and not fetched:
                                # The eldest, which held one.
                                fetched.append(NOTICE.findall(body)[0])
                                ask = (REOPEN, {"reference": fetched[0]})
                                return {"text": "", "tool": ask}
                            return {"text": "", "tool": ("look", {"n": 1})}
                        return {"text": "All done."}

                    clock = Clock()
                    model = FakeModel(script, clock=lambda: clock.at)
                    folder = tempfile.TemporaryDirectory()
                    self.addCleanup(folder.cleanup)
                    live = compache.Live(
                        policy(rounds=4, ratio=0.0), record=folder.name, clock=clock
                    )
                    app = Agent(live, shape(), model, stream=stream)
                    app.shape.ask("Click through the site.")
                    for k in range(1, 26):
                        _, calls = app.call()
                        if not calls:
                            break
                        clock.at += 5
                        found = self.pictured(shape, k) if k % 4 == 1 else result(k)
                        app.shape.answer([(c, found) for c, _ in calls])
                    self.assertEqual(live.stats["reopened"], 1)
                    self.assertEqual(live.stats["stood_down"], 0)
                    self.assertEqual(live.stats["retried"], 0)
                    last = shape.results(model.bodies[-1])
                    out = [noticed(r) for r in last]
                    self.assertEqual(out[-4:], [False] * 4)
                    self.assertTrue(all(out[:12]))
                    for found in last:
                        if noticed(found):
                            self.assertIsInstance(found, str)
                            self.assertLess(len(found), 260)
                    # What it was fetched back with holds the picture.
                    again = [
                        body
                        for body in model.bodies
                        if any(
                            isinstance(r, list) and "status" in json.dumps(r)
                            for r in shape.results(body)
                        )
                    ]
                    self.assertTrue(again)
                    held = next(
                        r
                        for r in shape.results(again[0])
                        if isinstance(r, list) and "status" in json.dumps(r)
                    )
                    self.assertEqual(len(wire.media_in(held)), 1)
                    self.assertIn("a line of file number 1 ", json.dumps(held))
                    (conversation,) = live.conversations()
                    flow = conversation.session._engine.flow
                    self.assertEqual(flow.size, sum(flow.shown))
                    rows = load_calls(folder.name)
                    left = [r.get("left_tokens", 0) for r in rows]
                    self.assertGreater(max(left), 4 * 1200)

    def test_a_screenshot_alone(self):
        live, model, folder = self.play(lambda k: [self.IMAGE])
        last = Claude.results(model.bodies[-1])
        self.assertEqual([noticed(r) for r in last], [True] * 20 + [False] * 4)
        self.assertEqual(live.stats["stood_down"], 0)
        (conversation,) = live.conversations()
        flow = conversation.session._engine.flow
        counted = tokens([model.bodies[-1]["messages"]])
        self.assertLess(abs(flow.size - counted) / counted, 0.25)
        # Its tokens were counted as it came, and count as left out.
        found = stats.savings(folder)
        (entry,) = found["unpriced"].values()
        whole = sum(tokens([b["messages"]]) for b in model.bodies)
        self.assertGreater(entry["left_out_tokens"], whole)
        threads, _ = load(folder)
        compache.load_threads(threads, model="agent-model")

    def test_one_with_something_else_stays(self):
        def content(k):
            text = {"type": "text", "text": result(k)}
            tool = {"type": "tool_reference", "tool_name": "x"}
            return [text, tool] if k % 4 == 0 else result(k)

        live, model, _ = self.play(content)
        kept = [r for r in Claude.results(model.bodies[-1]) if isinstance(r, list)]
        self.assertEqual(len(kept), 6)
        self.assertFalse(noticed(kept))
        self.assertTrue(noticed(model.bodies[-1]))

    def test_what_counts_as_an_image(self):
        for held, body in (
            (True, [{"content": [self.IMAGE]}]),
            (True, [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
            (True, [{"output": [{"type": "input_image", "image_url": "data:x"}]}]),
            (True, [{"parts": [{"inlineData": {"data": "x"}}]}]),
            (True, [{"parts": [{"inline_data": {"data": "x"}}]}]),
            (True, [{"parts": [{"functionResponse": {"parts": [{"fileData": {}}]}}]}]),
            (True, [{"output": {"type": "computer_screenshot", "image_url": "x"}}]),
            (True, [{"content": [{"type": "video_url", "video_url": {"url": "x"}}]}]),
            (
                True,
                [{"content": [{"content": {"content": {"type": "document"}}}]}],
            ),
            (False, [{"content": [{"type": "redacted_thinking", "data": "x"}]}]),
            (False, [{"content": [{"type": "tool_use", "input": {"file": "a"}}]}]),
            (False, [{"parts": [{"functionCall": {"args": {"data": 1}}}]}]),
            (False, [{"parts": [{"functionResponse": {"response": {"file": "a"}}}]}]),
        ):
            self.assertEqual(wire.holds_media(body), held, body)


class FollowUpTests(unittest.TestCase):
    """What follows a fetch carries all that the model said when it fetched."""

    SIGNED = {"google": {"thought_signature": "SIGNED"}}

    def asked(self, stream):
        """The second request of a conversation whose model fetches at once."""
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": REOPEN, "arguments": '{"reference": "words"}'},
            "extra_content": self.SIGNED,
        }
        done = {"role": "assistant", "content": "Done."}
        bodies: list = []

        def chunk(delta, reason=None):
            choice = {"index": 0, "delta": delta, "finish_reason": reason}
            return b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n"

        def send(body):
            bodies.append(body)
            if stream and len(bodies) == 1:
                pieces = [
                    chunk({"role": "assistant", "reasoning_content": "Let me "}),
                    chunk({"reasoning_content": "think."}),
                    chunk({"tool_calls": [{"index": 0, **call}]}),
                    chunk({}, "tool_calls"),
                ]
                return Result(200, events=iter([*pieces, b"data: [DONE]\n\n"]))
            if stream:
                pieces = [chunk({"content": "Done."}), chunk({}, "stop")]
                return Result(200, events=iter([*pieces, b"data: [DONE]\n\n"]))
            message = done
            if len(bodies) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Let me think.",
                    "tool_calls": [call],
                }
            choice = {"message": message, "finish_reason": "stop"}
            return Result(200, json={"choices": [choice]})

        live = compache.Live(policy(rounds=2))
        body = {**Chat().body(stream), "messages": [{"role": "user", "content": "hi"}]}
        answered = live.exchange("/v1/chat/completions", body, send)
        if answered.events is not None:
            list(answered.events)
        return bodies[1]["messages"][1]

    def test_a_chat_completion_s_reasoning_and_what_stands_beside_a_call(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                said = self.asked(stream)
                self.assertEqual(said["reasoning_content"], "Let me think.")
                (call,) = said["tool_calls"]
                self.assertEqual(call["extra_content"], self.SIGNED)
                self.assertEqual(call["function"]["name"], REOPEN)

    def test_reasoning_in_pieces(self):
        pieces = [
            {"type": "reasoning.text", "index": 0, "text": "One "},
            {"type": "reasoning.text", "index": 0, "text": "two."},
            {"type": "reasoning.text", "index": 0, "signature": "S"},
            {"type": "reasoning.encrypted", "index": 1, "data": "E"},
        ]
        held = None
        for piece in pieces:
            held = wire._joined(held, [piece])
        self.assertEqual(
            held,
            [
                {
                    "type": "reasoning.text",
                    "index": 0,
                    "text": "One two.",
                    "signature": "S",
                },
                {"type": "reasoning.encrypted", "index": 1, "data": "E"},
            ],
        )

    def test_a_tool_of_the_provider_s_own_keeps_its_input(self):
        def event(kind, data):
            return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()

        def block(index, shell, *pieces):
            yield event("content_block_start", {"index": index, "content_block": shell})
            for piece in pieces:
                delta = {"type": "input_json_delta", "partial_json": piece}
                yield event("content_block_delta", {"index": index, "delta": delta})
            yield event("content_block_stop", {"index": index})

        search = {
            "type": "server_tool_use",
            "id": "s1",
            "name": "web_search",
            "input": {},
        }
        ours = {"type": "tool_use", "id": "t1", "name": REOPEN, "input": {}}
        ending = {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}
        bodies: list = []

        def send(body):
            bodies.append(body)
            opening = {"message": {"model": "m", "usage": {"input_tokens": 5}}}
            first = [event("message_start", opening)]
            if len(bodies) == 1:
                first += block(0, search, '{"query": ', '"news"}')
                first += block(1, ours, '{"reference": "words"}')
            else:
                text = {"index": 0, "content_block": {"type": "text", "text": "Done."}}
                first += [event("content_block_start", text)]
                first += [event("content_block_stop", {"index": 0})]
            first += [event("message_delta", ending), event("message_stop", {})]
            return Result(200, events=iter(first))

        live = compache.Live(policy(rounds=2))
        shape = Claude()
        shape.ask("What is new?")
        list(live.exchange(shape.path, shape.body(True), send).events)
        said = bodies[1]["messages"][-2]["content"]
        self.assertEqual(said[0]["input"], {"query": "news"})


class LeftTests(unittest.TestCase):
    """Streams that nobody reads to their end."""

    def test_one_that_is_dropped_is_let_go_of_elsewhere(self):
        """It may be collected by a thread that holds a lock, or by the loop that would have to close it."""
        for reads in (0, 3):
            with self.subTest(reads=reads):
                closed: list = []
                model = FakeModel(lambda request: {"text": "words " * 50})
                live = compache.Live(policy(rounds=2))
                send = model.send(Claude.path)

                def keeping(body, send=send, closed=closed):
                    answered = send(body)
                    answered.close = lambda: closed.append(threading.current_thread())
                    return answered

                shape = Claude()
                shape.ask("Hi")
                answered = live.exchange(
                    shape.path, shape.body(True), keeping, thread="t"
                )
                for _ in range(reads):
                    next(answered.events)
                (conversation,) = live.conversations()
                pending = conversation.pending
                # What collects it here holds what it would need.
                with live._lock, conversation.lock:
                    del answered
                    gc.collect()
                until = time.time() + 5
                while (pending.in_flight or not closed) and time.time() < until:
                    time.sleep(0.01)
                self.assertEqual(pending.in_flight, 0)
                self.assertIsNot(closed[0], threading.current_thread())

    def test_one_that_is_closed_is_let_go_of_at_once(self):
        closed: list = []
        model = FakeModel(lambda request: {"text": "words " * 50})
        live = compache.Live(policy(rounds=2))
        send = model.send(Claude.path)

        def keeping(body):
            answered = send(body)
            answered.close = lambda: closed.append(True)
            return answered

        shape = Claude()
        shape.ask("Hi")
        answered = live.exchange(shape.path, shape.body(True), keeping, thread="t")
        next(answered.events)
        answered.events.close()
        (conversation,) = live.conversations()
        self.assertEqual((conversation.pending.in_flight, closed), (0, [True]))

    def held(self, closing=None):
        """A stream whose reader waits in another thread until `gate` is set."""
        gate, closed = threading.Event(), []

        def events():
            yield sse({"choices": [{"index": 0, "delta": {"content": "one"}}]})
            gate.wait(10)
            yield sse({"choices": [{"index": 0, "delta": {"content": "two"}}]})

        live = compache.Live(policy(rounds=2))
        body = {
            "model": "agent-model",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        answered = live.exchange(
            Chat.path,
            body,
            lambda sent: Result(
                200, events=events(), close=closing or (lambda: closed.append(True))
            ),
            thread="t",
        )
        next(answered.events)
        found: list = []

        def read():
            found.append(next(answered.events, None))

        reader = threading.Thread(target=read)
        reader.start()
        time.sleep(0.1)
        (conversation,) = live.conversations()
        return answered, conversation, gate, closed, reader, found

    def test_one_that_another_thread_reads_can_be_closed(self):
        answered, conversation, gate, closed, reader, found = self.held()
        _let_go(answered.events, answered.close)
        self.assertEqual(closed, [True])
        gate.set()
        reader.join()
        # Its reader closes it, and is given no more of it.
        self.assertEqual(found, [None])
        self.assertEqual(conversation.pending.in_flight, 0)
        # Nothing is left for whoever collects it, which may not wait for the conversation.
        taken, over = threading.Event(), threading.Event()

        def another():
            with conversation.lock:
                taken.set()
                over.wait(5)

        threading.Thread(target=another).start()
        taken.wait()
        begun = time.time()
        del answered, reader
        gc.collect()
        self.assertLess(time.time() - begun, 1)
        over.set()

    def test_a_response_that_cannot_be_closed_leaves_no_turn_open(self):
        def closing():
            raise TimeoutError("its loop does not run")

        answered, conversation, gate, _, reader, _ = self.held(closing)
        gate.set()
        reader.join()
        answered.events.close()
        self.assertEqual(conversation.pending.in_flight, 0)

    def test_one_that_nobody_reads_is_let_go_of_once(self):
        """Even if its response cannot be closed the first time."""

        def closing():
            raise TimeoutError("its loop does not run")

        live = compache.Live(policy())
        body = {"model": "agent-model", "stream": True}
        body["messages"] = [{"role": "user", "content": "Hi"}]
        answered = live.exchange(
            Chat.path,
            body,
            lambda sent: Result(200, events=iter(()), close=closing),
            thread="t",
        )
        with self.assertRaises(TimeoutError):
            answered.events.close()
        answered.events.close()
        (conversation,) = live.conversations()
        self.assertEqual(conversation.pending.in_flight, 0)
        del answered
        gc.collect()
        time.sleep(0.1)
        self.assertEqual(conversation.pending.in_flight, 0)

    def test_one_that_has_to_wait_holds_up_no_other(self):
        first, waits, gate, _, reader, _ = self.held()
        second, conversation, other, closed, reading, _ = self.held()
        for one in (gate, other):
            one.set()
        reader.join()
        reading.join()
        with waits.lock:
            del first, second
            gc.collect()
            until = time.time() + 5
            while conversation.pending.in_flight and time.time() < until:
                time.sleep(0.01)
            self.assertEqual(conversation.pending.in_flight, 0)
            self.assertEqual(waits.pending.in_flight, 1)
        while waits.pending.in_flight and time.time() < until:
            time.sleep(0.01)
        self.assertEqual(waits.pending.in_flight, 0)


class ReasoningItemTests(unittest.TestCase):
    """What a reply reasoned goes out once, however much of that reply the application sees."""

    @staticmethod
    def call(name, k, arguments="{}"):
        function = {"name": name, "arguments": arguments}
        return {"id": f"call_{k}{name:a<20}", "type": "function", "function": function}

    def play(self, stream, beside, part):
        """Six calls that reason. The fifth looks back. Returns what the last request holds."""
        live, app, sent = compache.Live(policy(rounds=2)), Chat(), []
        app.ask("Go")

        def send(body):
            sent.append(body)
            k = len(sent)
            calls = [self.call("read", k)]
            if k == 5:
                handle = NOTICE.search(json.dumps(body["messages"])).group()
                looks = self.call(REOPEN, k, json.dumps({"reference": handle}))
                calls = [looks, *calls] if beside else [looks]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            message["reasoning_details"] = [part(k, calls[0]["id"])]
            usage = {
                "prompt_tokens": len(json.dumps(body)) // 4,
                "completion_tokens": 5,
            }
            if not body.get("stream"):
                choice = {"index": 0, "message": message, "finish_reason": "tool_calls"}
                return Result(200, json={"choices": [choice], "usage": usage})
            deltas = [{"reasoning_details": message["reasoning_details"]}]
            deltas += [
                {"tool_calls": [{**call, "index": i}]} for i, call in enumerate(calls)
            ]
            data = [{"choices": [{"index": 0, "delta": d}]} for d in deltas]
            last = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
            data.append({"choices": [last], "usage": usage})
            return Result(200, events=iter([*map(sse, data), b"data: [DONE]\n\n"]))

        for k in range(6):
            answered = live.exchange(Chat.path, app.body(stream), send, thread="t")
            if answered.events is None:
                message = answered.json["choices"][0]["message"]
            else:
                events = [d for _, _, d in wire.sse_events(answered.events) if d]
                message = Chat.gathered(events)["choices"][0]["message"]
                message["reasoning_details"] = [
                    found
                    for data in events
                    for found in data["choices"][0]["delta"].get("reasoning_details")
                    or ()
                ]
            app.said.append(message)
            app.answer([(call["id"], result(k)) for call in message["tool_calls"]])
        self.assertEqual(live.stats["reopened"], 1)
        return len(sent), sent[-1]["messages"]

    def test_each_item_once(self):
        def item(k, call):
            return {"type": "reasoning.encrypted", "id": f"rs_{k}", "data": "x" * 40}

        for stream in (False, True):
            for beside in (False, True):
                with self.subTest(stream=stream, beside=beside):
                    count, said = self.play(stream, beside, item)
                    held = json.dumps(said)
                    for k in range(1, count):
                        self.assertEqual(held.count(f'"rs_{k}"'), 1, k)

    def test_what_belongs_to_a_call_of_ours_stays_with_it(self):
        """As the signature that Gemini wants back with every call."""

        def signature(k, call):
            return {"type": "reasoning.encrypted", "id": call, "data": "x" * 40}

        for stream in (False, True):
            _, said = self.play(stream, True, signature)
            (ours,) = (
                m
                for m in said
                if REOPEN in json.dumps(m.get("tool_calls") or "")
                and m["role"] == "assistant"
            )
            names = {part["id"] for part in ours["reasoning_details"]}
            self.assertEqual(names, {ours["tool_calls"][0]["id"]})
            if not stream:
                # And is not in the application's, which does not have the call.
                held = [m for m in said if m.get("reasoning_details")]
                named = [p["id"] for m in held for p in m["reasoning_details"]]
                self.assertEqual(named.count(*names), 1)


class FailTests(unittest.TestCase):
    """What compache sends of its own may fail, and then the application has to hear what it needs to."""

    def play(self, shape, stream, failing, text="", count=9):
        """An agent whose model looks back once. `failing(body)` says what comes of a request, if not a reply."""
        model = FakeModel(fetching(5, count, text))
        live = compache.Live(policy(rounds=2))

        def send_for(path):
            send = FakeModel.send(model, path)

            def sending(body):
                found = failing(body)
                if isinstance(found, Exception):
                    raise found
                return send(body) if found is None else found

            return sending

        model.send = send_for
        return live, model, Agent(live, shape(), model, stream=stream)

    @staticmethod
    def follows(body):
        return '\\"status\\": \\"ok\\"' in json.dumps(body).replace("\\\\\\", "\\")

    def test_what_follows_a_fetch_may_fail(self):
        for shape in SHAPES:
            for stream in (False, True):
                for text in ("", "Let me look back."):
                    for status in (400, 529, ConnectionError("gone")):
                        failed: list = []

                        def failing(body):
                            if "reopen" in json.dumps(body) and self.follows(body):
                                if status == 400 or not failed:
                                    failed.append(1)
                                    if isinstance(status, Exception):
                                        return status
                                    said = {"error": {"message": "no"}}
                                    return Result(status, json=said)
                            return None

                        with self.subTest(shape.__name__, stream=stream, status=status):
                            live, _, app = self.play(shape, stream, failing, text)
                            self.assertIn("All done.", app.run("Read nine files"))
                            self.assertEqual(app.made, 9)
                            self.assertTrue(failed)
                            self.assertEqual(live.stats["turns"], 1)

    def test_where_nothing_works_the_provider_s_error_is_passed_on(self):
        for shape in SHAPES:
            for stream in (False, True):
                down: list = []

                def failing(body):
                    if down or self.follows(body):
                        down.append(1)
                        said = {"message": "overloaded", "type": "overloaded_error"}
                        return Result(529, json={"type": "error", "error": said})
                    return None

                with self.subTest(shape.__name__, stream=stream):
                    live, model, app = self.play(shape, stream, failing, "Hm.")
                    app.shape.ask("Read nine files")
                    path = shape.path
                    if stream:
                        path = path.replace(":generate", ":streamGenerate")
                    for k in range(1, 7):
                        body = app.shape.body(stream)
                        answered = live.exchange(path, body, model.send(path))
                        if not stream:
                            if answered.status != 200:
                                break
                            _, calls = app.shape.take(answered.json)
                        else:
                            raw = b"".join(answered.events)
                            events = [d for _, _, d in wire.sse_events([raw]) if d]
                            if down:
                                break
                            _, calls = app.shape.take(app.shape.gathered(events))
                        app.shape.answer([(call, result(k)) for call, _ in calls])
                    if stream:
                        self.assertIn(b"overloaded", raw)
                        self.assertNotIn(b"message_stop", raw)
                        self.assertNotIn(b"[DONE]", raw)
                        self.assertNotIn(b'"response.completed"', raw)
                        self.assertNotIn(b'"function_call"', raw)
                        self.assertNotIn(b"tool_use", raw)
                    else:
                        self.assertEqual(answered.status, 529)
                    self.assertEqual(live.stats["turns"], 0)
                    # It asks again once the provider is back, and the turn goes on.
                    down.clear()
                    self.follows = lambda body: False
                    try:
                        body = app.shape.body(False)
                        again = live.exchange(shape.path, body, model.send(shape.path))
                        _, calls = app.shape.take(again.json)
                        self.assertTrue(calls)
                        self.assertEqual(len(live.conversations()), 1)
                    finally:
                        del self.follows

    def test_where_nothing_can_be_sent_the_application_hears_why(self):
        for stream in (False, True):
            down: list = []

            def failing(body):
                if down or self.follows(body):
                    down.append(1)
                    return ConnectionResetError("the network is down")
                return None

            live, _, app = self.play(Claude, stream, failing, "Hm.")
            with self.assertRaises(ConnectionResetError):
                app.run("Read nine files")
            self.assertEqual(app.made, 5)

    def test_a_refusal_that_is_none_of_the_policy_s_doing(self):
        limit, credit = 60_000, [True]

        def failing(body):
            said = None
            if not credit:
                said = "Your credit balance is too low"
            elif len(json.dumps(body)) > limit:
                said = "prompt is too long"
            return said and Result(400, json={"error": {"message": said}})

        model = FakeModel(worker(200))
        live = compache.Live(policy(rounds=3))
        send = FakeModel.send(model, Claude.path)
        model.send = lambda path: lambda body: failing(body) or send(body)
        app = Agent(live, Claude(), model)
        app.shape.ask("Read")
        for _ in range(60):
            _, calls = app.call()
            app.made += 1
            app.shape.answer([(call, result(app.made)) for call, _ in calls])
        body = app.shape.body(False)
        self.assertGreater(len(json.dumps(body)), limit)
        credit.clear()
        answered = live.exchange(Claude.path, body, model.send(Claude.path))
        self.assertEqual(answered.status, 400)
        credit.append(True)
        answered = live.exchange(Claude.path, body, model.send(Claude.path))
        self.assertEqual(answered.status, 200)
        self.assertEqual(live.stats["stood_down"], 0)

    def test_a_reply_that_breaks_off_after_a_fetch_is_no_answer(self):
        for shape in SHAPES:
            broken: list = []
            model = FakeModel(fetching(5, 9))
            live = compache.Live(policy(rounds=2))

            def send_for(path):
                send = FakeModel.send(model, path)

                def sending(body):
                    made = send(body)
                    if self.follows(body) and not broken and made.events:
                        broken.append(1)
                        events = list(made.events)

                        def some():
                            yield from events[: len(events) // 2]
                            raise ConnectionResetError("the connection dropped")

                        return Result(200, events=some())
                    return made

                return sending

            model.send = send_for
            app = Agent(live, shape(), model, stream=True)
            with self.subTest(shape.__name__):
                with self.assertRaises(ConnectionResetError):
                    app.run("Read nine files")
                self.assertEqual(live.stats["turns"], 0)
                _, calls = app.call()
                self.assertTrue(calls)
                sent = json.dumps(model.bodies[-1])
                self.assertIn("a line of file number 1 that", sent)
                self.assertEqual(len(live.conversations()), 1)

    def test_a_stream_that_is_closed_unread_lets_go(self):
        model = FakeModel(worker(3))
        live = compache.Live(policy())
        closed: list = []

        def send(body):
            made = FakeModel.send(model, Claude.path)(body)
            made.close = lambda: closed.append(1)
            return made

        app = Claude()
        app.ask("Read")
        for begun in (False, True):
            body = app.body(True)
            events = live.exchange(Claude.path, body, send, thread="t").events
            if begun:
                next(events)
            events.close()
        self.assertEqual(len(closed), 2)
        self.assertEqual(live._latest["t"].pending.in_flight, 0)


class MarkTests(unittest.TestCase):
    """Cache marks beside those of the application."""

    def strict(self, model, path, marks, refused):
        send = FakeModel.send(model, path)

        def sending(body):
            found = marks(body)
            short = False
            wrong = len(found) > 4
            for mark in found:
                short = short or mark.get("ttl", "5m") == "5m"
                wrong = wrong or (short and mark.get("ttl") == "1h")
            if wrong:
                refused.append(found)
                return Result(400, json={"error": {"message": "cache_control"}})
            return send(body)

        return sending

    @staticmethod
    def marks(body):
        found = [t["cache_control"] for t in body["tools"] if "cache_control" in t]
        for key in ("system", "messages"):
            value = body.get(key)
            for item in value if isinstance(value, list) else ():
                pieces = item.get("content")
                pieces = pieces if isinstance(pieces, list) else []
                for piece in (item, *pieces):
                    if "cache_control" in piece:
                        found.append(piece["cache_control"])
        return [*found, *([body["cache_control"]] if "cache_control" in body else [])]

    def play(self, shape, stream, **recipe):
        clock, refused = Clock(), []
        model = FakeModel(fetching(5, 8), clock=lambda: clock.at)
        path = shape.path
        model.send = lambda _: self.strict(model, path, self.marks, refused)
        live = compache.Live(for_an_hour(**recipe), clock=clock)
        app = Agent(live, shape(), model, stream=stream, clock=clock)
        self.assertEqual(app.run("Read eight files"), "All done.")
        self.assertEqual(refused, [])
        self.assertEqual(live.stats["reopened"], 1)
        self.assertEqual(live.stats["stood_down"], 0)
        return live, model, app, clock

    def test_the_mark_for_a_whole_request_keeps_its_lifetime(self):
        class App(Claude):
            own = False

            def body(self, stream):
                body = {**super().body(stream), "cache_control": MARK}
                if self.own:
                    text = {"type": "text", "text": "You are an agent."}
                    body["system"] = [{**text, "cache_control": MARK}]
                return body

        class Own(App):
            own = True

        for shape in (App, Own):
            for stream in (False, True):
                _, model, _, _ = self.play(shape, stream)
                for body in model.bodies:
                    self.assertEqual(body["cache_control"], MARK)
                    self.assertEqual(self.marks(body)[0].get("ttl"), "1h")

    def test_chat_completions_for_claude_keeps_to_the_rules(self):
        def marked(message):
            content = message.get("content")
            if not isinstance(content, str) or not content:
                return message
            piece = {"type": "text", "text": content, "cache_control": MARK}
            return {**message, "content": [piece]}

        class Base(Chat):
            model = "anthropic/claude-agent"

        class Rolling(Base):
            def body(self, stream):
                body = super().body(stream)
                said = list(body["messages"])
                last = [i for i, m in enumerate(said) if m["role"] in ("user", "tool")]
                for i in last[-2:]:
                    said[i] = marked(said[i])
                return {**body, "messages": said}

        class Two(Base):
            def body(self, stream):
                body = super().body(stream)
                said = list(body["messages"])
                fixed = {"type": "text", "text": "Fixed.", "cache_control": MARK}
                moment = {"type": "text", "text": "Of the moment."}
                said[0] = {"role": "system", "content": [fixed, moment]}
                said[-1] = marked(said[-1])
                return {**body, "messages": said}

        class Tool(Base):
            def body(self, stream):
                body = super().body(stream)
                said = list(body["messages"])
                said[-1] = marked(said[-1])
                tools = [{**body["tools"][0], "cache_control": MARK}]
                return {**body, "messages": said, "tools": tools}

        for shape in (Rolling, Two, Tool):
            for stream in (False, True):
                with self.subTest(shape.__name__, stream=stream):
                    self.play(shape, stream)

    def test_the_mark_at_the_cut_does_not_give_way_to_the_next_one(self):
        class App(Claude):
            def body(self, stream):
                body = super().body(stream)
                text = {"type": "text", "text": "A long prompt. " * 60}
                body["system"] = [{**text, "cache_control": MARK}] * 2
                said = [dict(m) for m in body["messages"]]
                for i in [i for i, m in enumerate(said) if m["role"] == "user"][-2:]:
                    content = said[i]["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    last = {**content[-1], "cache_control": MARK}
                    said[i]["content"] = [*content[:-1], last]
                return {**body, "messages": said}

        clock, usages = Clock(), []
        model = FakeModel(worker(60), clock=lambda: clock.at)
        send = FakeModel.send(model, Claude.path)

        def keeping(body):
            self.assertLessEqual(len(self.marks(body)), 4)
            answered = send(body)
            usages.append(answered.json["usage"])
            return answered

        model.send = lambda path: keeping
        live = compache.Live(policy(rounds=12, ratio=1.5), clock=clock)
        Agent(live, App(), model, clock=clock).run("Read sixty files", most=80)
        gone = [sum(map(noticed, Claude.results(b))) for b in model.bodies]
        cuts = [n for n in range(1, len(gone)) if gone[n] > gone[n - 1]]
        self.assertGreaterEqual(len(cuts), 2)
        for n in cuts:
            self.assertGreater(usages[n]["cache_read_input_tokens"], 0)


class ShapeTests(unittest.TestCase):
    """Shapes of requests and replies that agents and providers have of their own."""

    def test_an_instruction_of_the_moment_does_not_move_what_was_fetched(self):
        note = {"role": "system", "content": "Tokens left: 100."}
        aside = {"type": "message", "role": "developer", "content": []}

        class ChatE(Chat):
            def body(self, stream):
                body = super().body(stream)
                return {**body, "messages": [*body["messages"], note]}

        class ClaudeE(Claude):
            def body(self, stream):
                body = super().body(stream)
                return {**body, "messages": [*body["messages"], note]}

        class CodexE(Codex):
            def body(self, stream):
                body = super().body(stream)
                return {**body, "input": [*body["input"], aside]}

        for shape in (ChatE, ClaudeE, CodexE):
            for beside in ((), (("read", {"file": 5}),)):
                for stream in (False, True):
                    model = FakeModel(fetching(5, 9, beside=beside))
                    live = compache.Live(policy(rounds=2))
                    app = Agent(live, shape(), model, stream=stream)
                    self.assertEqual(app.run("Read nine files"), "All done.")
                    for body in model.bodies:
                        self.paired(body)

    def paired(self, body):
        """Every call is followed by its result."""
        waiting: set = set()
        for item in body.get("messages") or body.get("input"):
            kind = item.get("type")
            results = {item["tool_call_id"]} if item.get("role") == "tool" else set()
            if kind == "function_call_output":
                results = {item["call_id"]}
            content = item.get("content")
            pieces = content if isinstance(content, list) else []
            results |= {
                p["tool_use_id"] for p in pieces if p.get("type") == "tool_result"
            }
            self.assertLessEqual(results, waiting)
            if not results and item.get("role") not in ("system", "developer"):
                if kind != "function_call":
                    self.assertFalse(waiting, "a call without its result")
            waiting -= results
            waiting |= {c["id"] for c in item.get("tool_calls") or ()}
            waiting |= {p["id"] for p in pieces if p.get("type") == "tool_use"}
            waiting |= {item["call_id"]} if kind == "function_call" else set()

    def test_what_was_fetched_for_one_api_is_not_put_into_another_s_request(self):
        model = FakeModel(fetching(4, 5))
        live = compache.Live(policy(rounds=3))
        app = Agent(live, Claude(), model, thread="session")
        app.run("Read five files")
        self.assertEqual(live.stats["reopened"], 1)
        said = [{"role": "system", "content": "You are an agent."}]
        for message in app.shape.said:
            content, role = message["content"], message["role"]
            if isinstance(content, str):
                said.append({"role": role, "content": content})
            elif role == "assistant":
                calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        },
                    }
                    for b in content
                    if b["type"] == "tool_use"
                ]
                text = "".join(b["text"] for b in content if b["type"] == "text")
                made = {"role": role, "content": text or None}
                said.append({**made, "tool_calls": calls} if calls else made)
            else:
                for b in content:
                    one = {"role": "tool", "tool_call_id": b["tool_use_id"]}
                    said.append({**one, "content": b["content"]})
        other = Agent(live, Chat(), model, thread="session")
        other.shape.said = said
        other.run("And now with another model")
        self.assertNotIn("tool_result", json.dumps(model.bodies[-1]))
        self.assertEqual(live.stats["policy_refused"], 0)

    def test_an_answer_in_pieces_is_the_same_answer(self):
        """Said something, fetched, answered: the next turn goes on from there."""
        for shape in SHAPES:
            for stream in (False, True):
                model = FakeModel(fetching(6, 6))
                live = compache.Live(policy(rounds=2))
                app = Agent(live, shape(), model, stream=stream, thread="session")
                app.run("Read six files")
                model.script = worker(8)
                app.run("And two more")
                with self.subTest(shape.__name__, stream=stream):
                    self.assertEqual(live._names, {"session": 1})
                    self.assertEqual(live.stats["turns"], 2)

    def test_a_reply_of_two_messages_is_one_answer(self):
        def send_for(output):
            events = [{"type": "response.created", "response": {"model": "m"}}]
            for i, item in enumerate(output):
                text = item["content"][0]["text"]
                at = {"output_index": i, "content_index": 0}
                done = {"output_index": i, "item": item}
                events += [
                    {"type": "response.output_item.added", **done},
                    {"type": "response.output_text.delta", "delta": text, **at},
                    {"type": "response.output_item.done", **done},
                ]
            usage = {"input_tokens": 10, "output_tokens": 2}
            whole = {"output": output, "usage": usage}
            events.append({"type": "response.completed", "response": whole})
            sent = [wire.sse_bytes(e["type"], e) for e in events]
            return lambda body: Result(200, events=iter(sent))

        with tempfile.TemporaryDirectory() as folder:
            live, said = compache.Live(record=folder), []
            for n in range(1, 4):
                said.append({"role": "user", "content": f"Question {n}"})
                output = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                    for text in (f"Thinking about {n}.", f"Answer {n}.")
                ]
                body = {"model": "m", "input": list(said), "stream": True}
                answered = live.exchange("/v1/responses", body, send_for(output))
                list(answered.events)
                said += output
            self.assertEqual([len(t["runs"]) for t in load(folder)[0]], [3])

    def test_the_tools_that_the_api_has_of_its_own_are_calls_too(self):
        kinds = {
            "local_shell_call": {"action": {"type": "exec", "command": ["cat", "f"]}},
            "apply_patch_call": {"operation": {"type": "update_file", "path": "f"}},
            "shell_call": {"action": {"commands": ["cat f"]}},
            "custom_tool_call": {"name": "patch", "input": "*** Begin"},
        }
        for kind, what in kinds.items():
            for stream in (False, True):
                sent: list = []

                def send(body):
                    sent.append(body)
                    n = len(sent)
                    done = sum(
                        "_call_output" in str(i.get("type")) for i in body["input"]
                    )
                    item = {"type": kind, "id": f"x_{n}", "call_id": f"c{n}", **what}
                    if done >= 8:
                        text = [{"type": "output_text", "text": "All done."}]
                        item = {"type": "message", "role": "assistant", "content": text}
                    size = len(json.dumps(body)) // 4
                    usage = {"input_tokens": size, "output_tokens": 20}
                    whole = {"model": "m", "output": [item], "usage": usage}
                    if not stream:
                        return Result(200, json=whole)
                    events = [
                        {"type": "response.created", "response": {}},
                        {"type": "response.output_item.added", "output_index": 0},
                        {"type": "response.output_item.done", "output_index": 0},
                        {"type": "response.completed", "response": whole},
                    ]
                    for event in events[1:3]:
                        event["item"] = item
                    return Result(
                        200, events=iter([wire.sse_bytes(e["type"], e) for e in events])
                    )

                live = compache.Live(policy(rounds=2))
                said = [{"role": "user", "content": "Do it"}]
                tool = {"type": "function", "name": "read", "parameters": {}}
                for k in range(1, 12):
                    body = {"model": "m", "store": False, "input": list(said)}
                    body |= {"tools": [tool], **({"stream": True} if stream else {})}
                    answered = live.exchange("/v1/responses", body, send)
                    if stream:
                        events = [d for _, _, d in wire.sse_events(answered.events)]
                        done = "response.output_item.done"
                        output = [e["item"] for e in events if e["type"] == done]
                    else:
                        output = answered.json["output"]
                    said += output
                    if output[0]["type"] != kind:
                        break
                    text = result(k)
                    found = [{"stdout": text}] if kind == "shell_call" else text
                    made = {"type": kind + "_output", "call_id": output[0]["call_id"]}
                    said.append({**made, "output": found})
                with self.subTest(kind, stream=stream):
                    self.assertEqual(len(sent), 9)
                    self.assertEqual(len(NOTICE.findall(json.dumps(sent[-1]))), 6)
                    self.assertEqual(live.stats["turns"], 1)

    def test_reasoning_that_led_to_our_call_alone_is_not_passed_on(self):
        def thought(n):
            return {"type": "reasoning", "id": f"rs_{n}", "encrypted_content": "e"}

        def call(n, name):
            made = {"type": "function_call", "id": f"fc_{n}", "call_id": f"c{n}"}
            return {**made, "name": name, "arguments": "{}"}

        cases = (
            (
                [thought(1), call(1, "read"), thought(2), call(2, REOPEN)],
                ["rs_1", "fc_1"],
            ),
            ([thought(1), call(2, REOPEN), call(1, "read")], ["rs_1", "fc_1"]),
            (
                [thought(1), call(2, REOPEN), thought(2), call(1, "read")],
                ["rs_2", "fc_1"],
            ),
        )
        for output, kept in cases:
            whole = wire.OpenAIResponses.without_tool({"output": output}, REOPEN)
            self.assertEqual([i["id"] for i in whole["output"]], kept)
            events = [{"type": "response.created", "response": {}}]
            for i, item in enumerate(output):
                at = {"output_index": i, "item": item}
                events.append({"type": "response.output_item.added", **at})
                events.append({"type": "response.output_item.done", **at})
            done = {"output": output}
            events.append({"type": "response.completed", "response": done})
            stream = wire.OpenAIResponses.Stream(wire.OpenAIResponses, REOPEN)
            passed: list = []
            for k, event in enumerate(events):
                event["sequence_number"] = k
                name = event["type"]
                passed += stream.feed(wire.sse_bytes(name, event), name, event)
            got = [d for _, _, d in wire.sse_events(passed)]
            shown = [d["item"]["id"] for d in got if "done" in d["type"]]
            self.assertEqual(shown, kept)
            self.assertEqual([i["id"] for i in got[-1]["response"]["output"]], kept)
            self.assertEqual([d["sequence_number"] for d in got], list(range(len(got))))

    def test_what_followed_reasoning_is_not_named_without_it(self):
        items = [
            {"role": "user", "content": "Fix it"},
            {"type": "reasoning", "id": "rs_1"},
            {"type": "function_call", "id": "fc_1", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
            {"type": "reasoning", "id": "rs_2"},
            {"type": "message", "role": "assistant", "id": "msg_2", "content": []},
        ]
        body, gone = wire.OpenAIResponses.without_thoughts({"input": items}, [1])
        self.assertEqual(gone, [1])
        names = [item.get("id") for item in body["input"]]
        self.assertEqual(names, [None, None, None, "rs_2", "msg_2"])

    def test_the_calls_an_application_sees_count_from_nought(self):
        asked: list = []

        def script(request):
            if len(request["texts"]) > 4 and not asked:
                asked.append(NOTICE.search(json.dumps(request["body"])).group())
                return {
                    "text": "",
                    "tools": [(REOPEN, {"reference": asked[0]}), ("read", {"file": 7})],
                }
            return {"text": "", "tool": ("read", {"file": 1})}

        model = FakeModel(script)
        live = compache.Live(policy(rounds=1))
        app = Agent(live, Chat(), model, stream=True)
        app.shape.ask("Read")
        while not asked:
            path = Chat.path
            answered = live.exchange(path, app.shape.body(True), model.send(path))
            data = [d for _, _, d in wire.sse_events(answered.events) if d]
            _, calls = app.shape.take(app.shape.gathered(data))
            app.shape.answer([(call, result(1)) for call, _ in calls])
        numbers = {
            call["index"]
            for event in data
            for choice in event["choices"]
            for call in choice["delta"].get("tool_calls") or ()
        }
        self.assertEqual(numbers, {0})

    def test_two_fetches_without_ids_get_an_answer_each(self):
        for beside in ((), (("read", {"file": 9}),)):
            for stream in (False, True):
                done: list = []

                def extra(request, seen):
                    if seen == 5 and not done:
                        done.extend(NOTICE.findall(json.dumps(request["body"]))[:2])
                        looked = [(REOPEN, {"reference": h}) for h in done]
                        return {"text": "", "tools": [*beside, *looked]}
                    return None

                model = FakeModel(worker(8, extra))
                live = compache.Live(policy(rounds=2))
                Agent(live, Gemini(), model, stream=stream).run("Read eight files")
                self.assertEqual(live.stats["reopened"], 2)
                (body,) = [
                    b
                    for b in model.bodies
                    if json.dumps(b).count('"name": "reopen"') > 3
                ][:1]
                answers = [
                    json.loads(part["functionResponse"]["response"]["output"])["handle"]
                    for content in body["contents"]
                    for part in content["parts"]
                    if part.get("functionResponse", {}).get("name") == REOPEN
                ]
                self.assertEqual(answers, done)

    def test_gemini_s_signature_stays_on_the_first_call(self):
        ours = {"functionCall": {"name": REOPEN, "args": {}, "id": "a"}}
        theirs = {"functionCall": {"name": "read", "args": {}, "id": "b"}}
        signed = {**ours, "thoughtSignature": "SIGNED"}

        def reply(parts, ends=True):
            found: dict = {"content": {"role": "model", "parts": parts}}
            return {
                "candidates": [{**found, **({"finishReason": "STOP"} if ends else {})}]
            }

        gemini = wire.Gemini
        kept = gemini._parts(gemini.without_tool(reply([signed, theirs]), REOPEN))
        self.assertEqual(kept, [{**theirs, "thoughtSignature": "SIGNED"}])
        stream, passed = gemini.Stream(gemini, REOPEN), []
        for parts, ends in (([signed], False), ([theirs], True)):
            data = reply(parts, ends)
            passed += stream.feed(wire.sse_bytes(None, data), None, data)
        shown = [p for _, _, d in wire.sse_events(passed) for p in gemini._parts(d)]
        self.assertEqual(shown, [{**theirs, "thoughtSignature": "SIGNED"}])
        mixed = reply([{**theirs, "thoughtSignature": "SIGNED"}, ours])
        (call,) = gemini.fetched(mixed, {"a": "text"})[0]["parts"]
        self.assertTrue(call["thoughtSignature"])

    def test_claude_on_vertex_ai_is_claude(self):
        class Vertex(Claude):
            path = (
                "/v1/projects/p/locations/global/publishers/anthropic/models/"
                "claude-haiku-5-5@20260901:rawPredict"
            )

            def body(self, stream):
                body = {k: v for k, v in super().body(stream).items() if k != "model"}
                return {**body, "anthropic_version": "vertex-2023-10-16"}

        for stream in (False, True):
            model = FakeModel(worker(6))
            with tempfile.TemporaryDirectory() as folder:
                live = compache.Live(policy(rounds=2), record=folder)
                app = Agent(live, Vertex(), model, stream=stream)
                app.run("Read six files")
                gone = [noticed(t) for t in Claude.results(model.bodies[-1])]
                self.assertEqual(gone, [True] * 4 + [False] * 2)
                (thread,), _ = load(folder)
                self.assertEqual(thread["requested_model"], "claude-haiku-5-5")
        count = Vertex.path.replace("claude-haiku-5-5@20260901", "count-tokens")
        self.assertIsNone(wire.dialect_for(count))
        other = Vertex.path.replace("anthropic", "mistralai")
        self.assertIsNone(wire.dialect_for(other))

    def test_a_remark_is_what_stands_in_a_tag(self):
        for text, turns in (("<system-reminder>go on", 1), ("And then?", 2)):
            model = FakeModel(worker(3))
            live = compache.Live(policy(rounds=2))
            app = Agent(live, Claude(), model)
            app.shape.ask("Read three files")
            _, calls = app.call()
            app.shape.answer([(call, result(1)) for call, _ in calls], text)
            app.call()
            self.assertEqual(live.stats["turns"], turns - 1)


class ChatTests(unittest.TestCase):
    """A chat's application has one line of it, whatever it renders anew."""

    def play(self, shape, change):
        """A chat of eight turns, each of which searches twice."""
        from test_live import SYSTEM, fitted, passage

        if not hasattr(ChatTests, "policy"):
            ChatTests.policy = fitted()
            ChatTests.runs = 0
        ChatTests.runs += 1
        clock = Clock(1_950_000_000.0 + 1e7 * ChatTests.runs)
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(ChatTests.policy, record=folder, clock=clock)
            model = FakeModel(clock=clock)
            said: list = []
            for n in range(1, 9):
                said.append({"role": "user", "content": f"Question {n}?"})
                for k in range(3):
                    tool = ("search", {"k": k}) if k < 2 else None
                    made = {"text": "" if tool else f"Answer {n}.", "tool": tool}
                    model.script = lambda request, made=made: made
                    body = {"model": "claude-sonnet-5-5", "max_tokens": 99}
                    body |= {"system": SYSTEM, "messages": change(said, n, k)}
                    body["tools"] = [{"name": "search", "input_schema": {}}]
                    answered = live.exchange(
                        Claude.path,
                        body,
                        model.send(Claude.path),
                        thread=f"chat {ChatTests.runs}",
                    ).json
                    said.append({"role": "assistant", "content": answered["content"]})
                    found = [
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"{n}.{k} " + passage(n),
                        }
                        for block in answered["content"]
                        if block["type"] == "tool_use"
                    ]
                    if found:
                        said.append({"role": "user", "content": found})
                    clock.at += 2
                clock.at += 7_200
            runs = [len(t["runs"]) for t in load(folder)[0]]
            last = sum(map(noticed, Claude.results(model.bodies[-1])))
            return live, runs, last

    def test_as_it_was(self):
        live, runs, gone = self.play(Claude, lambda said, n, k: list(said))
        self.assertEqual((runs, live.stats["stood_down"]), ([8], 0))
        self.assertGreater(gone, 8)

    def test_an_earlier_result_of_the_turn_is_shortened(self):
        def shortened(said, n, k):
            for message in said[:-1]:
                if isinstance(message["content"], list):
                    for block in message["content"]:
                        if block["type"] == "tool_result":
                            block["content"] = block["content"][:300]
            return json.loads(json.dumps(said))

        live, runs, gone = self.play(Claude, shortened)
        self.assertEqual((sum(runs), live.stats["stood_down"]), (8, 0))
        self.assertEqual(live.stats["turns"], 8)
        self.assertGreater(gone, 4)

    def test_the_question_holds_the_time(self):
        def timed(said, n, k):
            made = list(said)
            at = max(i for i, m in enumerate(made) if isinstance(m["content"], str))
            text = f"[12:00:{n}{k}] {made[at]['content']}"
            made[at] = {**made[at], "content": text}
            return made

        live, runs, _ = self.play(Claude, timed)
        self.assertEqual(live.stats["turns"], 8)
        self.assertEqual(sum(runs), 8)

    def test_the_next_question_stands_beside_a_result(self):
        """Anthropic's API has the answer to a tool the application shows something with in the user's message."""
        from test_live import SYSTEM, fitted, passage

        clock = Clock(1_980_000_000.0)
        live = compache.Live(fitted(), clock=clock)
        model = FakeModel(clock=clock)
        said: list = []
        last = None
        for n in range(1, 7):
            opens: list = [{"type": "text", "text": f"Question {n}?"}]
            if n > 1:
                shown = {"type": "tool_result", "tool_use_id": last, "content": "shown"}
                opens.insert(0, shown)
            said.append({"role": "user", "content": opens})
            for name in ("search", "show_chart"):
                model.script = lambda request, name=name: {
                    "text": "Hm.",
                    "tool": (name, {}),
                }
                body = {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 99,
                    "system": SYSTEM,
                }
                body |= {"messages": list(said), "tools": []}
                answered = live.exchange(Claude.path, body, model.send(Claude.path))
                said.append({"role": "assistant", "content": answered.json["content"]})
                last = answered.json["content"][-1]["id"]
                if name == "search":
                    found = {"type": "tool_result", "tool_use_id": last}
                    found["content"] = passage(n)
                    said.append({"role": "user", "content": [found]})
            clock.at += 3_600
        self.assertEqual(live.stats["turns"], 5)
        self.assertNotIn("Clause 001", json.dumps(model.bodies[-1]))


if __name__ == "__main__":
    unittest.main()
