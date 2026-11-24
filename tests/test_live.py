"""compache in the request path: what goes to the provider, what comes back, what is kept."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import compache
from compache import notes
from compache.live import Live, Result
from compache.record import load, load_calls
from compache.wrap import start_pings
from fakes import NOTE_ASKED, NOTICE, FakeModel, noticed, read_stream

SYSTEM = "Answer from the documents you search."
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the documents.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


def passage(n):
    return f"Clause {n:03d} of the agreement. " + "wording " * 600


class Clock:
    def __init__(self, at=1_790_000_000.0):
        self.at = at

    def __call__(self):
        return self.at


def looping(request):
    """A model that searches first and answers once it has a result."""
    messages = request["body"]["messages"]
    last = messages[-1]
    n = sum(
        m["role"] == "user" for m in messages if NOTE_ASKED not in str(m.get("content"))
    )
    if (
        NOTE_ASKED in str(last.get("content"))
        and messages[-2]["role"] == "assistant"
        and not messages[-2].get("tool_calls")
    ):
        # A note asked of a finished answer, in a request of its own.
        return {"text": "", "refs": "{R1.1}" if n > 1 else "none"}
    if last["role"] == "tool" or (
        last["role"] == "user"
        and NOTE_ASKED in str(last.get("content"))
        and request["body"]["messages"][-2]["role"] == "tool"
    ):
        return {"text": f"Answer {n}.", "refs": "{R1.1}" if n > 1 else "none"}
    return {"text": "", "tool": ("search_docs", {"query": f"q{n}"})}


class ChatApp:
    """An application with a chat-completions tool loop that keeps its own message list."""

    def __init__(
        self, live, model, *, thread=None, stream=False, name="gpt-6-luna", tools=True
    ):
        self.live, self.model, self.thread, self.stream, self.name = (
            live,
            model,
            thread,
            stream,
            name,
        )
        self.tools = [SEARCH] if tools else None
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.seen = []

    def call(self):
        body = {"model": self.name, "messages": copy.deepcopy(self.messages)}
        if self.tools:
            body["tools"] = self.tools
        if self.stream:
            body["stream"] = True
        before = json.dumps(body)
        result = self.live.exchange(
            "/v1/chat/completions", body, self.model.send(), thread=self.thread
        )
        assert json.dumps(body) == before, (
            "the application's request was changed in place"
        )
        if self.stream:
            got = read_stream(result.events)
            message = {"role": "assistant", "content": got["text"] or None}
            if got["calls"]:
                message["tool_calls"] = [
                    {"id": f"call_{i}", "type": "function", "function": c}
                    for i, c in enumerate(got["calls"])
                ]
            self.seen.append(got)
            return message
        self.seen.append(result.json)
        return result.json["choices"][0]["message"]

    def turn(self, question, found=None):
        self.messages.append({"role": "user", "content": question})
        while True:
            message = self.call()
            if not message.get("tool_calls"):
                self.messages.append(
                    {"role": "assistant", "content": message["content"]}
                )
                return message["content"]
            self.messages.append(message)
            for call in message["tool_calls"]:
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": found}
                )


def fitted(model="claude-sonnet-5-5"):
    """A policy that keeps retrieved text for one turn, under a cache that lives five minutes."""
    threads, labels = [], []
    for i in range(5):
        runs = []
        for n in range(1, 10):
            at = 1_780_000_000 + n * 3600
            runs.append(
                {
                    "started_at": at,
                    "ended_at": at + 6,
                    "input": "What does the next section say?",
                    "retrievals": [passage(n)],
                    "output": "It says the supplier carries the cost.",
                }
            )
            needs = [[f"t{i}:{n - 1}:retrieval:1"]] if n > 1 else []
            labels.append({"thread_id": f"t{i}", "run": n, "needs": needs})
        threads.append({"thread_id": f"t{i}", "runs": runs})
    return compache.fit(
        threads,
        labels,
        model=model,
        system_prompt=SYSTEM,
        share_headers=False,
        lifetimes=(),
    )


class ObserveTests(unittest.TestCase):
    def test_without_a_policy_requests_pass_unchanged_and_a_record_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock, model = Clock(), FakeModel(looping)
            live = Live(record=tmp, clock=clock)
            app = ChatApp(live, model, thread="conversation-7")
            for n in range(1, 5):
                self.assertEqual(app.turn(f"Question {n}?", passage(n)), f"Answer {n}.")
                clock.at += 120
            # Every request reached the provider exactly as the application wrote it.
            self.assertEqual(len(model.bodies), 8)
            self.assertEqual(model.bodies[-1]["messages"], app.messages[:-1])
            self.assertEqual(live.stats["turns"], 4)
            threads, labels = load(tmp)
            self.assertEqual(len(threads), 1)
            runs = threads[0]["runs"]
            self.assertEqual([r["calls"] for r in runs], [2, 2, 2, 2])
            self.assertEqual(runs[2]["input"], {"bytes": len("Question 3?")})
            self.assertEqual(runs[2]["retrievals"], [{"bytes": len(passage(3))}])
            self.assertEqual(runs[2]["output"], {"bytes": len("Answer 3.")})
            self.assertEqual(threads[0]["system_bytes"], len(SYSTEM))
            # The usage of a turn: the last call's input and what the first call found cached.
            first, second = runs[0]["usage"], runs[1]["usage"]
            self.assertEqual(first["cache_read"], 0)
            self.assertGreater(second["cache_read"], 1000)
            self.assertGreater(second["input"], second["cache_read"])
            self.assertEqual([c["kind"] for c in load_calls(tmp)], ["answer"] * 8)
            written = (Path(tmp) / "turns.jsonl").read_text()
            for word in ("Clause", "Question", "Answer", "conversation-7"):
                self.assertNotIn(word, written)
            # Only the first turn is labelled: no reply was asked for a note.
            self.assertEqual(
                labels, [{"thread_id": threads[0]["thread_id"], "run": 1, "needs": []}]
            )

    def test_a_conversation_is_recognized_by_what_it_has_said(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(looping)
            live = Live(record=tmp, clock=Clock())
            one, two = ChatApp(live, model), ChatApp(live, model)
            # Two conversations that open alike, taking turns.
            for n in range(1, 4):
                one.turn("Hello?", passage(n))
                two.turn("Hello?", passage(n + 10))
            threads, _ = load(tmp)
            self.assertEqual(sorted(len(t["runs"]) for t in threads), [3, 3])
            sizes = sorted(t["runs"][1]["retrievals"][0]["bytes"] for t in threads)
            self.assertEqual(sizes, [len(passage(2)), len(passage(12))])

    def test_what_is_not_a_conversation_goes_through_untouched(self):
        live = Live()
        seen = []

        def send(body):
            seen.append(body)
            return Result(200, json={"ok": True})

        for path, body in (
            ("/v1/embeddings", {"input": "x"}),
            ("/v1/chat/completions", {"messages": "not a list"}),
            (
                "/v1/chat/completions",
                {"messages": [{"role": "assistant", "content": "prefill"}]},
            ),
            ("/v1/chat/completions", b"raw"),
            ("/v1/responses", {"previous_response_id": "r1", "input": "next"}),
        ):
            self.assertEqual(live.exchange(path, body, send).json, {"ok": True})
            self.assertIs(seen[-1], body)
        self.assertEqual(live.stats["passed_through"], 5)

    def test_a_fit_from_the_record_of_live_traffic(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock, model = Clock(), FakeModel(looping)
            live = Live(record=tmp, clock=clock, sample=1.0)
            for i in range(4):
                app = ChatApp(live, model, thread=f"user-{i}", name="claude-sonnet-5-5")
                for n in range(1, 9):
                    app.turn(f"Question {n}?", passage(n))
                    live.drain()
                    clock.at += 3600
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (28, 28))
            policy = compache.fit(tmp, lifetimes=())
            summary = policy.summary
            self.assertEqual(
                (summary["threads"], summary["turns"], summary["labelled_turns"]),
                (4, 32, 32),
            )
            self.assertEqual(summary["model"], "claude-sonnet-5-5")
            # Every later turn named the first turn's retrieved text, and the fit kept it in view.
            self.assertIsNotNone(summary["keep"])
            self.assertEqual(summary["coverage"], 1.0)


class NoteTests(unittest.TestCase):
    def app(self, tmp, script=looping, **options):
        """An application whose replies carry the note themselves."""
        clock, model = Clock(), FakeModel(script)
        options.setdefault("notes", "inline")
        live = Live(record=tmp, clock=clock, sample=1.0, **options)
        return live, model, ChatApp(live, model, thread="t")

    def test_a_sampled_reply_is_asked_and_the_note_never_reaches_the_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            live, model, app = self.app(tmp)
            self.assertEqual(app.turn("Question 1?", passage(1)), "Answer 1.")
            # A first turn has nothing earlier to name, so nothing is asked.
            self.assertFalse(any(NOTE_ASKED in json.dumps(b) for b in model.bodies))
            self.assertEqual(app.turn("Question 2?", passage(2)), "Answer 2.")
            asked = model.bodies[-1]["messages"]
            self.assertEqual(asked[:-1], app.messages[:-1])
            self.assertIn(NOTE_ASKED, asked[-1]["content"])
            self.assertIn("{R1.1} Clause 001 of the agreement.", asked[-1]["content"])
            self.assertIn("{R2.1} Clause 002", asked[-1]["content"])
            # What the application got back holds no trace of it.
            self.assertNotIn("refs", json.dumps(app.seen))
            self.assertNotIn(NOTE_ASKED, json.dumps(app.messages))
            threads, labels = load(tmp)
            name = threads[0]["thread_id"]
            self.assertEqual(
                labels[1],
                {"thread_id": name, "run": 2, "needs": [[f"{name}:1:retrieval:1"]]},
            )
            self.assertEqual(
                threads[0]["runs"][1]["cited_part_ids"], [f"{name}:1:retrieval:1"]
            )
            self.assertEqual(live.stats["noted"], 1)

    def test_only_the_sampled_share_is_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock, model = Clock(), FakeModel(looping)
            live = Live(record=tmp, clock=clock, sample=0.25, notes="inline")
            for i in range(40):
                app = ChatApp(live, model, thread=f"t{i}")
                for n in range(1, 4):
                    app.turn(f"Question {n}?", passage(n))
            asked = live.stats["asked"]
            self.assertTrue(8 <= asked <= 32, asked)
            self.assertEqual(asked, live.stats["noted"])
            self.assertEqual(
                len([row for row in load(tmp)[1] if row["run"] > 1]), asked
            )

    def test_a_streamed_reply_is_asked_apart_and_streams_as_the_provider_sent_it(self):
        def script(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {R1.1} -->"}
            return looping(request)

        with tempfile.TemporaryDirectory() as tmp:
            clock, model = Clock(), FakeModel(script)
            live = Live(record=tmp, clock=clock, sample=1.0)
            app = ChatApp(live, model, thread="t", stream=True)
            app.turn("Question 1?", passage(1))
            self.assertEqual(app.turn("Question 2?", passage(2)), "Answer 2.")
            live.drain()
            # What has been shown cannot be taken back, so nothing was added to the requests
            # that stream: every one went out as the application wrote it.
            streamed = [b for b in model.bodies if b.get("stream")]
            self.assertEqual(len(streamed), 4)
            self.assertNotIn(NOTE_ASKED, json.dumps(streamed))
            # The streams are byte for byte what the provider sends without compache.
            direct = ChatApp(Live(), FakeModel(looping), stream=True)
            direct.turn("Question 1?", passage(1))
            direct.turn("Question 2?", passage(2))
            self.assertEqual(
                [got["raw"] for got in app.seen], [got["raw"] for got in direct.seen]
            )
            # The note was asked once the reply was in, in a request of its own.
            asked = model.bodies[-1]
            self.assertNotIn("stream", asked)
            self.assertIn(NOTE_ASKED, asked["messages"][-1]["content"])
            self.assertEqual(
                load(tmp)[1][1]["needs"][0],
                [f"{load(tmp)[0][0]['thread_id']}:1:retrieval:1"],
            )

    def test_a_label_in_the_visible_reply_makes_the_question_be_asked_again_plainly(
        self,
    ):
        def leaky(request):
            made = looping(request)
            if made.get("refs") and NOTE_ASKED in request["last"]:
                made["text"] = "As {R1.1} says: " + made["text"]
            return made

        with tempfile.TemporaryDirectory() as tmp:
            live, model, app = self.app(tmp, leaky)
            app.turn("Question 1?", passage(1))
            self.assertEqual(app.turn("Question 2?", passage(2)), "Answer 2.")
            self.assertEqual(
                (live.stats["leaked"], live.stats["retried"], live.stats["noted"]),
                (1, 1, 0),
            )
            self.assertNotIn(NOTE_ASKED, json.dumps(model.bodies[-1]))
            self.assertEqual(
                [c["kind"] for c in load_calls(tmp)][-2:], ["answer", "retry"]
            )
            self.assertEqual(len(load(tmp)[1]), 1)

    def test_a_reply_the_note_spoiled_is_asked_again_without_it(self):
        spoiled = {
            # The output limit ended it: the note took room the answer may have needed.
            "cut short": {"text": "Answer 2 was going to", "cut": True},
            # Nothing is left once the note is out.
            "only a note": {"text": "<!-- refs: {R1.1} -->"},
            "nothing at all": {"text": ""},
        }
        for name, made in spoiled.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:

                def script(request, made=made):
                    answering = request["body"]["messages"][-2]["role"] == "tool"
                    if NOTE_ASKED in request["last"] and answering:
                        return dict(made)
                    return looping(request)

                live, model, app = self.app(tmp, script)
                app.turn("Question 1?", passage(1))
                self.assertEqual(app.turn("Question 2?", passage(2)), "Answer 2.")
                self.assertEqual((live.stats["retried"], live.stats["noted"]), (1, 0))
                self.assertEqual(live.stats["fell_back"], 0)
                self.assertNotIn(NOTE_ASKED, json.dumps(model.bodies[-1]))
                self.assertEqual(
                    [c["kind"] for c in load_calls(tmp)][-2:], ["answer", "retry"]
                )
                # The turn was asked and not answered, so it carries no label.
                self.assertEqual([row["run"] for row in load(tmp)[1]], [1])

    def test_a_request_refused_with_the_note_is_sent_again_without_it(self):
        def strict(request):
            if NOTE_ASKED in request["last"]:
                return {"status": 400, "error": "unexpected message"}
            return looping(request)

        with tempfile.TemporaryDirectory() as tmp:
            live, model, app = self.app(tmp, strict)
            for n in range(1, 7):
                self.assertEqual(app.turn(f"Question {n}?", passage(n)), f"Answer {n}.")
            # Three refusals, and notes are switched off for good.
            self.assertEqual(live.stats["retried"], 3)
            self.assertEqual(sum(NOTE_ASKED in json.dumps(b) for b in model.bodies), 3)
            self.assertEqual(live.stats["turns"], 6)

    def test_a_reply_that_ignores_the_request_is_simply_not_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            live, model, app = self.app(
                tmp, lambda request: {"text": "Fine."} if True else None
            )
            app.tools = None
            app.turn("Question 1?")
            self.assertEqual(app.turn("Question 2?"), "Fine.")
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (1, 0))
            self.assertEqual([row["run"] for row in load(tmp)[1]], [1])

    def test_a_json_reply_carries_the_labels_in_a_property_of_its_own(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        seen = []

        def structured(request):
            given = request["body"]["response_format"]["json_schema"]["schema"]
            seen.append(list(given["properties"]))
            answer = {"answer": f"Answer {len(seen)}."}
            if notes.KEY in given["properties"]:
                answer[notes.KEY] = ["Q1"] if "property" in request["last"] else []
            return {"text": json.dumps(answer)}

        with tempfile.TemporaryDirectory() as tmp:
            live, model, app = self.app(tmp, structured)
            app.tools = None
            body = {
                "model": "gpt-6-luna",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": schema},
                },
            }

            def ask(question):
                app.messages.append({"role": "user", "content": question})
                request = {**body, "messages": copy.deepcopy(app.messages)}
                result = live.exchange(
                    "/v1/chat/completions", request, model.send(), thread="t"
                )
                text = result.json["choices"][0]["message"]["content"]
                app.messages.append({"role": "assistant", "content": text})
                return text

            self.assertEqual(ask("Question 1?"), '{"answer": "Answer 1."}')
            self.assertEqual(ask("Question 2?"), '{"answer": "Answer 2."}')
            # The schema is the same on every turn, asked or not, so the provider caches one.
            self.assertEqual(seen, [["answer", notes.KEY]] * 2)
            self.assertEqual(schema["required"], ["answer"])
            name = load(tmp)[0][0]["thread_id"]
            self.assertEqual(load(tmp)[1][1]["needs"], [[f"{name}:1:input"]])

            # A reply whose labels cannot be cut out is asked again with the application's own schema.
            def clumsy(request):
                given = request["body"]["response_format"]["json_schema"]["schema"]
                if notes.KEY in given["properties"]:
                    return {"text": '{"answer": "A", "_refs": ["Q1"]', "cut": True}
                return {"text": '{"answer": "whole"}'}

            model.script = clumsy
            self.assertEqual(ask("Question 2b?"), '{"answer": "whole"}')
            self.assertEqual(live.stats["retried"], 1)
            live.stats["retried"] = 0

            # A provider that refuses the wider schema gets the application's own from then on.
            def refusing(request):
                given = request["body"]["response_format"]["json_schema"]["schema"]
                if notes.KEY in given["properties"]:
                    return {"status": 400, "error": "schema not supported"}
                return {"text": '{"answer": "ok"}'}

            model.script = refusing
            self.assertEqual(ask("Question 3?"), '{"answer": "ok"}')
            self.assertEqual(ask("Question 4?"), '{"answer": "ok"}')
            self.assertEqual(live.stats["retried"], 1)


if __name__ == "__main__":
    unittest.main()


CLAUDE_SEARCH = {
    "name": "search_docs",
    "description": "Search the documents.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}


def claude_looping(request):
    """The same model on the Messages API: search first, answer once there is a result."""
    messages = request["body"]["messages"]
    last = messages[-1]["content"]
    n = sum(
        m["role"] == "user"
        and (isinstance(m["content"], str) or m["content"][0]["type"] == "text")
        for m in messages
    )
    if isinstance(last, list) and last[0]["type"] == "tool_result":
        return {"text": f"Answer {n}.", "refs": "{R1.1}" if n > 1 else "none"}
    if isinstance(last, str) and NOTE_ASKED in last:
        # A note asked of a finished answer, in a request of its own: it is not a turn.
        return {"text": "", "refs": "{R1.1}" if n > 2 else "none"}
    return {"text": "I will search.", "tool": ("search_docs", {"query": f"q{n}"})}


class ClaudeApp:
    """An application on the Messages API, with the tool loop of Anthropic's guide."""

    def __init__(self, live, model, *, thread=None, stream=False):
        self.live, self.model, self.thread, self.stream = live, model, thread, stream
        self.messages = []
        self.seen = []

    def call(self):
        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 1024,
            "system": SYSTEM,
            "tools": [CLAUDE_SEARCH],
            "messages": copy.deepcopy(self.messages),
        }
        if self.stream:
            body["stream"] = True
        before = json.dumps(body)
        result = self.live.exchange(
            "/v1/messages", body, self.model.send("/v1/messages"), thread=self.thread
        )
        assert json.dumps(body) == before, (
            "the application's request was changed in place"
        )
        if not self.stream:
            self.seen.append(result.json)
            return result.json["content"], result.json["stop_reason"]
        got = read_stream(result.events)
        self.seen.append(got)
        content = [{"type": "text", "text": got["text"]}] if got["text"] else []
        for i, call in enumerate(got["calls"]):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{len(self.seen)}_{i}",
                    "name": call["name"],
                    "input": json.loads(call["arguments"] or "{}"),
                }
            )
        return content, got["ended"]

    def turn(self, question, found=None):
        self.messages.append({"role": "user", "content": question})
        while True:
            content, stop = self.call()
            if stop != "tool_use":
                # Kept as the SDK hands it over: a list of blocks.
                self.messages.append({"role": "assistant", "content": content})
                return "".join(b["text"] for b in content if b["type"] == "text")
            self.messages.append({"role": "assistant", "content": content})
            results = [
                {"type": "tool_result", "tool_use_id": b["id"], "content": found}
                for b in content
                if b["type"] == "tool_use"
            ]
            self.messages.append({"role": "user", "content": results})


def gone(body) -> list[bool]:
    """For each tool result of a request, whether its place holds a notice."""
    found = []
    for message in body["messages"]:
        content = message.get("content")
        if message.get("role") == "tool":
            found.append(noticed(content))
        elif isinstance(content, list):
            found += [noticed(b) for b in content if b.get("type") == "tool_result"]
    return found


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = fitted()
        cls.runs = 0

    def start(self, app=ChatApp, script=None, gap=3600, **options):
        """A conversation under the policy, on a clock of its own."""
        type(self).runs += 1
        self.clock = Clock(1_790_000_000.0 + self.runs * 1e7)
        self.model = FakeModel(
            script or (looping if app is ChatApp else claude_looping), clock=self.clock
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.live = Live(self.policy, record=self.tmp.name, clock=self.clock, **options)
        more = {"name": "claude-sonnet-5-5"} if app is ChatApp else {}
        self.gap = gap
        return app(self.live, self.model, thread=f"thread-{self.runs}", **more)

    def play(self, app, turns):
        for n in range(1, turns + 1):
            self.assertEqual(app.turn(f"Question {n}?", passage(n)), f"Answer {n}.")
            self.clock.at += self.gap

    def test_archived_parts_reach_the_provider_as_notices_and_the_application_keeps_its_own(
        self,
    ):
        for kind in (ChatApp, ClaudeApp):
            with self.subTest(app=kind.__name__):
                app = self.start(kind)
                self.play(app, 4)
                # Retrieved text stays for one turn. From the third turn the first is a notice.
                self.assertEqual(
                    [gone(b) for b in self.model.bodies[::2]],
                    [[], [False], [True, False], [True, True, False]],
                )
                self.assertEqual(gone({"messages": app.messages}), [False] * 4)
                sent = json.dumps(self.model.bodies[-1])
                self.assertIn('reopen(\\"ctx:', sent)
                self.assertNotIn("Clause 001", sent)
                self.assertIn("Clause 004", sent)
                # The tool to fetch a part back comes with the first notice, and stays.
                offered = []
                for body in self.model.bodies:
                    names = [
                        t.get("name") or t["function"]["name"] for t in body["tools"]
                    ]
                    self.assertEqual(
                        names,
                        ["search_docs", "reopen"] if noticed(body) else ["search_docs"],
                    )
                    offered.append("reopen" in names)
                self.assertEqual(offered, [False] * 4 + [True] * 4)
                # The record says which parts went out as a notice, by label, and holds the sizes
                # of what was said, not of what was sent.
                last = load_calls(self.tmp.name)[-1]
                self.assertEqual(
                    (last["turn"], last["archived"]), (4, ["R1.1", "R2.1"])
                )
                runs = load(self.tmp.name)[0][0]["runs"]
                self.assertEqual(runs[0]["retrievals"], [{"bytes": len(passage(1))}])

    def test_a_tool_result_in_several_pieces_is_one_part(self):
        clock = Clock(1_990_000_000.0)
        model = FakeModel(
            lambda request: {"text": f"Answer {request['n']}."}, clock=clock
        )
        live = Live(self.policy, clock=clock)
        messages = [{"role": "system", "content": SYSTEM}]
        for n in range(1, 4):
            call = {
                "id": f"c{n}",
                "type": "function",
                "function": {"name": "search_docs", "arguments": "{}"},
            }
            found = [
                {"type": "text", "text": passage(n)},
                {"type": "text", "text": f"Second piece {n}. " * 80},
            ]
            messages += [
                {"role": "user", "content": f"Question {n}?"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": f"c{n}", "content": found},
            ]
            body = {
                "model": "claude-sonnet-5-5",
                "messages": copy.deepcopy(messages),
                "tools": [SEARCH],
            }
            reply = live.exchange(
                "/v1/chat/completions", body, model.send(), thread="pieces"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.json["choices"][0]["message"]["content"],
                }
            )
            clock.at += 3600
        first = model.bodies[-1]["messages"][3]["content"]
        # The first piece carries the notice and the second a stand-in, so the message keeps its shape.
        self.assertIn('Call reopen("ctx:', first[0]["text"])
        self.assertEqual(first[1]["text"], "…")
        self.assertEqual(len(model.bodies[-1]["messages"][-1]["content"]), 2)

    def test_the_end_of_the_conversation_is_marked_cacheable_and_a_warm_cache_is_left_alone(
        self,
    ):
        for kind in (ChatApp, ClaudeApp):
            with self.subTest(app=kind.__name__):
                app = self.start(kind, gap=60)
                self.play(app, 4)
                for body in self.model.bodies:
                    marks = ["cache_control" in json.dumps(m) for m in body["messages"]]
                    self.assertEqual(marks, [False] * (len(marks) - 1) + [True])
                # Users are back within the cache's lifetime: a cut would rewrite what is cached, so none is made.
                self.assertEqual(sum(map(sum, map(gone, self.model.bodies))), 0)
                reads = [c["usage"]["cache_read"] for c in load_calls(self.tmp.name)]
                self.assertEqual(reads[0], 0)
                # From the second turn on every call reads what the one before it left.
                self.assertTrue(all(read > 300 for read in reads[2:]), reads)
                # Nothing was archived, so the model was never handed the tool to fetch it back.
                self.assertNotIn("reopen", json.dumps(self.model.bodies))

    def test_a_new_process_takes_the_cache_to_have_expired(self):
        app = self.start(gap=5)
        self.play(app, 4)
        # Users were back while the cache was warm, so nothing was cut.
        self.assertEqual(sum(map(sum, map(gone, self.model.bodies))), 0)
        # After a deploy, on the command line or in a function, a process meets a conversation in
        # the middle. It cannot know that the cache is warm, and a cut is made at once.
        before = len(self.model.bodies)
        app.live = Live(compache.Policy(self.policy.artifact), clock=self.clock)
        app.turn("Question 5?", passage(5))
        self.assertEqual(gone(self.model.bodies[before]), [True, True, True, False])
        # From there it knows what it sent: the next turn sends the same start and reads it from the cache.
        self.clock.at += 5
        app.turn("Question 6?", passage(6))
        self.assertEqual(
            gone(self.model.bodies[-1]), [True, True, True, False, False, False]
        )
        answered, returned = (reply["usage"] for reply in app.seen[-3:-1])
        self.assertGreater(
            returned["prompt_tokens_details"]["cached_tokens"],
            0.9 * answered["prompt_tokens"],
        )

    def test_a_turn_asked_again_finds_the_cache_as_this_process_left_it(self):
        app = self.start(gap=5)
        self.play(app, 5)
        # The user asks the last turn again: the same history, and a cache that is seconds old.
        app.messages = app.messages[
            : app.messages.index({"role": "user", "content": "Question 5?"})
        ]
        before = len(self.model.bodies)
        app.turn("Question 5, put differently?", passage(5))
        self.assertEqual(gone(self.model.bodies[before]), [False] * 4)
        self.assertGreater(
            app.seen[-2]["usage"]["prompt_tokens_details"]["cached_tokens"], 1000
        )
        # Where the request was already trimmed, what is cached is not the whole history anyway.
        app = self.start()
        self.play(app, 5)
        app.messages = app.messages[
            : app.messages.index({"role": "user", "content": "Question 5?"})
        ]
        before = len(self.model.bodies)
        app.turn("Question 5, put differently?", passage(5))
        self.assertEqual(gone(self.model.bodies[before]), [True, True, True, False])

    def test_instructions_inside_a_conversation_do_not_hide_it(self):
        """A harness may end every request on an instruction of its own, as Claude Code does."""

        def script(request):
            said = [m for m in request["body"]["messages"] if m["role"] != "system"]
            return claude_looping({**request, "body": {"messages": said}})

        app = self.start(ClaudeApp, script)
        send = app.live.exchange

        def told(path, body, *args, **options):
            note = {"role": "system", "content": f"The time is {self.clock.at}."}
            return send(
                path, {**body, "messages": [*body["messages"], note]}, *args, **options
            )

        app.live.exchange = told
        self.play(app, 4)
        self.assertEqual(
            [gone(b) for b in self.model.bodies[::2]],
            [[], [False], [True, False], [True, True, False]],
        )
        self.assertEqual(self.live.stats["passed_through"], 0)
        self.assertEqual(self.live.stats["turns"], 4)

    def test_a_model_without_a_cache_to_mark_gets_no_mark(self):
        app = self.start(ChatApp)
        app.name = "gpt-6-luna"
        self.play(app, 2)
        self.assertNotIn("cache_control", json.dumps(self.model.bodies))

    @staticmethod
    def reopening(base):
        """A model that fetches back the notice it finds before it goes on."""

        def script(request):
            made = base(request)
            sent = json.dumps(request["body"])
            if request["last"].startswith('{"status": "ok"'):
                # The archived text came back: answer the question.
                return {
                    "text": made["text"]
                    if not made.get("tool")
                    else f"Answer {request['n']}."
                }
            if noticed(sent) and made.get("tool"):
                handle = NOTICE.search(sent).group()
                return {"text": made["text"], "tool": ("reopen", {"reference": handle})}
            return made

        return script

    def test_the_model_fetches_an_archived_part_back_without_the_application_seeing_it(
        self,
    ):
        for kind, base in ((ChatApp, looping), (ClaudeApp, claude_looping)):
            for stream in (False, True):
                with self.subTest(app=kind.__name__, stream=stream):
                    app = self.start(kind, self.reopening(base))
                    app.stream = stream
                    for n in (1, 2):
                        app.turn(f"Question {n}?", passage(n))
                        self.clock.at += 3600
                    before = len(self.model.bodies)
                    answer = app.turn("Question 3?", passage(3))
                    self.assertTrue(answer.endswith("."), answer)
                    self.assertEqual(self.live.stats["reopened"], 1)
                    # The provider was asked once more, with the original text as the tool's result.
                    first, second = self.model.bodies[before:]
                    self.assertNotIn("Clause 001 of the agreement", json.dumps(first))
                    self.assertIn("Clause 001 of the agreement", json.dumps(second))
                    self.assertIn("reopen", json.dumps(second["messages"][-2]))
                    # The model is shown the text, not the names parts go by in here.
                    self.assertNotIn(app.thread, json.dumps(second))
                    # The application made one call and saw one answer, with no tool call in it.
                    seen = app.seen[-1]
                    shown = seen["raw"].decode() if "raw" in seen else json.dumps(seen)
                    self.assertNotIn("reopen", shown)
                    self.assertNotIn("tool_use", shown.replace("tool_use_id", ""))
                    self.assertNotIn(
                        "tool_calls", shown.replace('"tool_calls": null', "")
                    )
                    self.assertNotIn("reopen", json.dumps(app.messages))
                    self.assertEqual(
                        [c["kind"] for c in load_calls(self.tmp.name)][-2:],
                        ["answer", "reopen"],
                    )
                    # The conversation goes on as one.
                    self.clock.at += 3600
                    app.turn("Question 4?", passage(4))
                    self.assertEqual(len(load(self.tmp.name)[0][0]["runs"]), 4)

    def test_a_part_the_reply_fetched_back_counts_as_one_it_relied_on(self):
        def script(request):
            texts = [t for t in request["texts"] if NOTE_ASKED not in t]
            if texts[-1].startswith('{"status": "ok"'):
                # The archived text came back. The note still says the reply needed nothing.
                return {"text": "Answer 3.\n<!-- refs: none -->"}
            sent = json.dumps(request["body"])
            if noticed(sent) and texts[-1].endswith("?"):
                return {"tool": ("reopen", {"reference": NOTICE.search(sent).group()})}
            return looping(request)

        app = self.start(ChatApp, script, sample=1.0, notes="inline")
        for n in (1, 2, 3):
            app.turn(f"Question {n}?", passage(n))
            self.clock.at += 3600
        self.assertEqual(self.live.stats["reopened"], 1)
        _, labels = load(self.tmp.name)
        third = next(row for row in labels if row["run"] == 3)
        self.assertEqual(len(third["needs"]), 1)
        self.assertRegex(third["needs"][0][0], r":1:retrieval:1$")

    def test_a_search_that_finds_nothing_says_what_to_try(self):
        seen = []

        def script(request):
            texts = [t for t in request["texts"] if NOTE_ASKED not in t]
            if texts[-1].startswith('{"status"'):
                seen.append(json.loads(texts[-1]))
                return {"text": "Answer."}
            if noticed(json.dumps(request["body"])) and texts[-1].endswith("?"):
                return {"tool": ("reopen", {"reference": "gearbox oil"})}
            return looping(request)

        app = self.start(ChatApp, script)
        for n in (1, 2, 3):
            app.turn(f"Question {n}?", passage(n))
            self.clock.at += 3600
        self.assertEqual(seen[0]["status"], "not_found")
        self.assertIn("other or fewer distinctive words", seen[0]["hint"])

    @staticmethod
    def never_done(times):
        """A model that fetches again and again when a question meets a notice: `times` questions long."""
        left = [times]

        def script(request):
            body = request["body"]
            last = [
                m for m in body["messages"] if NOTE_ASKED not in str(m.get("content"))
            ][-1]
            sent = json.dumps(body)
            if noticed(sent) and "reopen" in json.dumps(body.get("tools")):
                # A cache mark turns the question's text into a list of one text block.
                said = last.get("content")
                said = (
                    said
                    if isinstance(said, str)
                    else "".join(p.get("text", "") for p in said or ())
                )
                asked = last["role"] == "user" and said.endswith("?")
                if asked and left[0]:
                    left[0] -= 1
                    left.append("fetching")
                if left[-1] == "fetching":
                    return {
                        "tool": ("reopen", {"reference": NOTICE.search(sent).group()})
                    }
            if left[-1] == "fetching":
                # The request came as the application wrote it: nothing to fetch any more.
                left.append("done")
            return looping(request)

        return script

    def test_a_model_that_goes_on_fetching_is_asked_as_the_application_wrote_it(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                app = self.start(ChatApp, self.never_done(1))
                app.stream = stream
                self.play(app, 4)
                # Three fetches were answered. The fourth made it send the application's own request.
                self.assertEqual(self.live.stats["reopened"], 3)
                self.assertEqual(self.live.stats["fell_back"], 1)
                self.assertEqual(self.live.stats["stood_down"], 0)
                written = [b for b in self.model.bodies if not noticed(b)]
                self.assertNotIn("reopen", json.dumps(written))
                self.assertNotIn("reopen", json.dumps(app.messages))
                kinds = [c["kind"] for c in load_calls(self.tmp.name) if c["turn"] == 3]
                self.assertEqual(
                    kinds, ["answer", "reopen", "reopen", "reopen", "retry", "answer"]
                )
                # The policy goes on being applied to the turns after it.
                self.assertTrue(noticed(self.model.bodies[-1]))

    def test_a_conversation_that_needs_this_twice_is_left_alone(self):
        app = self.start(ChatApp, self.never_done(2))
        self.play(app, 5)
        self.assertEqual(self.live.stats["fell_back"], 2)
        self.assertEqual(self.live.stats["stood_down"], 1)
        # From then on its requests go out as the application writes them.
        self.assertFalse(noticed(self.model.bodies[-1]))
        self.assertEqual(len(load(self.tmp.name)[0][0]["runs"]), 5)

    def test_an_empty_reply_to_a_changed_request_is_asked_again_as_written(self):
        def script(request):
            made = looping(request)
            if noticed(json.dumps(request["body"])) and not made.get("tool"):
                return {"text": ""}
            return made

        app = self.start(ChatApp, script)
        self.play(app, 3)
        self.assertEqual(self.live.stats["fell_back"], 1)
        self.assertEqual(self.live.stats["stood_down"], 0)

    def test_an_empty_reply_under_a_widened_schema_is_asked_as_written(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        def script(request):
            given = request["body"]["response_format"]["json_schema"]["schema"]
            # Nothing comes back for a request with the wider schema, which carries no note here.
            return (
                {"text": ""}
                if notes.KEY in given["properties"]
                else {"text": '{"answer": "ok"}'}
            )

        clock = Clock(1_975_000_000.0)
        model = FakeModel(script, clock=clock)
        live = Live(
            record=self.tmp_dir(),
            clock=clock,
            sample=0.01,
            notes="inline",
            salt="never",
        )
        body = {
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "Question 1?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": schema},
            },
        }
        result = live.exchange("/v1/chat/completions", body, model.send(), thread="w")
        # A first turn is never asked, so the schema was only widened: the empty reply was not
        # the note's doing, and the application's own request went out instead.
        self.assertEqual(
            result.json["choices"][0]["message"]["content"], '{"answer": "ok"}'
        )
        self.assertEqual(model.bodies[-1], body)
        self.assertEqual(live.stats["fell_back"], 1)

    def test_an_answer_that_says_nothing_still_ends_its_turn(self):
        def script(request):
            made = looping(request)
            return {"text": ""} if made["text"] == "Answer 2." else made

        clock = Clock(1_985_000_000.0)
        model = FakeModel(script, clock=clock)
        live = Live(record=self.tmp_dir(), clock=clock)
        app = ChatApp(live, model)
        said = [app.turn(f"Question {n}?", passage(n)) or "" for n in (1, 2, 3)]
        self.assertEqual(said, ["Answer 1.", "", "Answer 3."])
        # It is one conversation of three turns, recognized by what it has said, silence included.
        self.assertEqual(live.stats["turns"], 3)
        threads, _ = load(live.recorder.path)
        self.assertEqual([len(t["runs"]) for t in threads], [3])

    def tmp_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_a_request_the_provider_refuses_with_the_policy_goes_out_as_the_application_wrote_it(
        self,
    ):
        def strict(request):
            if "reopen" in json.dumps(request["body"].get("tools")):
                return {"status": 400, "error": "unknown tool"}
            return looping(request)

        app = self.start(ChatApp, strict)
        self.play(app, 3)
        self.assertEqual(self.live.stats["policy_refused"], 1)
        # One refusal, and from then on this conversation is only recorded.
        self.assertEqual(len(self.model.bodies), 7)
        self.assertEqual(self.model.bodies[-1]["messages"], app.messages[:-1])
        self.assertEqual(len(load(self.tmp.name)[0][0]["runs"]), 3)


class PingTests(unittest.TestCase):
    def test_a_due_ping_reads_the_cache_and_adds_the_answer(self):
        threads = []
        for i in range(4):
            runs = []
            for n in range(1, 7):
                at = 1_780_000_000 + n * 400
                runs.append(
                    {
                        "started_at": at,
                        "ended_at": at + 5,
                        "input": "Next?",
                        "retrievals": [passage(n)],
                        "output": "Done.",
                    }
                )
            threads.append({"thread_id": f"p{i}", "runs": runs})
        policy = compache.fit(
            threads,
            model="claude-sonnet-5-5",
            system_prompt=SYSTEM,
            share_headers=False,
            lifetimes=(),
        )
        self.assertGreater(policy.max_pings, 0)
        for kind in (ChatApp, ClaudeApp):
            with self.subTest(app=kind.__name__), tempfile.TemporaryDirectory() as tmp:
                clock = Clock(1_795_000_000.0 + (kind is ClaudeApp) * 1e6)
                model = FakeModel(
                    looping if kind is ChatApp else claude_looping, clock=clock
                )
                live = Live(policy, record=tmp, clock=clock)
                app = kind(
                    live,
                    model,
                    thread=f"ping-{kind.__name__}",
                    **({"name": "claude-sonnet-5-5"} if kind is ChatApp else {}),
                )
                app.turn("Question 1?", passage(1))
                due = live.next_ping_at()
                self.assertIsNotNone(due)
                self.assertEqual(live.tick(), 0)
                clock.at = due
                self.assertEqual(live.tick(), 1)
                ping = model.bodies[-1]
                self.assertEqual(ping["max_tokens"], 1)
                self.assertNotIn("stream", ping)
                last = ping["messages"][-1]
                self.assertEqual((last["role"], last["content"]), ("user", "."))
                self.assertIn("Answer 1.", json.dumps(ping["messages"][-2]))
                self.assertIn("cache_control", json.dumps(ping["messages"][-2]))
                call = load_calls(tmp)[-1]
                self.assertEqual(call["kind"], "ping")
                # It read what the answer's request had left, and wrote the answer after it.
                self.assertGreater(call["usage"]["cache_read"], 1000)
                self.assertLess(call["usage"]["cache_write"], 100)
                # The user returns after the first lifetime would have lapsed, and finds the cache warm.
                clock.at += 200
                app.turn("Question 2?", passage(2))
                first = [c for c in load_calls(tmp) if c["turn"] == 2][0]
                self.assertGreater(first["usage"]["cache_read"], 1000)
                self.assertEqual(live.stats["pings"], 1)
        # One thread sends a Live's pings, however many clients share it.
        shared = Live(policy)
        start_pings(shared)
        first = shared.pinger
        self.addCleanup(first.stop)
        start_pings(shared)
        self.assertIs(shared.pinger, first)
        for quiet in (Live(), Live(fitted())):
            start_pings(quiet)
            self.assertIsNone(quiet.pinger)


class LifetimeTests(unittest.TestCase):
    """A policy fitted for another cache lifetime than the provider's default asks for it."""

    @staticmethod
    def fit(gap):
        threads, labels = [], []
        for i in range(5):
            runs = []
            for n in range(1, 10):
                at = 1_780_000_000 + n * gap
                runs.append(
                    {
                        "started_at": at,
                        "ended_at": at + 6,
                        "input": "What does the next section say?",
                        "retrievals": [passage(n)],
                        "output": "It says the supplier carries the cost.",
                    }
                )
                needs = [[f"t{i}:{n - 1}:retrieval:1"]] if n > 1 else []
                labels.append({"thread_id": f"t{i}", "run": n, "needs": needs})
            threads.append({"thread_id": f"t{i}", "runs": runs})
        return compache.fit(
            threads,
            labels,
            model="claude-sonnet-5-5",
            system_prompt=SYSTEM,
            share_headers=False,
        )

    @classmethod
    def setUpClass(cls):
        cls.policy = cls.fit(1500)
        cls.started = 0

    def live(self, **options):
        type(self).started += 1
        self.clock = Clock(1_810_000_000.0 + self.started * 1e7)
        self.model = FakeModel(claude_looping, clock=self.clock, ttl=3600)
        fresh = compache.Policy(json.loads(json.dumps(self.policy.artifact)))
        return Live(fresh, clock=self.clock, **options)

    def marks(self, body):
        """Every cache mark in a request, wherever it is."""
        found = []

        def walk(value):
            if isinstance(value, dict):
                if "cache_control" in value:
                    found.append(value["cache_control"])
                for inner in value.values():
                    walk(inner)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner)

        walk(body)
        return found

    def test_requests_ask_for_it(self):
        self.assertEqual(self.policy.cache_lifetime, 3600)
        app = ClaudeApp(self.live(), self.model, thread="lasting")
        for n in range(1, 5):
            app.turn(f"Question {n}?", passage(n))
            self.clock.at += 1500
        hour = {"type": "ephemeral", "ttl": "1h"}
        for body in self.model.bodies:
            self.assertEqual(self.marks(body), [hour])
        # Users are back after 25 minutes and read what the turn before left.
        usage = app.seen[-2]["usage"]
        self.assertGreater(usage["cache_read_input_tokens"], 300)

    def test_the_application_s_own_marks_ask_for_it_too(self):
        # An entry that lives longer may not follow one that lives shorter, so all marks agree.
        live = self.live()
        sent = []

        def send(payload):
            sent.append(payload)
            return self.model.send("/v1/messages")(payload)

        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 64,
            "system": [
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Question 1?"}],
        }
        before = json.dumps(body)
        live.exchange("/v1/messages", body, send, thread="own")
        self.assertEqual(json.dumps(body), before)
        self.assertEqual(self.marks(sent[0]), [{"type": "ephemeral", "ttl": "1h"}])

    def test_a_provider_that_refuses_it_gets_the_application_s_own_request(self):
        live = self.live()
        sent = []

        def send(payload):
            sent.append(payload)
            if "ttl" in json.dumps(payload):
                return Result(400, json={"error": {"message": "ttl: unknown field"}})
            return self.model.send("/v1/messages")(payload)

        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Question 1?"}],
        }
        reply = live.exchange("/v1/messages", body, send, thread="refused")
        self.assertEqual(reply.status, 200)
        self.assertIs(sent[-1], body)
        self.assertEqual(live.stats["policy_refused"], 1)

    def test_pings_ask_for_it(self):
        # Users back after 70 minutes, just past the hour: one ping before it is over keeps the cache.
        late = self.fit(4200)
        self.assertEqual((late.cache_lifetime, late.max_pings), (3600, 1))
        live = self.live()
        live.policy = late
        app = ClaudeApp(live, self.model, thread="pinged")
        for n in range(1, 4):
            app.turn(f"Question {n}?", passage(n))
            due = live.next_ping_at()
            # Due shortly before the hour is over, not before the five minutes are.
            self.assertGreater(due, self.clock.at + 3000)
            self.assertLess(due, self.clock.at + 3600)
            self.clock.at = due
            self.assertEqual(live.tick(), 1)
            ping = self.model.bodies[-1]
            self.assertEqual(ping["max_tokens"], 1)
            self.assertEqual(self.marks(ping), [{"type": "ephemeral", "ttl": "1h"}])
            self.clock.at += 700
        # The first call of a turn reads what the ping before it left.
        self.assertGreater(app.seen[-2]["usage"]["cache_read_input_tokens"], 300)

    def test_without_a_policy_nothing_asks_for_anything(self):
        clock = Clock(1_820_000_000.0)
        model = FakeModel(claude_looping, clock=clock)
        app = ClaudeApp(Live(clock=clock), model, thread="plain")
        app.turn("Question 1?", passage(1))
        self.assertEqual([self.marks(b) for b in model.bodies], [[], []])


class ApartTests(unittest.TestCase):
    def test_a_note_asked_apart_leaves_the_reply_s_request_untouched(self):
        def script(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {R1.1} {A1} -->"}
            return looping(request)

        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(script)
            live = Live(record=tmp, clock=Clock(), sample=1.0, notes="apart")
            app = ChatApp(live, model, thread="t")
            app.turn("Question 1?", passage(1))
            self.assertEqual(app.turn("Question 2?", passage(2)), "Answer 2.")
            live.drain()
            # The two calls of the turn went out exactly as the application wrote them.
            first, second, asked = model.bodies[-3:]
            self.assertEqual(second["messages"], app.messages[:-1])
            self.assertNotIn(NOTE_ASKED, json.dumps([first, second]))
            # Then one short request of its own: the conversation, the answer, and the question.
            self.assertEqual(asked["messages"][:-2], app.messages[:-1])
            self.assertEqual(
                asked["messages"][-2], {"role": "assistant", "content": "Answer 2."}
            )
            self.assertIn(NOTE_ASKED, asked["messages"][-1]["content"])
            # Room for a model that reasons before it writes its one line.
            self.assertEqual(asked["max_tokens"], 2048)
            name = load(tmp)[0][0]["thread_id"]
            self.assertEqual(
                load(tmp)[1][1]["needs"],
                [[f"{name}:1:retrieval:1", f"{name}:1:output"]],
            )
            self.assertEqual(
                [c["kind"] for c in load_calls(tmp)][-3:], ["answer", "answer", "note"]
            )
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (1, 1))

    def test_an_answer_under_a_schema_is_asked_in_plain_text(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        def script(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {Q1} -->"}
            return {"text": '{"answer": "ok"}'}

        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(script)
            live = Live(record=tmp, clock=Clock(), sample=1.0)
            messages = []
            for n in (1, 2):
                messages.append({"role": "user", "content": f"Question {n}?"})
                body = {
                    "model": "gpt-6-luna",
                    "messages": list(messages),
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "a", "schema": schema},
                    },
                }
                at = len(model.bodies)
                result = live.exchange(
                    "/v1/chat/completions", body, model.send(), thread="s"
                )
                # The application's request, and its schema, went out as written.
                self.assertEqual(model.bodies[at], body)
                said = result.json["choices"][0]["message"]["content"]
                self.assertEqual(said, '{"answer": "ok"}')
                messages.append({"role": "assistant", "content": said})
                live.drain()
            asked = model.bodies[-1]
            self.assertNotIn("response_format", asked)
            self.assertIn(NOTE_ASKED, asked["messages"][-1]["content"])
            name = load(tmp)[0][0]["thread_id"]
            self.assertEqual(load(tmp)[1][1]["needs"], [[f"{name}:1:input"]])

    def test_a_streamed_reply_of_the_responses_api_is_asked_apart_too(self):
        def script(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {Q1} -->"}
            return {"text": f"Answer {request['n']}."}

        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(script)
            live = Live(record=tmp, clock=Clock(), sample=1.0)
            items = []
            for n in (1, 2):
                items.append({"role": "user", "content": f"Question {n}?"})
                body = {"model": "gpt-6-luna", "input": list(items), "stream": True}
                result = live.exchange(
                    "/v1/responses", body, model.send("/v1/responses"), thread="r"
                )
                raw = b"".join(result.events)
                # Nothing is put in a reply that streams.
                self.assertNotIn(b"refs", raw)
                self.assertIn(b"response.completed", raw)
                items.append(
                    {"role": "assistant", "content": f"Answer {len(model.bodies)}."}
                )
            live.drain()
            self.assertEqual(live.stats["noted"], 1)
            self.assertNotIn(NOTE_ASKED, json.dumps(model.bodies[:2]))
            self.assertIn(NOTE_ASKED, json.dumps(model.bodies[2]))

    def test_a_path_that_always_streams_is_a_stream_too(self):
        path = "/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent"

        def script(request):
            if NOTE_ASKED in request["last"]:
                return {"text": "<!-- refs: {Q1} -->"}
            return {"text": f"Answer {request['n']}."}

        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock(1_970_000_000.0)
            model = FakeModel(script, clock=clock)
            live = Live(fitted(), record=tmp, clock=clock, sample=1.0)
            contents = []
            for n in (1, 2, 3):
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": passage(n) + f" Question {n}?"}],
                    }
                )
                body, at = {"contents": list(contents)}, len(model.bodies)
                result = live.exchange(path, body, model.send(path), thread="g")
                raw = b"".join(result.events)
                # Nothing is added to what streams: no note, and no tool that could not be answered.
                self.assertNotIn(b"refs", raw)
                self.assertNotIn(NOTE_ASKED, json.dumps(model.bodies[at]))
                self.assertNotIn("reopen", json.dumps(model.bodies[at]))
                said = "".join(
                    part.get("text", "")
                    for line in raw.decode().split("\n\n")
                    if line.startswith("data: ")
                    for part in json.loads(line[6:])["candidates"][0]["content"][
                        "parts"
                    ]
                )
                contents.append({"role": "model", "parts": [{"text": said}]})
                live.drain()
                clock.at += 3600
            # The notes were asked apart, and read from the stream this path always answers with.
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (2, 2))
            self.assertEqual(
                [c["kind"] for c in load_calls(tmp)],
                ["answer", "answer", "note", "answer", "note"],
            )
            self.assertEqual(live.stats["turns"], 3)

    def test_a_policy_that_keeps_cited_text_says_so_when_nothing_is_asked(self):
        artifact = json.loads(json.dumps(fitted().artifact))
        for row in artifact["recipes"]:
            row["recipe"]["cited_answer_runs"] = 1
        policy = compache.Policy(artifact)
        with self.assertLogs("compache.live", "WARNING") as logged:
            Live(policy)
        self.assertIn("no reply is asked what it cited", logged.output[0])
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertNoLogs("compache.live", "WARNING"),
        ):
            Live(policy, record=tmp, sample=0.1)
            Live(fitted())

    def test_notes_without_a_record_are_refused(self):
        with self.assertRaisesRegex(ValueError, "record"):
            Live(sample=0.5)
        with self.assertRaises(ValueError):
            Live(notes="sometimes")


class ObservedTests(unittest.TestCase):
    """A call that something else made, such as LiteLLM: there is nothing to change, only books to keep."""

    def turns(self, live, count, *, ask=None, at=None):
        messages = []
        start = 1_900_000_000.0
        for n in range(1, count + 1):
            messages.append({"role": "user", "content": f"Question {n}?"})
            body = {"model": "claude-sonnet-5-5", "messages": list(messages)}
            reply = {
                "model": "claude-sonnet-5-5",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": f"Answer {n}."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100 * n,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {
                        "cached_tokens": 60,
                        "cache_write_tokens": 5,
                    },
                    "cost": 0.001 * n,
                },
            }
            begin = start + 600 * n
            live.observe(
                "/v1/chat/completions",
                body,
                reply,
                thread="talk",
                at=(begin, begin + 4.0),
                ask=ask,
            )
            messages.append({"role": "assistant", "content": f"Answer {n}."})
        live.drain()

    def test_an_observed_call_is_recorded_at_the_time_it_happened(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = Live(record=tmp, clock=Clock(1_950_000_000.0))
            self.turns(live, 3)
            threads, labels = load(tmp)
            self.assertEqual([len(t["runs"]) for t in threads], [3])
            # The call's own times, not the time it was told: the gap between turns is real.
            runs = threads[0]["runs"]
            self.assertAlmostEqual(
                runs[1]["started_at"] - runs[0]["ended_at"], 596.0, delta=0.01
            )
            self.assertAlmostEqual(
                runs[0]["ended_at"] - runs[0]["started_at"], 4.0, delta=0.01
            )
            self.assertEqual(
                runs[2]["usage"],
                {"input": 300, "output": 10, "cache_read": 60, "cache_write": 5},
            )
            calls = load_calls(tmp)
            self.assertEqual(sum(c["cost"] for c in calls), 0.006)
            self.assertEqual([c["kind"] for c in calls], ["answer"] * 3)

    def test_notes_are_asked_only_when_something_can_ask_them(self):
        def ask(body):
            self.asked.append(body)
            return Result(
                200, json={"choices": [{"message": {"content": "<!-- refs: {Q1} -->"}}]}
            )

        with tempfile.TemporaryDirectory() as tmp:
            self.asked = []
            live = Live(record=tmp, sample=1.0, clock=Clock(1_951_000_000.0))
            self.turns(live, 2)
            # No way to ask: not one request was made, and none is counted as refused.
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (0, 0))
            self.assertFalse(live._notes_off)
            live = Live(record=tmp + "/b", sample=1.0, clock=Clock(1_952_000_000.0))
            self.turns(live, 2, ask=ask)
            self.assertEqual(len(self.asked), 1)
            self.assertEqual((live.stats["asked"], live.stats["noted"]), (1, 1))
            self.assertIn("<!-- refs:", json.dumps(self.asked[0]["messages"][-1]))
            name = load(tmp + "/b")[0][0]["thread_id"]
            self.assertEqual(load(tmp + "/b")[1][1]["needs"], [[f"{name}:1:input"]])

    def test_a_note_in_the_reply_is_not_possible_once_the_call_was_made(self):
        def ask(body):
            self.asked.append(body)
            return Result(
                200, json={"choices": [{"message": {"content": "<!-- refs: {Q1} -->"}}]}
            )

        with tempfile.TemporaryDirectory() as tmp:
            self.asked = []
            live = Live(
                record=tmp, sample=1.0, notes="inline", clock=Clock(1_953_000_000.0)
            )
            self.turns(live, 2, ask=ask)
            # One request of its own, after the second reply: the call itself is not made again.
            self.assertEqual(len(self.asked), 1)
            self.assertIn("Answer 2.", json.dumps(self.asked[0]["messages"][-2]))
            answers = [c for c in load_calls(tmp) if c["kind"] == "answer"]
            self.assertEqual([c["usage"]["input"] for c in answers], [100, 200])

    def test_a_policy_has_nothing_to_change_in_an_observed_call(self):
        live = Live(fitted())
        with self.assertRaisesRegex(ValueError, "already made"):
            live.observe("/v1/chat/completions", {"messages": []}, {})


class ClockTests(unittest.TestCase):
    def test_a_clock_that_stands_still_or_steps_back_does_not_stop_the_policy(self):
        policy = fitted()
        clock = Clock(1_999_000_000.0)
        model = FakeModel(looping, clock=clock)
        live = Live(policy, clock=clock)
        app = ChatApp(live, model, thread="still", name="claude-sonnet-5-5")
        for n in range(1, 5):
            app.turn(f"Question {n}?", passage(n))
            clock.at -= 0.5
        # The policy went on being applied: it never stood down.
        self.assertEqual(live.stats["stood_down"], 0)
        self.assertEqual(live.stats["policy_refused"], 0)
        self.assertEqual(live.stats["turns"], 4)


class NoticeTests(unittest.TestCase):
    def test_the_notice_a_model_reads_is_as_long_as_the_one_the_policy_counted(self):
        from compache.live import _notice
        from compache.optimizer.parts import Source, placeholder

        counted = placeholder(
            "thread", Source("thread:1:retrieval:1", "retrieval", 1, "text")
        )
        shown = _notice(counted)
        self.assertEqual(len(shown), len(counted))
        self.assertIn('Call reopen("ctx:', shown)
        self.assertEqual(_notice("something else"), "something else")
