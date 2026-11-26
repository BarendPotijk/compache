"""compache as a Hermes context engine, against Hermes' own base class.

The base class is a stand-in with the part of its interface that matters. With
HERMES_REPO set to a checkout of hermes-agent, the tests use the real one.
"""

import json
import os
import sys
import tempfile
import types
import unittest

import compache
from compache import hermes
from compache.live import Live
from compache.record import load, load_calls
from fakes import FakeModel, noticed
from test_live import SEARCH, SYSTEM, Clock, fitted, looping, passage


def stand_in():
    """`agent.context_engine` as far as a context engine needs it: the abstract base and its defaults."""
    if os.environ.get("HERMES_REPO"):
        sys.path.insert(0, os.environ["HERMES_REPO"])
        import agent.context_engine

        return agent.context_engine
    from abc import ABC, abstractmethod

    module = types.ModuleType("agent.context_engine")

    class ContextEngine(ABC):
        last_prompt_tokens = last_completion_tokens = last_total_tokens = 0
        threshold_tokens = context_length = compression_count = 0
        threshold_percent = 0.75
        protect_first_n, protect_last_n = 3, 6

        @property
        @abstractmethod
        def name(self): ...

        @abstractmethod
        def update_from_response(self, usage): ...

        @abstractmethod
        def should_compress(self, prompt_tokens=None): ...

        @abstractmethod
        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ): ...

        def on_session_start(self, session_id, **kwargs): ...

        def on_session_reset(self):
            self.last_prompt_tokens = self.last_completion_tokens = (
                self.last_total_tokens
            ) = 0

        def select_context(
            self,
            request_messages,
            *,
            conversation_messages=None,
            incoming_message=None,
            budget_tokens=0,
        ):
            return None

        def on_turn_complete(self, messages, usage=None, **kwargs): ...

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, name, args, **kwargs):
            return json.dumps({"error": f"Unknown context engine tool: {name}"})

        def update_model(
            self,
            model,
            context_length,
            base_url="",
            api_key="",
            provider="",
            api_mode="",
        ):
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)

        def clone_for_agent(self):
            raise AssertionError(
                "the engine shares its state between its clones itself"
            )

    module.ContextEngine = ContextEngine
    package = types.ModuleType("agent")
    package.__path__ = []
    sys.modules["agent"], sys.modules["agent.context_engine"] = package, module
    return module


class HermesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = "HERMES_REPO" in os.environ
        cls.kept = {k: sys.modules.get(k) for k in ("agent", "agent.context_engine")}
        cls.base = stand_in()
        cls.policy = fitted()
        cls.started = 0

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.kept.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def engine(self, policy="policy", record=True):
        type(self).started += 1
        self.clock = Clock(1_880_000_000.0 + self.started * 1e7)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.record = tmp.name
        # A policy object keeps what it has seen of a session name, so every test gets its own.
        fresh = compache.Policy(json.loads(json.dumps(self.policy.artifact)))
        self.live = Live(
            fresh if policy == "policy" else None,
            record=tmp.name if record else None,
            clock=self.clock,
        )
        made = hermes.make(self.live)
        made.on_session_start(
            "session-1", hermes_home=".", platform="cli", model="claude-sonnet-5-5"
        )
        made.update_model("claude-sonnet-5-5", 200_000)
        return made

    def host(self, engine, model, questions):
        """The agent loop of a host that lets the engine choose what each request holds."""
        history, seen = [], []
        for n, question in enumerate(questions, 1):
            history.append({"role": "user", "content": question})
            while True:
                request = [{"role": "system", "content": SYSTEM}, *history]
                shown = (
                    engine.select_context(request, conversation_messages=list(history))
                    or request
                )
                seen.append(shown)
                status, reply = model.chat(
                    {"model": "claude-sonnet-5-5", "messages": shown, "tools": [SEARCH]}
                )
                assert status == 200
                engine.update_from_response(
                    {
                        "prompt_tokens": reply["usage"]["prompt_tokens"],
                        "completion_tokens": reply["usage"]["completion_tokens"],
                        "total_tokens": reply["usage"]["total_tokens"],
                        "cache_read_tokens": reply["usage"]["prompt_tokens_details"][
                            "cached_tokens"
                        ],
                        "cache_write_tokens": reply["usage"]["prompt_tokens_details"][
                            "cache_write_tokens"
                        ],
                    }
                )
                message = reply["choices"][0]["message"]
                history.append({k: v for k, v in message.items() if v is not None})
                for call in message.get("tool_calls") or ():
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": passage(n),
                        }
                    )
                if not message.get("tool_calls"):
                    break
            engine.on_turn_complete(list(history), usage=None, turn_id=str(n))
            self.clock.at += 3600
        return history, seen

    def test_archived_parts_leave_the_request_and_stay_in_the_host_s_history(self):
        engine = self.engine()
        model = FakeModel(looping, clock=self.clock)
        history, seen = self.host(
            engine, model, [f"Question {n}?" for n in range(1, 5)]
        )
        self.assertEqual(history[-1]["content"], "Answer 4.")
        # What the host keeps is untouched, whatever the requests held.
        self.assertFalse(noticed(history))
        self.assertEqual([bool(noticed(m)) for m in (seen[0], seen[-1])], [False, True])
        # The reopen instructions join the system prompt with the first notice, never before.
        told = [("reopen" in str(shown[0]["content"])) for shown in seen]
        self.assertEqual(told, [False] * 4 + [True] * (len(seen) - 4))
        # What was billed and how the conversation went is on record, without its text.
        threads, _ = load(self.record)
        self.assertEqual([len(t["runs"]) for t in threads], [4])
        self.assertEqual(
            threads[0]["runs"][0]["retrievals"], [{"bytes": len(passage(1))}]
        )
        calls = load_calls(self.record)
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[-1]["archived"], ["R1.1", "R2.1"])
        self.assertGreater(sum(c["usage"]["input"] for c in calls), 0)

    def test_the_model_fetches_an_archived_part_back(self):
        engine = self.engine()
        model = FakeModel(looping, clock=self.clock)
        self.host(engine, model, [f"Question {n}?" for n in range(1, 5)])
        # What the model is offered for an archived part, and how it is answered.
        schemas = engine.get_tool_schemas()
        self.assertEqual([s["name"] for s in schemas], ["reopen"])
        self.assertIn("reference", schemas[0]["parameters"]["properties"])
        found = json.loads(
            engine.handle_tool_call("reopen", {"reference": "Clause 001"})
        )
        self.assertEqual(found["status"], "ok")
        self.assertIn("Clause 001", found["text"])
        self.assertNotIn("session-1", json.dumps(found))
        # Words that match nothing say what to try.
        missing = json.loads(
            engine.handle_tool_call("reopen", {"reference": "gearbox oil"})
        )
        self.assertEqual(missing["status"], "not_found")
        self.assertIn("distinctive", missing["hint"])
        # What is not the engine's tool is refused in Hermes' own words.
        self.assertIn("error", json.loads(engine.handle_tool_call("other", {})))
        # Nothing to fetch from before any conversation is shown.
        fresh = self.engine()
        answer = json.loads(fresh.handle_tool_call("reopen", {"reference": "x"}))
        self.assertEqual(answer["status"], "error")

    def test_a_turn_that_waits_for_a_tool_has_not_ended(self):
        engine = self.engine()
        waiting = [
            {"role": "user", "content": "Question 1?"},
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_docs", "arguments": "{}"},
                    }
                ],
            },
        ]
        engine.select_context(waiting[:1], conversation_messages=waiting[:1])
        engine.on_turn_complete(waiting)
        self.assertEqual(self.live.stats["turns"], 0)
        # The tool's result comes in after all, and the turn ends with the answer.
        more = [
            *waiting,
            {"role": "tool", "tool_call_id": "c1", "content": passage(1)},
        ]
        engine.select_context(more, conversation_messages=more)
        engine.on_turn_complete([*more, {"role": "assistant", "content": "Answer 1."}])
        self.assertEqual(self.live.stats["turns"], 1)
        threads, _ = load(self.record)
        self.assertEqual([len(t["runs"]) for t in threads], [1])

    def test_nothing_that_goes_wrong_here_reaches_the_agent(self):
        engine = self.engine()
        model = FakeModel(looping, clock=self.clock)
        self.host(engine, model, [f"Question {n}?" for n in range(1, 5)])
        # A model may write anything into a tool call.
        answer = json.loads(
            engine.handle_tool_call(
                "reopen", {"reference": "Clause 001", "offset": "the start"}
            )
        )
        self.assertEqual(answer["status"], "error")

        # A record that cannot be written is not the agent's problem.
        def broken(*args, **kwargs):
            raise OSError("no space left on device")

        self.live.recorder.call = self.live.recorder.turn = broken
        with self.assertLogs("compache.live", "WARNING"):
            self.host(engine, model, ["Question 5?"])

    def test_a_count_that_is_not_reported_is_not_a_zero(self):
        from compache.threads import cache_read_reported

        counts = {"prompt_tokens": 50, "completion_tokens": 5}
        self.assertFalse(cache_read_reported(hermes._usage(counts)))
        self.assertTrue(
            cache_read_reported(hermes._usage({**counts, "cache_read_tokens": 0}))
        )

    def test_a_lifetime_that_hermes_does_not_ask_for_is_said(self):
        asked = {"prompt_caching": {"cache_ttl": "5m"}}
        config = types.ModuleType("hermes_cli.config")
        config.load_config_readonly = lambda: asked
        package = types.ModuleType("hermes_cli")
        package.__path__ = []
        names = ("hermes_cli", "hermes_cli.config")
        kept = {k: sys.modules.get(k) for k in names}

        def restore():
            for name, module in kept.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.addCleanup(restore)
        sys.modules.update(zip(names, (package, config)))
        # The policy counts on five minutes, and so does Hermes: nothing to say.
        with self.assertNoLogs("compache.hermes", "WARNING"):
            self.engine()
        # Hermes' own choice moves to an hour after a first pause.
        asked["prompt_caching"] = {}
        with self.assertLogs("compache.hermes", "WARNING") as said:
            self.engine()
        self.assertIn("prompt_caching.cache_ttl to 5m", said.output[0])
        # Without a policy nothing is counted on.
        with self.assertNoLogs("compache.hermes", "WARNING"):
            self.engine(policy=None)

    def test_with_no_policy_it_only_records(self):
        engine = self.engine(policy=None)
        self.assertEqual(engine.get_tool_schemas(), [])
        model = FakeModel(looping, clock=self.clock)
        _, seen = self.host(engine, model, ["Question 1?", "Question 2?"])
        self.assertFalse(any(noticed(shown) for shown in seen))
        threads, _ = load(self.record)
        self.assertEqual([len(t["runs"]) for t in threads], [2])
        # The reply to a question the host shows again is not a new turn.
        self.assertEqual(self.live.stats["turns"], 2)

    def test_sessions_are_kept_apart(self):
        engine = self.engine()
        model = FakeModel(looping, clock=self.clock)
        self.host(engine, model, ["Question 1?"])
        engine.on_session_start("session-2")
        self.host(engine, model, ["Question 1?"])
        threads, _ = load(self.record)
        self.assertEqual(sorted(len(t["runs"]) for t in threads), [1, 1])

    def test_the_host_s_own_compressor_handles_a_conversation_that_outgrows_the_model(
        self,
    ):
        engine = self.engine()
        seen = {}

        class ContextCompressor:
            def __init__(self, **route):
                seen["route"] = route

            def compress(
                self,
                messages,
                current_tokens=None,
                *,
                focus_topic=None,
                force=False,
                memory_context="",
            ):
                seen["args"] = (
                    len(messages),
                    current_tokens,
                    focus_topic,
                    force,
                    memory_context,
                )
                return messages[-1:]

        module = types.ModuleType("agent.context_compressor")
        module.ContextCompressor = ContextCompressor
        kept = sys.modules.get("agent.context_compressor")
        sys.modules["agent.context_compressor"] = module
        self.addCleanup(
            lambda: (
                sys.modules.pop("agent.context_compressor", None)
                if kept is None
                else sys.modules.__setitem__("agent.context_compressor", kept)
            )
        )
        # Well below the limit: the engine leaves it alone.
        engine.update_from_response(
            {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        )
        self.assertFalse(engine.should_compress())
        self.assertFalse(engine.should_compress(149_999))
        self.assertTrue(engine.should_compress(150_000))
        out = engine.compress(
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            160_000,
            focus_topic="b",
            force=True,
            memory_context="mc",
        )
        self.assertEqual(out, [{"role": "user", "content": "b"}])
        self.assertEqual(seen["args"], (2, 160_000, "b", True, "mc"))
        self.assertEqual(
            (seen["route"]["model"], seen["route"]["config_context_length"]),
            ("claude-sonnet-5-5", 200_000),
        )
        self.assertEqual(engine.compression_count, 1)

    def test_an_engine_without_the_built_in_compressor_changes_nothing(self):
        engine = self.engine()
        messages = [{"role": "user", "content": "a"}]
        kept = sys.modules.get("agent.context_compressor")
        sys.modules["agent.context_compressor"] = None  # not importable
        self.addCleanup(
            lambda: (
                sys.modules.pop("agent.context_compressor", None)
                if kept is None
                else sys.modules.__setitem__("agent.context_compressor", kept)
            )
        )
        self.assertIs(engine.compress(messages, 1), messages)

    def test_the_registered_engine_reads_its_settings_from_the_environment(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            policy = os.path.join(tmp, "policy.json")
            self.policy.save(policy)
            with mock.patch.dict(
                os.environ,
                {
                    "COMPACHE_POLICY": policy,
                    "COMPACHE_RECORD": os.path.join(tmp, "record"),
                },
            ):
                seen = []

                class Ctx:
                    def register_context_engine(self, engine):
                        seen.append(engine)

                hermes.register(Ctx())
                (engine,) = seen
                self.assertEqual(engine.name, "compache")
                self.assertIsInstance(engine, self.base.ContextEngine)
                self.assertIsNotNone(engine.live.policy)
                self.assertTrue(os.path.isdir(os.path.join(tmp, "record")))
                clone = engine.clone_for_agent()
                self.assertIs(clone.live, engine.live)


if __name__ == "__main__":
    unittest.main()


class HermesAgentPolicyTests(unittest.TestCase):
    def test_an_agent_policy_offers_the_tool_in_an_agent_s_words(self):
        stand_in()
        from compache.live import _AGENT_TOOL, _REOPEN_TOOL
        from test_agent_live import policy

        agentic = hermes.make(Live(policy(rounds=2))).get_tool_schemas()
        chat = hermes.make(Live(fitted())).get_tool_schemas()
        self.assertEqual(agentic[0]["description"], _AGENT_TOOL["function"]["description"])
        self.assertEqual(chat[0]["description"], _REOPEN_TOOL["function"]["description"])
        self.assertEqual(agentic[0]["name"], "reopen")
