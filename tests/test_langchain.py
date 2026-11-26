"""A LangChain agent on any chat model, with compache as its middleware."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace

import compache
from compache.record import load
from test_agent_live import SPEC, policy, result

HERE = all(
    importlib.util.find_spec(name)
    for name in ("langchain", "langchain_core", "langgraph")
)


@unittest.skipUnless(HERE, "needs LangChain")
class MiddlewareTests(unittest.TestCase):
    def agent(self, live, files=8, fetch=None, thread="t-1"):
        from langchain.agents import create_agent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        seen: list = []

        class Scripted(BaseChatModel):
            """Reads files one by one, and may look back once."""

            @property
            def _llm_type(self):
                return "scripted"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                seen.append([m.content for m in messages])
                tools = [m for m in messages if m.type == "tool"]
                read = sum(m.name == "read" for m in tools)
                looked = any(m.name == "reopen" for m in tools)
                usage = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
                said = {"content": "All done.", "usage_metadata": usage}
                if fetch and read == fetch and not looked:
                    notice = next(m.content for m in tools if "ctx:" in m.content)
                    handle = "ctx:" + notice.split("ctx:")[1][:24]
                    call = {"name": "reopen", "args": {"reference": handle}}
                    said = {"content": "", "tool_calls": [{**call, "id": "back"}]}
                elif read < files:
                    call = {"name": "read", "args": {"file": read + 1}}
                    said = {"content": "", "tool_calls": [{**call, "id": f"c{read}"}]}
                message = AIMessage(**said)
                return ChatResult(generations=[ChatGeneration(message=message)])

        @tool
        def read(file: int) -> str:
            """Read a file."""
            return result(file)

        made = create_agent(
            Scripted(),
            [read],
            middleware=[compache.langchain.middleware(live)],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 200}
        return made, config, seen

    def ask(self, agent, config, text="Read eight files"):
        return agent.invoke({"messages": [{"role": "user", "content": text}]}, config)

    def test_older_results_reach_the_model_as_notices_and_the_graph_keeps_its_own(self):
        live = compache.Live(policy(rounds=2))
        agent, config, seen = self.agent(live)
        state = self.ask(agent, config)
        self.assertEqual(state["messages"][-1].content, "All done.")
        gone = [
            "ctx:" in text
            for text in seen[-1]
            if "file number" in text or "ctx:" in text
        ]
        self.assertEqual(gone, [True] * 6 + [False] * 2)
        kept = [m.content for m in state["messages"] if m.type == "tool"]
        self.assertFalse(any("ctx:" in text for text in kept))
        self.assertEqual(live.stats["turns"], 1)
        self.assertEqual(live.stats["stood_down"], 0)

    def test_the_agent_fetches_back_with_a_tool_of_its_own(self):
        live = compache.Live(policy(rounds=2))
        agent, config, seen = self.agent(live, fetch=5)
        state = self.ask(agent, config)
        back = [
            m.content
            for m in state["messages"]
            if m.type == "tool" and m.name == "reopen"
        ]
        self.assertEqual(len(back), 1)
        self.assertIn("a line of file number 1 that only it holds", back[0])
        self.assertEqual(live.stats["reopened"], 1)
        # What it fetched is a result like any other, and goes when it is old.
        self.assertFalse(any("file number 1 that" in text for text in seen[-1]))

    def test_the_next_question_of_a_thread_goes_on(self):
        live = compache.Live(policy(rounds=2))
        agent, config, seen = self.agent(live, files=4)
        self.ask(agent, config)
        self.ask(agent, config, "And now?")
        self.assertEqual(live.stats["turns"], 2)
        gone = [
            "ctx:" in text
            for text in seen[-1]
            if "file number" in text or "ctx:" in text
        ]
        self.assertEqual(gone, [True, True, False, False])

    def test_reasoning_behind_a_cut_goes_with_it(self):
        """Claude takes its reasoning back only behind what stood ahead of it when it was made."""
        import hashlib
        import json

        from langchain.agents import create_agent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        kept: list = []
        # LangChain's own form of what Claude reasoned, and Claude's.
        form: list = []

        def signed(block):
            if block["type"] == "non_standard":
                return signed(block["value"])
            if block["type"] == "thinking":
                return block["signature"]
            return (block.get("extras") or {}).get("signature")

        def behind(messages):
            said = [[m.type, m.content, getattr(m, "tool_calls", [])] for m in messages]
            said = json.dumps(said, sort_keys=True, default=str).encode()
            return hashlib.sha256(said).hexdigest()[:16]

        class Claude(BaseChatModel):
            model: str = "claude-haiku-5-5"

            @property
            def _llm_type(self):
                return "scripted"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                held = 0
                for k, message in enumerate(messages):
                    blocks = (
                        message.content if isinstance(message.content, list) else ()
                    )
                    for block in filter(signed, blocks):
                        held += 1
                        assert signed(block) == behind(messages[:k])
                kept.append(held)
                read = sum(m.type == "tool" for m in messages)
                sign = form[0](behind(messages))
                usage = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
                said = {"content": [sign, {"type": "text", "text": "Step."}]}
                if read < 9:
                    call = {
                        "name": "read",
                        "args": {"file": read + 1},
                        "id": f"c{read}",
                    }
                    said["tool_calls"] = [call]
                message = AIMessage(**said, usage_metadata=usage)
                return ChatResult(generations=[ChatGeneration(message=message)])

        @tool
        def read(file: int) -> str:
            """Read a file."""
            return result(file)

        def thinking(signature):
            return {"type": "thinking", "thinking": "", "signature": signature}

        for made in (
            thinking,
            lambda signature: {
                "type": "reasoning",
                "reasoning": "",
                "extras": {"signature": signature},
            },
            lambda signature: {"type": "non_standard", "value": thinking(signature)},
        ):
            form[:], kept[:] = [made], []
            live = compache.Live(policy(rounds=5, ratio=1.0))
            agent = create_agent(
                Claude(),
                [read],
                middleware=[compache.langchain.middleware(live)],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "t-9"}, "recursion_limit": 200}
            with self.subTest(made("...")):
                self.ask(agent, config)
                self.assertEqual(kept[:6], [0, 1, 2, 3, 4, 5])
                self.assertLess(min(kept[6:]), 5, "a cut took it along")
                self.assertGreater(kept[-1], 0, "what came after the cut stays")
                self.assertEqual(live.stats["stood_down"], 0)

    def test_what_a_reply_says_beside_what_it_reasons(self):
        from compache.langchain import _said

        thinking = {"type": "thinking", "thinking": "Hm.", "signature": "s"}
        call = {"type": "tool_use", "id": "c", "name": "read", "input": {}}
        text = {"type": "text", "text": "It is in clause 4."}
        for content in (
            "It is in clause 4.",
            [text],
            [thinking, text, call],
            [{"type": "reasoning", "reasoning": "Hm."}, "It is in clause 4."],
        ):
            self.assertEqual(_said(content), "It is in clause 4.")
        self.assertEqual(_said([call]), "")

    def test_async_and_learning(self):
        import asyncio

        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(learn=folder)
            agent, config, _ = self.agent(live, files=3)
            question = {"messages": [{"role": "user", "content": "Read three"}]}
            state = asyncio.run(agent.ainvoke(question, config))
            self.assertEqual(state["messages"][-1].content, "All done.")
            threads, _ = load(folder)
            self.assertEqual(threads[0]["runs"][0]["calls"], 4)
            self.assertEqual(len(threads[0]["runs"][0]["steps"]), 4)

    def test_without_a_thread_nothing_is_changed(self):
        live = compache.Live(policy(rounds=1))
        agent, _, seen = self.agent(live)
        agent.checkpointer = None
        question = {"messages": [{"role": "user", "content": "Read"}]}
        state = agent.invoke(question, {"recursion_limit": 200})
        self.assertEqual(state["messages"][-1].content, "All done.")
        self.assertFalse(any("ctx:" in text for text in seen[-1]))


if __name__ == "__main__":
    unittest.main()


CLAUDE = HERE and importlib.util.find_spec("langchain_anthropic") is not None


@unittest.skipUnless(CLAUDE, "needs LangChain's Anthropic models")
class ClaudeMarkTests(unittest.TestCase):
    """Nothing else in a LangChain agent marks Claude's requests for the cache: the middleware does."""

    def run_agent(self, live, more=(), reads=6, before=()):
        import json as json_
        import time

        from fakes import FakeModel, serve
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        def script(request):
            made = sum(
                1
                for m in request["body"]["messages"]
                if isinstance(m["content"], list)
                for b in m["content"]
                if b.get("type") == "tool_result"
            )
            if made < reads:
                return {"text": "", "tool": ("read", {"file": made + 1})}
            return {"text": "All read."}

        @tool
        def read(file: int) -> str:
            """Read a file."""
            return result(file)

        model = FakeModel(script, clock=time.time)
        url, stop = serve(model)
        self.addCleanup(stop)
        middleware = [*before, compache.langchain.middleware(live), *more]
        agent = create_agent(
            ChatAnthropic(model="claude-sonnet-5-5", api_key="k", base_url=url, max_tokens=64),
            [read],
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        agent.invoke(
            {"messages": [{"role": "user", "content": "Read them"}]},
            {"configurable": {"thread_id": "t-1"}, "recursion_limit": 100},
        )
        return [json_.loads(json_.dumps(b)) for b in model.bodies]

    @staticmethod
    def marks(body):
        found = []

        def walk(value, at):
            if isinstance(value, dict):
                if "cache_control" in value:
                    found.append((at, value["cache_control"]))
                for key, inner in value.items():
                    walk(inner, f"{at}.{key}" if at else key)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner, at)

        walk(body, "")
        return found

    def test_every_request_is_marked_for_five_minutes(self):
        bodies = self.run_agent(compache.Live(policy(rounds=2)))
        self.assertGreater(len(bodies), 5)
        for body in bodies:
            self.assertEqual(body.get("cache_control"), {"type": "ephemeral", "ttl": "5m"})
            self.assertLessEqual(len(self.marks(body)), 4)

    def test_a_policy_fitted_for_an_hour_asks_for_an_hour(self):
        hour = replace(SPEC, cache_ttl_seconds=3600, write_multiplier=2.0)
        made = policy(rounds=2, spec=hour)
        made = compache.Policy({**made.artifact, "standard_spec": SPEC.to_dict()})
        self.assertEqual(made.cache_lifetime, 3600)
        for body in self.run_agent(compache.Live(made)):
            self.assertEqual(body.get("cache_control"), {"type": "ephemeral", "ttl": "1h"})

    def test_marks_of_the_agent_s_own_are_left_as_they_are(self):
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

        for first in (True, False):
            with self.subTest(compache_first=first):
                caching = [AnthropicPromptCachingMiddleware()]
                live = compache.Live(policy(rounds=2))
                if first:
                    bodies = self.run_agent(live, more=caching)
                else:
                    bodies = self.run_agent(live, before=caching)
                for body in bodies:
                    top = [m for at, m in self.marks(body) if at == ""]
                    self.assertEqual(top, [{"type": "ephemeral", "ttl": "5m"}])
                    self.assertLessEqual(len(self.marks(body)), 4)

    def test_without_a_policy_nothing_is_marked(self):
        for body in self.run_agent(compache.Live()):
            self.assertEqual(self.marks(body), [])
