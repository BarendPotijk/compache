"""compache inside the provider SDKs, run against a local stand-in for the provider.

These tests need the SDKs themselves and are skipped where they are not installed:
`pip install openai anthropic google-genai langchain langchain-openai langgraph`.
"""

import asyncio
import importlib.util
import json
import tempfile
import unittest

import compache
from compache.live import Live
from compache.record import load
from compache.wrap import thread, wrap
from fakes import NOTE_ASKED, FakeModel, noticed, serve
from test_live import (
    CLAUDE_SEARCH,
    SEARCH,
    SYSTEM,
    Clock,
    claude_looping,
    fitted,
    looping,
    passage,
)


def has(*modules) -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except ModuleNotFoundError:
        return False


class SdkCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = fitted()
        cls.started = 0

    def start(self, script, **options):
        """A stand-in provider, a `Live` on a clock of its own, and where its record goes."""
        type(self).started += 1
        self.model = FakeModel(script)
        self.url, stop = serve(self.model)
        self.addCleanup(stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.record = tmp.name
        self.clock = Clock(1_850_000_000.0 + self.started * 1e7)
        options.setdefault("sample", 1.0)
        self.live = Live(
            options.pop("policy", self.policy),
            record=tmp.name,
            clock=self.clock,
            **options,
        )
        return self.live

    def check(self, said, turns=4):
        """The policy was applied upstream, notes were taken, and nothing of either reached the application."""
        # A streamed reply is asked for its note once it is in.
        self.live.drain()
        sent = [json.dumps(b) for b in self.model.bodies]
        self.assertGreaterEqual(sum(map(noticed, sent)), turns - 1)
        self.assertTrue(any(NOTE_ASKED in s for s in sent))
        self.assertNotIn("refs", json.dumps(said, default=str))
        self.assertFalse(noticed(said))
        threads, labels = load(self.record)
        self.assertEqual([len(t["runs"]) for t in threads], [turns])
        self.assertEqual([row["run"] for row in labels], list(range(1, turns + 1)))
        self.assertEqual(self.live.stats["noted"], turns - 1)
        headers = {k.lower(): v for k, v in self.model.headers[-1].items()}
        self.assertNotIn("x-compache-thread", headers)
        return headers


@unittest.skipUnless(has("openai"), "needs the OpenAI SDK")
class OpenAITests(SdkCase):
    def turn(self, client, messages, n, stream):
        messages.append({"role": "user", "content": f"Question {n}?"})
        while True:
            if not stream:
                reply = (
                    client.chat.completions.create(
                        model="claude-sonnet-5-5", messages=messages, tools=[SEARCH]
                    )
                    .choices[0]
                    .message
                )
                text, calls = reply.content, reply.tool_calls or []
                # Many applications append the SDK's own object.
                messages.append(
                    reply if calls else {"role": "assistant", "content": text}
                )
                found = [c.id for c in calls]
            else:
                text, calls = "", {}
                for chunk in client.chat.completions.create(
                    model="claude-sonnet-5-5",
                    messages=messages,
                    tools=[SEARCH],
                    stream=True,
                ):
                    for choice in chunk.choices:
                        text += choice.delta.content or ""
                        for call in choice.delta.tool_calls or ():
                            entry = calls.setdefault(
                                call.index, {"id": None, "name": None, "arguments": ""}
                            )
                            entry["id"] = call.id or entry["id"]
                            entry["name"] = call.function.name or entry["name"]
                            entry["arguments"] += call.function.arguments or ""
                message = {"role": "assistant", "content": text or None}
                if calls:
                    message["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"],
                            },
                        }
                        for c in calls.values()
                    ]
                messages.append(message)
                found = [c["id"] for c in calls.values()]
            if not found:
                return text
            for call_id in found:
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": passage(n)}
                )

    def test_chat_completions_whole_and_streamed(self):
        import openai

        for stream in (False, True):
            with self.subTest(stream=stream):
                live = self.start(looping)
                client = wrap(
                    openai.OpenAI(
                        api_key="key", base_url=self.url + "/v1", max_retries=0
                    ),
                    live=live,
                )
                self.assertIs(client.compache, live)
                messages = [{"role": "system", "content": SYSTEM}]
                answers = []
                with thread(f"openai-{stream}"):
                    for n in range(1, 5):
                        answers.append(self.turn(client, messages, n, stream))
                        self.clock.at += 3600
                self.assertEqual(answers, [f"Answer {n}." for n in range(1, 5)])
                headers = self.check(messages)
                self.assertEqual(headers["authorization"], "Bearer key")

    def test_a_parsed_reply_validates_as_the_application_s_own_model(self):
        import openai
        from pydantic import BaseModel, ConfigDict

        class Answer(BaseModel):
            model_config = ConfigDict(extra="forbid")
            answer: str
            sure: bool

        def structured(request):
            schema = request["body"]["response_format"]["json_schema"]["schema"]
            out = {"answer": f"Answer {request['n']}.", "sure": True}
            if "_refs" in schema["properties"]:
                out["_refs"] = ["Q1"] if "property" in request["last"] else []
            return {"text": json.dumps(out)}

        live = self.start(structured, policy=None, notes="inline")
        client = wrap(
            openai.OpenAI(api_key="key", base_url=self.url + "/v1", max_retries=0),
            live=live,
        )
        messages = [{"role": "system", "content": SYSTEM}]
        with thread("parse"):
            for n in range(1, 4):
                messages.append({"role": "user", "content": f"Question {n}?"})
                reply = (
                    client.chat.completions.parse(
                        model="gpt-6-luna", messages=messages, response_format=Answer
                    )
                    .choices[0]
                    .message
                )
                self.assertEqual(reply.parsed, Answer(answer=f"Answer {n}.", sure=True))
                messages.append({"role": "assistant", "content": reply.content})
        self.assertNotIn("_refs", json.dumps(messages))
        self.assertEqual(live.stats["noted"], 2)
        name = load(self.record)[0][0]["thread_id"]
        self.assertEqual(load(self.record)[1][1]["needs"], [[f"{name}:1:input"]])

    def test_the_responses_api(self):
        import openai

        def script(request):
            items = request["body"]["input"]
            n = sum(
                i.get("role") == "user" and NOTE_ASKED not in json.dumps(i)
                for i in items
            )
            results = [i for i in items[-2:] if i.get("type") == "function_call_output"]
            if results:
                return {"text": f"Answer {n}.", "refs": "{R1.1}" if n > 1 else "none"}
            if (
                NOTE_ASKED in json.dumps(items[-1])
                and items[-2].get("role") == "assistant"
            ):
                # A note asked of a finished answer, in a request of its own.
                return {"text": "", "refs": "{R1.1}" if n > 1 else "none"}
            return {"text": "", "tool": ("search_docs", {"query": f"q{n}"})}

        tool = {
            "type": "function",
            "name": "search_docs",
            "description": "Search.",
            "parameters": {"type": "object"},
        }
        live = self.start(script)
        client = wrap(
            openai.OpenAI(api_key="key", base_url=self.url + "/v1", max_retries=0),
            live=live,
        )
        items = []
        with thread("responses"):
            for n in range(1, 5):
                items.append({"role": "user", "content": f"Question {n}?"})
                while True:
                    response = client.responses.create(
                        model="claude-sonnet-5-5",
                        instructions=SYSTEM,
                        input=items,
                        tools=[tool],
                    )
                    items += [i.model_dump(exclude_none=True) for i in response.output]
                    calls = [i for i in response.output if i.type == "function_call"]
                    if not calls:
                        break
                    items += [
                        {
                            "type": "function_call_output",
                            "call_id": c.call_id,
                            "output": passage(n),
                        }
                        for c in calls
                    ]
                self.assertEqual(response.output_text, f"Answer {n}.")
                self.clock.at += 3600
        self.check(items)

    def test_an_async_client_streams_through_it(self):
        import openai

        live = self.start(looping)

        async def run():
            client = wrap(
                openai.AsyncOpenAI(
                    api_key="key", base_url=self.url + "/v1", max_retries=0
                ),
                live=live,
            )
            messages = [{"role": "system", "content": SYSTEM}]
            with thread("async"):
                for n in range(1, 5):
                    messages.append({"role": "user", "content": f"Question {n}?"})
                    while True:
                        text, calls = "", {}
                        stream = await client.chat.completions.create(
                            model="claude-sonnet-5-5",
                            messages=messages,
                            tools=[SEARCH],
                            stream=True,
                        )
                        async for chunk in stream:
                            for choice in chunk.choices:
                                text += choice.delta.content or ""
                                for call in choice.delta.tool_calls or ():
                                    entry = calls.setdefault(
                                        call.index,
                                        {"id": None, "name": None, "arguments": ""},
                                    )
                                    entry["id"] = call.id or entry["id"]
                                    entry["name"] = call.function.name or entry["name"]
                                    entry["arguments"] += call.function.arguments or ""
                        message = {"role": "assistant", "content": text or None}
                        if calls:
                            message["tool_calls"] = [
                                {
                                    "id": c["id"],
                                    "type": "function",
                                    "function": {
                                        "name": c["name"],
                                        "arguments": c["arguments"],
                                    },
                                }
                                for c in calls.values()
                            ]
                        messages.append(message)
                        if not calls:
                            break
                        for c in calls.values():
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": c["id"],
                                    "content": passage(n),
                                }
                            )
                    self.clock.at += 3600
            # The notes of streamed replies are asked through the same client: wait for them.
            await asyncio.get_running_loop().run_in_executor(None, live.drain)
            await client.close()
            return messages

        messages = asyncio.run(run())
        self.assertEqual(messages[-1]["content"], "Answer 4.")
        self.check(messages)


@unittest.skipUnless(has("anthropic"), "needs the Anthropic SDK")
class AnthropicTests(SdkCase):
    def test_messages_whole_and_through_the_stream_helper(self):
        import anthropic

        for helper in (False, True):
            with self.subTest(stream=helper):
                live = self.start(claude_looping)
                client = wrap(
                    anthropic.Anthropic(
                        api_key="key", base_url=self.url, max_retries=0
                    ),
                    live=live,
                )
                messages, shown = [], ""
                with thread(f"claude-{helper}"):
                    for n in range(1, 5):
                        messages.append({"role": "user", "content": f"Question {n}?"})
                        while True:
                            request = dict(
                                model="claude-sonnet-5-5",
                                max_tokens=512,
                                system=SYSTEM,
                                tools=[CLAUDE_SEARCH],
                                messages=messages,
                            )
                            if helper:
                                with client.messages.stream(**request) as stream:
                                    shown += "".join(stream.text_stream)
                                    response = stream.get_final_message()
                            else:
                                response = client.messages.create(**request)
                            messages.append(
                                {"role": "assistant", "content": response.content}
                            )
                            if response.stop_reason != "tool_use":
                                break
                            results = [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": b.id,
                                    "content": passage(n),
                                }
                                for b in response.content
                                if b.type == "tool_use"
                            ]
                            messages.append({"role": "user", "content": results})
                        self.assertEqual(response.content[-1].text, f"Answer {n}.")
                        self.clock.at += 3600
                self.assertNotIn("refs", shown)
                said = [
                    m
                    if isinstance(m["content"], (str, list))
                    and not hasattr(m["content"][0], "model_dump")
                    else [b.model_dump() for b in m["content"]]
                    for m in messages
                ]
                self.check(said)


@unittest.skipUnless(has("google.genai"), "needs the Google GenAI SDK")
class GeminiTests(SdkCase):
    def test_generate_content_whole_and_streamed(self):
        from google import genai
        from google.genai import types

        def script(request):
            contents = request["body"]["contents"]
            n = sum(
                c.get("role") == "user" and "text" in c["parts"][0] for c in contents
            )
            if any("functionResponse" in p for p in contents[-1]["parts"]):
                return {"text": f"Answer {n}.", "refs": "{R1.1}" if n > 1 else "none"}
            asked = NOTE_ASKED in str(contents[-1]["parts"][0].get("text"))
            if asked and contents[-2].get("role") == "model":
                # A note asked of a finished answer, in a request of its own: it is not a turn.
                return {"text": "", "refs": "{R1.1}" if n > 2 else "none"}
            return {"text": "", "tool": ("search_docs", {"query": f"q{n}"})}

        declared = types.FunctionDeclaration(
            name="search_docs",
            description="Search.",
            parameters={"type": "OBJECT", "properties": {}},
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=[declared])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                live = self.start(script)
                client = wrap(
                    genai.Client(
                        api_key="key", http_options=types.HttpOptions(base_url=self.url)
                    ),
                    live=live,
                )
                contents = []
                with thread(f"gemini-{stream}"):
                    for n in range(1, 5):
                        contents.append(
                            types.Content(
                                role="user", parts=[types.Part(text=f"Question {n}?")]
                            )
                        )
                        while True:
                            if stream:
                                parts = []
                                for chunk in client.models.generate_content_stream(
                                    model="gemini-2.5-flash-lite",
                                    contents=contents,
                                    config=config,
                                ):
                                    parts += chunk.candidates[0].content.parts or []
                                text = "".join(p.text or "" for p in parts)
                                calls = [p for p in parts if p.function_call]
                                said = types.Content(
                                    role="model",
                                    parts=([types.Part(text=text)] if text else [])
                                    + calls,
                                )
                            else:
                                said = (
                                    client.models.generate_content(
                                        model="gemini-2.5-flash-lite",
                                        contents=contents,
                                        config=config,
                                    )
                                    .candidates[0]
                                    .content
                                )
                                text = "".join(p.text or "" for p in said.parts)
                                calls = [p for p in said.parts if p.function_call]
                            contents.append(said)
                            if not calls:
                                break
                            answers = [
                                types.Part.from_function_response(
                                    name=p.function_call.name,
                                    response={"result": passage(n)},
                                )
                                for p in calls
                            ]
                            contents.append(types.Content(role="user", parts=answers))
                        self.assertEqual(text, f"Answer {n}.")
                        self.clock.at += 3600
                self.check([c.model_dump(exclude_none=True) for c in contents])


@unittest.skipUnless(
    has("langchain", "langchain_openai", "langgraph"), "needs LangChain"
)
class LangChainTests(SdkCase):
    def test_an_agent_is_named_by_its_thread_and_served_by_the_same_client(self):
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import InMemorySaver

        turn = {"n": 0}

        @tool
        def search_docs(query: str) -> str:
            """Search the documents."""
            return passage(turn["n"])

        for mode in ("invoke", "stream"):
            with self.subTest(mode=mode):
                live = self.start(looping)
                chat = ChatOpenAI(
                    model="claude-sonnet-5-5",
                    api_key="key",
                    base_url=self.url + "/v1",
                    max_retries=0,
                    stream_usage=True,
                    **compache.langchain.clients(live),
                )
                agent = create_agent(
                    chat,
                    [search_docs],
                    system_prompt=SYSTEM,
                    middleware=[compache.langchain.middleware()],
                    checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": f"agent-{mode}"}}
                answers = []
                for n in range(1, 5):
                    turn["n"] = n
                    asked = {
                        "messages": [{"role": "user", "content": f"Question {n}?"}]
                    }
                    if mode == "invoke":
                        answers.append(
                            agent.invoke(asked, config)["messages"][-1].content
                        )
                    else:
                        text = ""
                        for chunk, _ in agent.stream(
                            asked, config, stream_mode="messages"
                        ):
                            if chunk.type == "AIMessageChunk" and isinstance(
                                chunk.content, str
                            ):
                                text += chunk.content
                        answers.append(text)
                    self.clock.at += 3600
                self.assertEqual(answers, [f"Answer {n}." for n in range(1, 5)])
                self.check(answers)
                # The graph's thread ID named the conversation: one thread, under that name's hash.
                state = agent.get_state(config).values["messages"]
                self.assertNotIn("refs", json.dumps([m.content for m in state]))


if __name__ == "__main__":
    unittest.main()
