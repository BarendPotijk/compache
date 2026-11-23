"""Small chat applications, each written the way its stack's own documentation shows.

None of them knows compache. It goes in from outside: by wrapping the client
an application is given, or by giving it a client whose base URL is the proxy.

    tool-loop      OpenAI SDK, Chat Completions, a search tool the model calls
    stuffed        OpenAI SDK, Chat Completions, the passages pasted into the question
    claude-loop    Anthropic SDK, Messages API, the tool loop of Anthropic's guide
    responses      OpenAI SDK, Responses API, a search tool the model calls
    agent          LangChain `create_agent` on `ChatOpenAI`, with a checkpointer

`turn(question, passages)` plays one user message and returns the answer. The
passages are what the application's search finds for it.
"""

from __future__ import annotations

import json

SEARCH = {
    "name": "search_docs",
    "description": "Search the document and return the most relevant passages. Call it for every question.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
INSTRUCTIONS = (
    " Search the document for every question before you answer. Answer only from what the search "
    "returns and from the conversation so far. If that does not hold the answer, say that you do "
    "not have it. Be brief: a few sentences."
)
NOTHING = "No further results."


class App:
    """What every application here shares: its client, its model and how many tokens an answer may take."""

    name = ""

    def __init__(
        self,
        client,
        model: str,
        system: str,
        *,
        limit: int = 1500,
        extra: dict | None = None,
    ):
        self.client, self.model, self.limit = client, model, limit
        self.system = system + INSTRUCTIONS
        # Sent with every request: how a gateway is told which provider to use.
        self.extra = extra or {}
        self.calls = 0

    def turn(self, question: str, passages: list[str]) -> str:
        raise NotImplementedError


class ToolLoop(App):
    """The manual tool loop of OpenAI's guide: every model call sends the growing message list."""

    name = "tool-loop"
    stream = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{"role": "system", "content": self.system}]

    def _call(self) -> dict:
        self.calls += 1
        request = dict(
            model=self.model,
            messages=self.messages,
            tools=[{"type": "function", "function": SEARCH}],
            max_tokens=self.limit,
            extra_body=self.extra,
        )
        if not self.stream:
            reply = self.client.chat.completions.create(**request).choices[0].message
            return reply.model_dump(exclude_none=True)
        text, calls = "", {}
        for chunk in self.client.chat.completions.create(stream=True, **request):
            for choice in chunk.choices:
                text += choice.delta.content or ""
                for call in choice.delta.tool_calls or ():
                    entry = calls.setdefault(
                        call.index, {"id": None, "name": None, "arguments": ""}
                    )
                    entry["id"] = call.id or entry["id"]
                    if call.function is not None:
                        entry["name"] = call.function.name or entry["name"]
                        entry["arguments"] += call.function.arguments or ""
        message = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for c in calls.values()
            ]
        return message

    def turn(self, question, passages):
        self.messages.append({"role": "user", "content": question})
        searched = False
        for _ in range(4):
            reply = self._call()
            calls = reply.get("tool_calls") or []
            if not calls:
                self.messages.append(
                    {"role": "assistant", "content": reply.get("content") or ""}
                )
                return reply.get("content") or ""
            self.messages.append(
                {
                    "role": "assistant",
                    "content": reply.get("content"),
                    "tool_calls": calls,
                }
            )
            for call in calls:
                # The search answers once a turn, one piece of text per passage.
                found = (
                    [{"type": "text", "text": p} for p in passages]
                    if passages and not searched
                    else NOTHING
                )
                searched = True
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": found}
                )
        self.messages.append({"role": "assistant", "content": ""})
        return ""


class StreamedToolLoop(ToolLoop):
    name = "tool-loop-streamed"
    stream = True


class Stuffed(App):
    """Retrieve, paste the passages into the prompt, and keep the prompts as the history."""

    name = "stuffed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{"role": "system", "content": self.system}]

    def turn(self, question, passages):
        self.calls += 1
        context = "\n\n".join(passages)
        self.messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        )
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.limit,
            extra_body=self.extra,
        )
        text = reply.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": text})
        return text


class ClaudeLoop(App):
    """The manual tool loop of Anthropic's guide, with the SDK's own blocks kept as the history."""

    name = "claude-loop"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []
        self.tools = [
            {
                "name": SEARCH["name"],
                "description": SEARCH["description"],
                "input_schema": SEARCH["parameters"],
            }
        ]

    def turn(self, question, passages):
        self.messages.append({"role": "user", "content": question})
        searched = False
        for _ in range(4):
            self.calls += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.limit,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                extra_body=self.extra,
            )
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    found = (
                        [{"type": "text", "text": p} for p in passages]
                        if passages and not searched
                        else NOTHING
                    )
                    searched = True
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": found,
                        }
                    )
            self.messages.append({"role": "user", "content": results})
        return "".join(b.text for b in response.content if b.type == "text")


class Responses(App):
    """The Responses API with the conversation sent in full, as a stateless application does."""

    name = "responses"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = []

    def turn(self, question, passages):
        self.items.append({"role": "user", "content": question})
        searched = False
        for _ in range(4):
            self.calls += 1
            response = self.client.responses.create(
                model=self.model,
                instructions=self.system,
                input=self.items,
                tools=[{"type": "function", **SEARCH}],
                max_output_tokens=self.limit,
                extra_body=self.extra,
            )
            self.items += [
                item.model_dump(exclude_none=True) for item in response.output
            ]
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            for call in calls:
                found = "\n\n".join(passages) if passages and not searched else NOTHING
                searched = True
                self.items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": found,
                    }
                )
        return response.output_text


class Agent(App):
    """A LangChain agent. The thread's history lives in the graph's checkpointer."""

    name = "agent"

    def __init__(self, chat, model, system, *, thread_id: str, middleware=(), **kwargs):
        super().__init__(chat, model, system, **kwargs)
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        self.found: list[str] = []
        self.searched = False

        @tool
        def search_docs(query: str) -> str:
            """Search the document and return the most relevant passages. Call it for every question."""
            found = (
                "\n\n".join(self.found) if self.found and not self.searched else NOTHING
            )
            self.searched = True
            return found

        self.agent = create_agent(
            chat,
            [search_docs],
            system_prompt=self.system,
            middleware=list(middleware),
            checkpointer=InMemorySaver(),
        )
        self.config = {"configurable": {"thread_id": thread_id}}

    def turn(self, question, passages):
        self.found, self.searched = passages, False
        state = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, self.config
        )
        self.calls += 1
        content = state["messages"][-1].content
        return content if isinstance(content, str) else json.dumps(content)


APPS = {
    app.name: app
    for app in (ToolLoop, StreamedToolLoop, Stuffed, ClaudeLoop, Responses, Agent)
}
