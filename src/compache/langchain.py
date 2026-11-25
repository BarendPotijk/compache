"""compache for LangChain and LangGraph agents.

Any chat model, through a middleware of the agent:

    live = compache.Live(learn="compache-record")
    agent = create_agent(model, tools, middleware=[compache.langchain.middleware(live)], checkpointer=saver)
    agent.invoke({"messages": [...]}, {"configurable": {"thread_id": "conversation-17"}})

What the graph keeps is left as it is. Archived results go to the model as a notice, and the
agent gets a `reopen` tool for them, which runs like any other. The thread ID names the
conversation. compache makes no requests of its own here, so it sends no keep-warm pings.

A chat model that takes an HTTP client, as `ChatOpenAI` does, can have compache inside
instead. Fetching back is then unseen by the graph, and pings go out:

    model = ChatOpenAI(model="gpt-6-luna", **compache.langchain.clients(live))
    agent = create_agent(model, tools, middleware=[compache.langchain.middleware()])
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .live import _AGENT_TOOL, _REOPEN, Live
from .wrap import async_http_client, http_client, thread

__all__ = ["clients", "middleware"]

logger = logging.getLogger(__name__)

_PATH = "/v1/chat/completions"


def clients(live: Live, *, pings: bool = True) -> dict:
    """The `http_client` and `http_async_client` arguments of `ChatOpenAI`, with `live` in the path.

    `pings` lets a background thread send the policy's keep-warm pings.
    """
    return {
        "http_client": http_client(live, pings=pings),
        "http_async_client": async_http_client(live, pings=pings),
    }


def _thread_id(request=None) -> str | None:
    """The thread ID a model call runs under, from the agent's runtime or the graph's config."""
    info = getattr(getattr(request, "runtime", None), "execution_info", None)
    found = getattr(info, "thread_id", None)
    if found is None:
        try:
            from langgraph.config import get_config

            found = (get_config().get("configurable") or {}).get("thread_id")
        except Exception:
            found = None
    return None if found is None else str(found)


def _body(request) -> tuple[dict, list]:
    """A model request as the JSON of a chat completion, and which message each of its messages came from."""
    from langchain_core.messages import convert_to_openai_messages
    from langchain_core.utils.function_calling import convert_to_openai_tool

    said, owners = [], []
    if getattr(request, "system_message", None) is not None:
        said += convert_to_openai_messages([request.system_message])
        owners += [None] * len(said)
    for k, message in enumerate(request.messages):
        made = convert_to_openai_messages([message])
        thoughts = _thoughts(message)
        if thoughts and made:
            content = made[0].get("content")
            if isinstance(content, list):
                content = [p for p in content if not _thinks(p)] or None
            held = {"content": content, "thinking_blocks": thoughts}
            made = [{**made[0], **held}, *made[1:]]
        said += made
        owners += [k] * len(made)
    tools = []
    for tool in request.tools or ():
        try:
            tools.append(
                tool if isinstance(tool, dict) else convert_to_openai_tool(tool)
            )
        except Exception:
            tools.append({"type": "function", "function": {"name": str(tool)}})
    model = getattr(request.model, "model_name", None) or getattr(
        request.model, "model", None
    )
    body = {"model": model, "messages": said}
    return {**body, "tools": tools} if tools else body, owners


def _thinks(block: Any) -> bool:
    return isinstance(block, dict) and str(block.get("type")).endswith("thinking")


def _thoughts(message) -> list:
    """What Claude reasoned in a message, enclosed."""
    content = getattr(message, "content", None)
    if getattr(message, "type", "") != "ai" or not isinstance(content, list):
        return []
    return [block for block in content if _thinks(block)]


def _text(content: Any) -> str | None:
    """The text of a chat-completions message, where it is text only."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" for p in content
    ):
        return "\n\n".join(p.get("text") or "" for p in content)
    return None


def _shown(request, body: dict, shown: dict, owners: list) -> list:
    """The request's messages with what is archived replaced by its notice."""
    messages = list(request.messages)
    for before, after, owner in zip(body["messages"], shown["messages"], owners):
        if owner is None or before is after or before == after:
            continue
        text = _text(after.get("content"))
        if before.get("content") != after.get("content") and text is not None:
            messages[owner] = messages[owner].model_copy(update={"content": text})
        elif before.get("thinking_blocks") and not after.get("thinking_blocks"):
            kept = [b for b in messages[owner].content if not _thinks(b)]
            kept = kept or [{"type": "text", "text": "\u2026"}]
            messages[owner] = messages[owner].model_copy(update={"content": kept})
    return messages


def _usage(message) -> dict | None:
    """What a reply was billed, in the form of the OpenAI API. What is not reported is left out."""
    found = getattr(message, "usage_metadata", None)
    if not found:
        return None
    details = found.get("input_token_details") or {}
    kept = {
        ours: details[theirs]
        for ours, theirs in (
            ("cached_tokens", "cache_read"),
            ("cache_write_tokens", "cache_creation"),
        )
        if details.get(theirs) is not None
    }
    return {
        "prompt_tokens": found.get("input_tokens") or 0,
        "completion_tokens": found.get("output_tokens") or 0,
        **({"prompt_tokens_details": kept} if kept else {}),
    }


def _replied(shown, name: str, body: dict, response) -> None:
    """Keep the books on a reply, and close the turn if it ends it."""
    from langchain_core.messages import convert_to_openai_messages

    said = getattr(response, "result", None)
    said = [response] if said is None else said
    reply = next((m for m in reversed(said) if getattr(m, "type", "") == "ai"), None)
    if reply is None:
        return
    calls = [
        {"id": c.get("id"), "arguments": c.get("args")}
        for c in getattr(reply, "tool_calls", None) or ()
    ]
    said = _text(reply.content) or ""
    shown.heard(_usage(reply), model=body.get("model"), said=said, calls=calls)
    if not calls:
        messages = [*body["messages"], *convert_to_openai_messages([reply])]
        shown.conclude({**body, "messages": messages}, thread=name)


def _reopen_tool(live: Live):
    """The agent's `reopen` tool, which reads what a notice stands for."""
    from langchain_core.tools import StructuredTool

    def reopen(reference: str, offset: int = 0) -> str:
        found = live.reopen(_thread_id() or "", reference, offset)
        return json.dumps(found, ensure_ascii=False)

    function = _AGENT_TOOL["function"]
    return StructuredTool.from_function(
        reopen,
        name=function["name"],
        description=function["description"],
        args_schema=function["parameters"],
    )


def middleware(live: Live | None = None):
    """An agent middleware that puts `live` between the agent and its model, whatever model that is.

    Without `live` it only names every model call's conversation after the thread it runs in, for
    a model that has compache in its HTTP client. Either way it can go anywhere in the list.
    """
    from langchain.agents.middleware import AgentMiddleware

    class Compache(AgentMiddleware):
        tools = [_reopen_tool(live)] if live is not None else []

        def _before(self, request):
            """The request to make instead, and what keeps the books on its reply."""
            name = _thread_id(request)
            if live is None or name is None:
                return name, request, None
            try:
                body, owners = _body(request)
                shown = live.show(_PATH, body, thread=name)
            except Exception:
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                return name, request, None
            try:
                changed = _shown(request, body, shown.body, owners)
                return name, request.override(messages=changed), (shown, body)
            except Exception:
                shown.over()
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                return name, request, None

        def _after(self, name, books, response):
            if books is not None:
                shown, body = books
                try:
                    _replied(shown, name, body, response)
                except Exception:
                    logger.warning("compache: a reply was not recorded", exc_info=True)
                finally:
                    shown.over()
            return response

        @staticmethod
        def _failed(books):
            if books is not None:
                books[0].over()

        def wrap_model_call(self, request, handler):
            name, request, books = self._before(request)
            try:
                with thread(name):
                    response = handler(request)
            except BaseException:
                self._failed(books)
                raise
            return self._after(name, books, response)

        async def awrap_model_call(self, request, handler):
            name, request, books = self._before(request)
            try:
                with thread(name):
                    response = await handler(request)
            except BaseException:
                self._failed(books)
                raise
            return self._after(name, books, response)

        def _answer(self, request):
            """The answer to a call of our tool, from the conversation whose model made it."""
            call = request.tool_call
            if live is None or call.get("name") != _REOPEN:
                return None
            from langchain_core.messages import ToolMessage

            given = call.get("args") or {}
            found = live.reopen(
                _thread_id(request) or "",
                str(given.get("reference", "")),
                given.get("offset") or 0,
                call=call.get("id"),
            )
            text = json.dumps(found, ensure_ascii=False)
            return ToolMessage(text, tool_call_id=call.get("id"), name=_REOPEN)

        def wrap_tool_call(self, request, handler):
            answer = self._answer(request)
            return handler(request) if answer is None else answer

        async def awrap_tool_call(self, request, handler):
            answer = self._answer(request)
            return await handler(request) if answer is None else answer

    return Compache()
