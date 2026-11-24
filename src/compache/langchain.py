"""compache for LangChain and LangGraph agents.

LangChain's chat models send through the provider SDKs, so compache goes in at
the same place as for the SDKs themselves: the HTTP client. What an agent adds
is a thread ID, and `middleware` passes it on as the conversation's name.

    live = compache.Live("policy.json", record="compache-record", sample=0.1)
    model = ChatOpenAI(model="gpt-6-luna", **compache.langchain.clients(live))
    agent = create_agent(model, tools, middleware=[compache.langchain.middleware()])
    agent.invoke({"messages": [...]}, {"configurable": {"thread_id": "conversation-17"}})

A chat model that takes no HTTP client, such as `ChatAnthropic`, is pointed at
`compache proxy` with its base URL instead.
"""

from __future__ import annotations

from .live import Live
from .wrap import async_http_client, http_client, thread

__all__ = ["clients", "middleware"]


def clients(live: Live, *, pings: bool = True) -> dict:
    """The `http_client` and `http_async_client` arguments of `ChatOpenAI`, with `live` in the path.

    `pings` lets a background thread send the policy's keep-warm pings.
    """
    return {
        "http_client": http_client(live, pings=pings),
        "http_async_client": async_http_client(live, pings=pings),
    }


def _thread_id(request) -> str | None:
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


def middleware():
    """An agent middleware that names every model call's conversation after the thread it runs in.

    It changes nothing else, so it can go anywhere in the list.
    """
    from langchain.agents.middleware import AgentMiddleware

    class Compache(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            with thread(_thread_id(request)):
                return handler(request)

        async def awrap_model_call(self, request, handler):
            with thread(_thread_id(request)):
                return await handler(request)

    return Compache()
