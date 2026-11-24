"""compache for LiteLLM.

Two ways in, alone or together.

`install` puts a `Live` in the path of LiteLLM's OpenAI-compatible calls (OpenAI,
OpenRouter, Azure and every other provider LiteLLM reaches that way), sync and async,
whole and streamed. The policy is applied, notes are asked and pings go out as with
`compache.wrap`:

    import compache, litellm
    live = compache.Live("policy.json", record="compache-record", sample=0.1)
    compache.litellm.install(live)
    with compache.thread("conversation-17"):
        litellm.completion(model="openrouter/anthropic/claude-haiku-5.5", messages=messages)

`Callback` is a LiteLLM callback that only records, whatever the provider, for teams
that want to learn a policy first. It reads the conversation's name from `metadata`:

    litellm.callbacks = [compache.litellm.Callback(record="compache-record", sample=0.1)]
    litellm.completion(..., metadata={"session_id": "conversation-17"})

In a LiteLLM proxy, name `compache.litellm.callback` under `litellm_settings.callbacks`
and set COMPACHE_RECORD to a folder (COMPACHE_SAMPLE and COMPACHE_NOTES are optional).

A provider that LiteLLM calls over its own HTTP handlers, such as Anthropic's Messages
API, goes through `compache proxy` with `api_base="http://127.0.0.1:8787"`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
from typing import Any

from .live import Live, Result
from .wrap import async_http_client, http_client, per_loop_transport

__all__ = ["Callback", "install"]

logger = logging.getLogger(__name__)

_CHAT_CALLS = ("completion", "acompletion", "", None)
_numbers = itertools.count(1)


def install(live: Live, *, pings: bool = True) -> Live:
    """Put `live` in the path of every OpenAI-compatible call that LiteLLM makes from now on.

    It sets `litellm.client_session` and `litellm.aclient_session`, which LiteLLM gives
    to the OpenAI SDK, and forgets the clients it made before. `pings` lets a background
    thread send the policy's keep-warm pings. Returns `live`.
    """
    import httpx
    import litellm

    options = {"timeout": httpx.Timeout(600.0, connect=10.0), "follow_redirects": True}
    litellm.client_session = http_client(
        live, inner=httpx.Client(**options), pings=pings
    )
    # One client for every event loop of the process, so its connections belong to the loop that uses them.
    litellm.aclient_session = async_http_client(
        live, inner=per_loop_transport(httpx), pings=pings, **options
    )
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    for name in ("flush_cache", "clear"):
        if callable(getattr(cache, name, None)):
            getattr(cache, name)()
            break
    return live


def _plain(value: Any) -> Any:
    """A message or reply as plain JSON: LiteLLM keeps replies as objects of its own."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _usage(usage: Any, cost: Any) -> dict | None:
    """LiteLLM's usage in the form of the OpenAI API, which counts what was read or written in the total.

    What the provider does not report is left out: a missing count is not a zero.
    """
    if usage is None:
        return None
    usage = _plain(usage)
    details = usage.get("prompt_tokens_details") or {}
    found = {
        "cached_tokens": usage.get(
            "cache_read_input_tokens", details.get("cached_tokens")
        ),
        "cache_write_tokens": usage.get(
            "cache_creation_input_tokens", details.get("cache_write_tokens")
        ),
    }
    out = {
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
    }
    if any(v is not None for v in found.values()):
        out["prompt_tokens_details"] = {k: v for k, v in found.items() if v is not None}
    # LiteLLM says 0 where it has no price for the model.
    if isinstance(cost, (int, float)) and cost > 0:
        out["cost"] = float(cost)
    return out


def _seconds(moment: Any) -> float:
    return moment.timestamp() if hasattr(moment, "timestamp") else float(moment)


def _metadata(kwargs: dict) -> dict:
    metadata = (kwargs.get("litellm_params") or {}).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _session(kwargs: dict) -> str | None:
    """The name of the conversation a call belongs to, from its metadata."""
    metadata = _metadata(kwargs)
    for value in (
        metadata.get("session_id"),
        (kwargs.get("litellm_params") or {}).get("litellm_session_id"),
        metadata.get("litellm_session_id"),
    ):
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return None


class _Observer:
    """Sees LiteLLM's finished calls and hands them to a `Live` that only records."""

    def __init__(self, live: Live):
        self.live = live

    def see(self, kwargs: dict, response: Any, start: Any, end: Any) -> None:
        """One finished call, whole or streamed. Never raises: a callback must not break a call."""
        try:
            self._see(kwargs, response, start, end)
        except Exception:
            logger.warning("compache: a LiteLLM call was not recorded", exc_info=True)

    def _see(self, kwargs, response, start, end) -> None:
        if kwargs.get("call_type") not in _CHAT_CALLS or _metadata(kwargs).get(
            "compache_note"
        ):
            return
        if kwargs.get("cache_hit"):
            # LiteLLM answered from a cache of its own. No model was called.
            return
        streamed = kwargs.get("complete_streaming_response") or kwargs.get(
            "async_complete_streaming_response"
        )
        reply = _plain(streamed or response)
        if not isinstance(reply, dict) or not reply.get("choices"):
            return
        reply["usage"] = _usage(
            getattr(streamed or response, "usage", None) or reply.get("usage"),
            kwargs.get("response_cost"),
        )
        # A LiteLLM proxy answers under its own name for the model. What was called is in the call.
        reply["model"] = kwargs.get("model") or reply.get("model")
        optional = kwargs.get("optional_params") or {}
        body = {
            "model": kwargs.get("model"),
            "messages": [_plain(m) for m in kwargs.get("messages") or ()],
        }
        tools = optional.get("tools") or kwargs.get("tools")
        if tools:
            body["tools"] = _plain(tools)
        self.live.observe(
            "/v1/chat/completions",
            body,
            reply,
            thread=_session(kwargs),
            at=(_seconds(start), _seconds(end)),
            ask=_asker(kwargs) if self.live.sample else None,
        )


# How a call reaches its provider, as LiteLLM keeps it with the call.
_CONNECTION = (
    "api_base",
    "api_key",
    "api_version",
    "azure_ad_token",
    "azure_ad_token_provider",
    "azure_password",
    "azure_username",
    "client_id",
    "client_secret",
    "extra_headers",
    "organization",
    "region_name",
    "ssl_verify",
    "tenant_id",
    "timeout",
)
_CONNECTIONS = ("aws_", "vertex_", "watsonx_")


def _asker(kwargs: dict):
    """A function that asks the model what its reply needed, through LiteLLM with this call's own settings."""
    params = {
        **(kwargs.get("optional_params") or {}),
        **(kwargs.get("litellm_params") or {}),
    }
    connection = {
        key: value
        for key, value in params.items()
        if value is not None and (key in _CONNECTION or key.startswith(_CONNECTIONS))
    }

    def ask(body: dict) -> Result:
        import litellm

        provider = kwargs.get("custom_llm_provider")
        model = body.get("model") or kwargs.get("model") or ""
        if provider and not str(model).startswith(f"{provider}/"):
            model = f"{provider}/{model}"
        request = {
            **connection,
            "model": model,
            "messages": body["messages"],
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "metadata": {"compache_note": True},
            "tools": body.get("tools"),
            # A note is worth one try.
            "num_retries": 0,
        }
        try:
            response = litellm.completion(
                **{k: v for k, v in request.items() if v is not None}
            )
        except Exception as error:
            # Counted with the refusals: after three of them no more notes are asked.
            logger.info("compache: a note was not asked", exc_info=True)
            status = getattr(error, "status_code", None)
            return Result(status if isinstance(status, int) and status >= 400 else 502)
        return Result(200, json=_plain(response))

    return ask


def Callback(  # named like a class: it makes an instance of one that LiteLLM has to provide
    live: Live | None = None, *, record=None, sample: float = 0.0, notes: str = "apart"
):
    """A LiteLLM callback that records every call, and asks notes when `sample` is set.

    `Callback(record="compache-record", sample=0.1)` is an instance to put in
    `litellm.callbacks`. LiteLLM is needed only when it is made. Its `Live` is
    on it as `compache`.
    """
    from litellm.integrations.custom_logger import CustomLogger

    if live is None:
        live = Live(record=record, sample=sample, notes=notes)
    seen = _Observer(live)

    class _Compache(CustomLogger):
        """Records what LiteLLM did. The text of the calls goes nowhere."""

        compache = live

        def __init__(self):
            super().__init__()
            # LiteLLM drops a callback whose class name and plain attributes it already has.
            # A number of its own keeps two of these, which record in different places, apart.
            self.number = next(_numbers)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            seen.see(kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            await asyncio.to_thread(
                seen.see, kwargs, response_obj, start_time, end_time
            )

    return _Compache()


def __getattr__(name: str) -> Any:
    """`compache.litellm.callback`: an instance made from the environment, for a LiteLLM proxy's config."""
    if name != "callback":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    folder = os.environ.get("COMPACHE_RECORD")
    if not folder:
        raise RuntimeError(
            "compache.litellm.callback: set COMPACHE_RECORD to the folder to record into"
        )
    made = Callback(
        record=folder,
        sample=float(os.environ.get("COMPACHE_SAMPLE") or 0),
        notes=os.environ.get("COMPACHE_NOTES") or "apart",
    )
    globals()["callback"] = made
    return made
