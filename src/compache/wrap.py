"""compache inside a provider's SDK client.

The OpenAI, Anthropic and Google SDKs all send their requests through an
`httpx` client. `wrap` gives a client one that passes every model call through
a `Live`, so nothing else in the application changes: `create`, `parse`,
`stream`, tools, structured output, async clients and anything built on top of
them, such as LangChain's chat models, keep working as they did.

    client = compache.wrap(OpenAI(), "policy.json")
    with compache.thread("conversation-17"):
        client.chat.completions.create(model=..., messages=messages)

`thread` names the conversation. Without it, a request header
`x-compache-thread` does, and without either the conversation is recognized by
what it has said so far.

The HTTP library is the SDK's own, `httpx` or its successor `httpx2`, found
from the client that is wrapped. compache itself needs neither.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import json
import logging
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any

from . import wire
from .live import _LATER, Live, Result

__all__ = ["async_http_client", "http_client", "start_pings", "thread", "wrap"]

HEADER = "x-compache-thread"
_thread: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "compache_thread", default=None
)
logger = logging.getLogger(__name__)
# Headers of a response that describe a body this module may have rewritten.
_BODY_HEADERS = ("content-length", "content-encoding", "transfer-encoding")
# Async calls do their bookkeeping here, one worker per call in flight and per stream being read.
_MOST_WORKERS = 2048
_workers = ThreadPoolExecutor(max_workers=_MOST_WORKERS, thread_name_prefix="compache")
# The replies that are being streamed.
_open: dict[int, Any] = {}
# How long the end of a stream that nobody reads is waited for.
_CLOSING_SECONDS = 30


@contextmanager
def thread(thread_id: str | None):
    """Name the conversation that the model calls made inside this block belong to."""
    token = _thread.set(thread_id)
    try:
        yield
    finally:
        _thread.reset(token)


def current_thread() -> str | None:
    return _thread.get()


def _named(request) -> str | None:
    return _thread.get() or request.headers.get(HEADER)


def signed(headers) -> bool:
    """Whether a request is signed over its body, as AWS signs it: changing it would break the signature."""
    return str(headers.get("authorization") or "").startswith("AWS4-HMAC-SHA256")


def _untouched(live: Live, request) -> bool:
    """Whether a request has to go out exactly as it came. Says so once."""
    if not signed(request.headers):
        return False
    live.passed(
        "signed",
        "requests signed with AWS SigV4 pass unchanged: what is signed cannot be changed",
    )
    return True


def _headers(response) -> list:
    return [
        (k, v)
        for k, v in response.headers.multi_items()
        if k.lower() not in _BODY_HEADERS
    ]


def _outgoing(httpx, request, raw: bytes, body: Any, payload: Any):
    """The request to send for `payload`: the original bytes if nothing changed."""
    content = raw if payload is body else json.dumps(payload).encode()
    headers = [
        (k, v)
        for k, v in request.headers.multi_items()
        if k.lower() not in ("content-length", "transfer-encoding", HEADER)
    ]
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=content,
        extensions=request.extensions,
    )


def _reply(httpx, request, result: Result, stream_type):
    """An `httpx` response for what `Live` handed back."""
    if result.events is not None:
        return httpx.Response(
            result.status,
            headers=result.headers,
            stream=stream_type(result),
            request=request,
        )
    content = result.raw if result.raw is not None else json.dumps(result.json).encode()
    return httpx.Response(
        result.status, headers=result.headers, content=content, request=request
    )


def _result(response, events=None, close=None) -> Result:
    """What a provider answered, as `Live` reads it."""
    headers = _headers(response)
    if events is not None:
        return Result(response.status_code, headers, events=events, close=close)
    result = Result(response.status_code, headers, raw=response.content)
    if "json" in response.headers.get("content-type", ""):
        try:
            result.json = json.loads(result.raw)
        except ValueError:
            pass
    return result


def _streams(response) -> bool:
    return response.status_code < 400 and "text/event-stream" in response.headers.get(
        "content-type", ""
    )


def _library(inner=None):
    """The HTTP library a client is built on: `httpx2` or `httpx`. Without a client, whichever is installed."""
    for base in type(inner).__mro__ if inner is not None else ():
        root = base.__module__.split(".")[0]
        if root in ("httpx2", "httpx"):
            return importlib.import_module(root)
    for name in ("httpx2", "httpx"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(
        "compache.wrap needs httpx2 or httpx, which the provider SDKs install"
    )


def http_client(live: Live, *, inner=None, pings: bool = True, **options):
    """An HTTP client that passes model calls through `live`, for anything that takes one.

    `inner` is the client or transport that does the sending: the SDK's own
    client, so that its proxies, certificates and limits stay in force. `pings`
    lets a background thread send the policy's keep-warm pings. Other options
    are the HTTP client's own.
    """
    httpx = _library(inner)
    if pings:
        start_pings(live)

    send = inner.send if isinstance(inner, httpx.Client) else None
    transport = (
        inner if isinstance(inner, httpx.BaseTransport) else httpx.HTTPTransport()
    )

    class Stream(httpx.SyncByteStream):
        def __init__(self, result):
            self._result = result

        def __iter__(self):
            yield from self._result.events

        def close(self):
            _let_go(self._result.events, self._result.close)

    class Transport(httpx.BaseTransport):
        def _forward(self, request):
            if send is not None:
                return send(request, stream=True)
            return transport.handle_request(request)

        def handle_request(self, request):
            dialect = wire.dialect_for(request.url.path)
            if request.method != "POST" or dialect is None:
                return self._forward(request)
            if _untouched(live, request):
                return self._forward(request)
            raw = request.read()
            try:
                body = json.loads(raw)
            except ValueError:
                return self._forward(request)

            def perform(payload):
                response = self._forward(_outgoing(httpx, request, raw, body, payload))
                if _streams(response):
                    return _result(response, response.iter_bytes(), response.close)
                response.read()
                response.close()
                return _result(response)

            result = live.exchange(
                request.url.path, body, perform, thread=_named(request)
            )
            return _reply(httpx, request, result, Stream)

        def close(self):
            if send is None:
                transport.close()

    if isinstance(inner, httpx.Client):
        options.setdefault("timeout", inner.timeout)
        options.setdefault("follow_redirects", inner.follow_redirects)
    made = httpx.Client(transport=Transport(), **options)
    # So that nothing puts compache in its path twice, and another `Live` can take over.
    made._compache, made._compache_inner = live, inner
    return made


def _let_go(events, close) -> None:
    """End a stream before its end: what reads it, and the response behind it."""
    closing = getattr(events, "close", None)
    try:
        if closing is not None:
            closing()
    except ValueError:
        # It is being read at this moment, in another thread. That ends with the response.
        pass
    finally:
        if close is not None:
            close()


def per_loop_transport(httpx):
    """An async transport with connections of their own for every event loop that uses it.

    A connection belongs to the loop that opened it, and an application may run
    several: one `asyncio.run` a request, a worker's, a test's.
    """

    class PerLoop(httpx.AsyncBaseTransport):
        def __init__(self):
            self._made = weakref.WeakKeyDictionary()
            self._lock = threading.Lock()

        async def handle_async_request(self, request):
            loop = asyncio.get_running_loop()
            with self._lock:
                transport = self._made.get(loop)
                if transport is None:
                    transport = self._made[loop] = httpx.AsyncHTTPTransport()
            return await transport.handle_async_request(request)

        async def aclose(self):
            with self._lock:
                transport = self._made.pop(asyncio.get_running_loop(), None)
            if transport is not None:
                await transport.aclose()

    return PerLoop()


def async_http_client(live: Live, *, inner=None, pings: bool = True, **options):
    """An async HTTP client that passes model calls through `live`.

    `Live` does its work on a worker thread and sends through `inner` on the
    event loop, so the loop is never blocked. Otherwise as `http_client`.
    """
    httpx = _library(inner)
    if pings:
        start_pings(live)

    send = inner.send if isinstance(inner, httpx.AsyncClient) else None
    transport = (
        inner
        if isinstance(inner, httpx.AsyncBaseTransport)
        else per_loop_transport(httpx)
    )
    done = object()

    class Stream(httpx.AsyncByteStream):
        def __init__(self, result):
            self._events, self._close = iter(result.events), result.close
            self._context = contextvars.copy_context()
            self._over = False
            _LATER.ready()

        def __aiter__(self):
            return self

        async def __anext__(self):
            loop = asyncio.get_running_loop()
            chunk = await loop.run_in_executor(
                _workers, self._context.copy().run, next, self._events, done
            )
            if chunk is done:
                self._over = True
                raise StopAsyncIteration
            return chunk

        async def aclose(self):
            if not self._over:
                self._over = True
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    _workers,
                    self._context.copy().run,
                    _let_go,
                    self._events,
                    self._close,
                )

        def __del__(self):
            if not self._over:
                _LATER.do(
                    partial(
                        self._context.copy().run, _let_go, self._events, self._close
                    )
                )

    class Transport(httpx.AsyncBaseTransport):
        async def _forward(self, request):
            if send is not None:
                return await send(request, stream=True)
            return await transport.handle_async_request(request)

        async def handle_async_request(self, request):
            dialect = wire.dialect_for(request.url.path)
            if request.method != "POST" or dialect is None:
                return await self._forward(request)
            if _untouched(live, request):
                return await self._forward(request)
            raw = await request.aread()
            try:
                body = json.loads(raw)
            except ValueError:
                return await self._forward(request)
            loop = asyncio.get_running_loop()
            # What this call has under way on the loop, so that cancelling the call cancels it.
            under_way: set = set()

            def on_loop(coroutine, most=None):
                made = asyncio.run_coroutine_threadsafe(coroutine, loop)
                under_way.add(made)
                try:
                    return made.result(most)
                finally:
                    under_way.discard(made)

            async def fetch(payload):
                response = await self._forward(
                    _outgoing(httpx, request, raw, body, payload)
                )
                if _streams(response):
                    return response, response.aiter_bytes()
                await response.aread()
                await response.aclose()
                return response, None

            def perform(payload):
                response, chunks = on_loop(fetch(payload))
                if chunks is None:
                    return _result(response)

                # Not to be collected by itself: it is closed in one place, when what reads it is.
                _open[id(chunks)] = chunks

                async def step():
                    return await chunks.__anext__()

                def events():
                    while True:
                        try:
                            yield on_loop(step())
                        except StopAsyncIteration:
                            _open.pop(id(chunks), None)
                            return

                async def over():
                    try:
                        await chunks.aclose()
                    except RuntimeError:
                        # It is being read at this moment.
                        pass
                    finally:
                        _open.pop(id(chunks), None)
                        await response.aclose()

                def closing():
                    try:
                        here = asyncio.get_running_loop() is loop
                    except RuntimeError:
                        here = False
                    if here:
                        # Its own thread cannot wait for what it has to do itself.
                        loop.create_task(over())
                    elif not loop.is_closed():
                        on_loop(over(), _CLOSING_SECONDS)

                return _result(response, events(), closing)

            named = _named(request)
            try:
                result = await loop.run_in_executor(
                    _workers,
                    contextvars.copy_context().run,
                    partial(
                        live.exchange, request.url.path, body, perform, thread=named
                    ),
                )
            except asyncio.CancelledError:
                # The application gave up on it: so does the provider, as it would without compache.
                for made in list(under_way):
                    made.cancel()
                raise
            return _reply(httpx, request, result, Stream)

        async def aclose(self):
            if send is None:
                await transport.aclose()

    if isinstance(inner, httpx.AsyncClient):
        options.setdefault("timeout", inner.timeout)
        options.setdefault("follow_redirects", inner.follow_redirects)
    made = httpx.AsyncClient(transport=Transport(), **options)
    made._compache, made._compache_inner = live, inner
    return made


class Pinger:
    """A background thread that sends a `Live`'s keep-warm pings when they fall due."""

    def __init__(self, live: Live, every: float = 1.0):
        self.live, self.every = live, every
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="compache-pings", daemon=True
        )
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.every):
            due = self.live.next_ping_at()
            if due is not None and due <= self.live.clock():
                self.live.tick()

    def stop(self):
        self._stop.set()


# The `Live`s whose pings a thread sends, to start that thread again in a forked worker.
_pinging: weakref.WeakSet = weakref.WeakSet()


def start_pings(live: Live) -> None:
    """Let a background thread send `live`'s keep-warm pings.

    One thread for a `Live`, however many clients share it. Nothing is started
    for a policy that sends no pings.
    """
    if live.learner is None and (live.policy is None or not live.policy.max_pings):
        return
    with live._lock:
        if live.pinger is None or not live.pinger._thread.is_alive():
            live.pinger = Pinger(live)
        _pinging.add(live)


def _forked() -> None:
    """In a worker forked from a process that had started them: threads do not survive a fork."""
    global _workers
    _workers = ThreadPoolExecutor(
        max_workers=_MOST_WORKERS, thread_name_prefix="compache"
    )
    for live in list(_pinging):
        live.pinger = None
        try:
            start_pings(live)
        except Exception:
            logger.warning(
                "compache: keep-warm pings did not start again", exc_info=True
            )


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forked)


def wrap(
    client,
    policy=None,
    *,
    record=None,
    learn=None,
    sample: float = 0.0,
    notes: str = "apart",
    pings: bool = True,
    live: Live | None = None,
):
    """The client with compache in its request path.

    `client` is a client of the OpenAI, Anthropic or Google GenAI SDK, sync or
    async. The other arguments are those of `Live`: a policy to apply, a
    folder to record into or to learn from, and the share of turns to ask for a note. Pass a
    `live` to share one between several clients. `pings` lets a background
    thread send the policy's keep-warm pings. The `Live` is on the returned
    client as `client.compache`.
    """
    if live is None:
        live = Live(policy, record=record, learn=learn, sample=sample, notes=notes)
    inner = getattr(client, "_client", None)
    api = getattr(client, "_api_client", None)
    httpx = _library(
        inner if inner is not None else getattr(api, "_httpx_client", None)
    )
    if hasattr(client, "with_options") and isinstance(inner, httpx.Client):
        wrapped = client.with_options(
            http_client=http_client(live, inner=inner, pings=pings)
        )
    elif hasattr(client, "with_options") and isinstance(inner, httpx.AsyncClient):
        wrapped = client.with_options(
            http_client=async_http_client(live, inner=inner, pings=pings)
        )
    elif hasattr(api, "_httpx_client"):
        # The Google GenAI SDK takes its HTTP clients when it is built, so its own are replaced.
        api._httpx_client = http_client(live, inner=api._httpx_client, pings=pings)
        if isinstance(getattr(api, "_async_httpx_client", None), httpx.AsyncClient):
            api._async_httpx_client = async_http_client(
                live, inner=api._async_httpx_client, pings=pings
            )
            options = getattr(api, "_http_options", None)
            if options is not None and hasattr(options, "httpx_async_client"):
                # Given as the application's own, it is the one async calls take, and not aiohttp.
                options.httpx_async_client = api._async_httpx_client
        wrapped = client
    else:
        raise TypeError(
            "wrap takes a client of the OpenAI, Anthropic or Google GenAI SDK. "
            "For anything else that sends through httpx, pass it compache.http_client(live)."
        )
    wrapped.compache = live
    return wrapped
