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
# Headers of a response that describe a body this module may have rewritten.
_BODY_HEADERS = ("content-length", "content-encoding", "transfer-encoding")
# Async calls do their bookkeeping here, one worker per call in flight.
_workers = ThreadPoolExecutor(max_workers=128, thread_name_prefix="compache")
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
    return httpx.Client(transport=Transport(), **options)


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
            self._over = False
            _LATER.ready()

        def __aiter__(self):
            return self

        async def __anext__(self):
            loop = asyncio.get_running_loop()
            chunk = await loop.run_in_executor(_workers, next, self._events, done)
            if chunk is done:
                self._over = True
                raise StopAsyncIteration
            return chunk

        async def aclose(self):
            if not self._over:
                self._over = True
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(_workers, _let_go, self._events, self._close)

        def __del__(self):
            if not self._over:
                _LATER.do(partial(_let_go, self._events, self._close))

    class Transport(httpx.AsyncBaseTransport):
        async def _forward(self, request):
            if send is not None:
                return await send(request, stream=True)
            return await transport.handle_async_request(request)

        async def handle_async_request(self, request):
            dialect = wire.dialect_for(request.url.path)
            if request.method != "POST" or dialect is None:
                return await self._forward(request)
            raw = await request.aread()
            try:
                body = json.loads(raw)
            except ValueError:
                return await self._forward(request)
            loop = asyncio.get_running_loop()

            def on_loop(coroutine, most=None):
                made = asyncio.run_coroutine_threadsafe(coroutine, loop)
                return made.result(most)

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
            result = await loop.run_in_executor(
                _workers,
                lambda: live.exchange(request.url.path, body, perform, thread=named),
            )
            return _reply(httpx, request, result, Stream)

        async def aclose(self):
            if send is None:
                await transport.aclose()

    if isinstance(inner, httpx.AsyncClient):
        options.setdefault("timeout", inner.timeout)
        options.setdefault("follow_redirects", inner.follow_redirects)
    return httpx.AsyncClient(transport=Transport(), **options)


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


def start_pings(live: Live) -> None:
    """Let a background thread send `live`'s keep-warm pings.

    One thread for a `Live`, however many clients share it. Nothing is started
    for a policy that sends no pings.
    """
    if live.learner is None and (live.policy is None or not live.policy.max_pings):
        return
    with live._lock:
        if live.pinger is None:
            live.pinger = Pinger(live)


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
        wrapped = client
    else:
        raise TypeError(
            "wrap takes a client of the OpenAI, Anthropic or Google GenAI SDK. "
            "For anything else that sends through httpx, pass it compache.http_client(live)."
        )
    wrapped.compache = live
    return wrapped
