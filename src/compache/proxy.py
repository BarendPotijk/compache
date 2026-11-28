"""compache behind a base URL.

    compache proxy --policy policy.json --record compache-record --sample 0.1

An application, an agent harness or an SDK that lets its base URL be set needs
no other change: point it at the proxy and every model call passes through a
`Live`. The proxy speaks the provider's own API on the same paths and forwards
each request, with the caller's credentials, to the provider:

    …/v1/chat/completions, …/v1/responses        OpenAI, or any compatible gateway
    …/v1/messages                                Anthropic
    …/v1beta/models/<model>:generateContent      Google Gemini

Everything else is passed on untouched. A conversation is named by the header
`x-compache-thread`, or by a session header or `prompt_cache_key` that agent
harnesses already send. Without any, it is recognized by what it has said so far.

Standard library only. It is one process with one thread per request, meant to
run beside the application it serves.
"""

from __future__ import annotations

import base64
import gzip
import http.client
import importlib
import io
import json
import logging
import os
import re
import select
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import _tls, wire
from .live import Live, Result
from .wrap import HEADER, signed, start_pings

__all__ = ["DEFAULTS", "serve"]

logger = logging.getLogger(__name__)

# Where each API goes when no upstream is named.
DEFAULTS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
# Headers that name a conversation, in the order they are tried. Harnesses send the later ones.
THREAD_HEADERS = (
    HEADER,
    "x-session-affinity",
    "session_id",
    "session-id",
    "x-session-id",
    "x-claude-code-session-id",
    "x-opencode-session-id",
    "agent-session-id",
)
_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}
# Paths as they come from an SDK whose base URL was set without the API's version.
_UNVERSIONED = ("/messages", "/chat/completions", "/responses", "/models")
_BODY = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def _family(path: str, headers, other: str = "openai") -> str:
    """Which provider's API a request speaks. `other` is the answer where nothing says."""
    dialect = wire.dialect_for(path)
    if dialect is not None:
        return {"anthropic": "anthropic", "gemini": "gemini"}.get(
            dialect.name, "openai"
        )
    key = headers.get("x-api-key", headers.get("authorization", ""))
    if (
        "anthropic-version" in headers
        or "sk-ant-" in key
        or "/publishers/anthropic/" in path
        or ("x-api-key" in headers and "authorization" not in headers)
    ):
        return "anthropic"
    if (
        "x-goog-api-key" in headers
        or "/v1beta/" in path
        or "/publishers/google/" in path
    ):
        return "gemini"
    return other


# What the keys of a provider look like. Any other is for another server, such as a gateway.
_KEYS = {
    "openai": re.compile(r"T3BlbkFJ|^sk-(?:proj|svcacct|admin)-"),
    "anthropic": re.compile(r"^sk-ant-"),
    "gemini": re.compile(r"^(?:AIza|AQ\.|ya29\.)"),
}


def _elsewhere(api: str, headers, path: str) -> bool:
    """Whether a request's key is none of the provider's whose API it speaks."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    keys = [headers.get("authorization", "").removeprefix("Bearer ").strip()]
    keys += [headers.get(name, "") for name in ("x-api-key", "x-goog-api-key")]
    keys = [k for k in (*keys, *query.get("key", ())) if k]
    return bool(keys) and not any(_KEYS[api].search(k) for k in keys)


def _decoded(raw: bytes, encoding: str | None) -> bytes:
    """A request body as JSON text. One in an encoding this Python does not read passes as it came."""
    encoding = (encoding or "identity").strip().lower()
    if encoding == "gzip":
        return gzip.decompress(raw)
    if encoding == "zstd":
        for name in ("compression.zstd", "backports.zstd", "zstandard"):
            try:
                found = importlib.import_module(name)
            except ImportError:
                continue
            if name == "zstandard":
                # It may not say how large it is, which the plain call needs.
                found = found.ZstdDecompressor()
                return found.stream_reader(io.BytesIO(raw)).read()
            return found.decompress(raw)
        raise ValueError("no zstd here")
    if encoding != "identity":
        raise ValueError(f"unknown encoding {encoding}")
    return raw


def _proxy(url: urllib.parse.SplitResult) -> urllib.parse.SplitResult | None:
    """What this machine reaches a host through, as `HTTPS_PROXY` names it. None where it needs nothing."""
    host = url.hostname or ""
    if host == "localhost" or host.startswith("127.") or host == "::1":
        return None
    given = urllib.request.getproxies().get(url.scheme)
    if not given or urllib.request.proxy_bypass(host):
        return None
    found = urllib.parse.urlsplit(given if "://" in given else f"http://{given}")
    if found.scheme == "https":
        # Its name and password would go out in plain sight.
        raise ValueError(
            f"{host} is reached through a proxy over TLS, which is not spoken"
        )
    return found if found.scheme == "http" and found.hostname else None


def _credentials(proxy: urllib.parse.SplitResult) -> dict:
    if not proxy.username:
        return {}
    name = urllib.parse.unquote(proxy.username)
    said = f"{name}:{urllib.parse.unquote(proxy.password or '')}".encode()
    return {"Proxy-Authorization": "Basic " + base64.b64encode(said).decode()}


def _metrics(live: Live) -> str:
    """What was done so far, as Prometheus reads it."""
    found = {f"compache_{name}_total": n for name, n in live.stats.items()}
    found |= {f"compache_{name}_tokens_total": n for name, n in live.tokens.items()}
    return "".join(f"# TYPE {name} counter\n{name} {n}\n" for name, n in found.items())


def _thread(headers, body) -> str | None:
    for name in THREAD_HEADERS:
        if headers.get(name):
            return headers[name]
    if isinstance(body, dict):
        for key in ("prompt_cache_key", "session_id"):
            if isinstance(body.get(key), str) and body[key]:
                return body[key]
    return None


# Connections to a provider kept open between calls, as an SDK keeps them: how many, and for how long.
_POOL_MOST = 8
_POOL_IDLE = 50.0
# What a connection that the other side has closed meanwhile raises.
_STALE = (
    http.client.RemoteDisconnected,
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


def _dropped(connection) -> bool:
    """Whether the other side has closed an idle connection: it is readable, with nothing asked of it."""
    sock = getattr(connection, "sock", None)
    if sock is None:
        return True
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


class _Pool:
    """Idle connections to providers, by where they lead. Safe to share between threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._kept: dict = {}

    def take(self, key):
        with self._lock:
            found = self._kept.get(key) or []
            while found:
                connection, since = found.pop()
                if time.monotonic() - since < _POOL_IDLE and not _dropped(connection):
                    return connection
                connection.close()
        return None

    def give(self, key, connection) -> None:
        with self._lock:
            found = self._kept.setdefault(key, [])
            found.append((connection, time.monotonic()))
            while len(found) > _POOL_MOST:
                found.pop(0)[0].close()


_POOL = _Pool()


class _Upstream:
    """One request to a provider, and its response as `Live` reads it."""

    def __init__(
        self, base: str, method: str, path: str, headers: list, timeout: float
    ):
        self.url = urllib.parse.urlsplit(base.rstrip("/") + path)
        self.method, self.headers, self.timeout = method, headers, timeout

    def _open(self, host, port, proxy):
        if self.url.scheme == "https":
            connection = http.client.HTTPSConnection(
                host, port, timeout=self.timeout, context=_tls.FOUND["context"]
            )
            if proxy is not None:
                connection.set_tunnel(
                    self.url.hostname, self.url.port, _credentials(proxy)
                )
            return connection
        return http.client.HTTPConnection(host, port, timeout=self.timeout)

    def _sent(self, body: bytes | None, headers: dict):
        """The provider's response to the request, the connection it came on, and where that leads."""
        target = self.url.path + ("?" + self.url.query if self.url.query else "")
        host, port, proxy = self.url.hostname, self.url.port, _proxy(self.url)
        if proxy is not None:
            host, port = proxy.hostname, proxy.port or 80
            if self.url.scheme != "https":
                target = f"http://{self.url.netloc}{target}"
                headers = {**headers, "Host": self.url.netloc, **_credentials(proxy)}
        key = (
            self.url.scheme,
            host,
            port,
            self.url.netloc,
            None if proxy is None else proxy.geturl(),
            id(_tls.FOUND["context"]),
        )
        for attempt in range(2):
            connection = _POOL.take(key) if attempt == 0 else None
            reused = connection is not None
            if connection is None:
                connection = self._open(host, port, proxy)
            try:
                connection.request(self.method, target, body=body, headers=headers)
                response = connection.getresponse()
            except ssl.SSLError as error:
                connection.close()
                if reused:
                    continue
                # The handshake failed, so nothing was sent: the request can go out again.
                if _tls.unverified(error) and _tls.fall_back():
                    return self._sent(body, headers)
                raise
            except _STALE:
                connection.close()
                # One that was kept open had been closed by the other side: it goes out on a new one.
                if reused:
                    continue
                raise
            return connection, response, key
        raise AssertionError("not reached")

    def __call__(self, payload, raw: bytes | None = None) -> Result:
        body = raw
        if raw is None and payload is not None:
            body = json.dumps(payload).encode()
        headers = {k: v for k, v in self.headers}
        headers["Accept-Encoding"] = "identity"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        connection, response, key = self._sent(body, headers)
        kept = [(k, v) for k, v in response.getheaders() if k.lower() not in _BODY]
        over = {"done": False}

        def finish(whole: bool) -> None:
            """Keep the connection for the next call once its response is read to the end, else close it."""
            if over["done"]:
                return
            over["done"] = True
            if whole and getattr(response, "will_close", True) is False:
                _POOL.give(key, connection)
            else:
                connection.close()

        if response.status < 400 and "text/event-stream" in (
            response.getheader("Content-Type") or ""
        ):

            def events():
                whole = False
                try:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            whole = True
                            return
                        yield chunk
                finally:
                    finish(whole)

            return Result(
                response.status, kept, events=events(), close=lambda: finish(False)
            )
        try:
            data = response.read()
        except BaseException:
            finish(False)
            raise
        finish(True)
        result = Result(response.status, kept, raw=data)
        if "json" in (response.getheader("Content-Type") or ""):
            try:
                result.json = json.loads(data)
            except ValueError:
                pass
        return result


def serve(
    live: Live,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    upstream: str | None = None,
    upstreams: dict | None = None,
    pings: bool = True,
    timeout: float = 600.0,
) -> ThreadingHTTPServer:
    """An HTTP server that passes model calls through `live`. Call its `serve_forever()`.

    `upstream` is one base URL for everything, such as a gateway's. Otherwise
    each API goes to its provider, or to the URL `upstreams` names for
    `openai`, `anthropic` or `gemini`. `server.live` is the `Live`.
    """
    targets = {**DEFAULTS, **(upstreams or {})}
    forward: dict = {"all": upstream, "other": "openai", "known": set(DEFAULTS)}
    forward["told"] = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _answer(self, status: int, headers, body: bytes):
            self.send_response(status)
            for name, value in headers or ():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, result: Result):
            self.send_response(result.status)
            for name, value in result.headers or ():
                self.send_header(name, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in result.events:
                    if chunk:
                        self.wfile.write(
                            f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
                        )
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # The caller went away. Closing the events releases the provider's connection.
                self.close_connection = True
            finally:
                closing = getattr(result.events, "close", None)
                if closing is not None:
                    closing()

        def _read(self) -> bytes | None:
            """The request's body, whole or in chunks."""
            if "chunked" not in (self.headers.get("Transfer-Encoding") or "").lower():
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else None
            pieces = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if not size:
                    break
                pieces.append(self.rfile.read(size))
                self.rfile.readline()
            while self.rfile.readline().strip():
                # Trailers, which nobody sends.
                pass
            return b"".join(pieces)

        def _handle(self):
            if (
                self.path in ("/compache/stats", "/compache/health")
                and self.command == "GET"
            ):
                said = {"status": "ok", **live.stats, "tokens": live.tokens}
                body = json.dumps(said).encode()
                return self._answer(200, [("Content-Type", "application/json")], body)
            if self.path == "/compache/metrics" and self.command == "GET":
                kind = [("Content-Type", "text/plain; version=0.0.4")]
                return self._answer(200, kind, _metrics(live).encode())
            if "websocket" in (self.headers.get("Upgrade") or "").lower():
                # An agent that tries this first, as Codex does, goes on over HTTP.
                return self._answer(426, [], b"compache speaks HTTP")
            raw = self._read()
            headers = {k.lower(): v for k, v in self.headers.items()}
            speaks = _family(self.path, headers, forward["other"])
            base = forward["all"] or targets[speaks]
            if speaks not in forward["known"] and _elsewhere(
                speaks, headers, self.path
            ):
                return self._refuse(speaks, base)
            if base in DEFAULTS.values() and self.path.startswith(_UNVERSIONED):
                # SDKs differ in whether a base URL holds the version. Their providers do not.
                self.path = "/v1" + self.path
            kept = [
                (k, v)
                for k, v in self.headers.items()
                if k.lower() not in _HOP and k.lower() != HEADER
            ]
            send = _Upstream(base, self.command, self.path, kept, timeout)
            body = None
            if signed(headers) and wire.dialect_for(self.path) is not None:
                live.passed(
                    "signed",
                    "requests signed with AWS SigV4 pass unchanged: what is signed cannot be changed",
                )
            elif (
                self.command == "POST"
                and raw
                and wire.dialect_for(self.path) is not None
            ):
                try:
                    body = json.loads(_decoded(raw, headers.get("content-encoding")))
                except Exception:
                    # Whatever is wrong with it is for the provider to say.
                    body = None
            if body is None:
                result = send(None, raw)
            else:
                # The caller's bytes go out as they came whenever nothing in them was changed.
                plain = _Upstream(
                    base,
                    self.command,
                    self.path,
                    [(k, v) for k, v in kept if k.lower() != "content-encoding"],
                    timeout,
                )
                result = live.exchange(
                    self.path,
                    body,
                    lambda payload: (
                        send(payload, raw) if payload is body else plain(payload)
                    ),
                    thread=_thread(headers, body),
                )
            if result.events is not None:
                return self._stream(result)
            content = (
                result.raw
                if result.raw is not None
                else json.dumps(result.json).encode()
            )
            self._answer(result.status, result.headers, content)

        def _refuse(self, speaks: str, base: str):
            """Where a key belongs is not known, so it goes nowhere."""
            message = (
                f"compache: this key does not look like one for {base}, so it was not sent there. Say "
                "where it belongs with --upstream URL"
            )
            if speaks not in forward["told"]:
                forward["told"].add(speaks)
                print(message, file=sys.stderr, flush=True)
            said = {"type": "permission_error", "message": message}
            body = json.dumps({"type": "error", "error": said}).encode()
            self._answer(403, [("Content-Type", "application/json")], body)

        def _guarded(self):
            try:
                self._handle()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as error:
                logger.warning("compache proxy: a request failed", exc_info=True)
                message = json.dumps(
                    {
                        "error": {
                            "message": f"compache proxy: {error}",
                            "type": "proxy_error",
                        }
                    }
                )
                try:
                    self._answer(
                        502, [("Content-Type", "application/json")], message.encode()
                    )
                except Exception:
                    self.close_connection = True

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _guarded

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # A caller that hangs up is not an error worth a traceback.
            pass

        @staticmethod
        def forward(
            everything: str | None,
            named: dict,
            other: str = "openai",
            careful: bool = False,
        ) -> None:
            """Where requests go on to, once that is known: one base URL, or one by API.

            `other` is the API of those that do not say which they speak. Where it
            has to be `careful`, the providers' own addresses get only their own keys.
            """
            forward["all"], forward["other"] = everything, other
            targets.update(named)
            forward["known"] = (
                set(named) if careful and not everything else set(DEFAULTS)
            )

    server = Server((host, port), Handler)
    server.live = live
    if pings:
        start_pings(live)
    return server


def main(args) -> int:
    """`compache proxy`: run until interrupted."""
    live = Live(
        args.policy,
        record=args.record,
        learn=args.learn,
        sample=args.sample,
        notes=args.notes,
    )
    upstream = args.upstream or os.environ.get("COMPACHE_UPSTREAM")
    named = {
        key: value
        for key, value in (
            ("openai", args.openai_url),
            ("anthropic", args.anthropic_url),
            ("gemini", args.gemini_url),
        )
        if value
    }
    server = serve(
        live, host=args.host, port=args.port, upstream=upstream, upstreams=named
    )
    if args.careful:
        server.forward(upstream, named, careful=True)
    where = f"http://{args.host}:{server.server_address[1]}"
    doing = []
    if live.policy is not None:
        doing.append(f"applying {args.policy}")
    if live.learner is not None:
        doing.append(f"learning in {args.learn}")
    elif live.recorder is not None:
        doing.append(f"recording to {args.record}")
    if live.sample:
        doing.append(f"asking {live.sample:.0%} of turns what they needed")
    print(
        f"compache proxy on {where}: {', '.join(doing) or 'passing requests through'}.",
        flush=True,
    )
    print(
        f"Forwarding to {upstream or 'each provider'}. Set your base URL to {where}/v1 (Anthropic: {where}).",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
