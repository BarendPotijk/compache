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

import http.client
import json
import logging
import os
import ssl
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import _tls, wire
from .live import Live, Result
from .wrap import HEADER, start_pings

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
    "x-session-id",
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
_BODY = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def _family(path: str, headers) -> str:
    """Which provider's API a request speaks."""
    dialect = wire.dialect_for(path)
    if dialect is not None:
        return {"anthropic": "anthropic", "gemini": "gemini"}.get(
            dialect.name, "openai"
        )
    if "x-goog-api-key" in headers or "/v1beta/" in path:
        return "gemini"
    if "anthropic-version" in headers or (
        "x-api-key" in headers and "authorization" not in headers
    ):
        return "anthropic"
    return "openai"


def _thread(headers, body) -> str | None:
    for name in THREAD_HEADERS:
        if headers.get(name):
            return headers[name]
    if isinstance(body, dict):
        for key in ("prompt_cache_key", "session_id"):
            if isinstance(body.get(key), str) and body[key]:
                return body[key]
    return None


class _Upstream:
    """One request to a provider, and its response as `Live` reads it."""

    def __init__(
        self, base: str, method: str, path: str, headers: list, timeout: float
    ):
        self.url = urllib.parse.urlsplit(base.rstrip("/") + path)
        self.method, self.headers, self.timeout = method, headers, timeout

    def _sent(self, body: bytes | None, headers: dict):
        """A connection to the provider with the request on its way."""
        if self.url.scheme == "https":
            connection = http.client.HTTPSConnection(
                self.url.hostname,
                self.url.port,
                timeout=self.timeout,
                context=_tls.FOUND["context"],
            )
        else:
            connection = http.client.HTTPConnection(
                self.url.hostname, self.url.port, timeout=self.timeout
            )
        target = self.url.path + ("?" + self.url.query if self.url.query else "")
        try:
            connection.request(self.method, target, body=body, headers=headers)
        except ssl.SSLError as error:
            connection.close()
            # The handshake failed, so nothing was sent: the request can go out again.
            if _tls.unverified(error) and _tls.fall_back():
                return self._sent(body, headers)
            raise
        return connection

    def __call__(self, payload, raw: bytes | None = None) -> Result:
        body = raw if raw is not None else json.dumps(payload).encode()
        headers = {k: v for k, v in self.headers}
        headers["Accept-Encoding"] = "identity"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        connection = self._sent(body, headers)
        response = connection.getresponse()
        kept = [(k, v) for k, v in response.getheaders() if k.lower() not in _BODY]
        if response.status < 400 and "text/event-stream" in (
            response.getheader("Content-Type") or ""
        ):

            def events():
                try:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            return
                        yield chunk
                finally:
                    connection.close()

            return Result(
                response.status, kept, events=events(), close=connection.close
            )
        data = response.read()
        connection.close()
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

        def _handle(self):
            if (
                self.path in ("/compache/stats", "/compache/health")
                and self.command == "GET"
            ):
                body = json.dumps({"status": "ok", **live.stats}).encode()
                return self._answer(200, [("Content-Type", "application/json")], body)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else None
            headers = {k.lower(): v for k, v in self.headers.items()}
            base = upstream or targets[_family(self.path, headers)]
            kept = [
                (k, v)
                for k, v in self.headers.items()
                if k.lower() not in _HOP and k.lower() != HEADER
            ]
            send = _Upstream(base, self.command, self.path, kept, timeout)
            body = None
            if (
                self.command == "POST"
                and raw
                and wire.dialect_for(self.path) is not None
            ):
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
            if body is None:
                result = send(None, raw)
            else:
                # The caller's bytes go out as they came whenever nothing in them was changed.
                result = live.exchange(
                    self.path,
                    body,
                    lambda payload: send(payload, raw if payload is body else None),
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

    server = Server((host, port), Handler)
    server.live = live
    if pings:
        start_pings(live)
    return server


def main(args) -> int:
    """`compache proxy`: run until interrupted."""
    live = Live(args.policy, record=args.record, sample=args.sample, notes=args.notes)
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
    where = f"http://{args.host}:{server.server_address[1]}"
    doing = []
    if live.policy is not None:
        doing.append(f"applying {args.policy}")
    if live.recorder is not None:
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
