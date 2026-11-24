"""compache behind a base URL: what a caller of the proxy sends and gets."""

import http.client
import json
import ssl
import tempfile
import threading
import time
import unittest
import urllib.parse
from unittest import mock

from compache import _tls
from compache.cli import main
from compache.live import Live
from compache.proxy import _family, _thread, _Upstream, serve
from compache.record import load
from fakes import NOTE_ASKED, FakeModel, noticed, read_stream
from fakes import serve as serve_fake
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


def post(url, path, body, headers=None):
    """One POST as an application sends it: the status, the headers and the body or the stream."""
    target = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
    payload = json.dumps(body).encode()
    connection.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, dict(response.getheaders()), data


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = fitted()
        cls.started = 0

    def start(self, script, **options):
        type(self).started += 1
        self.model = FakeModel(script)
        upstream, stop = serve_fake(self.model)
        self.addCleanup(stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.record = tmp.name
        self.clock = Clock(1_860_000_000.0 + self.started * 1e7)
        self.live = Live(
            options.pop("policy", self.policy),
            record=tmp.name,
            sample=1.0,
            notes=options.pop("notes", "apart"),
            clock=self.clock,
        )
        server = serve(self.live, port=0, upstream=upstream, **options)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def settled(self, turns):
        """A stream's books are closed just after its last byte: wait for them."""
        for _ in range(200):
            if self.live.stats["turns"] >= turns:
                return
            time.sleep(0.01)
        self.fail("the conversation was not recorded")

    def test_a_chat_completions_caller_changes_only_its_base_url(self):
        def noting(request):
            """A model that also answers a note asked of a finished answer, as a streamed one is."""
            messages = request["body"]["messages"]
            if NOTE_ASKED in request["last"] and messages[-2]["role"] == "assistant":
                return {"text": "<!-- refs: {R1.1} -->"}
            return looping(request)

        for stream in (False, True):
            with self.subTest(stream=stream):
                url = self.start(noting)
                headers = {
                    "Authorization": "Bearer the-callers-key",
                    "x-session-affinity": f"session-{stream}",
                }
                messages = [{"role": "system", "content": SYSTEM}]
                for n in range(1, 5):
                    messages.append({"role": "user", "content": f"Question {n}?"})
                    while True:
                        body = {
                            "model": "claude-sonnet-5-5",
                            "messages": messages,
                            "tools": [SEARCH],
                        }
                        if stream:
                            body["stream"] = True
                        status, got, data = post(
                            url, "/v1/chat/completions", body, headers
                        )
                        self.assertEqual(status, 200)
                        self.assertNotIn(b"refs", data)
                        if stream:
                            self.assertEqual(got["Content-Type"], "text/event-stream")
                            read = read_stream([data])
                            text = read["text"]
                            calls = [
                                {"id": f"call_{i}", "type": "function", "function": c}
                                for i, c in enumerate(read["calls"])
                            ]
                        else:
                            message = json.loads(data)["choices"][0]["message"]
                            text, calls = (
                                message["content"],
                                message.get("tool_calls") or [],
                            )
                        if not calls:
                            messages.append({"role": "assistant", "content": text})
                            break
                        messages.append(
                            {"role": "assistant", "content": text, "tool_calls": calls}
                        )
                        for call in calls:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": passage(n),
                                }
                            )
                    self.assertEqual(text, f"Answer {n}.")
                    self.settled(n)
                    self.live.drain()
                    self.clock.at += 3600
                sent = [json.dumps(b) for b in self.model.bodies]
                self.assertGreaterEqual(sum(map(noticed, sent)), 3)
                self.assertTrue(any(NOTE_ASKED in s for s in sent))
                # The caller's credentials go on to the provider, and compache's own header does not.
                seen = {k.lower(): v for k, v in self.model.headers[-1].items()}
                self.assertEqual(seen["authorization"], "Bearer the-callers-key")
                self.assertEqual(self.model.paths[-1], "/v1/chat/completions")
                threads, labels = load(self.record)
                self.assertEqual([len(t["runs"]) for t in threads], [4])
                self.assertEqual(len(labels), 4)

    def test_the_messages_api_and_what_is_not_a_model_call(self):
        url = self.start(claude_looping)
        headers = {
            "x-api-key": "the-callers-key",
            "anthropic-version": "2023-06-01",
            "x-compache-thread": "c1",
        }
        messages = []
        for n in range(1, 4):
            messages.append({"role": "user", "content": f"Question {n}?"})
            while True:
                body = {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 256,
                    "system": SYSTEM,
                    "tools": [CLAUDE_SEARCH],
                    "messages": messages,
                }
                status, _, data = post(url, "/v1/messages", body, headers)
                reply = json.loads(data)
                self.assertEqual(status, 200)
                messages.append({"role": "assistant", "content": reply["content"]})
                if reply["stop_reason"] != "tool_use":
                    break
                results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": passage(n),
                    }
                    for b in reply["content"]
                    if b["type"] == "tool_use"
                ]
                messages.append({"role": "user", "content": results})
            self.live.drain()
            self.clock.at += 3600
        self.assertNotIn("refs", json.dumps(messages))
        seen = {k.lower(): v for k, v in self.model.headers[-1].items()}
        self.assertEqual(seen["x-api-key"], "the-callers-key")
        self.assertNotIn("x-compache-thread", seen)
        before = len(self.model.bodies)
        # Another endpoint passes through byte for byte, and so does a refusal.
        status, _, data = post(
            url,
            "/v1/embeddings",
            {"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
        self.assertEqual((status, self.model.paths[-1]), (200, "/v1/embeddings"))
        self.assertEqual(len(self.model.bodies), before + 1)
        self.model.script = lambda request: {"status": 429, "error": "slow down"}
        status, _, data = post(
            url, "/v1/messages", {**body, "messages": messages[:1]}, headers
        )
        self.assertEqual(
            (status, json.loads(data)["error"]["message"]), (429, "slow down")
        )
        target = urllib.parse.urlsplit(url)
        connection = http.client.HTTPConnection(
            target.hostname, target.port, timeout=10
        )
        connection.request("GET", "/compache/stats")
        stats = json.loads(connection.getresponse().read())
        self.assertEqual((stats["status"], stats["turns"]), ("ok", 3))

    def test_a_request_leaves_the_same_whichever_way_compache_went_in(self):
        from test_live import ChatApp

        # With the note in the reply's own request, so that the requests come in one order.
        url = self.start(looping, notes="inline")
        headers = {"x-compache-thread": "same"}
        messages = [{"role": "system", "content": SYSTEM}]
        for n in range(1, 5):
            messages.append({"role": "user", "content": f"Question {n}?"})
            while True:
                body = {
                    "model": "claude-sonnet-5-5",
                    "messages": messages,
                    "tools": [SEARCH],
                }
                message = json.loads(
                    post(url, "/v1/chat/completions", body, headers)[2]
                )["choices"][0]["message"]
                if not message.get("tool_calls"):
                    messages.append(
                        {"role": "assistant", "content": message["content"]}
                    )
                    break
                messages.append(message)
                for call in message["tool_calls"]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": passage(n),
                        }
                    )
            self.clock.at += 3600
        through_proxy = self.model.bodies
        # The same conversation with compache called in process, as a wrapped client does.
        clock = Clock(self.clock.at + 1e7)
        direct = FakeModel(looping)
        live = Live(
            self.policy,
            record=tempfile.mkdtemp(),
            sample=1.0,
            notes="inline",
            clock=clock,
        )
        app = ChatApp(live, direct, thread="same", name="claude-sonnet-5-5")
        for n in range(1, 5):
            app.turn(f"Question {n}?", passage(n))
            clock.at += 3600
        self.assertEqual(len(through_proxy), len(direct.bodies))
        for one, other in zip(through_proxy, direct.bodies):
            self.assertEqual(one, other)
        self.assertGreater(sum(map(noticed, direct.bodies)), 0)

    def test_an_upstream_that_cannot_be_reached_is_an_error_the_caller_can_read(self):
        live = Live()
        server = serve(live, port=0, upstream="http://127.0.0.1:9")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        status, _, data = post(
            url,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(status, 502)
        self.assertIn("compache proxy", json.loads(data)["error"]["message"])

    def test_a_python_without_root_certificates_uses_another_bundle(self):
        other, seen = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT), []

        class Response:
            status = 200

            def getheaders(self):
                return [("Content-Type", "application/json")]

            def getheader(self, name):
                return dict(self.getheaders()).get(name)

            def read(self):
                return b'{"ok": true}'

        class Connection:
            def __init__(self, host, port=None, *, timeout=None, context=None):
                self.context = context

            def request(self, method, target, body=None, headers=None):
                seen.append(self.context)
                if self.context is None:
                    raise ssl.SSLCertVerificationError(
                        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
                    )

            def getresponse(self):
                return Response()

            def close(self):
                pass

        def call():
            send = _Upstream("https://provider.example", "POST", "/v1/models", [], 30)
            return send({"messages": []})

        self.addCleanup(_tls.FOUND.update, context=_tls.FOUND["context"])
        _tls.FOUND["context"] = None
        with mock.patch("compache.proxy.http.client.HTTPSConnection", Connection):
            with mock.patch("compache._tls._other", return_value=None):
                with self.assertRaises(ssl.SSLCertVerificationError):
                    call()
            with mock.patch("compache._tls._other", return_value=other):
                self.assertEqual(call().json, {"ok": True})
                self.assertEqual(call().json, {"ok": True})
        # Python's own certificates first, the other bundle from then on.
        self.assertEqual(seen, [None, None, other, other])

    def test_where_a_request_goes_and_what_names_its_conversation(self):
        self.assertEqual(_family("/v1/chat/completions", {}), "openai")
        self.assertEqual(_family("/v1/responses", {}), "openai")
        self.assertEqual(_family("/v1/messages", {}), "anthropic")
        self.assertEqual(_family("/v1beta/models/g:generateContent", {}), "gemini")
        self.assertEqual(_family("/v1/models", {"x-api-key": "k"}), "anthropic")
        self.assertEqual(_family("/v1beta/models", {"x-goog-api-key": "k"}), "gemini")
        self.assertEqual(_family("/v1/models", {"authorization": "Bearer k"}), "openai")
        self.assertEqual(
            _thread({"x-compache-thread": "a", "x-session-affinity": "b"}, {}), "a"
        )
        # What the harnesses themselves send: Claude Code, Codex, OpenCode.
        for name in ("x-claude-code-session-id", "session-id", "x-opencode-session-id"):
            self.assertEqual(_thread({name: "c"}, {"prompt_cache_key": "d"}), "c")
        self.assertEqual(_thread({"session_id": "s"}, {"prompt_cache_key": "p"}), "s")
        self.assertEqual(_thread({}, {"prompt_cache_key": "p"}), "p")
        self.assertIsNone(_thread({}, {"messages": []}))

    def test_the_command_refuses_what_it_cannot_run(self):
        self.assertEqual(
            main(["proxy", "--policy", "/no/such/policy.json", "--port", "0"]), 1
        )


if __name__ == "__main__":
    unittest.main()
