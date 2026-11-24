"""`compache wrap`: an agent started behind compache, told where its model is."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import compache
from compache import doctor, launch
from compache.cli import main
from compache.proxy import _decoded, serve as proxied
from compache.record import load, load_calls
from fakes import FakeModel, serve
from test_agent_live import worker

# An agent that reads its base URL from the environment and works through a loop of tool calls.
AGENT = """
import json, os, sys, urllib.request
base = os.environ["ANTHROPIC_BASE_URL"]
said = [{"role": "user", "content": "Read " + sys.argv[-1]}]
for k in range(60):
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": said,
            "tools": [{"name": "read", "input_schema": {"type": "object"}}]}
    request = urllib.request.Request(base + "/v1/messages", json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request))
    said.append({"role": "assistant", "content": reply["content"]})
    calls = [b for b in reply["content"] if b["type"] == "tool_use"]
    if not calls:
        break
    text = "result %d: " % k + "a line of file number %d that only it holds\\n" % k * 60
    said.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": text} for c in calls]})
sys.exit(7)
"""


class SetupTests(unittest.TestCase):
    """What each agent is told, and where its requests go on to."""

    def told(self, name, env=None):
        return launch.prepared(
            launch.AGENTS[name],
            "http://127.0.0.1:9999",
            {"HOME": "/none", **(env or {})},
        )

    def test_claude_code(self):
        changes, ahead, upstreams = self.told("claude")
        self.assertEqual(changes["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9999")
        self.assertEqual(changes["ENABLE_TOOL_SEARCH"], "true")
        self.assertEqual((ahead, upstreams), ([], {}))
        changes, _, _ = self.told("claude", {"ENABLE_TOOL_SEARCH": "auto:5"})
        self.assertNotIn("ENABLE_TOOL_SEARCH", changes)

    def test_a_base_url_that_is_set_already_is_where_requests_go_on_to(self):
        changes, _, upstreams = self.told(
            "claude", {"ANTHROPIC_BASE_URL": "https://gateway.example/api/"}
        )
        self.assertEqual(changes["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9999/api")
        self.assertEqual(upstreams, {"anthropic": "https://gateway.example"})
        changes, _, upstreams = self.told(
            "qwen", {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"}
        )
        self.assertEqual(changes["OPENAI_BASE_URL"], "http://127.0.0.1:9999/api/v1")
        self.assertEqual(upstreams, {"openai": "https://openrouter.ai"})

    def test_codex_with_a_key(self):
        changes, ahead, upstreams = self.told("codex", {"OPENAI_API_KEY": "sk-x"})
        said = " ".join(ahead)
        self.assertEqual(changes, {})
        self.assertEqual(upstreams, {"openai": "https://api.openai.com"})
        self.assertIn('base_url="http://127.0.0.1:9999/v1"', said)
        self.assertIn('env_key="OPENAI_API_KEY"', said)
        self.assertIn("supports_websockets=false", said)
        self.assertNotIn("sk-x", said)

    def test_codex_with_a_key_of_its_own_or_a_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            auth = Path(folder) / "auth.json"
            auth.write_text(json.dumps({"auth_mode": "apikey"}))
            _, ahead, upstreams = self.told("codex", {"CODEX_HOME": folder})
            self.assertIn("requires_openai_auth=true", " ".join(ahead))
            self.assertEqual(upstreams, {"openai": "https://api.openai.com"})
            for found in ({"auth_mode": "chatgpt"}, {"tokens": {"id_token": "x"}}):
                auth.write_text(json.dumps(found))
                _, ahead, upstreams = self.told("codex", {"CODEX_HOME": folder})
                said = " ".join(ahead)
                self.assertIn(
                    'openai_base_url="http://127.0.0.1:9999/backend-api/codex"', said
                )
                self.assertNotIn("model_provider", said)
                self.assertEqual(upstreams, {"openai": "https://chatgpt.com"})
            auth.write_text("{")
            _, ahead, _ = self.told("codex", {"CODEX_HOME": folder})
            self.assertIn("model_provider", " ".join(ahead))

    def test_the_sign_in_of_a_plan_goes_to_its_own_place_and_nowhere_else(self):
        others = (
            {},
            {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"},
            {"OPENAI_API_BASE": "http://localhost:11434/v1"},
        )
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "auth.json").write_text('{"auth_mode": "chatgpt"}')
            for word in ("codex", "/opt/homebrew/bin/codex"):
                for env in others:
                    name, agent, _ = launch._given([word])
                    env = {"HOME": "/none", "CODEX_HOME": folder, **env}
                    _, ahead, upstreams = launch.prepared(agent, "http://h:1", env)
                    self.assertEqual(upstreams, {"openai": "https://chatgpt.com"})
                    self.assertIn(
                        'openai_base_url="http://h:1/backend-api/codex"', ahead
                    )

    def test_codex_with_a_provider_of_its_own(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.toml").write_text(
                'profile = "work"\n'
                '[profiles.work]\nmodel_provider = "gateway"\n'
                "[model_providers.gateway]\n"
                'base_url = "https://gateway.example/openai/v1"\nenv_key = "KEY"\n'
            )
            _, ahead, upstreams = self.told("codex", {"CODEX_HOME": folder})
            said = " ".join(ahead)
            self.assertIn(
                'model_providers.gateway.base_url="http://127.0.0.1:9999/openai/v1"',
                said,
            )
            self.assertNotIn("model_provider=", said)
            self.assertEqual(upstreams, {"openai": "https://gateway.example"})
            (Path(folder) / "config.toml").write_text(
                'openai_base_url = "https://eu.example/v2"'
            )
            _, ahead, upstreams = self.told("codex", {"CODEX_HOME": folder})
            self.assertIn('base_url="http://127.0.0.1:9999/v2"', " ".join(ahead))
            self.assertEqual(upstreams, {"openai": "https://eu.example"})

    def test_goose_s_own_hosts_are_where_its_requests_go_on_to(self):
        changes, _, upstreams = self.told(
            "goose",
            {
                "OPENAI_HOST": "https://gateway.example",
                "OPENAI_BASE_URL": "https://others.example/v1",
            },
        )
        self.assertEqual(changes["OPENAI_HOST"], "http://127.0.0.1:9999")
        self.assertEqual(upstreams, {"openai": "https://gateway.example"})

    def test_behind_a_proxy_of_the_machine_s_this_one_is_reached_directly(self):
        self.assertNotIn("NO_PROXY", self.told("claude")[0])
        changes, _, _ = self.told(
            "claude", {"HTTPS_PROXY": "http://proxy:3128", "NO_PROXY": "intranet"}
        )
        for name in ("no_proxy", "NO_PROXY"):
            self.assertEqual(changes[name], "intranet,127.0.0.1,localhost")

    def test_opencode_keeps_the_configuration_it_is_given(self):
        given = {"autoupdate": False, "provider": {"openai": {"options": {"x": 1}}}}
        changes, _, _ = self.told(
            "opencode", {"OPENCODE_CONFIG_CONTENT": json.dumps(given)}
        )
        made = json.loads(changes["OPENCODE_CONFIG_CONTENT"])
        self.assertFalse(made["autoupdate"])
        self.assertEqual(made["provider"]["openai"]["options"]["x"], 1)
        self.assertEqual(
            made["provider"]["anthropic"]["options"]["baseURL"],
            "http://127.0.0.1:9999/v1",
        )

    def test_gemini_aider_and_goose(self):
        self.assertEqual(
            self.told("gemini")[0]["GOOGLE_GEMINI_BASE_URL"], "http://127.0.0.1:9999"
        )
        changes = self.told("aider")[0]
        self.assertEqual(changes["OPENAI_API_BASE"], "http://127.0.0.1:9999/v1")
        self.assertEqual(changes["ANTHROPIC_API_BASE"], "http://127.0.0.1:9999")
        self.assertEqual(self.told("goose")[0]["OPENAI_HOST"], "http://127.0.0.1:9999")

    def test_any_other_command_is_told_by_the_variables_of_the_sdks(self):
        name, found, words = launch._given(["--", "my-agent", "--fast", "--", "x"])
        self.assertEqual((name, words), ("my-agent", ["--fast", "--", "x"]))
        changes, _, _ = launch.prepared(found, "http://127.0.0.1:9999", {})
        self.assertEqual(changes["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9999")
        self.assertEqual(changes["OPENAI_BASE_URL"], "http://127.0.0.1:9999/v1")
        name, found, words = launch._given(["claude", "--", "-p", "hi"])
        self.assertEqual(found, launch.AGENTS["claude"])
        self.assertEqual(words, ["-p", "hi"])
        # A path to a known agent is that agent.
        name, found, _ = launch._given(["/opt/bin/codex"])
        self.assertEqual((name, found.command), ("codex", "/opt/bin/codex"))
        with self.assertRaises(ValueError):
            launch._given([])


class Wrapped(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.home = Path(self.folder.name)
        script = self.home / "claude"
        script.write_text(f"#!{sys.executable}\n{AGENT}")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.model = FakeModel(worker(40))
        self.url, stop = serve(self.model)
        self.addCleanup(stop)
        self.env = {
            "PATH": f"{self.home}{os.pathsep}{os.environ['PATH']}",
            "COMPACHE_HOME": str(self.home / "kept"),
        }
        before = {k: os.environ.get(k) for k in (*self.env, "ANTHROPIC_BASE_URL")}
        os.environ.update(self.env)
        os.environ.pop("ANTHROPIC_BASE_URL", None)

        def restore():
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)

    def wrap(self, *words):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(["wrap", *map(str, words)])
        return status, out.getvalue(), err.getvalue()

    def ask(self, *words):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = main([str(w) for w in words])
        return status, out.getvalue()


class RunTests(Wrapped):
    def test_it_runs_the_agent_learns_and_says_what_was_saved(self):
        status, _, err = self.wrap("--upstream", self.url, "claude", "--", "files")
        self.assertEqual(status, 7, "the agent's own status")
        self.assertIn("compache: 41 calls", err)
        self.assertIn("less", err)
        folder = self.home / "kept" / "claude"
        threads, _ = load(folder)
        self.assertEqual(threads[0]["runs"][0]["calls"], 41)
        self.assertTrue(any(r.get("left_bytes", 0) > 0 for r in load_calls(folder)))
        self.assertIn("reopen", json.dumps(self.model.bodies[0]["tools"]))
        status, out = self.ask("stats")
        self.assertIn("41 calls in 1 conversation", out)
        self.assertIn("the default, until there are 200 calls", out)
        status, out = self.ask("stats", "--json")
        self.assertEqual(
            json.loads(out)[str(folder)]["models"]["claude-haiku-5-5"]["calls"], 41
        )

    def test_watching_changes_nothing(self):
        status, _, err = self.wrap(
            "--watch", "--upstream", self.url, "claude", "--", "files"
        )
        self.assertEqual(status, 7)
        self.assertNotIn("reopen", json.dumps(self.model.bodies))
        self.assertIn("compache: 41 calls", err)

    def test_a_policy_of_one_s_own(self):
        path = self.home / "policy.json"
        compache.default_policy("claude-haiku-5-5", agent=True).save(path)
        status, _, _ = self.wrap(
            "--policy", path, "--quiet", "--upstream", self.url, "claude", "files"
        )
        self.assertEqual(status, 7)
        self.assertTrue(any("ctx:" in json.dumps(b) for b in self.model.bodies))

    def test_an_agent_that_is_not_there(self):
        status, _, err = self.wrap("--", "no-such-agent-here")
        self.assertEqual(status, 1)
        self.assertIn("not on the PATH", err)

    def test_what_to_do_by_hand(self):
        status, out, _ = self.wrap("--print", "claude")
        self.assertEqual(status, 0)
        self.assertIn("compache proxy --port 8787 --learn", out)
        self.assertIn("export ANTHROPIC_BASE_URL='http://127.0.0.1:8787'", out)


class DoctorTests(Wrapped):
    def test_it_says_what_is_here_and_what_to_run(self):
        self.wrap("--upstream", self.url, "claude", "--", "files")
        status, out = self.ask("doctor", "--port", "1")
        self.assertEqual(status, 0)
        self.assertIn("a request goes through a local proxy and back", out)
        self.assertIn("Claude Code: compache wrap claude", out)
        self.assertIn("calls on record", out)
        self.assertNotIn("problem", out)
        os.environ["ANTHROPIC_API_KEY"] = "sk-secret"
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY", None)
        status, out = self.ask("doctor", "--json", "--port", "1")
        self.assertNotIn("sk-secret", out)
        self.assertIn("ANTHROPIC_API_KEY", out)
        self.assertEqual(json.loads(out)["problems"], 0)

    def test_a_record_that_cannot_be_read_is_a_problem(self):
        folder = self.home / "kept" / "broken"
        folder.mkdir(parents=True)
        (folder / "turns.jsonl").write_text("{}\n")
        (folder / "calls.jsonl").write_text("not json\n")
        found = doctor.checked(port=1)
        self.assertTrue(any(f["level"] == "problem" for f in found))
        row = {"v": 1, "thread": "t", "turn": 1, "kind": "answer", "at": 1.0}
        (folder / "calls.jsonl").write_text(json.dumps(row) + "\nnot json\n")
        found = doctor.checked(port=1)
        self.assertFalse(any(f["level"] == "problem" for f in found))
        self.assertTrue(any("1 lines cannot be read" in f["text"] for f in found))


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        url, stop = serve(self.model)
        self.addCleanup(stop)
        self.server = proxied(compache.Live(), port=0, upstream=url)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def post(self, data, headers=None):
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data,
            {"Content-Type": "application/json", **(headers or {})},
        )
        return json.load(urllib.request.urlopen(request))

    def test_a_compressed_request_is_read(self):
        raw = gzip.compress(json.dumps(self.body).encode())
        self.assertEqual(_decoded(raw, "gzip"), json.dumps(self.body).encode())
        self.assertEqual(_decoded(b"{}", None), b"{}")
        with self.assertRaises(ValueError):
            _decoded(b"", "br")

    def test_a_body_that_cannot_be_read_goes_on_as_it_came(self):
        heard = []

        class Other(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self):
                size = int(self.headers.get("Content-Length") or 0)
                heard.append((self.command, self.rfile.read(size)))
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET = do_DELETE = answer

        other = ThreadingHTTPServer(("127.0.0.1", 0), Other)
        self.addCleanup(other.server_close)
        self.addCleanup(other.shutdown)
        threading.Thread(target=other.serve_forever, daemon=True).start()
        self.server.forward(f"http://127.0.0.1:{other.server_address[1]}", {})
        path = "/v1/chat/completions"
        for encoding in ("gzip", "zstd", "br"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post(b"not that", {"Content-Encoding": encoding})
            self.assertEqual(caught.exception.code, 400, "the provider's own answer")
            caught.exception.close()
        for method in ("GET", "DELETE"):
            request = urllib.request.Request(self.base + path + "/x", method=method)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            caught.exception.close()
        self.assertEqual(
            heard, [("POST", b"not that")] * 3 + [("GET", b""), ("DELETE", b"")]
        )

    def test_a_machine_s_proxy_is_gone_through(self):
        asked = []

        class Machine(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                size = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(size))
                asked.append((self.path, self.headers["Proxy-Authorization"], body))
                status, reply = FakeModel().chat(body)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_CONNECT(self):
                asked.append(self.path)
                self.send_response(403)
                self.end_headers()

        machine = ThreadingHTTPServer(("127.0.0.1", 0), Machine)
        self.addCleanup(machine.server_close)
        self.addCleanup(machine.shutdown)
        threading.Thread(target=machine.serve_forever, daemon=True).start()
        name = f"http://me:it%40s@127.0.0.1:{machine.server_address[1]}"
        given = {"http_proxy": name, "https_proxy": name, "no_proxy": ""}
        before = {k: os.environ.get(k) for k in given}
        os.environ.update(given)

        def restore():
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def ask():
            request = urllib.request.Request(
                self.base + "/v1/chat/completions",
                json.dumps(self.body).encode(),
                {"Content-Type": "application/json"},
            )
            return json.load(opener.open(request))

        self.server.forward("http://provider.example", {})
        self.assertTrue(ask()["choices"])
        self.server.forward("https://provider.example", {})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            ask()
        caught.exception.close()
        path = "http://provider.example/v1/chat/completions"
        self.assertEqual(asked[0], (path, "Basic bWU6aXRAcw==", self.body))
        self.assertEqual(asked[1], "provider.example:443")

    def test_a_request_in_chunks_is_read(self):
        pieces = iter(
            [json.dumps(self.body).encode()[:9], json.dumps(self.body).encode()[9:]]
        )
        reply = self.post(pieces, {"Transfer-Encoding": "chunked"})
        self.assertEqual(reply["choices"][0]["message"]["content"], "Answer 1.")
        self.assertEqual(self.model.bodies, [self.body])

    def test_a_websocket_is_turned_away_at_once(self):
        request = urllib.request.Request(
            self.base + "/v1/responses",
            headers={"Upgrade": "websocket", "Connection": "Upgrade"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 426)
        caught.exception.close()


if __name__ == "__main__":
    unittest.main()
