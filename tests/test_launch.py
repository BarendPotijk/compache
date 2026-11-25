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
from compache import doctor, launch, proxy
from compache.cli import main
from compache.proxy import _decoded, serve as proxied
from compache.record import load, load_calls, thread_name
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


class Told(unittest.TestCase):
    """A home and a project of its own, so that no agent's files are there but those of the test."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.home = Path(folder.name).resolve()
        self.project = self.home / "work" / "project"
        self.project.mkdir(parents=True)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.project)

    def put(self, path, text):
        file = self.home / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text if isinstance(text, str) else json.dumps(text))

    def told(self, name, env=None):
        return launch.prepared(
            launch.AGENTS[name],
            "http://127.0.0.1:9999",
            {"HOME": str(self.home), **(env or {})},
        )


class SetupTests(Told):
    """What each agent is told, and where its requests go on to."""

    def test_claude_code(self):
        changes, ahead, upstreams = self.told("claude")
        self.assertEqual(changes["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9999")
        self.assertEqual(changes["ENABLE_TOOL_SEARCH"], "true")
        self.assertEqual((ahead, upstreams), ([], {}))
        changes, _, _ = self.told("claude", {"ENABLE_TOOL_SEARCH": "auto:5"})
        self.assertNotIn("ENABLE_TOOL_SEARCH", changes)

    def test_claude_code_on_other_clouds(self):
        changes, _, upstreams = self.told(
            "claude", {"CLAUDE_CODE_USE_VERTEX": "1", "CLOUD_ML_REGION": "us-east5"}
        )
        base = {"ANTHROPIC_VERTEX_BASE_URL": "http://127.0.0.1:9999/v1"}
        self.assertEqual(changes, base)
        to = "https://us-east5-aiplatform.googleapis.com"
        self.assertEqual(upstreams, {"anthropic": to})
        for region, host in (
            ("global", "aiplatform.googleapis.com"),
            ("eu", "aiplatform.eu.rep.googleapis.com"),
        ):
            env = {"CLAUDE_CODE_USE_VERTEX": "true", "CLOUD_ML_REGION": region}
            _, _, upstreams = self.told("claude", env)
            self.assertEqual(upstreams, {"anthropic": f"https://{host}"})
        changes, _, upstreams = self.told(
            "claude",
            {"CLAUDE_CODE_USE_FOUNDRY": "1", "ANTHROPIC_FOUNDRY_RESOURCE": "ours"},
        )
        base = {"ANTHROPIC_FOUNDRY_BASE_URL": "http://127.0.0.1:9999/anthropic"}
        base["ANTHROPIC_FOUNDRY_RESOURCE"] = ""
        self.assertEqual(changes, base)
        to = "https://ours.services.ai.azure.com"
        self.assertEqual(upstreams, {"anthropic": to})
        with self.assertRaisesRegex(ValueError, "AWS"):
            self.told("claude", {"CLAUDE_CODE_USE_BEDROCK": "1"})

    def test_credentials_go_to_the_api_they_are_for(self):
        family = proxy._family
        google = "/v1/projects/p/locations/global/publishers"
        counted = google + "/anthropic/models/count-tokens:rawPredict"
        bearer = {"authorization": "Bearer x"}
        for path, headers, speaks in (
            ("/v1/messages", bearer, "anthropic"),
            (
                "/api/oauth/usage",
                {"authorization": "Bearer sk-ant-oat01-x"},
                "anthropic",
            ),
            ("/v1/files", {"x-api-key": "sk-ant-x", **bearer}, "anthropic"),
            (counted, bearer, "anthropic"),
            (google + "/google/models/gemini:countTokens", bearer, "gemini"),
            ("/v1/embeddings", bearer, "openai"),
        ):
            self.assertEqual(family(path, headers), speaks, path)
        self.assertEqual(family("/anything", bearer, "anthropic"), "anthropic")

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
            "goose", {"OPENAI_HOST": "https://gateway.example/openai"}
        )
        self.assertEqual(changes["OPENAI_HOST"], "http://127.0.0.1:9999/openai")
        self.assertEqual(upstreams, {"openai": "https://gateway.example"})
        changes, _, upstreams = self.told(
            "goose", {"OPENAI_BASE_URL": "https://gateway.example/v1"}
        )
        self.assertEqual(changes["OPENAI_BASE_URL"], "http://127.0.0.1:9999/v1")
        self.assertEqual(upstreams, {"openai": "https://gateway.example"})
        both = {"OPENAI_HOST": "https://one.example"}
        with self.assertRaisesRegex(ValueError, "different things"):
            self.told("goose", {**both, "OPENAI_BASE_URL": "https://two.example/v1"})

    def test_what_an_agent_does_not_read_says_nothing_of_it(self):
        """LiteLLM's variables are for what LiteLLM calls."""
        for name in ("claude", "opencode", "qwen", "gemini", "pi"):
            _, _, upstreams = self.told(
                name,
                {
                    "ANTHROPIC_API_BASE": "https://others.example",
                    "OPENAI_API_BASE": "https://others.example/v1",
                },
            )
            self.assertEqual(upstreams, {}, name)
        _, _, upstreams = self.told(
            "aider", {"ANTHROPIC_API_BASE": "https://gateway.example"}
        )
        self.assertEqual(upstreams, {"anthropic": "https://gateway.example"})
        _, found, _ = launch._given(["--", "my-agent"])
        changes, _, upstreams = launch.prepared(
            found, "http://127.0.0.1:9999", {"OPENAI_API_BASE": "https://one.example"}
        )
        self.assertEqual(upstreams, {})
        self.assertNotIn("OPENAI_API_BASE", changes)
        with self.assertRaisesRegex(ValueError, "OPENAI_API_ENDPOINT"):
            self.told("crush", {"OPENAI_BASE_URL": "https://gateway.example/v1"})

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

    def test_kilo_takes_the_same_configuration_under_a_name_of_its_own(self):
        (variable,) = self.told("kilo")[0]
        self.assertEqual(variable, "KILO_CONFIG_CONTENT")

    def test_copilot_with_a_model_of_one_s_own(self):
        with self.assertRaisesRegex(ValueError, "COPILOT_PROVIDER_BASE_URL"):
            self.told("copilot")
        for kind, api in (("", "openai"), ("azure", "openai"), ("anthropic",) * 2):
            changes, _, upstreams = self.told(
                "copilot",
                {
                    "COPILOT_PROVIDER_BASE_URL": "https://eu.example/v1",
                    "COPILOT_PROVIDER_TYPE": kind,
                },
            )
            base = changes["COPILOT_PROVIDER_BASE_URL"]
            self.assertEqual(base, "http://127.0.0.1:9999/v1")
            self.assertEqual(upstreams, {api: "https://eu.example"})

    def test_pi_is_told_by_an_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            before = os.environ.get("COMPACHE_HOME")
            os.environ["COMPACHE_HOME"] = folder
            try:
                changes, ahead, _ = self.told("pi")
            finally:
                os.environ.pop("COMPACHE_HOME")
                if before is not None:
                    os.environ["COMPACHE_HOME"] = before
            self.assertEqual(ahead[0], "-e")
            self.assertIn("registerProvider", Path(ahead[1]).read_text())
        told = json.loads(changes["COMPACHE_BASES"])
        self.assertEqual(told["anthropic"], "http://127.0.0.1:9999")
        self.assertEqual(told["openai"], "http://127.0.0.1:9999/v1")

    def test_openhands(self):
        changes, ahead, upstreams = self.told("openhands")
        self.assertEqual(changes, {"LLM_BASE_URL": "http://127.0.0.1:9999"})
        self.assertEqual((ahead, upstreams), (["--override-with-envs"], {}))
        changes, _, upstreams = self.told(
            "openhands", {"LLM_BASE_URL": "https://gateway.example/api/v1"}
        )
        self.assertEqual(changes, {"LLM_BASE_URL": "http://127.0.0.1:9999/api/v1"})
        self.assertEqual(set(upstreams.values()), {"https://gateway.example"})

    def test_crush(self):
        changes, _, upstreams = self.told(
            "crush", {"OPENAI_API_ENDPOINT": "https://gateway.example/api/v1"}
        )
        self.assertEqual(
            changes,
            {
                "OPENAI_API_ENDPOINT": "http://127.0.0.1:9999/api/v1",
                "ANTHROPIC_API_ENDPOINT": "http://127.0.0.1:9999",
            },
        )
        self.assertEqual(upstreams, {"openai": "https://gateway.example"})

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


class OwnFilesTests(Told):
    """Where an agent's own files name a server, its requests and its key go on to that."""

    GATEWAY = "https://gateway.example"

    def test_files_that_name_a_server(self):
        cases = (
            (
                "qwen",
                ".qwen/.env",
                "OPENAI_API_KEY=k\nexport OPENAI_BASE_URL='{0}/mode/v1'\n",
                {"openai": self.GATEWAY},
                ("OPENAI_BASE_URL", "/mode/v1"),
            ),
            (
                "qwen",
                ".qwen/settings.json",
                '{{"security": {{"auth": {{"baseUrl": "{0}/v1"}}}}}}',
                {"openai": self.GATEWAY},
                ("OPENAI_BASE_URL", "/v1"),
            ),
            (
                "gemini",
                ".gemini/.env",
                "GOOGLE_GEMINI_BASE_URL={0}\n",
                {"gemini": self.GATEWAY},
                ("GOOGLE_GEMINI_BASE_URL", ""),
            ),
            (
                "goose",
                ".config/goose/config.yaml",
                "GOOSE_PROVIDER: openai\nOPENAI_HOST: {0}/x\n",
                {"openai": self.GATEWAY},
                ("OPENAI_HOST", "/x"),
            ),
            (
                "goose",
                ".config/goose/config.yaml",
                'OPENAI_BASE_URL: "{0}/v1"\n',
                {"openai": self.GATEWAY},
                ("OPENAI_BASE_URL", "/v1"),
            ),
            (
                "openhands",
                ".openhands/agent_settings.json",
                '{{"llm": {{"model": "litellm_proxy/x", "base_url": "{0}/v1"}}}}',
                {"openai": self.GATEWAY, "anthropic": self.GATEWAY},
                ("LLM_BASE_URL", "/v1"),
            ),
        )
        for name, path, text, goes, (variable, behind) in cases:
            with self.subTest(agent=name, file=path):
                self.put(path, text.format(self.GATEWAY))
                changes, _, upstreams = self.told(name)
                self.assertEqual(upstreams, goes)
                self.assertEqual(changes[variable], "http://127.0.0.1:9999" + behind)
                (self.home / path).unlink()

    def test_what_is_in_the_environment_comes_first(self):
        self.put(".config/goose/config.yaml", "OPENAI_HOST: https://others.example\n")
        _, _, upstreams = self.told("goose", {"OPENAI_HOST": self.GATEWAY})
        self.assertEqual(upstreams, {"openai": self.GATEWAY})

    def test_opencode_s_configuration(self):
        self.put(
            ".config/opencode/opencode.jsonc",
            """{
              // which may hold comments, and "https://in.a.string" that is none
              "provider": {
                "openai": {"options": {"baseURL": "https://one.example/v1", "apiKey": "k"}},
                "anthropic": {"options": {"baseURL": "{env:OURS}/v1"},},
              }, /* and commas */
            }""",
        )
        changes, _, upstreams = self.told("opencode", {"OURS": self.GATEWAY})
        self.assertEqual(
            upstreams, {"openai": "https://one.example", "anthropic": self.GATEWAY}
        )
        told = json.loads(changes["OPENCODE_CONFIG_CONTENT"])["provider"]
        bases = {name: own["options"]["baseURL"] for name, own in told.items()}
        self.assertEqual(
            bases,
            {
                "openai": "http://127.0.0.1:9999/v1",
                "anthropic": "http://127.0.0.1:9999/v1",
                "google": "http://127.0.0.1:9999/v1beta",
            },
        )
        # A project's comes after the user's.
        options = {"options": {"baseURL": "https://two.example/v1"}}
        self.put("work/opencode.json", {"provider": {"openai": options}})
        self.assertEqual(self.told("opencode")[2]["openai"], "https://two.example")
        self.put("work/opencode.json", "{ not json")
        with self.assertRaisesRegex(ValueError, "could not be read"):
            self.told("opencode")

    def test_opencode_s_variables_hold_the_version(self):
        changes, _, upstreams = self.told(
            "opencode", {"ANTHROPIC_BASE_URL": self.GATEWAY + "/v1"}
        )
        self.assertEqual(upstreams, {"anthropic": self.GATEWAY})
        told = json.loads(changes["OPENCODE_CONFIG_CONTENT"])["provider"]
        self.assertEqual(
            told["anthropic"]["options"]["baseURL"], "http://127.0.0.1:9999/v1"
        )

    def test_pi_s_models(self):
        providers = {"anthropic": {"baseUrl": self.GATEWAY + "/a", "apiKey": "k"}}
        self.put(".pi/agent/models.json", {"providers": providers})
        before = os.environ.get("COMPACHE_HOME")
        os.environ["COMPACHE_HOME"] = str(self.home / "kept")
        try:
            changes, _, upstreams = self.told("pi")
        finally:
            os.environ.pop("COMPACHE_HOME")
            if before is not None:
                os.environ["COMPACHE_HOME"] = before
        self.assertEqual(upstreams, {"anthropic": self.GATEWAY})
        bases = json.loads(changes["COMPACHE_BASES"])
        self.assertEqual(bases["anthropic"], "http://127.0.0.1:9999/a")
        self.assertEqual(bases["openai"], "http://127.0.0.1:9999/v1")

    def test_claude_code_s_settings_come_before_its_environment(self):
        settings = {"env": {"ANTHROPIC_BASE_URL": self.GATEWAY + "/api"}}
        self.put(".claude/settings.json", settings)
        _, ahead, upstreams = self.told("claude")
        self.assertEqual(upstreams, {"anthropic": self.GATEWAY})
        self.assertEqual(ahead[0], "--settings")
        base = json.loads(ahead[1])["env"]["ANTHROPIC_BASE_URL"]
        self.assertEqual(base, "http://127.0.0.1:9999/api")

    def test_a_project_s_file_is_not_followed(self):
        """It may come with somebody else's repository."""
        self.put("work/project/.qwen/.env", f"OPENAI_BASE_URL={self.GATEWAY}/v1\n")
        with self.assertRaisesRegex(ValueError, "OPENAI_BASE_URL"):
            self.told("qwen")
        _, _, upstreams = self.told("qwen", {"OPENAI_BASE_URL": "https://one.example"})
        self.assertEqual(upstreams, {"openai": "https://one.example"})
        # It is the only one that is read, so one that says nothing leaves nothing said.
        self.put("work/project/.qwen/.env", "OTHER=1\n")
        self.put(".qwen/.env", f"OPENAI_BASE_URL={self.GATEWAY}/v1\n")
        self.assertEqual(self.told("qwen")[2], {})

    def test_codex_on_a_provider_of_its_own(self):
        self.put(".codex/config.toml", 'model_provider = "ollama"\n')
        with self.assertRaisesRegex(ValueError, "ollama"):
            self.told("codex", {"OPENAI_API_KEY": "sk-x"})

    def test_openhands_on_openrouter(self):
        changes, _, upstreams = self.told(
            "openhands", {"LLM_MODEL": "openrouter/qwen/qwen3-coder"}
        )
        self.assertEqual(upstreams["openai"], "https://openrouter.ai")
        self.assertEqual(changes["LLM_BASE_URL"], "http://127.0.0.1:9999/api/v1")


class KeyTests(unittest.TestCase):
    """A key that is none of a provider's does not go to that provider because nothing else is known."""

    def setUp(self):
        self.model = FakeModel(lambda request: {"text": "Hello."})
        self.url, stop = serve(self.model)
        self.addCleanup(stop)
        before = dict(proxy.DEFAULTS)
        self.addCleanup(proxy.DEFAULTS.update, before)
        proxy.DEFAULTS.update({api: self.url for api in before})
        self.server = proxied(compache.Live(), port=0, pings=False)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def ask(self, path, headers, body=None):
        body = body or {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
        request = urllib.request.Request(
            self.base + path,
            json.dumps({"max_tokens": 9, **body}).encode(),
            {"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request) as reply:
                return reply.status, json.load(reply)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_only_its_own_keys_go_to_a_provider(self):
        self.server.forward(None, {}, "openai", careful=True)
        chat, claude = "/v1/chat/completions", "/v1/messages"
        gemini = "/v1beta/models/g:generateContent"
        asks = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        with contextlib.redirect_stderr(io.StringIO()) as said:
            for path, headers, goes in (
                (chat, {"Authorization": "Bearer sk-abcT3Blbk" + "FJdef"}, True),
                (chat, {"Authorization": "Bearer sk-proj-abc"}, True),
                (chat, {"Authorization": "Bearer sk-or-v1-abc"}, False),
                (chat, {"Authorization": "Bearer sk-1234"}, False),
                (chat, {}, True),
                (claude, {"x-api-key": "sk-ant-api03-abc"}, True),
                (claude, {"Authorization": "Bearer sk-ant-oat01-abc"}, True),
                (claude, {"x-api-key": "sk-GATEWAY"}, False),
                (claude, {"Authorization": "Bearer corp-token"}, False),
                (gemini, {"x-goog-api-key": "AIzaabc"}, True),
                (gemini, {"x-goog-api-key": "corp"}, False),
                (gemini + "?key=corp", {}, False),
            ):
                before = len(self.model.bodies)
                body = asks if path.startswith("/v1beta") else None
                status, reply = self.ask(path, headers, body)
                self.assertEqual(status, 200 if goes else 403, (path, headers))
                self.assertEqual(len(self.model.bodies) - before, int(goes))
                if not goes:
                    self.assertIn("--upstream", reply["error"]["message"])
        # Said once for each.
        self.assertEqual(said.getvalue().count("does not look like one"), 3)

    def test_where_it_is_known_where_they_go_any_key_goes(self):
        key = {"Authorization": "Bearer sk-1234"}
        for everything, named in ((None, {"openai": self.url}), (self.url, {})):
            self.server.forward(everything, named, "openai", careful=True)
            self.assertEqual(self.ask("/v1/chat/completions", key)[0], 200)
        # And a proxy that somebody set up takes what it is sent.
        self.server.forward(None, {})
        self.assertEqual(self.ask("/v1/chat/completions", key)[0], 200)


class Wrapped(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.home = Path(self.folder.name)
        if os.name == "nt":
            (self.home / "claude.py").write_text(AGENT)
            starts = f'@"{sys.executable}" "%~dp0claude.py" %*\n'
            (self.home / "claude.cmd").write_text(starts)
        else:
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


class LineTests(Wrapped):
    """The status line of an agent."""

    def line(self, said):
        before, sys.stdin = sys.stdin, io.StringIO(said)
        try:
            return self.ask("stats", "claude", "--line")
        finally:
            sys.stdin = before

    def test_a_session_s_savings_in_a_few_words(self):
        self.assertEqual(self.line(""), (0, "compache\n"))
        self.wrap("--upstream", self.url, "claude", "--", "files")
        status, out = self.line("{}")
        self.assertRegex(out, r"^compache \d+% less, \$0\.\d+ of \$0\.\d+$")
        self.assertEqual(self.line('{"session_id": "another"}')[1], "compache\n")
        folder = self.home / "kept" / "claude"
        row = {"v": 1, "thread": thread_name("mine"), "turn": 1, "kind": "answer"}
        usage = {"input": 9_000, "output": 1, "cache_read": 0, "cache_write": 0}
        row |= {"at": 1.0, "model": "claude-haiku-5-5", "usage": usage}
        with open(folder / "calls.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        _, out = self.line('{"session_id": "mine"}')
        self.assertIn("$0.0009 of $0.0009", out)


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
        # One that is itself spoken to over TLS gets no name and password in plain sight.
        os.environ["https_proxy"] = name.replace("http://", "https://")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            ask()
        self.assertIn("over TLS", caught.exception.read().decode())
        caught.exception.close()
        self.assertEqual(len(asked), 2)

    def test_a_base_url_without_the_version_works_too(self):
        heard = []

        class Heard:
            def __init__(self, base, method, path, *more):
                heard.append((base, path))

            def __call__(self, payload, raw=None):
                return compache.live.Result(200, json=FakeModel().chat(payload)[1])

        before = proxy._Upstream
        proxy._Upstream = Heard
        self.addCleanup(setattr, proxy, "_Upstream", before)
        for upstream in (None, "https://gateway.example"):
            self.server.forward(upstream, {})
            request = urllib.request.Request(
                self.base + "/chat/completions",
                json.dumps(self.body).encode(),
                {"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request).close()
        self.assertEqual(
            list(dict.fromkeys(heard)),
            [
                ("https://api.openai.com", "/v1/chat/completions"),
                ("https://gateway.example", "/chat/completions"),
            ],
        )

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
