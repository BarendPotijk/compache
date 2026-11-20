import unittest
from pathlib import Path
import sys
import os
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from appserver_rag_transport import parse_turn, check_empty_history, isolated_command


class AppServerTests(unittest.TestCase):
    def test_external_tools_are_disabled_at_server_start(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.toml").write_text(
                '[mcp_servers.context7]\nurl="https://example.test"\n[mcp_servers.computer-use]\ncommand="example"\n'
            )
            with patch.dict(os.environ, {"CODEX_HOME": folder}):
                command = isolated_command("codex")
            self.assertIn('web_search="disabled"', command)
            self.assertIn("mcp_servers.context7.enabled=false", command)
            self.assertIn("mcp_servers.computer-use.enabled=false", command)
            self.assertEqual(command[-1], "app-server")

    def setUp(self):
        self.done = {
            "status": "completed",
            "items": [{"type": "agentMessage", "phase": "final_answer", "text": "OK"}],
        }
        self.usage = {
            "last": {"inputTokens": 100, "cachedInputTokens": 50, "outputTokens": 5},
            "modelContextWindow": 828400,
        }

    def test_valid_last_usage(self):
        r = parse_turn(self.done, self.usage, [])
        self.assertTrue(r["ok"])
        self.assertEqual(r["usage"]["cached_input_tokens"], 50)

    def test_fail_closed(self):
        self.assertFalse(parse_turn(self.done, None, [])["ok"])
        self.assertFalse(
            parse_turn(self.done, self.usage, [{"method": "thread/compacted"}])["ok"]
        )
        self.assertFalse(
            parse_turn(
                self.done,
                self.usage,
                [
                    {
                        "method": "item/completed",
                        "params": {"item": {"type": "commandExecution"}},
                    }
                ],
            )["ok"]
        )
        self.assertFalse(
            parse_turn({"status": "failed", "items": []}, self.usage, [])["ok"]
        )

    def test_revert_history_guard(self):
        check_empty_history({"data": [], "nextCursor": None})
        with self.assertRaises(RuntimeError):
            check_empty_history({"data": [{"id": "old"}]})
        with self.assertRaises(RuntimeError):
            check_empty_history({"data": [], "nextCursor": "more"})


if __name__ == "__main__":
    unittest.main()
