import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_rag_transport import interpret, redact_stderr, run


def events(*items):
    return "\n".join(json.dumps(item) for item in items) + "\n"


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.message = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "answer"},
        }
        self.complete = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 30,
                "output_tokens": 5,
            },
        }

    def test_valid_usage(self):
        result = interpret(
            events(
                {"type": "thread.started", "thread_id": "1"},
                self.message,
                self.complete,
            ),
            0,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["usage"]["cached_input_tokens"], 30)

    def test_fail_closed(self):
        bad = [
            events(self.message, {"type": "turn.completed", "usage": {}}),
            events(
                self.message,
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1,
                        "cached_input_tokens": 2,
                        "output_tokens": 1,
                    },
                },
            ),
            events(
                self.message,
                {"type": "item.completed", "item": {"type": "command_execution"}},
                self.complete,
            ),
            events(self.message, self.complete, self.complete),
            events(self.message, {"type": "context.compacted"}, self.complete),
            events(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": None},
                },
                self.complete,
            ),
            events(self.message, {"type": "error", "message": "retry"}, self.complete),
            "not json\n",
        ]
        for output in bad:
            with self.subTest(output=output):
                self.assertFalse(interpret(output, 0)["ok"])
        self.assertFalse(interpret(events(self.message, self.complete), 1)["ok"])

    def test_failure_record_and_stable_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run(
                "hello", path / "run", working_dir=path, codex=str(path / "missing")
            )
            self.assertFalse(result["ok"])
            self.assertIsNone(result["returncode"])
            self.assertEqual(result["working_dir"], str(path.resolve()))
            self.assertIn("model_context_window=872000", result["cli_args"])
            self.assertIn("model_auto_compact_token_limit=872000", result["cli_args"])
            self.assertEqual(result["reasoning_effort"], "low")
            self.assertEqual((path / "run" / "prompt.txt").read_text(), "hello")
            self.assertTrue((path / "run" / "events.jsonl").exists())
            self.assertIsNone(
                json.loads((path / "run" / "result.json").read_text())["cost_usd"]
            )

    def test_redaction(self):
        s = redact_stderr(
            'Authorization: Bearer abcdef api_key="secretvalue" sk-abcdefghijklmnop'
        )
        self.assertNotIn("abcdef", s)
        self.assertNotIn("secretvalue", s)
        self.assertNotIn("sk-abcdefghijklmnop", s)


if __name__ == "__main__":
    unittest.main()
