import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from openai_rag_transport import (
    make_payload,
    usage_cost,
    interpret,
    OpenAISession,
    Budget,
)


class TransportTests(unittest.TestCase):
    def test_boundaries(self):
        view = (
            "header\nCONVERSATION\n"
            + json.dumps({"role": "user", "text": "q"})
            + "\n"
            + json.dumps({"role": "tool", "text": "evidence"})
            + "\n"
            + json.dumps({"role": "assistant", "text": "answer"})
            + "\nTASK\nAnswer the latest substantive user question."
        )
        p = make_payload(view, "key")
        self.assertEqual(
            [m["role"] for m in p["input"]], ["developer", "user", "user", "assistant"]
        )
        ping = make_payload(
            view.replace("Answer the latest substantive user question.", "ping"), "key"
        )
        self.assertEqual(ping["input"][:-1], p["input"])
        self.assertEqual(
            make_payload("summary prompt", "key")["input"][0]["content"],
            "summary prompt",
        )

    def test_cost(self):
        u = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
            "output_tokens": 10,
        }
        self.assertAlmostEqual(
            usage_cost(u)[1], (50 * 0.1 + 20 * 0.01 + 30 * 0.125 + 10 * 0.5) / 1e6
        )
        u["input_tokens"] = 300000
        self.assertAlmostEqual(
            usage_cost(u)[1],
            ((300000 - 50) * 0.1 * 2 + 20 * 0.01 * 2 + 30 * 0.125 * 2 + 10 * 0.5 * 1.5)
            / 1e6,
        )
        del u["input_tokens_details"]["cache_write_tokens"]
        self.assertEqual(usage_cost(u), (None, None))
        u["input_tokens_details"]["cache_write_tokens"] = 300001
        self.assertEqual(usage_cost(u), (None, None))

    def test_failed_and_key_never_saved(self):
        self.assertFalse(interpret({"status": "incomplete", "output": []})["ok"])
        with tempfile.TemporaryDirectory() as d:
            s = OpenAISession("stable", api_key="SECRET_KEY_VALUE")
            with patch(
                "openai_rag_transport.urlopen", side_effect=URLError("SECRET_KEY_VALUE")
            ):
                r = s.run("fake", Path(d) / "run")
            self.assertFalse(r["ok"])
            self.assertIsNone(r["cost_usd"])
            for f in (Path(d) / "run").iterdir():
                self.assertNotIn("SECRET_KEY_VALUE", f.read_text())

    def test_budget_reservations(self):
        b = Budget(".52")
        a = b.reserve()
        b.reserve()
        with self.assertRaises(RuntimeError):
            b.reserve()
        b.reconcile(a, 0.1)
        self.assertEqual(str(b.spent), "0.1")
        self.assertEqual(str(b.reserved), "0.26")

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            s = OpenAISession("stable", api_key="secret")
            with patch("openai_rag_transport.urlopen", side_effect=TimeoutError()):
                self.assertFalse(s.run("fake", Path(d) / "run")["ok"])


if __name__ == "__main__":
    unittest.main()
