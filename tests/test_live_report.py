import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_live_comparison import build, write_report, audit_events


class ReportTests(unittest.TestCase):
    def fixture(self, base):
        (base / "plan.json").write_text(
            json.dumps({"model": "gpt-6-luna", "planned_answers_per_arm": 1})
        )
        for arm in ("opencode", "compache"):
            for rep in range(1, 4):
                p = base / "runs" / f"{arm}-{rep}"
                p.mkdir(parents=True)
                counts = {"answer": 1, "summary": 0, "ping": 0, "recovery": 0}
                row = {
                    "arm": arm,
                    "replicate": rep,
                    "sequence": 1,
                    "role": "answer",
                    "run_index": 1,
                    "logical_at": 1,
                    "ok": True,
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "output_tokens": 5,
                    },
                    "cache_blocks": ["a" * 64],
                }
                (p / "ledger.jsonl").write_text(json.dumps(row) + "\n")
                (p / "answers.jsonl").write_text(
                    json.dumps({"run_index": 1, "answer": "a"}) + "\n"
                )
                (p / "expected-counts.json").write_text(json.dumps(counts))
                (p / "complete.json").write_text(
                    json.dumps({"answers": 1, "calls": 1, "counts": counts})
                )

    def test_complete_report(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.fixture(p)
            r = write_report(p)
            self.assertTrue(r["complete"])
            self.assertEqual(len(r["timeline"]), 12)
            self.assertEqual(r["modeled_paired_savings"]["300"]["aggregate_savings"], 0)
            self.assertEqual(r["timeline"][0]["cumulative_input_tokens"], 100)
            self.assertEqual(
                r["diagnostics"]["opencode-1"]["response_formats"]["unknown_legacy"], 1
            )
            self.assertIn("OpenCode credits", (p / "report.md").read_text())
            self.assertTrue((p / "timeline.tsv").exists())

    def test_integrated_retry_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.fixture(p)
            retry = p / "excluded-transport-attempts/opencode-1-call-0002"
            retry.mkdir(parents=True)
            u = {"inputTokens": 1000, "cachedInputTokens": 400, "outputTokens": 10}
            (retry / "overhead.json").write_text(
                json.dumps({"model_calls": [u], "totals": u})
            )
            (p / "setup-overhead.json").write_text(json.dumps({"scope": "partial"}))
            events = p / "runs/opencode-1/call-0001"
            events.mkdir()
            ledger = p / "runs/opencode-1/ledger.jsonl"
            row = json.loads(ledger.read_text())
            row["session_id"] = "thread"
            ledger.write_text(json.dumps(row) + "\n")
            counters = {"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 5}
            event = {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread",
                    "tokenUsage": {"total": counters.copy(), "last": counters.copy()},
                },
            }
            (events / "events.jsonl").write_text(json.dumps(event) + "\n")
            event["params"]["tokenUsage"]["last"] = {
                "inputTokens": 40,
                "cachedInputTokens": 0,
                "outputTokens": 2,
            }
            event["params"]["tokenUsage"]["total"] = {
                "inputTokens": 140,
                "cachedInputTokens": 0,
                "outputTokens": 7,
            }
            first = {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread",
                    "tokenUsage": {"total": counters.copy(), "last": counters.copy()},
                },
            }
            row["usage"] = {
                "input_tokens": 40,
                "cached_input_tokens": 0,
                "output_tokens": 2,
            }
            ledger.write_text(json.dumps(row) + "\n")
            (events / "events.jsonl").write_text(
                json.dumps(first) + "\n" + json.dumps(event) + "\n"
            )
            result = write_report(p)
            overhead = result["retry_overhead"]["opencode"]
            self.assertEqual(result["codex_additional_invocations"]["model_calls"], 1)
            self.assertEqual(
                result["measured"]["aggregate"]["opencode"]["actual"]["usage"][
                    "input_tokens"
                ],
                340,
            )
            self.assertTrue(
                any(x["role"] == "codex_overhead" for x in result["timeline"])
            )
            self.assertEqual(overhead["model_calls"], 1)
            self.assertEqual(overhead["combined_input_tokens"], 1340)
            self.assertAlmostEqual(
                overhead["combined_credit_equivalent"],
                result["measured"]["aggregate"]["opencode"]["actual"][
                    "standard_rate_credit_equivalent"
                ]
                + 0.001725,
            )
            self.assertEqual(result["partial_setup_overhead"]["scope"], "partial")
            self.assertIn("Retry model calls", (p / "report.md").read_text())
            event["params"]["tokenUsage"]["total"]["inputTokens"] = 141
            (events / "events.jsonl").write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "usage mismatch"):
                build(p)

    def test_raw_total_must_equal_last(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            row = {
                "session_id": "t",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 2,
                    "output_tokens": 1,
                },
            }
            u = {"inputTokens": 100, "cachedInputTokens": 2, "outputTokens": 1}
            e = {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "t",
                    "tokenUsage": {"total": u.copy(), "last": u.copy()},
                },
            }
            p.write_text(json.dumps(e) + "\n")
            self.assertTrue(audit_events(p, row))
            e["params"]["tokenUsage"]["total"]["inputTokens"] = 200
            p.write_text(json.dumps(e) + "\n")
            with self.assertRaises(ValueError):
                audit_events(p, row)

    def test_missing_complete_and_usage_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.fixture(p)
            (p / "runs/opencode-2/complete.json").unlink()
            with self.assertRaises(ValueError):
                build(p)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.fixture(p)
            f = p / "runs/opencode-2/ledger.jsonl"
            row = json.loads(f.read_text())
            row["usage"] = None
            f.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                build(p)


if __name__ == "__main__":
    unittest.main()
