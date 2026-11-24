"""The policy to start from before there is anything to fit on."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import compache
from compache.cli import main
from compache.defaults import WORKLOAD, default_policy, default_workload
from compache.threads import load_threads


class DefaultTests(unittest.TestCase):
    def test_the_default_workload_is_the_calculator_s_and_always_the_same(self):
        threads, labels = default_workload()
        self.assertEqual((threads, labels), default_workload())
        self.assertNotEqual(threads, default_workload(seed=7)[0])
        self.assertEqual(len(threads), WORKLOAD["threads"])
        lengths = sorted(len(t["runs"]) for t in threads)
        self.assertTrue(6 <= lengths[len(lengths) // 2] <= 10, lengths)
        sizes = sorted(
            r["retrievals"][0]["bytes"] / 4 for t in threads for r in t["runs"]
        )
        self.assertTrue(20_000 < sizes[len(sizes) // 2] < 30_000)
        # Sizes only: nothing in it is text.
        loaded = load_threads(threads, model="claude-sonnet-5-5")
        self.assertEqual(set(loaded[0].runs[0].input.text), {"."})
        later = [row for row in labels if row["run"] > 1]
        share = sum(bool(row["needs"]) for row in later) / len(later)
        self.assertTrue(0.25 < share < 0.5, share)
        # The site's defaults are where these numbers come from.
        site = (Path(__file__).parent.parent / "site" / "engine.mjs").read_text()
        for given in (
            "plannedTurns: 8, plannedTurnsP95: 20",
            "tokenMedian: 25000, tokenP95: 45000",
            "inputTokens: 200, outputTokens: 500",
            "gapMedianMinutes: 4, gapP95Minutes: 60",
            "retrievalDependencyProbability: 0.15, retrievalDependencyTurnsBack: 3, retrievalDependencyTurnsBackP95: 30",
        ):
            self.assertIn(given, site)

    def test_a_default_policy_cuts_and_can_be_made_nearer_to_what_is_expected(self):
        policy = default_policy("claude-sonnet-5-5", threads=10)
        summary = policy.summary
        self.assertEqual(summary["model"], "claude-sonnet-5-5")
        self.assertIsNotNone(summary["keep"])
        self.assertFalse(summary["keep"]["matching_retrieved"])
        self.assertGreater(summary["saving"], 0.3)
        small = default_policy(
            "claude-sonnet-5-5", threads=10, retrieved_tokens=(300, 600)
        )
        self.assertLess(
            small.summary["cost_per_1000_turns"]["keep_everything"],
            summary["cost_per_1000_turns"]["keep_everything"] / 5,
        )
        session = policy.session("first-day")
        turn = session.prepare("A question?", ["A retrieved passage."], at=1.0)
        self.assertEqual(turn.action, "keep")

    def test_the_command_writes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main(
                    ["default", "--model", "claude-haiku-5-5", "-o", str(path)]
                )
            self.assertEqual(status, 0)
            self.assertIn(
                "Fitted to the default workload, not to yours", out.getvalue()
            )
            self.assertEqual(
                compache.Policy.load(path).summary["model"], "claude-haiku-5-5"
            )
            self.assertIn("solve", json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
