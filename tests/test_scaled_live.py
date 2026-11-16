import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_scaled_live import matched_prefix, first_sentence
from run_live_comparison import bind_live_requirements
from compache.optimizer import Source


class ScaledLiveTests(unittest.TestCase):
    def test_admission_stop_does_not_count_skipped_turns_as_savings(self):
        completed = {
            "opencode-1": 96,
            "compache-1": 20,
            "opencode-2": 96,
            "compache-2": 96,
            "opencode-3": 96,
            "compache-3": 0,
        }
        rows = [
            {"arm": arm, "replicate": rep, "run_index": i, "role": "answer"}
            for arm in ("opencode", "compache")
            for rep in (1, 2, 3)
            for i in range(1, 97)
        ]
        rows += [
            {"arm": "opencode", "replicate": 1, "run_index": 18, "role": "summary"},
            {"arm": "opencode", "replicate": 1, "run_index": 21, "role": "summary"},
        ]
        selected, n = matched_prefix(rows, completed)
        self.assertEqual(n, {1: 20, 2: 96, 3: 0})
        self.assertEqual(sum(r["role"] == "answer" for r in selected), 232)
        self.assertEqual(
            [r["run_index"] for r in selected if r["role"] == "summary"], [18]
        )
        self.assertFalse(any(r["replicate"] == 3 for r in selected))

    def test_prior_output_quote_uses_the_recorded_text(self):
        self.assertEqual(
            first_sentence("A different live response! Additional content."),
            "A different live response!",
        )

    def test_fabricated_output_is_not_treated_as_live_evidence(self):
        label = {
            "status": "complete",
            "task_requirements": [
                {
                    "requirement_id": "r",
                    "source_kind": "output",
                    "alternatives": [
                        {"sources": [{"part_id": "o", "quote": "Fabricated answer."}]},
                    ],
                }
            ],
        }
        sources = [
            Source("o", "output", 1, "Actual first sentence. Actual second sentence.")
        ]
        bound, diagnostic = bind_live_requirements(label, sources)
        self.assertEqual(bound["task_requirements"][0]["alternatives"], [])
        self.assertEqual(diagnostic["unassessable_requirements"], ["r"])
        bound, diagnostic = bind_live_requirements(
            label, sources, quote_actual_output=True
        )
        self.assertEqual(
            bound["task_requirements"][0]["alternatives"][0]["sources"][0]["quote"],
            "Actual first sentence.",
        )
        self.assertEqual(
            label["task_requirements"][0]["alternatives"][0]["sources"][0]["quote"],
            "Fabricated answer.",
        )


if __name__ == "__main__":
    unittest.main()
