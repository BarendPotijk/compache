import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_scaled_rag import summarize


class ScaledEvaluationTest(unittest.TestCase):
    def test_direct_lookup_not_in_historical_denominator(self):
        label = {
            "target_run_id": "r",
            "task_requirements": [
                {
                    "requirement_id": "direct",
                    "historical_context_needed": "no",
                    "alternatives": [{}],
                },
                {
                    "requirement_id": "past",
                    "historical_context_needed": "yes",
                    "alternatives": [{}],
                    "question_quote": "planning note in turn",
                },
            ],
            "historical_dependencies": [{"source_run_id": "thread-run-001"}],
        }
        record = {
            "run_id": "r",
            "run_index": 10,
            "dependency_assessable": True,
            "missed_requirements": ["past"],
            "prefix_transition": "unchanged",
            "tokens_estimated": 100,
            "admitted": True,
            "historical_requirements_assessed": 1,
            "historical_requirements_retained": 0,
        }
        groups, _ = summarize(None, [label], {"runs": [record]}, {}, 0.2, 1000)
        self.assertEqual(groups["all"]["historical_requirements"], 1)
        self.assertEqual(groups["all"]["historical_missed"], 1)
        self.assertEqual(groups["lag:6-20"]["historical_missed"], 1)
        self.assertEqual(groups["family:input"]["historical_requirements"], 1)

    def test_skipped_history_is_not_scored_as_retained(self):
        labels = [
            {
                "target_run_id": "thread-run-090",
                "task_requirements": [
                    {
                        "requirement_id": "x",
                        "historical_context_needed": "yes",
                        "alternatives": [{}],
                    }
                ],
            }
        ]
        metadata = {
            "run_design": [{"run_id": "thread-run-090", "family": "output", "lag": 55}]
        }
        groups, _ = summarize(None, labels, {"runs": []}, metadata, 0.2, 1000)
        self.assertEqual(groups["all"]["historical_requirements_unassessed_skipped"], 1)
        self.assertEqual(
            groups["lag:51+"]["historical_requirements_unassessed_skipped"], 1
        )
        self.assertNotIn("historical_requirements", groups["all"])


if __name__ == "__main__":
    unittest.main()
