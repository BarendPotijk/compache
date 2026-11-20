import unittest

from compache.setup_metadata import evaluate_metadata, fit_metadata
from compache.traces import ContractError
from compache.workflow_presets import (
    WORKFLOW_NAMES,
    describe_workflow,
    example_metadata,
    example_model_spec,
)


class WorkflowExampleTests(unittest.TestCase):
    def test_presets_keep_the_same_conservative_default(self):
        for name in WORKFLOW_NAMES:
            self.assertEqual(describe_workflow(name)["default_actions"], {
                "compaction": "host", "maintenance_pings": False
            })
        with self.assertRaises(ContractError):
            describe_workflow("unknown-workflow")

    def test_reproducible_split_has_disjoint_threads_and_only_counts_or_scopes(self):
        allowed = {"thread_id", "model_id", "route_id", "run_index", "started_at", "ended_at",
                   "added_tokens", "retrieved_tokens", "visible_output_tokens", "output_tokens", "calls_per_run"}
        for name in WORKFLOW_NAMES:
            train, cutoff = example_metadata(name)
            self.assertEqual((train, cutoff), example_metadata(name))
            heldout, _ = example_metadata(name, split="heldout")
            self.assertLess(cutoff, min(r["started_at"] for r in heldout))
            self.assertTrue({r["thread_id"] for r in train}.isdisjoint(r["thread_id"] for r in heldout))
            for row in train + heldout:
                self.assertEqual(set(row), allowed)
                self.assertEqual(row["model_id"], example_model_spec().model_id)
                self.assertLessEqual(row["retrieved_tokens"], row["added_tokens"])
                self.assertGreaterEqual(row["output_tokens"], row["visible_output_tokens"])

    def test_counter_only_demos_make_different_heldout_decisions(self):
        spec = example_model_spec()
        for name in ("research", "support"):
            training, cutoff = example_metadata(name)
            artifact, _ = fit_metadata(training, spec, observation_end=cutoff)
            heldout, cutoff = example_metadata(name, split="heldout")
            result = evaluate_metadata(heldout, artifact, observation_end=cutoff)
            self.assertEqual(result["status"], "modeled")
            self.assertEqual(artifact["recipes"], [])
            if name == "research":
                self.assertGreater(result["ping_calls"], 0)
                self.assertLess(result["candidate_cost"], result["baseline_cost"])
            else:
                self.assertEqual(result["ping_calls"], 0)
                self.assertEqual(result["candidate_cost"], result["baseline_cost"])
