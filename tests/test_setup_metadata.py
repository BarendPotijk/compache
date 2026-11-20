import json
import unittest
from datetime import datetime, timedelta, timezone

from compache.optimizer import ModelSpec, WorkloadPolicy
from compache.optimizer.contracts import digest
from compache.optimizer.distribution import fit_distribution
from compache.setup_metadata import (
    evaluate_metadata,
    fit_metadata,
    plan_metadata_idle,
    validate_metadata_row,
)
from compache.traces import ContractError
from test_optimizer import thread


SPEC = ModelSpec(
    "test",
    context_tokens=500000,
    cache_ttl_seconds=300,
    cache_min_tokens=100,
    ping_output_tokens=3,
    read_multiplier=0.1,
)
ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = "2026-01-04T00:00:00Z"


def row(tid, pos, start, **edits):
    return {
        "thread_id": tid,
        "route_id": "scope1",
        "model_id": "test",
        "run_index": pos,
        "started_at": (ORIGIN + timedelta(seconds=start)).isoformat(),
        "ended_at": (ORIGIN + timedelta(seconds=start + 5)).isoformat(),
        "added_tokens": 10000 if pos == 1 else 200,
        "output_tokens": 10,
        "visible_output_tokens": 8,
        **edits,
    }


class SetupMetadataTests(unittest.TestCase):
    def test_fits_existing_planner_with_same_numeric_demand(self):
        source = thread(tid="train", n=3)
        sample = []
        for run in source.runs:
            sample.append(
                row(
                    "train",
                    run.run_index,
                    10,
                    started_at=run.generation.started_at,
                    ended_at=run.ended_at,
                    output_tokens=SPEC.text_tokens(run.output.text),
                    visible_output_tokens=SPEC.part_tokens(run.output.text),
                    added_tokens=sum(
                        SPEC.part_tokens(p.text) for p in (run.input, *run.retrievals)
                    ),
                )
            )
        artifact, profile = fit_metadata(reversed(sample), SPEC, observation_end=END)
        original = fit_distribution([source], SPEC, observation_end=END)
        self.assertEqual(artifact["demand"]["by_position"], original["by_position"])
        self.assertEqual(artifact["training_threads"], [digest("train")])
        self.assertEqual(WorkloadPolicy(artifact).recipes, {})
        self.assertEqual(profile["runs"], 3)
        self.assertNotIn("train-r", json.dumps(artifact))
        self.assertNotIn("scope1", json.dumps(artifact))

    def test_censoring_and_optional_fields(self):
        sample = [row("a", 1, 0), row("a", 2, 600, retrieved_tokens=120)]
        del sample[0]["visible_output_tokens"]
        sample[0]["calls_per_run"] = 4
        cutoff = (ORIGIN + timedelta(seconds=610)).isoformat()
        a, profile = fit_metadata(
            sample, SPEC, observation_end=cutoff, horizon_seconds=600
        )
        self.assertEqual(profile["return_within_horizon_fraction"], 1)
        self.assertEqual(profile["censored_horizons"], 1)
        self.assertEqual(profile["median_return_gap_seconds"], 595)
        self.assertEqual(profile["median_retrieved_tokens"], 120)
        self.assertEqual(profile["runs_with_retrieved_tokens"], 1)
        self.assertEqual(profile["runs_assuming_visible_equals_billed_output"], 1)
        self.assertEqual(WorkloadPolicy(a).forecast.call_forecast(2)[0], 4)

    def test_rejects_unknown_raw_fields_models_routes_and_bad_timelines(self):
        for invalid in [
            row("a", 1, 0, accidental_private_field="private"),
            row("real@email", 1, 0),
            row("a", 1, 0, added_tokens=False),
            row("a", 1, 0, retrieved_tokens=10001),
            row("a", 1, 0, calls_per_run=0),
        ]:
            with self.assertRaises(ContractError) as exc:
                validate_metadata_row(invalid)
            self.assertNotIn("private", str(exc.exception))
            self.assertNotIn("real@email", str(exc.exception))
        normal = row("a", 1, 0)
        for extra in [
            normal,
            row("b", 1, 0, model_id="different"),
            row("a", 3, 300),
            row("b", 1, 0, route_id="other"),
            row("a", 2, 2),
        ]:
            with self.assertRaises(ContractError):
                fit_metadata([normal, extra], SPEC, observation_end=END)

    def test_live_timer_requires_verified_capacity_and_honors_cache_latch(self):
        train = [row("fit", 1, 0), row("fit", 2, 400)]
        artifact, _ = fit_metadata(train, SPEC, observation_end=END)
        policy = WorkloadPolicy(artifact)
        args = dict(
            run_index=1,
            input_tokens=10000,
            added_tokens=10000,
            visible_output_tokens=8,
            duration_seconds=5,
        )
        self.assertEqual(plan_metadata_idle(policy, **args)["pings"], 0)
        self.assertEqual(
            plan_metadata_idle(policy, **args, reusable_input_tokens=10000)["pings"], 1
        )
        self.assertEqual(
            plan_metadata_idle(
                policy, **args, reusable_input_tokens=10000, ping_allowed=False
            )["pings"],
            0,
        )

    def test_numeric_endpoint_observations_fit_and_score_separate_splits(self):
        train = [row("fit", 1, 0, next_endpoint_reuse_fraction=1), row("fit", 2, 400)]
        artifact, metadata = fit_metadata(train, SPEC, observation_end=END)
        self.assertEqual(metadata["runs_with_next_endpoint_reuse_fraction"], 1)
        held = [row("test", 1, 0, next_endpoint_reuse_fraction=0), row("test", 2, 400)]
        missed = evaluate_metadata(held, artifact, observation_end=END)
        self.assertEqual(missed["ping_calls"], 1)
        self.assertLess(missed["savings_fraction"], 0)
        train[0]["next_endpoint_reuse_fraction"] = 0
        artifact, _ = fit_metadata(train, SPEC, observation_end=END)
        self.assertEqual(
            evaluate_metadata(held, artifact, observation_end=END)["ping_calls"], 0
        )

    def test_holdout_models_pings_expiry_ties_and_terminal_charges(self):
        train = [row("fit", 1, 0), row("fit", 2, 400)]
        policy, _ = fit_metadata(train, SPEC, observation_end=END)
        heldout = [row("score", 1, 0), row("score", 2, 400)]
        measured = evaluate_metadata(heldout, policy, observation_end=END)
        self.assertEqual(measured["ping_calls"], 1)
        self.assertEqual(measured["status"], "modeled")
        self.assertGreater(measured["savings_fraction"], 0)
        tie = evaluate_metadata(
            [heldout[0], row("score", 2, 255)], policy, observation_end=END
        )
        self.assertEqual(tie["ping_calls"], 0)
        self.assertEqual(tie["savings_fraction"], 0)
        terminal = evaluate_metadata(heldout[:1], policy, observation_end=END)
        self.assertEqual(terminal["ping_calls"], 1)
        self.assertLess(terminal["savings_fraction"], 0)
        with self.assertRaises(ContractError):
            evaluate_metadata(train, policy, observation_end=END)
        with self.assertRaises(ContractError):
            evaluate_metadata(
                [row("score", 1, 0, route_id="other")], policy, observation_end=END
            )
        self.assertEqual(
            evaluate_metadata([], policy, observation_end=END)["status"],
            "insufficient_evidence",
        )
        large = evaluate_metadata(
            [row("score", 1, 0, added_tokens=SPEC.context_tokens)],
            policy,
            observation_end=END,
        )
        self.assertEqual(large["status"], "context_limit")
        self.assertIsNone(large["savings_fraction"])
