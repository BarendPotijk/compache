import copy
import unittest
from dataclasses import replace

from compache.optimizer.distribution import (
    fit_distribution,
    freeze_distribution,
    plan_distribution,
    plan_distribution_idle,
    repeat_input_usage,
    validate_calls_per_run,
)
from compache.optimizer.workload import fit_workload
from compache.traces import ContractError
from test_distribution_planner import SPEC, cohort, view
from helpers import END, labels_for, thread


def with_counts(profile, **counts):
    result = copy.deepcopy(profile)
    result["schema_version"] = "compache-thread-demand-v2"
    for pos, row in result["by_position"].items():
        row["calls_per_run"] = counts[pos]
    return result


class WithinRunCallsTests(unittest.TestCase):
    def test_fit_preserves_unknowns_and_learns_position_without_heldout_ids(self):
        a, b, c = (thread(tid=tid, n=2) for tid in "abc")
        artifact = fit_workload(
            [c, a, b],
            [*labels_for(a), *labels_for(b), *labels_for(c)],
            SPEC,
            observation_end=END,
            calls_per_run={"a-r1": 1, "b-r1": 3, "c-r2": 8, "a-r2": None},
        )
        demand = artifact["demand"]
        self.assertEqual(demand["by_position"]["1"]["calls_per_run"], [1, 3, None])
        self.assertEqual(demand["by_position"]["2"]["calls_per_run"], [None, None, 8])
        result = plan_distribution(
            demand,
            SPEC,
            run_index=1,
            candidates=[view("keep", 10, 0, 10)],
            warm=False,
            lookahead=3,
        )["calls_per_run"]
        self.assertEqual(result["training_observed_runs"], 3)
        self.assertEqual(result["training_unknown_runs"], 3)
        self.assertEqual(
            result["by_position"],
            {
                "1": {
                    "expected_calls": 2.0,
                    "source": "position",
                    "observed_runs": 2,
                    "unknown_runs": 1,
                },
                "2": {
                    "expected_calls": 8.0,
                    "source": "position",
                    "observed_runs": 1,
                    "unknown_runs": 2,
                },
                "3": {
                    "expected_calls": 4.0,
                    "source": "pooled",
                    "observed_runs": 0,
                    "unknown_runs": 0,
                },
            },
        )
        bad = (
            {"foreign": 2},
            {"foreign": None},
            {"a-r1": False},
            {"a-r1": 0},
            {"a-r1": 1.5},
            {"a-r1": "3"},
            [],
        )
        for sidecar in bad:
            with self.subTest(sidecar=sidecar), self.assertRaises(ContractError):
                fit_distribution([a], SPEC, observation_end=END, calls_per_run=sidecar)
        self.assertEqual(
            validate_calls_per_run({"a": 1, "b": None}, ["a", "b"]), {"a": 1}
        )

    def test_legacy_and_entirely_unknown_profiles_predict_one(self):
        legacy = cohort([(500, 0, 0)])
        forecast = with_counts(legacy, **{"1": [None], "2": [None]})
        options = dict(
            run_index=1, candidates=[view("keep", 100, 0, 100)], warm=False, lookahead=2
        )
        old = plan_distribution(legacy, SPEC, **options)
        new = plan_distribution(forecast, SPEC, **options)
        self.assertEqual(old["costs"], new["costs"])
        self.assertEqual(old["calls_per_run"], new["calls_per_run"])
        for row in old["calls_per_run"]["by_position"].values():
            self.assertEqual(
                (row["source"], row["expected_calls"]), ("one_call_assumption", 1)
            )
        for invalid in ([], [None, 2], [0], [True], "1"):
            broken = with_counts(legacy, **{"1": invalid, "2": [1]})
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                freeze_distribution(broken, SPEC)

    def test_block_rounding_minimum_disabled_cache_and_output_counted_once(self):
        spec = replace(
            SPEC, cache_block_tokens=128, cache_min_tokens=1024, write_multiplier=1.25
        )
        self.assertEqual(
            repeat_input_usage(spec, 1100, 3),
            {
                "input_tokens": 3300,
                "fresh_input_tokens": 0,
                "cache_write_tokens": 228,
                "cache_read_tokens": 3072,
                "cost_input_equivalents": 3 * (102.4 + 95),
            },
        )
        for model, size in ((spec, 999), (replace(spec, cache_ttl_seconds=0), 1100)):
            usage = repeat_input_usage(model, size, 2)
            self.assertEqual(
                (usage["fresh_input_tokens"], usage["cost_input_equivalents"]),
                (2 * size, 2 * size),
            )
            self.assertEqual(
                usage["cache_read_tokens"] + usage["cache_write_tokens"], 0
            )
        # Three foreground calls, one billed answer. Only the first send is cold.
        demand = with_counts(cohort([], 1), **{"1": [3]})
        demand["by_position"]["1"]["response"] = [[30, 20, 5]]
        result = plan_distribution(
            demand,
            spec,
            run_index=1,
            warm=False,
            candidates=[view("keep", 1100, 0, 1100)],
            lookahead=1,
        )
        expected = 1100 * 1.25 + 2 * (102.4 + 95) + 30 * 5
        self.assertAlmostEqual(result["expected_cost"], expected)

    def test_expected_repeats_can_advance_cut_at_a_five_minute_ttl(self):
        legacy = cohort([(600, 0, 0)])
        fitted = with_counts(legacy, **{"1": [2], "2": [2]})
        choices = [view("keep", 1000, 1000, 1000), view("cut", 200, 100, 100)]
        plans = []
        for ttl in (300, 1800):
            spec = replace(SPEC, cache_ttl_seconds=ttl)
            params = dict(
                run_index=1, candidates=choices, warm=True, max_pings=0, lookahead=2
            )
            plans.append(
                [plan_distribution(d, spec, **params) for d in (legacy, fitted)]
            )
        (short_old, short_new), (long_old, long_new) = plans
        self.assertEqual(short_old["action"], "keep")
        self.assertEqual(short_new["action"], "cut")
        self.assertEqual(
            (short_new["costs"], long_new["costs"]),
            ({"keep": 420, "cut": 350}, {"keep": 330, "cut": 170}),
        )
        self.assertEqual((long_old["action"], long_new["action"]), ("cut", "cut"))

    def test_idle_prices_next_runs_without_multiplying_ping_or_charging_past_calls(
        self,
    ):
        demand = with_counts(cohort([(301, 0, 0)]), **{"1": [100], "2": [None]})
        idle = plan_distribution_idle(
            demand,
            SPEC,
            run_index=1,
            candidates=[view("keep", 1000, 1000, 1000)],
            answer_tokens=0,
            duration_seconds=0,
            lookahead=1,
        )
        # A missing position's mean is pooled from the observed first run.
        # A real return pays 99 additional reads whichever timer was chosen.
        self.assertEqual(idle["costs"], {0: 1000 + 9900, 1: 200 + 9900})
        self.assertEqual(
            idle["calls_per_run"]["by_position"],
            {
                "2": {
                    "expected_calls": 100.0,
                    "source": "pooled",
                    "observed_runs": 0,
                    "unknown_runs": 1,
                },
            },
        )
        stopped = plan_distribution_idle(
            with_counts(cohort([], stops=1), **{"1": [100]}),
            SPEC,
            run_index=1,
            candidates=[view("keep", 1000, 1000, 1000)],
            answer_tokens=0,
            duration_seconds=0,
            lookahead=1,
        )
        self.assertEqual(stopped["costs"], {0: 0, 1: 100})


if __name__ == "__main__":
    unittest.main()
