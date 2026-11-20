"""Marginal idle economics across expiry, lookup reachability and tool rounds."""

import copy
import unittest
from dataclasses import replace

from test_distribution_planner import SPEC, cohort, view
from helpers import END, thread

from compache.optimizer.distribution import (
    fit_distribution,
    freeze_distribution,
    plan_distribution_idle,
)
from compache.traces import ContractError


def fitted(returns, stops=0):
    profile = cohort(returns, stops=stops, horizon=6000)
    profile["schema_version"] = "compache-thread-demand-v3"
    for row in profile["by_position"].values():
        row["calls_per_run"] = [4] * len(row["response"])
    return profile


def schedule(returns, ttl=300, stops=0, **options):
    defaults = {
        "run_index": 1,
        "candidates": [view("keep", 1000, 1000, 1000)],
        "answer_tokens": 0,
        "duration_seconds": 0,
        "branches": 1,
        "lookahead": 1,
    }
    return plan_distribution_idle(
        fitted(returns, stops),
        replace(SPEC, cache_ttl_seconds=ttl),
        **(defaults | options),
    )


class EndpointPayoffTests(unittest.TestCase):
    def test_only_reachable_returns_in_the_extension_create_value(self):
        for ttl in (300, 1800):
            due = ttl - 45
            for fraction, expected in ((1, 1), (0, 0), (0.05, 0), (0.5, 1)):
                with self.subTest(ttl=ttl, fraction=fraction):
                    result = schedule([(ttl + 5, 0, 0, fraction)], ttl)
                    economics = result["ping_economics"][1]
                    self.assertEqual(result["pings"], expected)
                    self.assertEqual(economics["expected_ping_cost"], 100)
                    self.assertAlmostEqual(
                        economics["max_expected_foreground_savings"], 900 * fraction
                    )
                    self.assertEqual(
                        economics["beneficial_return_probability"], int(fraction > 0)
                    )
                    # Three additional calls are already warm on both paths.
                    self.assertAlmostEqual(
                        result["costs"][0] - result["costs"][1], 900 * fraction - 100
                    )
            returns = [
                (due, 0, 0, 1),
                (ttl - 1, 0, 0, 1),
                (ttl, 0, 0, 1),
                (due + ttl, 0, 0, 1),
            ]
            result = schedule(returns, ttl, stops=1)
            econ = result["ping_economics"][1]
            self.assertEqual(econ["dispatch_probability"], 4 / 5)
            self.assertEqual(econ["beneficial_return_probability"], 1 / 5)
            self.assertEqual(econ["expected_ping_cost"], 80)
            self.assertEqual(econ["expected_net_savings"], 100)

    def test_payment_uses_all_return_landmarks_before_sampling_growth(self):
        # One growth sample can draw the single profitable return. The
        # timer still incurs the charge on all 20 eligible schedules.
        demand = [(320, 0, 0, 1)] + [(800, 0, 0, 1)] * 19
        for seed in range(6):
            result = schedule(demand, seed=seed)
            self.assertEqual(result["ping_economics"][1]["expected_net_savings"], -55)
            self.assertEqual(result["pings"], 0)

    def test_gap_and_eligibility_stay_paired(self):
        good = schedule([(100, 0, 0, 0), (350, 0, 0, 1)])
        bad = schedule([(100, 0, 0, 1), (350, 0, 0, 0)])
        self.assertEqual((good["pings"], bad["pings"]), (1, 0))

    def test_forced_rewrite_at_return_only_credits_its_retained_prefix(self):
        candidates = [view("keep", 9900, 9900, 9900), view("cut", 500, 100, 100)]
        result = schedule([(320, 0, 200, 1)], candidates=candidates)
        econ = result["ping_economics"][1]
        self.assertEqual(econ["max_expected_foreground_savings"], 90)
        self.assertEqual(econ["expected_ping_cost"], 990)
        self.assertEqual(result["pings"], 0)

    def test_first_private_ping_cannot_credit_the_new_answer_to_foreground(self):
        demand = [(280, 0, 0, 1)]
        public = schedule(demand, answer_tokens=250)
        private = schedule(demand, answer_tokens=250, ping_has_private_suffix=True)
        self.assertEqual(
            public["ping_economics"][1]["max_expected_foreground_savings"], 225
        )
        self.assertEqual(
            private["ping_economics"][1]["max_expected_foreground_savings"], 0
        )
        self.assertEqual(private["ping_economics"][1]["expected_ping_cost"], 350)

    def test_endpoint_sidecar_keeps_unknowns_distinct_and_checks_source_ids(self):
        t = thread(n=4, gap=120)
        source = t.runs[0].run_id
        old = fit_distribution([t], SPEC, observation_end=END)
        new = fit_distribution(
            [t], SPEC, observation_end=END, next_endpoint_reuse_fraction={source: 0}
        )
        self.assertEqual(old["schema_version"], "compache-thread-demand-v2")
        self.assertEqual(new["schema_version"], "compache-thread-demand-v3")
        self.assertEqual(new["by_position"]["1"]["returns"][0][-1], 0)
        self.assertIsNone(new["by_position"]["2"]["returns"][0][-1])
        freeze_distribution(new, SPEC)
        for obs in (
            {"foreign": 1},
            {t.runs[-1].run_id: 0},
            {source: True},
            {source: -0.1},
        ):
            with self.assertRaises(ContractError):
                fit_distribution(
                    [t], SPEC, observation_end=END, next_endpoint_reuse_fraction=obs
                )
        malformed = copy.deepcopy(new)
        malformed["by_position"]["1"]["returns"][0][-1] = 1.01
        with self.assertRaises(ContractError):
            freeze_distribution(malformed, SPEC)
