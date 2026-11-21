import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.distribution import (
    fit_distribution,
    freeze_distribution,
    plan_distribution,
    plan_distribution_idle,
    request_charge,
)
from compache.errors import ContractError
from helpers import END, thread

SPEC = ModelSpec(
    "test",
    context_tokens=10000,
    cache_min_tokens=1,
    cache_block_tokens=1,
    cache_ttl_seconds=300,
    write_multiplier=1,
    read_multiplier=0.1,
    output_multiplier=5,
    bytes_per_token=1,
    part_overhead_tokens=0,
)


def view(name, size, shared, common, **extra):
    return {
        "name": name,
        "input_tokens": size,
        "shared_prefix_tokens": shared,
        "rewrite_prefix_tokens": common,
        **extra,
    }


def cohort(returns, stops=0, *, next_outcomes=None, horizon=1000):
    """Each synthetic training thread is followed throughout the idle horizon."""
    n = len(returns) + stops
    rows = {
        "1": {
            "response": [[0, 0, 0] for _ in range(n)],
            "returns": [list(x) for x in returns],
            "no_return": stops,
            "censored": 0,
        }
    }
    if returns:
        rows["2"] = {
            "response": next_outcomes or [[0, 0, 0] for x in returns],
            "returns": [],
            "no_return": len(returns),
            "censored": 0,
        }
    return {
        "schema_version": "compache-thread-demand-v1",
        "model_id": "test",
        "token_ruler": [1, 0],
        "observation_end": END,
        "horizon_seconds": horizon,
        "training_threads": n,
        "training_runs": n + len(returns),
        "reported_output_runs": 0,
        "by_position": rows,
    }


class DistributionPlannerTests(unittest.TestCase):
    def setUp(self):
        self.views = [view("keep", 1000, 1000, 1000), view("cut", 200, 100, 100)]

    def test_upfront_premium_is_worthwhile_over_multiple_requests(self):
        demand = cohort([(10, 0, 0)])
        short = plan_distribution(
            demand, SPEC, run_index=1, candidates=self.views, warm=True, lookahead=1
        )
        long = plan_distribution(
            demand, SPEC, run_index=1, candidates=self.views, warm=True, lookahead=2
        )
        self.assertEqual(short["costs"], {"keep": 100, "cut": 110})
        self.assertEqual(long["costs"], {"keep": 200, "cut": 130})
        self.assertEqual((short["action"], long["action"]), ("keep", "cut"))

    def test_rewrite_can_be_deferred_until_observing_a_cold_return(self):
        result = plan_distribution(
            cohort([(301, 0, 0)]),
            SPEC,
            run_index=1,
            candidates=self.views,
            warm=True,
            lookahead=2,
            max_pings=0,
        )
        self.assertEqual(result["costs"], {"keep": 300, "cut": 310})
        cold = plan_distribution(
            cohort([], 1), SPEC, run_index=1, candidates=self.views, warm=False
        )
        self.assertEqual(cold["action"], "cut")

    def test_timer_decision_integrates_idle_exits_without_foresight(self):
        for stops, expected, selected in (
            (1, {0: 500, 1: 150}, 1),
            (9, {0: 100, 1: 110}, 0),
        ):
            with self.subTest(stops=stops):
                demand = cohort([(301, 0, 0)], stops)
                result = plan_distribution_idle(
                    demand,
                    SPEC,
                    run_index=1,
                    candidates=self.views[:1],
                    answer_tokens=0,
                    duration_seconds=0,
                    lookahead=1,
                )
                self.assertEqual(result["costs"], expected)
                self.assertEqual(result["pings"], selected)
                self.assertEqual(
                    result["due_after_start_seconds"], [255] if selected else []
                )

    def test_known_short_prefix_changes_first_ping_and_warm_return_price(self):
        options = dict(
            run_index=1,
            candidates=self.views[:1],
            answer_tokens=0,
            duration_seconds=0,
            lookahead=1,
        )
        late = cohort([(301, 0, 0)])
        full = plan_distribution_idle(late, SPEC, **options)
        partial = plan_distribution_idle(late, SPEC, **options, held_cache_tokens=40)
        self.assertEqual((full["pings"], partial["pings"]), (1, 0))
        self.assertEqual(partial["costs"], {0: 1000, 1: 1064})
        soon = plan_distribution_idle(
            cohort([(20, 0, 0)]), SPEC, **options, held_cache_tokens=40
        )
        self.assertEqual(soon["costs"][0], 964)
        with self.assertRaisesRegex(ContractError, "held_cache_tokens"):
            plan_distribution_idle(late, SPEC, **options, held_cache_tokens=-1)

    def test_inhibited_pings_do_not_enter_search_and_timer_ties_cancel(self):
        demand = cohort([(320, 0, 0)])
        blocked = plan_distribution_idle(
            demand,
            SPEC,
            run_index=1,
            candidates=self.views[:1],
            answer_tokens=0,
            duration_seconds=0,
            ping_allowed=False,
        )
        self.assertEqual(blocked["costs"], {0: 1000})
        tied = plan_distribution_idle(
            cohort([(250, 0, 0)]),
            SPEC,
            run_index=1,
            candidates=self.views[:1],
            answer_tokens=0,
            duration_seconds=5,
        )
        self.assertEqual(tied["costs"], {0: 100, 1: 100})
        self.assertEqual(tied["pings"], 0)
        expired = plan_distribution_idle(
            cohort([(0, 0, 0)]),
            SPEC,
            run_index=1,
            candidates=self.views[:1],
            answer_tokens=0,
            duration_seconds=300,
        )
        self.assertEqual(expired["costs"], {0: 1000})

    def test_canceled_ping_at_arrival_and_first_write_of_answer(self):
        result = plan_distribution_idle(
            cohort([(301, 10, 20)]),
            SPEC,
            run_index=1,
            candidates=self.views[:1],
            answer_tokens=50,
            duration_seconds=0,
            lookahead=1,
        )
        # Ping reads 1000, writes 50. Next request reads 1050 and writes 20.
        self.assertEqual(result["costs"], {0: 1070, 1: 275})
        self.assertEqual(
            request_charge(
                replace(SPEC, cache_block_tokens=128, cache_min_tokens=1024),
                1300,
                1100,
                0,
            ),
            378.4,
        )

    def test_incomplete_search_returns_no_partial_winner(self):
        with patch("compache.optimizer.distribution.MAX_SEARCH_STATES", 1):
            result = plan_distribution(
                cohort([(10, 0, 0)]),
                SPEC,
                run_index=1,
                candidates=self.views,
                warm=True,
            )
        self.assertEqual(result["status"], "search_budget_exhausted")
        self.assertIsNone(result["action"])
        self.assertEqual(result["costs"], {})

    def test_missing_projection_support_is_not_priced_as_zero(self):
        d = cohort([(10, 0, 9800)])
        r = plan_distribution(
            d,
            SPEC,
            run_index=1,
            candidates=self.views,
            warm=True,
            lookahead=2,
            reserve_tokens=1,
        )
        self.assertIsNone(r["action"])
        self.assertEqual(r["costs"], {"keep": None, "cut": None})
        supplied = [dict(self.views[0], admitted=False), self.views[1]]
        self.assertEqual(
            plan_distribution(
                cohort([], 1), SPEC, run_index=1, candidates=supplied, warm=True
            )["action"],
            "cut",
        )

    def test_cross_thread_root_header_and_warm_thread_prefix_overlap(self):
        profile = cohort([], 1)
        kw = {
            "run_index": 1,
            "candidates": self.views[:1],
            "lookahead": 1,
            "shared_header_tokens": 800,
        }
        self.assertEqual(
            plan_distribution(profile, SPEC, warm=False, **kw)["expected_cost"], 280
        )
        self.assertEqual(
            plan_distribution(profile, SPEC, warm=True, **kw)["expected_cost"], 100
        )
        self.assertEqual(
            plan_distribution(
                profile,
                SPEC,
                warm=False,
                run_index=1,
                candidates=[view("tiny", 100, 0, 100)],
                shared_header_tokens=1200,
            )["expected_cost"],
            10,
        )

    def test_ambient_provider_hit_can_remove_the_value_of_a_keep_warm(self):
        # The first request vanishes from the thread cache before the return.
        # A separate thread can keep the 950-token header warm. A scheduled
        # ping still costs 100 and the post-ping request costs another 100.
        kw = {
            "run_index": 1,
            "candidates": self.views[:1],
            "answer_tokens": 0,
            "duration_seconds": 0,
            "lookahead": 1,
            "ambient_header_tokens": 950,
        }
        profile = cohort([(305, 0, 0)])
        observed = plan_distribution_idle(profile, SPEC, **kw, shared_header_tokens=950)
        half = plan_distribution_idle(profile, SPEC, **kw, ambient_hit_probability=0.5)
        certain = plan_distribution_idle(profile, SPEC, **kw, ambient_hit_probability=1)
        self.assertEqual(observed["costs"], {0: 1000, 1: 200})
        self.assertEqual(half["costs"], {0: 572.5, 1: 200})
        self.assertEqual(certain["costs"], {0: 145, 1: 200})
        self.assertEqual(certain["pings"], 0)

    def test_training_fit_censors_landmarks_without_full_followup(self):
        early = thread(tid="early", n=3, gap=600)
        demand = fit_distribution(
            [early], SPEC, observation_end="2026-01-01T00:30:00Z", horizon_seconds=1000
        )
        # Position 3 has no full idle horizon yet and does not train a zero return.
        self.assertEqual(demand["by_position"]["3"]["no_return"], 0)
        self.assertEqual(demand["by_position"]["3"]["censored"], 1)
        self.assertEqual(demand["by_position"]["1"]["returns"][0][0], 570.0)
        freeze_distribution(demand, SPEC)
        with self.assertRaisesRegex(ContractError, "fully observed"):
            fit_distribution(
                [early],
                SPEC,
                observation_end="2026-01-01T00:30:00Z",
                horizon_seconds=86400,
            )
        with self.assertRaisesRegex(ContractError, "duplicate"):
            fit_distribution([early, early], SPEC, observation_end=END)

    def test_brownian_log_increment_scales_to_current_observed_demand(self):
        # log1p(39)-log1p(19)=log(2); starting at 9 predicts a next arrival of 19.
        d = freeze_distribution(cohort([(10, 19, 39)]), SPEC)
        args = {
            "run_index": 1,
            "candidates": self.views[:1],
            "answer_tokens": 0,
            "duration_seconds": 0,
            "lookahead": 1,
            "sampler": "brownian",
            "current_arrival_tokens": 9,
        }
        self.assertAlmostEqual(
            plan_distribution_idle(d, SPEC, **args)["expected_cost"], 119
        )
        args["sampler"] = "empirical"
        self.assertAlmostEqual(
            plan_distribution_idle(d, SPEC, **args)["expected_cost"], 139
        )
        invalid = copy.deepcopy(cohort([(10, 0, 0)]))
        invalid["by_position"]["1"]["no_return"] = 1
        with self.assertRaisesRegex(ContractError, "counts"):
            freeze_distribution(invalid, SPEC)

    def test_rewrite_charge_is_charged_once_and_output_is_revealed_after_view_choice(
        self,
    ):
        d = cohort([(10, 0, 0), (10, 0, 0)], next_outcomes=[[5, 5, 0], [60, 60, 0]])
        # 60 tokens cannot fit after the uncut request. At arrival the cut
        # must be selected once for both possible next responses.
        common = [view("keep", 100, 100, 100), view("cut", 55, 45, 45, extra_cost=20)]
        model = replace(SPEC, context_tokens=130)
        result = plan_distribution_idle(
            d,
            model,
            run_index=1,
            candidates=common,
            answer_tokens=0,
            duration_seconds=0,
            lookahead=1,
        )
        self.assertAlmostEqual(result["expected_cost"], 20 + 4.5 + 10 + 32.5 * 5)


if __name__ == "__main__":
    unittest.main()
