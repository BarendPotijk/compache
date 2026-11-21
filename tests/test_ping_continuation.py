"""Marginal timer economics after an observed cache refresh."""

import unittest
from dataclasses import replace

from test_distribution_planner import SPEC, cohort, view

from compache.optimizer.distribution import plan_distribution_idle, repeat_ping_charge
from compache.errors import ContractError


def receipt(
    *, slots=(510,), dispatch=255, observed=None, size=1000, reuse=1000, output=4
):
    result = {
        "elapsed_seconds": dispatch,
        "remaining_due_after_start_seconds": slots,
        "input_tokens": size,
        "reusable_tokens": reuse,
        "output_tokens": output,
    }
    if observed is not None:
        result["observed_elapsed_seconds"] = observed
    return result


def continuation(demand, measured, spec=SPEC, **options):
    settings = {
        "run_index": 1,
        "candidates": [view("keep", 1000, 1000, 1000)],
        "duration_seconds": 0,
        "answer_tokens": 0,
        "max_pings": 2,
        "lookahead": 1,
    }
    settings.update(options)
    return plan_distribution_idle(demand, spec, after_ping=measured, **settings)


class PingContinuationTests(unittest.TestCase):
    def test_measured_input_and_output_price_repeat_and_foreground_at_both_ttls(self):
        for ttl in (300, 1800):
            spec = replace(SPEC, cache_ttl_seconds=ttl, cache_block_tokens=100)
            interval = ttl - 45
            due = 2 * interval
            demand = cohort([(due + 90, 0, 70)], horizon=3 * ttl)
            for output, chosen in ((4, 1), (249, 0)):
                with self.subTest(ttl=ttl, output=output):
                    result = continuation(
                        demand,
                        receipt(
                            dispatch=interval + 8,
                            slots=(due,),
                            size=1254,
                            reuse=1200,
                            output=output,
                        ),
                        spec,
                        duration_seconds=11,
                        answer_tokens=50,
                        held_cache_tokens=900,
                    )
                    # The actual 1,254 input includes the answer. On a repeat
                    # 1,200 read + 54 write. The next user adds 70 fresh tokens.
                    repeat = 120 + 54 + 5 * output
                    self.assertEqual(result["costs"], {0: 1324, 1: repeat + 244})
                    self.assertEqual(result["pings"], chosen)
                    self.assertEqual(result["due_after_start_seconds"], [due] * chosen)

    def test_observed_uncached_mix_can_change_whether_repeat_pays(self):
        spec = replace(
            SPEC, cache_ttl_seconds=1800, cache_block_tokens=100, write_multiplier=2
        )
        demand = cohort([(3700, 0, 0)], horizon=5400)
        measured = receipt(
            dispatch=1755, slots=(3510,), size=1050, reuse=1000, output=345
        )
        # Repeat reads 1,000 tokens; the 50-token tail has seen only ordinary
        # fresh charges, a mix, or all writes. Foreground retains model pricing.
        for rate, cost, pings in (
            (None, 2125, 0),
            (2, 2125, 0),
            (1.5, 2100, 0),
            (1, 2075, 1),
        ):
            with self.subTest(rate=rate):
                result = continuation(
                    demand, dict(measured, uncached_input_rate=rate), spec
                )
                self.assertEqual(result["costs"], {0: 2100, 1: cost})
                self.assertEqual(result["pings"], pings)
                self.assertEqual(
                    repeat_ping_charge(spec, 1050, 1049, 345, uncached_input_rate=rate),
                    cost - 200,
                )

    def test_condition_before_quadrature_includes_late_return_and_terminal_landmark(
        self,
    ):
        # Receipt observed at 275, with 10 seconds of foreground generation.
        # Every tied or earlier return is gone. Even with one quadrature
        # stratum the single remaining late return must survive sampling.
        demand = cohort([(265, 0, 0)] * 29 + [(600, 0, 0)], stops=1)
        result = continuation(
            demand, receipt(observed=275), duration_seconds=10, branches=1
        )
        self.assertEqual(
            result["support"], {"returns": 1, "no_return": 1, "return_probability": 0.5}
        )
        self.assertEqual(result["costs"], {0: 500, 1: 170})

    def test_lapse_uses_actual_dispatch_and_timer_ties_cancel(self):
        for arrival in (510, 560, 575):
            with self.subTest(arrival=arrival):
                result = continuation(cohort([(arrival, 0, 0)]), receipt(dispatch=275))
                if arrival == 510:
                    self.assertEqual(result["costs"], {0: 100, 1: 100})
                elif arrival == 560:
                    self.assertEqual(result["costs"], {0: 100, 1: 220})
                else:
                    # The actual refresh at 275 expires exactly at 575.
                    self.assertEqual(result["costs"], {0: 1000, 1: 220})
                self.assertEqual(result["pings"], int(arrival == 575))

    def test_conditional_tail_keeps_fitted_growth_and_current_arrival_size(self):
        # Both transitions double (tokens + 1). At a current arrival of nine,
        # the fitted Brownian forecast is 19, including on the late branch.
        demand = cohort([(200, 9, 19), (600, 99, 199)])
        for sampler, incoming in (("empirical", 199), ("brownian", 19)):
            with self.subTest(sampler=sampler):
                result = continuation(
                    demand,
                    receipt(),
                    current_arrival_tokens=9,
                    branches=1,
                    sampler=sampler,
                )
                self.assertEqual(result["support"]["returns"], 1)
                self.assertEqual(
                    result["costs"], {0: 1000 + incoming, 1: 220 + incoming}
                )

    def test_delayed_receipt_drops_old_or_expired_slots_without_restarting_budget(self):
        demand = cohort([(815, 0, 0)])
        slots = (510, 765)
        for dispatch, expected in ((275, []), (500, [765])):
            with self.subTest(dispatch=dispatch):
                result = continuation(
                    demand,
                    receipt(dispatch=dispatch, observed=702, slots=slots),
                    max_pings=3,
                )
                self.assertEqual(result["due_after_start_seconds"], expected)
                self.assertEqual(len(result["costs"]), 1 + len(expected))
        # A receipt ready exactly at the next deadline can still fund it.
        result = continuation(cohort([(610, 0, 0)]), receipt(observed=510))
        self.assertEqual(result["due_after_start_seconds"], [510])

    def test_jointly_profitable_chain_can_require_multiple_remaining_timers(self):
        result = continuation(
            cohort([(900, 0, 0)]), receipt(slots=(510, 765)), max_pings=3
        )
        self.assertEqual(result["costs"], {0: 1000, 1: 1120, 2: 340})
        self.assertEqual(result["due_after_start_seconds"], [510, 765])

    def test_deferred_cut_and_repeat_foreground_calls_can_make_lapse_cheapest(self):
        demand = cohort([(600, 0, 0)])
        demand["schema_version"] = "compache-thread-demand-v2"
        for key, row in demand["by_position"].items():
            row["calls_per_run"] = [3 if key == "2" else 1]
        result = continuation(
            demand,
            receipt(),
            candidates=[view("keep", 1000, 1000, 1000), view("cut", 180, 100, 100)],
        )
        # Cut on return: 180 cold or 90 warm, plus two recurring 18-token charges.
        self.assertEqual(result["costs"], {0: 216, 1: 246})
        self.assertEqual(result["pings"], 0)

    def test_new_foreground_run_keeps_original_ping_output_forecast(self):
        demand = cohort([(300, 0, 0)])
        next_run = demand["by_position"]["2"]
        next_run.update(returns=[[301, 0, 0]], no_return=0)
        demand["by_position"]["3"] = {
            "response": [[0, 0, 0]],
            "returns": [],
            "no_return": 1,
            "censored": 0,
        }
        demand["training_runs"] = 3
        result = continuation(demand, receipt(output=900), lookahead=2)
        # User arrives while current refresh is warm; the NEXT foreground run
        # can buy its own cheap modeled ping and bridge its 301-second gap.
        self.assertEqual(result["costs"], {0: 300, 1: 300})

    def test_no_evidence_and_expired_horizon_withhold_a_winner(self):
        unsupported = continuation(cohort([(255, 0, 0)]), receipt())
        self.assertEqual(unsupported["status"], "no_observed_idle_survivors")
        self.assertEqual(unsupported["costs"], {})
        self.assertIsNone(unsupported["support"]["return_probability"])
        elapsed = continuation(cohort([], stops=1), receipt(observed=1001))
        self.assertEqual(elapsed["status"], "idle_horizon_elapsed")
        self.assertEqual(elapsed["pings"], 0)
        all_stops = continuation(cohort([], stops=1), receipt())
        self.assertEqual(all_stops["costs"], {0: 0, 1: 120})

    def test_invalid_receipt_clocks_slots_and_token_units(self):
        for measured in (
            receipt(observed=254),
            receipt(slots=(510, 765)),
            receipt(reuse=1001),
            receipt(output=-1),
            dict(receipt(), uncached_input_rate=-0.1),
            dict(receipt(), uncached_input_rate=float("inf")),
        ):
            with self.subTest(measured=measured), self.assertRaises(ContractError):
                continuation(cohort([(600, 0, 0)]), measured)


if __name__ == "__main__":
    unittest.main()
