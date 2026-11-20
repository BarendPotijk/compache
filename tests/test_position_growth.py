import json
import unittest
from datetime import datetime, timedelta

from compache.optimizer import (
    ModelSpec, Policy, Request, Source, ThreadState, end_run, fit_position_growth,
    fit_profile, freeze_profile, optimize, position_risk, position_risk_policy, prepare, replay,
)
from compache.optimizer.contracts import digest
from compache.optimizer.solve import candidates
from compache.traces import ContractError
from test_optimizer import END, labels_for, thread


def observation(tid, pos, remaining, arrival=10, status="complete"):
    return {"thread_id": tid, "run_index": pos, "arrival_tokens": arrival,
            "remaining_tokens": remaining, "source": "observed_peak", "status": status}


def fitted(growth, *, reserve=.1, alpha=.05, ttl=300):
    spec = ModelSpec("test", context_tokens=1000, bytes_per_token=1,
                     part_overhead_tokens=0, cache_ttl_seconds=ttl)
    profile = fit_profile([thread()], [], spec, observation_end=END)
    profile["position_growth"] = fit_position_growth(growth)
    base = Policy(spec.to_dict(), digest(profile), retrieval_grace_runs=0,
                  target=.15, cold_trigger=.2, warm_trigger=.6, reserve=reserve,
                  max_pings=1, protect_query_matches=False)
    return base, position_risk_policy(base, profile, max_overflow_probability=alpha), profile


def paired_requests(added=40):
    history = (Source("question", "input", 1, "Q" * 10),
               Source("old", "retrieval", 1, "Z" * 450))
    first = Request("t", "r1", 1, "test", history)
    second = Request("t", "r2", 2, "test", history + (
        Source("answer", "output", 1, "A" * 10), Source("new", "input", 2, "Q" * 10),
        Source("retrieval", "retrieval", 2, "E" * added)))
    return first, second


def warm_view(policy, added=40):
    state = ThreadState("t")
    first, second = paired_requests(added)
    prepare(first, policy, state, 0)
    return prepare(second, policy, state, 1)


class PositionRiskTests(unittest.TestCase):
    def test_strict_exceedance_finite_sample_and_no_pooling(self):
        rows = [observation(str(i), 2, 100) for i in range(19)]
        rows += [observation(str(i), 3, 500) for i in range(18)]
        _, policy, _ = fitted(rows)
        equal = position_risk(policy, 2, 900, "remaining_tokens")
        self.assertEqual(equal["sample_exceedance_fraction_conservative"], 0)
        self.assertEqual(equal["rank_adjusted_exceedance"], .05)
        self.assertEqual(equal["warm_ceiling_tokens"], 900)
        self.assertFalse(equal["triggered"])
        self.assertTrue(position_risk(policy, 2, 901, "remaining_tokens")["triggered"])
        self.assertEqual(position_risk(policy, 3, 901, "remaining_tokens")["fallback"], "insufficient_position_threads")
        self.assertEqual(position_risk(policy, 4, 901, "remaining_tokens")["fallback"], "unseen_position")
        with self.assertRaises(ContractError):
            fit_position_growth(rows + rows[:1])

    def test_prepare_charges_known_arrivals_once_and_rolls_back_failed_admission(self):
        lows = [observation(str(i), 2, 1, arrival=900) for i in range(20)]
        base, low, profile = fitted(lows)
        old, risk = warm_view(base, 160), warm_view(low, 160)
        self.assertGreater(risk.tokens, 600)
        self.assertFalse(risk.warm_overflow_risk["before"]["triggered"])
        self.assertEqual(risk.action, "applied_warm")
        self.assertNotEqual(old.action, "applied_warm")
        # At the previous run end the incoming retrieval is still unknown.
        prediction = end_run(low, profile, held_input=450, answer_tokens=10,
                             cold_view=200, run_index=1, request_started_at=0, ended_at=1)
        self.assertTrue(prediction.next_run_overflow_risk["triggered"])
        self.assertEqual(prediction.next_run_overflow_risk["run_index"], 2)
        self.assertEqual(prediction.action, "lapse")
        self.assertEqual(low.hash, Policy.from_dict(json.loads(json.dumps(low.to_dict()))).hash)
        with self.assertRaises(TypeError):
            low.position_growth["by_run_index"]["2"]["threads"] = 0

        # Crossing the separate admission cap overrides a low growth forecast.
        partial = warm_view(low, 450)
        self.assertTrue(partial.admitted)
        self.assertEqual(partial.action, "replaced_warm_for_admission")
        self.assertTrue(partial.warm_overflow_risk["admission_override"])
        self.assertFalse(partial.warm_overflow_risk["before"]["triggered"])
        first, second = paired_requests(1000)
        state = ThreadState("t")
        prepare(first, low, state, 0)
        self.assertFalse(prepare(second, low, state, 1).admitted)
        self.assertEqual(state.last_run_id, "r1")
        self.assertEqual(state.removed, set())

    def test_high_tool_peak_triggers_below_fixed_threshold_and_survives_partial_target(self):
        rows = [observation(str(i), 2, 600) for i in range(20)]
        base, risk, _ = fitted(rows)
        fixed, prepared = warm_view(base), warm_view(risk)
        self.assertEqual(fixed.action, "applied_warm")
        self.assertLess(prepared.warm_overflow_risk["before"]["base_tokens"], 600)
        self.assertTrue(prepared.warm_overflow_risk["before"]["triggered"])
        self.assertFalse(prepared.warm_overflow_risk["after"]["triggered"])
        self.assertEqual(prepared.action, "replaced_warm")

        # The protected current input stops the cut above its target. Keep a
        # partial cut that reduces the forecast even if it cannot clear alpha.
        mixed = [observation(str(i), 2, 600 if i < 14 else 960) for i in range(20)]
        _, policy, _ = fitted(mixed)
        partial = warm_view(policy, 170)
        self.assertGreater(partial.tokens, 150)
        self.assertEqual(partial.action, "replaced_warm")
        before, after = (partial.warm_overflow_risk[k] for k in ("before", "after"))
        self.assertTrue(after["triggered"])
        self.assertLess(after["rank_adjusted_exceedance"], before["rank_adjusted_exceedance"])

    def test_sparse_fallback_matches_legacy_and_recovery_does_not_forecast_twice(self):
        legacy, sparse, _ = fitted([observation("only", 2, 1000)])
        fixed, view = warm_view(legacy, 160), warm_view(sparse, 160)
        self.assertEqual(view.wire, fixed.wire)
        self.assertIsNone(view.warm_overflow_risk["before"]["triggered"])
        state = ThreadState("t")
        first, second = paired_requests()
        prepare(first, sparse, state, 0)
        prepare(second, sparse, state, 1)
        self.assertIsNone(prepare(second, sparse, state, 2).warm_overflow_risk)

    def test_sidecar_counts_censoring_and_masks_without_imputing_zero(self):
        t = thread()
        spec = ModelSpec("test", context_tokens=1000)
        peaks = [{"run_id": t.runs[0].run_id, "status": "complete",
                  "start_input_tokens": 200, "peak_context_tokens": 800},
                 {"run_id": t.runs[1].run_id, "status": "censored",
                  "start_input_tokens": 250, "peak_context_tokens": 300}]
        prof = fit_profile([t], [], spec, observation_end=END, peak_observations=peaks)
        rows = prof["position_growth"]["by_run_index"]
        self.assertEqual(rows["1"]["remaining_tokens"], [600])
        self.assertEqual(rows["2"]["statuses"], {"censored": 1})
        self.assertEqual(rows["3"]["statuses"], {"missing": 1})
        proxy = fit_profile([t], [], spec, observation_end=END)["position_growth"]
        self.assertEqual(proxy["by_run_index"]["1"]["sources"], {"single_generation_proxy": 1})
        with self.assertRaises(ContractError):
            fit_profile([t], [], spec, observation_end=END, peak_observations=[dict(peaks[0], run_id="unknown")])
        unknown = [observation(str(i), 2, 20) for i in range(18)] + [
            observation("censored", 2, 1, status="censored")]
        _, policy, _ = fitted(unknown)
        result = position_risk(policy, 2, 10, "remaining_tokens")
        self.assertTrue(result["triggered"])
        self.assertEqual(result["unbounded_observations"], 1)
        self.assertIsNone(result["growth_quantile_tokens"])
        self.assertEqual(result["sample_exceedance_fraction_conservative"], 1/19)

    def test_joint_next_admission_and_peak_count_opportunities_once(self):
        # All 19 upcoming runs fit the total window but none can preserve the
        # warm prefix at the separate admission cap. Positive EV cannot override it.
        base, policy, profile = fitted([observation(str(i), 2, 0, arrival=100) for i in range(19)])
        result = end_run(policy, profile, held_input=800, answer_tokens=40, cold_view=900,
                         run_index=1, request_started_at=0, ended_at=1)
        self.assertEqual(result.action, "lapse")
        r = result.next_run_overflow_risk
        self.assertEqual(r["bounded_pair_admission_exceedances"], 19)
        self.assertEqual(r["bounded_pair_peak_exceedances"], 0)
        self.assertEqual(r["bounded_pair_joint_exceedances"], 19)
        self.assertIn("admission or peak", result.reason)

        # Two distinct marginal failures plus an overlapping failure count as 3.
        mix = [observation(str(i), 2, rem, arrival=arr) for i, (arr, rem) in
               enumerate([(70, 0), (10, 160), (100, 100), (60, 100)] + [(0,0)] * 15)]
        _, mixed, _ = fitted(mix)
        both = position_risk(mixed, 2, 840, "next_admission_or_peak")
        self.assertEqual((both["bounded_pair_admission_exceedances"], both["bounded_pair_peak_exceedances"],
                          both["bounded_pair_joint_exceedances"]), (2,2,3))
        self.assertEqual(both["rank_adjusted_exceedance"], 4/20)
        # A v1 growth artifact round-trips and only has enough information for
        # a conservative test against the tighter cap.
        old = policy.to_dict()
        model = old["position_growth"]
        model["schema_version"] = "compache-position-growth-v1"
        for row in model["by_run_index"].values():
            del row["arrival_and_whole_tokens"]
        old_policy = Policy.from_dict(old)
        fallback = position_risk(old_policy, 2, 840, "next_admission_or_peak")
        self.assertEqual(fallback["joint_basis"], "whole_run_against_input_cap_conservative")
        self.assertTrue(fallback["triggered"])

    def test_supported_heldout_realized_overflows_veto_candidate_and_baseline(self):
        from compache.traces import Thread
        def spaced(tid, company):
            data = thread(tid, company=company, n=4, gap=60, body_size=360).to_dict()
            for run in data["runs"][2:]:
                for obj in (run, run["generation"]):
                    for key in ("started_at", "ended_at"):
                        obj[key] = (datetime.fromisoformat(obj[key]) + timedelta(seconds=540)).isoformat()
            return Thread.from_dict(data)
        train = [spaced(str(i), "train") for i in range(19)]
        test = spaced("test", "heldout")
        traces = train + [test]
        peaks = [{"run_id": r.run_id, "start_input_tokens": 10,
                  "peak_context_tokens": 990 if t is test else 20, "status": "complete"}
                 for t in traces for r in t.runs]
        result = optimize(traces, [row for t in traces for row in labels_for(t)], ModelSpec("test", context_tokens=1000),
                          heldout_company="heldout", observation_end=END, max_pings=0,
                          max_overflow_probability=.05, peak_observations=peaks)
        checks = result["heldout"]["summary"]["warm_outcome_validation"]
        self.assertEqual(checks["status"], "exceeds_budget")
        self.assertEqual(checks["supported_warm_opportunities"], 2)
        self.assertTrue(all(v["exceedances"] == 1 for v in checks["by_run_index"].values()))
        self.assertFalse(result["result"]["heldout_constraints_satisfied"])
        self.assertFalse(result["result"]["candidate_promoted"])
        self.assertIsNone(result["recommended_policy"])

    def test_warm_validation_missing_tails_and_fallback_have_explicit_denominators(self):
        from compache.optimizer.replay import warm_outcome_validation
        _, policy, _ = fitted([observation(str(i), 2, 10) for i in range(19)])
        forecast = position_risk(policy, 2, 200, "remaining_tokens")
        def outcome(identified):
            return {"window": "warm", "run_index": 2, "action": "applied_warm",
                    "warm_overflow_risk":{"before": forecast},
                    "frozen_growth_outcome":{"modeled_overflow": identified, "source": "observed_peak"}}
        def status(rows): return warm_outcome_validation(rows, policy)["status"]
        self.assertEqual(status([]), "no_supported_warm_opportunities")
        self.assertEqual(status([outcome(False)]), "insufficient_validation_threads")
        self.assertEqual(status([outcome(False)]*19), "within_empirical_budget")
        self.assertEqual(status([outcome(False)]*18 + [outcome(None)]), "inconclusive_missing")
        self.assertEqual(status([outcome(False)]*19 + [outcome(None)]), "within_empirical_budget")
        self.assertEqual(status([outcome(False)]*18 + [outcome(True)]), "exceeds_budget")

    def test_duplicate_run_keys_rejected_and_reported_output_preferred_to_proxy(self):
        from compache.traces import Thread
        t = thread()
        data = t.to_dict()
        data["thread_id"] = "different"
        with self.assertRaisesRegex(ContractError, "unique"):
            fit_profile([t, Thread.from_dict(data)], [], ModelSpec("test"), observation_end=END)
        data["runs"][0]["generation"]["usage"] = {"input": 80, "output": 600, "cache_read": 0, "cache_write": 0}
        profile = fit_profile([Thread.from_dict(data)], [], ModelSpec("test"), observation_end=END)
        row = profile["position_growth"]["by_run_index"]["1"]
        self.assertEqual(row["remaining_tokens"], [600])
        self.assertEqual(row["sources"], {"single_generation_reported_output": 1})

    def test_cli_solver_family_training_is_disjoint_and_profiles_are_reusable(self):
        train = thread("training", n=2, body_size=50)
        test = thread("heldout", company="heldout", n=2, body_size=50)
        spec = ModelSpec("test", context_tokens=5000, cache_ttl_seconds=1800)
        peaks = [{"run_id": r.run_id, "start_input_tokens": 20,
                  "peak_context_tokens": 30 if t is train else 4900, "status": "complete"}
                 for t in [train, test] for r in t.runs]
        result = optimize([train, test], labels_for(train) + labels_for(test), spec,
                          observation_end=END, heldout_company="heldout", max_pings=0,
                          max_overflow_probability=.05, peak_observations=peaks)
        self.assertFalse(result["selection"]["heldout_used_for_selection"])
        self.assertEqual(result["profile"]["position_growth"]["by_run_index"]["2"]["remaining_tokens"], [10])
        family = list(candidates(result["profile"], spec, max_pings=0, max_overflow_probability=.05))
        self.assertTrue(all(p.ingress_match_min_chars is None and p.position_growth is not None for p in family))
        self.assertEqual(result["heldout"]["runs"][0]["frozen_growth_outcome"]["remaining_tokens"], 4880)
        profile = freeze_profile(result["profile"])
        again = replay([test], labels_for(test), Policy.from_dict(result["policy"]), profile,
                       observation_end=END, peak_observations=peaks[2:])
        self.assertEqual(again["runs"], result["heldout"]["runs"])
        self.assertIs(profile, freeze_profile(profile))
        self.assertEqual(profile.fingerprint, digest(result["profile"]))


if __name__ == "__main__":
    unittest.main()
