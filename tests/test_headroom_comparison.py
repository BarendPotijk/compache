"""Comparison contract tests. Fake compressor results are never benchmark evidence."""
import copy
import unittest
from dataclasses import replace

from test_optimizer import thread, labels_for
from compache.comparison import compare_headroom, replay_headroom
from compache.optimizer import ModelSpec, Policy, fit_profile
from compache.optimizer.contracts import digest
from compache.traces import ContractError, Thread

END = "2026-01-03T00:00:00+00:00"


class IdentityCompressor:
    provenance = {"package": "test-double-not-headroom"}

    def __init__(self):
        self.calls = []

    def compress(self, messages, *, frozen, context_tokens):
        self.calls.append((copy.deepcopy(messages), frozen))
        return copy.deepcopy(messages), {"lossless_verified_by_headroom": True,
                                        "compression_seconds_measured": 0.0}


class HeadroomComparisonTests(unittest.TestCase):
    def fixture(self, gap=60, context=100000):
        t = thread(gap=gap)
        labels = labels_for(t, need_old=True)
        spec = ModelSpec("test", context_tokens=context)
        profile = fit_profile([t], labels, spec, observation_end=END)
        policy = Policy(spec.to_dict(), digest(profile))
        return t, labels, policy, profile

    def test_shared_accounting_and_source_coverage(self):
        t, labels, policy, profile = self.fixture()
        result = compare_headroom([t], labels, policy, profile, IdentityCompressor(), observation_end=END)
        for arm in result["baselines"].values():
            self.assertTrue(arm["comparable_cost"])
            self.assertAlmostEqual(arm["compache_saving_fraction"], 0)
            self.assertEqual(arm["ledger"], result["compache"]["ledger"])

    def test_warm_freeze_and_cold_refresh_are_explicit(self):
        t, labels, policy, _ = self.fixture(gap=400)
        for cold_refresh, expected in ((False, [0, 2, 5]), (True, [0, 0, 0])):
            backend = IdentityCompressor()
            replay_headroom([t], labels, policy, backend, observation_end=END,
                            cold_refresh=cold_refresh)
            self.assertEqual([n for _, n in backend.calls], expected)

    def test_state_is_isolated_per_thread(self):
        t, labels, policy, _ = self.fixture()
        other = thread(tid="second", gap=60)
        backend = IdentityCompressor()
        replay_headroom([t, other], labels + labels_for(other), policy, backend, observation_end=END)
        self.assertEqual([n for _, n in backend.calls], [0, 2, 5, 0, 2, 5])

    def test_missing_text_suppresses_comparison(self):
        t, labels, policy, profile = self.fixture()
        data = t.to_dict()
        data["runs"][-1]["output"]["text"] = None
        masked = Thread.from_dict(data)
        result = compare_headroom([masked], labels[:-1], policy, profile, IdentityCompressor(), observation_end=END)
        for arm in result["baselines"].values():
            self.assertFalse(arm["comparable_cost"])
            self.assertIsNone(arm["compache_saving_fraction"])
            self.assertEqual(arm["summary"]["unavailable_text_runs"], 1)

    def test_admission_failure_is_not_a_savings_claim(self):
        t, labels, policy, profile = self.fixture(context=100)
        result = compare_headroom([t], labels, policy, profile, IdentityCompressor(), observation_end=END)
        for arm in result["baselines"].values():
            self.assertFalse(arm["comparable_cost"])
            self.assertIsNone(arm["compache_saving_fraction"])
            self.assertEqual(arm["summary"]["admission_failures"], 1)

    def test_frozen_prefix_violation_is_rejected(self):
        t, labels, policy, _ = self.fixture()
        class Bad(IdentityCompressor):
            def compress(self, messages, **kwargs):
                output, receipt = super().compress(messages, **kwargs)
                if kwargs["frozen"]:
                    output[0]["content"] = "changed"
                return output, receipt
        with self.assertRaisesRegex(ContractError, "frozen prefix"):
            replay_headroom([t], labels, policy, Bad(), observation_end=END)

    def test_profile_mismatch_rejected_before_baseline(self):
        t, labels, policy, profile = self.fixture()
        backend = IdentityCompressor()
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            compare_headroom([t], labels, replace(policy, profile_hash="wrong"), profile,
                             backend, observation_end=END)
        self.assertEqual(backend.calls, [])
