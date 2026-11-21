from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

from compache.optimizer.contracts import ModelSpec, digest
from compache.optimizer.header_traffic import (
    fit_shared_header_traffic,
    header_traffic_signature,
)
from compache.threads import ContractError, Thread
from helpers import thread


SPEC = ModelSpec("test", cache_min_tokens=4, cache_block_tokens=4, cache_ttl_seconds=50)
HEADER = {"system_prompt": b"abcdefgh", "default_tools": b"TOOL", "tokenize": list}


def sample(tid, slots, company="acme"):
    data = thread(tid, company=company, n=len(slots), body_size=1).to_dict()
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for run, (start, end, total) in zip(data["runs"], slots):
        g = run["generation"]
        for key, val in (("started_at", start), ("ended_at", end)):
            run[key] = g[key] = (origin + timedelta(seconds=val)).isoformat()
        if total is not None:
            g["usage"] = {
                "input": total,
                "output": 1,
                "cache_read": 0,
                "cache_write": 0,
            }
    return Thread.from_dict(data)


class HeaderTraffic(unittest.TestCase):
    def test_condition_on_cold_return_events_and_keep_first_run_denominator_separate(
        self,
    ):
        # A2 returns after A1's TTL and before B1's TTL. A3 returns while A2 is warm.
        # B2 is a subsequent cold return with no surviving neighbor.
        a = sample("A", [(0, 2, 20), (90, 94, 20), (100, 102, 20)])
        b = sample("B", [(60, 61, 20), (220, 221, 20)])
        c = sample("C", [(92, 94, None)])
        fit = fit_shared_header_traffic([c, b, a], SPEC, **HEADER)
        self.assertEqual(fit["first_run_hit_probability"], 1 / 3)
        self.assertEqual(fit["ambient_hit_probability"], 0.5)
        self.assertEqual(fit["cold_returns"]["eligible_runs"], 2)
        self.assertEqual(fit["cold_returns"]["matched_header_tokens"], 12)
        self.assertEqual(fit["warm_returns"]["runs"], 1)
        self.assertIsNone(fit["warm_returns"]["probability"])
        self.assertEqual(
            (fit["reported_input_runs"], fit["estimated_input_runs"]), (5, 1)
        )
        group = fit["by_scope"][digest(["acme", "declared", "test", ""])]
        self.assertEqual(group["ambient_hit_probability"], 0.5)

    def test_active_only_after_completion_and_strict_start_based_expiry(self):
        # B starts during A and C starts exactly when A has expired; B is ineligible.
        a = sample("A", [(0, 20, 20)])
        b = sample("B", [(15, 19, 0)])
        c = sample("C", [(50, 51, 20)])
        e = sample("E", [(0, 0, 20)], company="other")
        d = sample("D", [(0, 0, 20)], company="other")
        fit = fit_shared_header_traffic([a, b, c, d, e], SPEC, **HEADER)
        self.assertEqual(fit["first_runs"]["runs"], 5)
        self.assertEqual(fit["first_runs"]["eligible_runs"], 4)
        self.assertEqual(fit["first_run_hit_probability"], 0)
        self.assertEqual(fit["first_runs"]["support"], "no_cross_thread_matches")

    def test_same_header_is_scoped_to_provider_route_company_and_model(self):
        a = sample("confidential-thread", [(10, 11, 20)])
        b = sample("B", [(14, 15, 20)])
        c = sample("C", [(16, 17, 20)], company="OTHER")
        fit = fit_shared_header_traffic(
            [a, b, c], SPEC, provider="p", cache_key="route", **HEADER
        )
        groups = fit["by_scope"]
        self.assertEqual(
            groups[digest(["acme", "p", "test", "route"])]["first_run_hit_probability"],
            0.5,
        )
        self.assertEqual(
            groups[digest(["OTHER", "p", "test", "route"])][
                "first_run_hit_probability"
            ],
            0,
        )
        for private in ("confidential-thread", "abcdefgh", "TOOL", "acme", "route"):
            self.assertNotIn(private, json.dumps(fit))
        with self.assertRaises(ContractError):
            fit_shared_header_traffic([a, a], SPEC, **HEADER)
        with self.assertRaises(ContractError):
            fit_shared_header_traffic([a], replace(SPEC, model_id="other"), **HEADER)

    def test_partial_prior_capacity_and_shorter_offers_do_not_replace_stronger_ones(
        self,
    ):
        a = sample("A", [(0, 1, 30)])
        b = sample("B", [(30, 32, 7)])
        c = sample("C", [(40, 41, 12), (100, 101, 12)])
        d = sample("D", [(97, 98, 7)])
        fit = fit_shared_header_traffic([a, b, c, d], SPEC, **HEADER)
        self.assertEqual(fit["first_runs"]["full_header_hits"], 2)
        self.assertEqual(fit["cold_returns"]["partial_header_hits"], 1)
        self.assertEqual(fit["cold_returns"]["hit_rate"], 1)
        self.assertEqual(fit["ambient_hit_probability"], 4 / 12)

    def test_unsupported_header_and_no_cold_return_fallback(self):
        a = sample("A", [(0, 3, 20), (7, 9, 20)])
        b = sample("B", [(90, 92, 20)])
        fit = fit_shared_header_traffic(
            [a, b], replace(SPEC, cache_min_tokens=16), **HEADER
        )
        self.assertEqual(fit["first_runs"]["support"], "no_eligible_runs")
        self.assertEqual(fit["cold_returns"]["support"], "no_runs")
        self.assertEqual(fit["ambient_hit_probability"], 0)
        self.assertEqual(fit["warm_returns"]["runs"], 1)
        cold = fit_shared_header_traffic(
            [a], replace(SPEC, cache_ttl_seconds=0), **HEADER
        )
        self.assertEqual(cold["cold_returns"]["support"], "no_eligible_runs")

    def test_signature_respects_order_serialization_and_component_boundaries(self):
        tools = [{"type": "a", "n": 5}, {"name": "b"}]
        signature = header_traffic_signature("sys", tools)
        self.assertEqual(
            signature,
            header_traffic_signature(b"sys", b'[{"type":"a","n":5},{"name":"b"}]'),
        )
        self.assertNotEqual(signature, header_traffic_signature("sys", tools[::-1]))
        self.assertNotEqual(
            signature,
            header_traffic_signature("sys", tools, ("default_tools", "system_prompt")),
        )
        self.assertNotEqual(
            header_traffic_signature("ab", "c"), header_traffic_signature("a", "bc")
        )
        self.assertEqual(
            header_traffic_signature("sys", b"ab"),
            header_traffic_signature("sys", None, tool_fragments=[b"a", b"b"]),
        )
