import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from helpers import END, labels_for, thread

from compache.optimizer.cache_breaks import CacheBreakChecker, CacheScope
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.parts import Request, Source
from compache.optimizer.workload import WorkloadPolicy, WorkloadSession, fit_workload
from compache.traces import ContractError


class WorkloadSessionTest(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            "test", context_tokens=100_000, cache_min_tokens=16, cache_block_tokens=8
        )
        train = thread(tid="train", n=3, gap=310)
        self.artifact = fit_workload(
            [train], labels_for(train), self.spec, observation_end=END
        )
        self.policy = WorkloadPolicy(self.artifact, lookahead=2, branches=1)
        self.db = tempfile.TemporaryDirectory()
        self.addCleanup(self.db.cleanup)
        self.check = CacheBreakChecker(Path(self.db.name) / "checker.db")
        self.addCleanup(self.check.close)
        self.system = b"S" * 64
        self.tools = [{"name": "search", "description": "Find records by ID"}]

    def start(self, tid, *, tenant="org", share=True, policy=None, verified=False):
        t = thread(tid=tid, n=3)
        s = WorkloadSession(
            policy or self.policy,
            CacheScope(tenant, "host", "test", tid),
            self.check,
            tokenize=list,
            share_headers=share,
            verified_contiguous_prefix_reads=verified,
        )
        return t, s

    def prepare(self, pair, index, at, **kwargs):
        t, session = pair
        run = t.runs[index - 1]
        kwargs = dict({"system_prompt": self.system, "tools": self.tools}, **kwargs)
        return session.prepare(
            Request.from_thread(t, run.run_id), started_at=at, **kwargs
        )

    def finish(self, pair, plan, *, at=None, **kwargs):
        run = pair[0].runs[plan.request.run_index - 1]
        answer = Source(run.output.part_id, "output", run.run_index, run.output.text)
        return pair[1].complete(
            plan, answer, ended_at=at or plan.started_at + 1, **kwargs
        )

    def test_first_runs_share_only_successfully_published_in_scope(self):
        a, b, other = self.start("a"), self.start("b"), self.start("c", tenant="other")
        first = self.prepare(a, 1, 10)
        self.assertEqual(first.reuse.eligible_tokens, 0)
        self.assertEqual(self.prepare(b, 1, 11).reuse.eligible_tokens, 0)
        self.finish(a, first, input_tokens=2000, cached_input_tokens=0)
        shared = self.prepare(b, 1, 13)
        self.assertTrue(shared.reuse.cross_thread)
        self.assertEqual(
            shared.decision["shared_header_tokens_now"], shared.reuse.eligible_tokens
        )
        self.assertEqual(
            shared.header.default_tool_tokens, sum(shared.header.tool_breakdown)
        )
        self.assertEqual(shared.reuse.system_tokens, 64)
        self.assertGreater(shared.reuse.default_tool_tokens, 0)
        self.assertEqual(self.prepare(other, 1, 13).reuse.eligible_tokens, 0)
        self.assertEqual(
            self.prepare(
                self.start("isolated", share=False), 1, 13
            ).reuse.eligible_tokens,
            0,
        )
        self.assertEqual(
            self.prepare(self.start("late"), 1, 310).reuse.eligible_tokens, 0
        )
        self.assertLess(
            shared.decision["expected_cost"], first.decision["expected_cost"]
        )

    def test_on_time_changed_schema_loses_history_and_disables_timers(self):
        a = self.start("a", verified=True)
        first = self.prepare(a, 1, 10)
        result = self.finish(a, first, at=40, input_tokens=2048, cached_input_tokens=0)
        self.assertTrue(result["due_times"])
        self.assertIsNotNone(
            a[1].ping_view(run_id=first.request.run_id, at=result["due_times"][0])
        )
        second = self.prepare(
            a, 2, 55, tools=[{"name": "different", "description": "longer" * 90}]
        )
        self.assertIsNone(
            a[1].ping_view(run_id=first.request.run_id, at=result["due_times"][0])
        )
        self.assertEqual(second.reuse.system_tokens, 64)
        self.assertLess(second.reuse.eligible_tokens, first.header.static_tokens)
        with self.assertLogs("compache.optimizer.cache_breaks", level="WARNING"):
            result = self.finish(
                a,
                second,
                input_tokens=3000,
                cached_input_tokens=second.reuse.eligible_tokens,
            )
        self.assertEqual(result["cache_event"]["outcome"], "on_time_history_cache_miss")
        self.assertEqual(result["cache_event"]["changed_components"], ["tools"])
        self.assertEqual(result["due_times"], ())
        with CacheBreakChecker(Path(self.db.name) / "checker.db") as reopened:
            self.assertFalse(reopened.can_ping(a[1].scope))
            self.assertEqual(len(reopened.events(a[1].scope)), 1)

    def test_large_provider_counter_beyond_estimator_is_valid(self):
        a = self.start("a")
        plan = self.prepare(a, 1, 10)
        count = plan.view.tokens + 1000
        self.finish(a, plan, input_tokens=count, cached_input_tokens=count)
        self.assertTrue(self.check.can_ping(a[1].scope))

    def test_future_rate_requires_same_header_provider_and_company(self):
        from compache.optimizer.contracts import digest
        from compache.optimizer.header_traffic import header_traffic_signature

        current = self.prepare(self.start("first"), 1, 1).header
        assumption = {
            "provider": "host",
            "header_signature": header_traffic_signature(self.system, self.tools),
            "count_source": current.source,
            "static_tokens": current.static_tokens,
            "by_scope": {
                digest(["org", "host", "test", ""]): {
                    "ambient_hit_probability": 0.75,
                    "cold_returns": {"support": "calendar_estimate"},
                }
            },
        }
        policy = WorkloadPolicy(dict(self.artifact, header_traffic=assumption))
        same = self.prepare(self.start("same", policy=policy), 1, 1)
        self.assertEqual(same.decision["ambient_hit_probability"], 0.75)
        self.assertEqual(same.decision["shared_header_tokens_now"], 0)
        for kwargs, identifier in (
            ({"system_prompt": b"different"}, "changed"),
            ({}, "unseen"),
        ):
            pair = self.start(
                identifier,
                tenant="unseen" if identifier == "unseen" else "org",
                policy=policy,
            )
            self.assertEqual(
                self.prepare(pair, 1, 1, **kwargs).decision["ambient_hit_probability"],
                0,
            )
        foreign_policy = WorkloadPolicy(
            dict(self.artifact, header_traffic={**assumption, "provider": "another"})
        )
        self.assertEqual(
            self.prepare(
                self.start("other-route", policy=foreign_policy), 1, 1
            ).decision["ambient_hit_probability"],
            0,
        )

    def test_verified_own_limit_preserves_independent_cross_thread_hit(self):
        a, b = self.start("a"), self.start("b")
        pa = self.prepare(a, 1, 10)
        self.finish(a, pa, input_tokens=1500)
        bounded = self.prepare(a, 2, 12, shared_prefix_upper_bound=0)
        self.assertEqual(bounded.reuse.eligible_tokens, 0)
        self.assertEqual(
            bounded.decision["header"]["planned_input_cost"]["components"][
                "system_prompt"
            ]["read_tokens"],
            0,
        )
        pb = self.prepare(b, 1, 13)
        self.finish(b, pb, input_tokens=1500)
        independent = self.prepare(a, 2, 20, shared_prefix_upper_bound=0)
        self.assertTrue(independent.reuse.cross_thread)
        self.assertGreater(independent.reuse.eligible_tokens, 0)

    def test_ineligible_authoritative_capacity_overrides_local_estimate(self):
        pair = self.start("a")
        first = self.prepare(pair, 1, 10)
        self.finish(pair, first, reusable_input_tokens=0)
        second = self.prepare(pair, 2, 15)
        self.assertEqual(second.reuse.eligible_tokens, 0)
        cost = second.decision["header"]["planned_input_cost"]
        self.assertEqual(
            cost["header_input_equivalents"],
            cost["same_header_fully_cold_input_equivalents"],
        )

    def test_transport_rejects_header_even_if_body_fits(self):
        spec = replace(self.spec, byte_limit=3000)
        artifact = dict(self.artifact, model_spec=spec.to_dict())
        a = self.start("a", policy=WorkloadPolicy(artifact))
        with self.assertRaisesRegex(ContractError, "no admissible"):
            self.prepare(a, 1, 10, system_prompt="X" * 3000)
