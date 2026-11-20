"""Foreground returns replace a native ping's maintenance-only tail."""

import tempfile
import unittest
from pathlib import Path

from compache.optimizer import (
    CacheBreakChecker,
    CacheScope,
    ModelSpec,
    Request,
    RetentionRecipe,
    Source,
    WorkloadPolicy,
    WorkloadSession,
    fit_workload,
)
from compache.traces import Thread
from test_distribution_planner import cohort
from helpers import END, labels_for, thread
from test_ping_continuation import continuation, receipt


class PrivatePingSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = thread(n=4, gap=700)
        cls.profile = fit_workload(
            [train],
            labels_for(train),
            ModelSpec(
                "test",
                context_tokens=131072,
                cache_min_tokens=1024,
                ping_output_tokens=5,
            ),
            observation_end=END,
            recipes=[RetentionRecipe()],
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "guard.sqlite"
        self.guard = CacheBreakChecker(self.database)
        self.addCleanup(self.guard.close)
        policy = WorkloadPolicy(self.profile, lookahead=2, branches=1, max_pings=2)
        self.scope = CacheScope("host", "provider", "test", "t")
        self.session = WorkloadSession(policy, self.scope, self.guard)
        self.question = Source("q", "input", 1, "q" * 32800)
        self.answer = Source("a", "output", 1, "answer")
        self.tools = [{"name": "retrieve"}]
        plan = self.session.prepare(
            Request("t", "r", 1, "test", (self.question,)),
            started_at=0,
            system_prompt="stable",
            tools=self.tools,
        )
        # Native tool-bearing seed receipt. The remaining 75 input tokens are fresh.
        self.session.complete(
            plan,
            self.answer,
            ended_at=1,
            input_tokens=8873,
            cached_input_tokens=0,
            cache_write_input_tokens=8798,
            response_id="seed",
        )
        self.assertEqual(self.session.next_ping_at, 255)

    def ping(self):
        return self.session.acknowledge_ping(
            run_id="r",
            at=255,
            ping_has_private_suffix=True,
            input_tokens=8921,
            cached_input_tokens=8798,
            cache_write_input_tokens=48,
            output_tokens=5,
            response_id="ping",
        )

    def test_live_tool_counters_and_exact_repeat(self):
        self.assertTrue(self.ping())
        event = self.session.last_ping_receipt
        self.assertEqual(event["refreshed_reusable_tokens"], 8846)
        self.assertEqual(event["foreground_reusable_tokens"], 8798)
        self.assertFalse(event["occupancy_anchored"])
        self.assertEqual(self.session.ping_view(run_id="r", at=510).tokens, 8921)
        self.assertAlmostEqual(
            event["receipt_cost_input_equivalents"], 75 + 8798 * 0.1 + 48 * 1.25 + 5 * 4
        )
        # Additional synthetic identical repeat: its larger prefix belongs to
        # the same private ping, even if another receipt omits the optional flag.
        self.assertTrue(
            self.session.acknowledge_ping(
                run_id="r",
                at=510,
                input_tokens=8921,
                cached_input_tokens=8846,
                cache_write_input_tokens=0,
                output_tokens=5,
                response_id="repeat",
            )
        )
        self.assertEqual(
            self.session.last_ping_receipt["expected_reusable_tokens"], 8846
        )
        self.assertEqual(
            self.session.last_ping_receipt["foreground_reusable_tokens"], 8798
        )
        plan = self.session.prepare(
            Request(
                "t",
                "return",
                2,
                "test",
                (
                    self.question,
                    self.answer,
                    Source("q2", "input", 2, "next question"),
                ),
            ),
            started_at=520,
            system_prompt="stable",
            tools=self.tools,
            shared_prefix_upper_bound=8798,
        )
        self.assertEqual(
            plan.decision["input_accounting"]["anchor_response_id"], "seed"
        )
        # Measured native return omits the private maintenance message.
        result = self.session.complete(
            plan,
            Source("a2", "output", 2, "next answer"),
            ended_at=521,
            input_tokens=8933,
            cached_input_tokens=8798,
            cache_write_input_tokens=60,
        )["cache_event"]
        self.assertEqual(result["expected_reusable_tokens"], 8798)
        self.assertEqual(result["outcome"], "expected_reuse_observed")
        self.assertFalse(result["pings_disabled"])

    def test_luna_response_defaults_to_private_branch_and_host_can_override(self):
        model = "gpt-6-luna"
        raw = thread(n=4, gap=700).to_dict()
        for run in raw["runs"]:
            run["generation"]["model"] = model
        train = Thread.from_dict(raw)
        profile = fit_workload(
            [train],
            labels_for(train),
            ModelSpec(
                model,
                context_tokens=131072,
                cache_min_tokens=1024,
                ping_output_tokens=5,
            ),
            observation_end=END,
            recipes=[RetentionRecipe()],
        )
        policy = WorkloadPolicy(profile, lookahead=2, branches=1, max_pings=2)
        usage = {
            "input_tokens": 8921,
            "input_tokens_details": {"cached_tokens": 8798, "cache_write_tokens": 48},
            "output_tokens": 5,
        }
        raw_response = {
            "id": "resp_ping",
            "model": model,
            "status": "completed",
            "usage": usage,
        }
        counters = dict(
            input_tokens=8921,
            cached_input_tokens=8798,
            cache_write_input_tokens=48,
            output_tokens=5,
        )
        modeled = dict(modeled_cached_input_tokens=8798, modeled_output_tokens=5)
        for tid, provider, kwargs, private in (
            ("native", "OpenAI", {"response": raw_response}, True),
            (
                "forced-off",
                "openai",
                {"response": raw_response, "ping_has_private_suffix": False},
                False,
            ),
            ("unknown", "gateway", {"response": raw_response}, False),
            ("counters", "openai", counters, False),
            ("modeled", "openai", modeled, False),
        ):
            with self.subTest(tid=tid):
                scope = CacheScope("host", provider, model, tid)
                session = WorkloadSession(policy, scope, self.guard)
                plan = session.prepare(
                    Request(tid, "r", 1, model, (self.question,)),
                    started_at=0,
                    system_prompt="stable",
                    tools=self.tools,
                )
                session.complete(
                    plan,
                    self.answer,
                    ended_at=1,
                    input_tokens=8873,
                    cached_input_tokens=0,
                    cache_write_input_tokens=8798,
                    response_id="resp_seed",
                )
                self.assertTrue(session.acknowledge_ping(run_id="r", at=255, **kwargs))
                event = session.last_ping_receipt
                self.assertEqual(event["ping_has_private_suffix"], private)
                if "response" in kwargs:
                    self.assertEqual(
                        event["foreground_reusable_tokens"], 8798 if private else 8846
                    )
                    self.assertEqual(event["occupancy_anchored"], not private)

    def test_foreground_other_worker_can_still_diagnose_a_real_miss(self):
        self.assertTrue(self.ping())
        with CacheBreakChecker(self.database) as other:
            self.assertTrue(
                other.begin_return(self.scope, run_id="return", started_at=515)
            )
            self.assertFalse(
                self.session.acknowledge_ping(
                    run_id="r",
                    at=510,
                    input_tokens=8921,
                    cached_input_tokens=8846,
                    cache_write_input_tokens=0,
                    output_tokens=5,
                )
            )
            with self.assertLogs("compache.optimizer.cache_breaks", level="WARNING"):
                result = other.observe_return(
                    self.scope,
                    run_id="return",
                    started_at=515,
                    system_prompt="stable",
                    tools=self.tools,
                    ttl_seconds=300,
                    reusable_input_tokens=8858,
                    cached_input_tokens=0,
                )
            self.assertEqual(result["expected_reusable_tokens"], 8798)
            self.assertEqual(result["outcome"], "on_time_cache_miss")
            self.assertTrue(result["pings_disabled"])

    def test_partial_private_repeat_restricts_the_next_foreground(self):
        self.assertTrue(self.ping())
        self.assertFalse(
            self.session.acknowledge_ping(
                run_id="r",
                at=510,
                input_tokens=8921,
                cached_input_tokens=2000,
                cache_write_input_tokens=6846,
                output_tokens=5,
            )
        )
        self.assertEqual(self.session.last_ping_receipt["outcome"], "partial_read")
        self.assertEqual(
            self.session.last_ping_receipt["foreground_reusable_tokens"], 2000
        )
        with CacheBreakChecker(self.database) as other:
            result = other.observe_return(
                self.scope,
                run_id="return",
                started_at=520,
                system_prompt="stable",
                tools=self.tools,
                ttl_seconds=300,
                reusable_input_tokens=8858,
                cached_input_tokens=2000,
            )
        self.assertEqual(result["outcome"], "expected_reuse_observed")
        self.assertEqual(result["expected_reusable_tokens"], 2000)

    def test_repricing_separates_full_repeat_and_canonical_foreground(self):
        # Adding 200 private tokens affects the repeat cost; the real user's
        # input and its 900 reusable shared tokens have the same cost in both runs.
        demand = cohort([(600, 0, 0)])
        observed = dict(receipt(size=1100, reuse=1000), foreground_reusable_tokens=900)
        short = continuation(demand, observed)
        longer = continuation(
            demand, dict(observed, input_tokens=1300, reusable_tokens=1200)
        )
        self.assertEqual(short["costs"][0], longer["costs"][0])
        self.assertAlmostEqual(longer["costs"][1] - short["costs"][1], 20)
