"""Validate only the previously reusable input on an actual keep-warm call."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compache.optimizer import (
    CacheBreakChecker,
    CacheScope,
    ModelSpec,
    Request,
    RetentionRecipe,
    SharedPrefixCatalog,
    Source,
    WorkloadPolicy,
    WorkloadSession,
    fit_workload,
)
from compache.errors import ContractError
from compache.optimizer.distribution import plan_distribution_idle
from helpers import END, labels_for, thread


class PingReceipts(unittest.TestCase):
    def policy_for(self, gap, max_pings, **model_settings):
        train = thread(tid="training", n=4, gap=gap)
        spec = ModelSpec(
            "test",
            context_tokens=10_000,
            cache_min_tokens=128,
            cache_block_tokens=128,
            ping_output_tokens=3,
        )
        spec = replace(spec, **model_settings)
        return WorkloadPolicy(
            fit_workload(
                [train],
                labels_for(train),
                spec,
                observation_end=END,
                recipes=[RetentionRecipe()],
            ),
            lookahead=2,
            branches=1,
            max_pings=max_pings,
        )

    def setUp(self):
        self.policy = self.policy_for(700, 2)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / "guard.sqlite"
        self.guard = CacheBreakChecker(self.db)
        self.addCleanup(self.guard.close)
        self.meter_calls = []
        self.question = Source("q", "input", 1, "q" * 1800)
        self.answer = Source("a", "output", 1, "a" * 300)

    def start(self, tid="thread", verified=True, due=(255, 510), foreground=None):
        def count(part):
            self.meter_calls.append(part.part_id)
            return len(part.text)

        def tokenize(wire):
            self.meter_calls.append("header")
            return list(wire)

        session = WorkloadSession(
            self.policy,
            CacheScope("installation", "provider", "test", tid),
            self.guard,
            tokenize=tokenize,
            count_part_tokens=count,
            verified_contiguous_prefix_reads=verified,
        )
        plan = session.prepare(
            Request(tid, "r", 1, "test", (self.question,)),
            started_at=0,
            system_prompt=b"s" * 256,
            tools=b"",
        )
        done = session.complete(
            plan,
            self.answer,
            ended_at=1,
            **(
                {"input_tokens": 2048, "response_id": "answer"}
                if foreground is None
                else foreground
            ),
        )
        self.assertEqual(done["due_times"], due)
        return session

    def next_plan(self, session, at):
        req = Request(
            session.scope.thread_id,
            "next",
            2,
            "test",
            (self.question, self.answer, Source("new-q", "input", 2, "q")),
        )
        return session.prepare(req, started_at=at, system_prompt=b"s" * 256, tools=b"")

    def next_account(self, session, at):
        return self.next_plan(session, at).decision["input_accounting"]

    def response(self, *, reads=2048, output=3, status="completed"):
        return {
            "type": f"response.{status}",
            "response": {
                "id": "ping",
                "model": "test",
                "status": status,
                "usage": {
                    "input_tokens": 2359,
                    "input_tokens_details": {"cached_tokens": reads},
                    "output_tokens": output,
                },
            },
        }

    def test_first_ping_reads_prior_blocks_then_repeat_reads_larger_capacity(self):
        s = self.start()
        calls = tuple(self.meter_calls)
        self.assertTrue(
            s.acknowledge_ping(run_id="r", at=255, response=self.response())
        )
        receipt = s.last_ping_receipt
        self.assertEqual(receipt["expected_reusable_tokens"], 2048)
        self.assertEqual(receipt["input_tokens"], 2359)
        self.assertEqual(receipt["outcome"], "refresh_confirmed")
        self.assertEqual(receipt["verified_static_read_tokens"], 256)
        self.assertEqual(receipt["verified_private_read_tokens"], 1792)
        self.assertTrue(
            s.acknowledge_ping(
                run_id="r",
                at=510,
                input_tokens=2359,
                cached_input_tokens=2304,
                output_tokens=0,
            )
        )
        self.assertEqual(s.last_ping_receipt["expected_reusable_tokens"], 2304)
        self.assertEqual(tuple(self.meter_calls), calls)

    def test_provider_write_boundary_for_seed_and_exact_ping_repeat(self):
        self.policy = self.policy_for(
            700, 2, cache_block_tokens=1, context_tokens=131072, ping_output_tokens=5
        )
        # Saved native Responses receipts across three independent Luna threads.
        # Each call had three ordinary input tokens outside its read/write prefix.
        # Inject synthetic reasoning usage to check output cannot inflate the prefix.
        for seed_writes, ping_writes in ((8493, 46), (8415, 47), (8559, 49)):
            with self.subTest(seed=seed_writes):
                s = self.start(
                    str(seed_writes),
                    foreground={
                        "response": {
                            "status": "completed",
                            "usage": {
                                "input_tokens": seed_writes + 3,
                                "input_tokens_details": {
                                    "cached_tokens": 0,
                                    "cache_write_tokens": seed_writes,
                                },
                                "output_tokens": 29,
                                "output_tokens_details": {"reasoning_tokens": 10},
                            },
                        },
                    },
                )
                published = seed_writes + ping_writes
                for at, reads, writes in (
                    (255, seed_writes, ping_writes),
                    (510, published, 0),
                ):
                    self.assertTrue(
                        s.acknowledge_ping(
                            run_id="r",
                            at=at,
                            input_tokens=published + 3,
                            cached_input_tokens=reads,
                            cache_write_input_tokens=writes,
                            output_tokens=5,
                        )
                    )
                    self.assertEqual(
                        s.last_ping_receipt["expected_reusable_tokens"], reads
                    )
                    self.assertEqual(
                        s.last_ping_receipt["refreshed_reusable_tokens"], published
                    )
                    self.assertEqual(s.last_ping_receipt["input_tokens"], published + 3)

    def test_empty_write_coverage_does_not_hide_observed_zero_reads(self):
        s = self.start()
        with self.assertLogs("compache.optimizer.cache_breaks", "WARNING"):
            self.assertFalse(
                s.acknowledge_ping(
                    run_id="r",
                    at=255,
                    input_tokens=2359,
                    cached_input_tokens=0,
                    cache_write_input_tokens=0,
                    output_tokens=3,
                )
            )
        self.assertEqual(s.last_ping_receipt["outcome"], "zero_read")
        self.assertTrue(s.last_ping_receipt["pings_disabled"])

    def test_smaller_of_verified_and_reported_boundary_and_missing_counters(self):
        for tid, foreground, expected in (
            (
                "capped",
                {
                    "input_tokens": 2048,
                    "cached_input_tokens": 512,
                    "cache_write_input_tokens": 1024,
                    "reusable_input_tokens": 2048,
                },
                1536,
            ),
            ("missing-write", {"input_tokens": 2048, "cached_input_tokens": 512}, 2048),
            (
                "missing-read",
                {"input_tokens": 2048, "cache_write_input_tokens": 512},
                2048,
            ),
        ):
            with self.subTest(tid=tid):
                s = self.start(tid, foreground=foreground)
                self.assertTrue(
                    s.acknowledge_ping(
                        run_id="r",
                        at=255,
                        input_tokens=2359,
                        cached_input_tokens=expected,
                        output_tokens=2,
                    )
                )
                self.assertEqual(
                    s.last_ping_receipt["expected_reusable_tokens"], expected
                )

    def test_foreground_reads_compared_with_prior_even_if_current_published_less(self):
        s = self.start()
        plan = self.next_plan(s, 200)
        result = s.complete(
            plan,
            Source("a2", "output", 2, "ok"),
            ended_at=201,
            input_tokens=2400,
            cached_input_tokens=1024,
            cache_write_input_tokens=0,
        )
        event = result["cache_event"]
        self.assertEqual(event["expected_reusable_tokens"], 2048)
        self.assertEqual(event["outcome"], "partial_reuse_observed")

    def test_mixed_retention_requires_host_verified_boundary(self):
        def response():
            raw = self.response()
            details = raw["response"]["usage"]["input_tokens_details"]
            details.update(
                cache_write_tokens=256,
                cache_write_tokens_details={
                    "ttl_30m_tokens": 128,
                    "ttl_12h_tokens": 128,
                },
            )
            return raw

        uncertain = self.start("mixed")
        self.assertFalse(
            uncertain.acknowledge_ping(run_id="r", at=255, response=response())
        )
        self.assertEqual(
            uncertain.last_ping_receipt["outcome"], "prefix_capacity_unknown"
        )
        self.assertFalse(uncertain.last_ping_receipt["pings_disabled"])
        known = self.start(
            "bounded",
            foreground={
                "input_tokens": 2048,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 2048,
                "reusable_input_tokens": 1024,
            },
        )
        self.assertTrue(
            known.acknowledge_ping(
                run_id="r",
                at=255,
                response=response(),
                reusable_input_tokens=1280,
            )
        )
        self.assertEqual(known.last_ping_receipt["expected_reusable_tokens"], 1024)
        self.assertEqual(known.last_ping_receipt["refreshed_reusable_tokens"], 1280)

    def test_zero_and_verified_header_only_latch_on_the_ping_itself(self):
        for tid, read, expected in (
            ("zero", 0, "zero_read"),
            ("header", 256, "static_only"),
        ):
            s = self.start(tid)
            with self.assertLogs("compache.optimizer.cache_breaks", "WARNING"):
                confirmed = s.acknowledge_ping(
                    run_id="r", at=255, response=self.response(reads=read)
                )
            self.assertFalse(confirmed)
            self.assertEqual(s.last_ping_receipt["outcome"], expected)
            with CacheBreakChecker(self.db) as other:
                self.assertFalse(other.can_ping(s.scope))
                self.assertEqual(other.events(s.scope), [s.last_ping_receipt])
            self.assertIsNone(s.ping_view(run_id="r", at=510))
            self.assertEqual(self.next_account(s, 550)["anchor_response_id"], "ping")

    def test_partial_and_unknown_receipts_stop_current_chain(self):
        for tid, read, output, expected in (
            ("partial", 1024, 3, "partial_read"),
            ("opaque", 256, 3, "partial_read"),
            ("missing-read", None, 0, "unknown"),
            ("no-output", 2048, None, "output_unknown"),
        ):
            with self.subTest(tid=tid):
                s = self.start(tid, verified=tid != "opaque")
                self.assertFalse(
                    s.acknowledge_ping(
                        run_id="r",
                        at=255,
                        response=self.response(reads=read, output=output),
                    )
                )
                receipt = s.last_ping_receipt
                self.assertEqual(receipt["outcome"], expected)
                self.assertEqual(receipt["observed_cached_tokens"], read)
                if tid == "opaque":
                    self.assertIsNone(receipt["verified_private_read_tokens"])
                with CacheBreakChecker(self.db) as other:
                    self.assertTrue(other.can_ping(s.scope))
                    self.assertFalse(other.can_ping(s.scope, run_id="r", at=260))
                    self.assertFalse(
                        other.note_refresh(s.scope, run_id="r", refreshed_at=260)
                    )
                self.assertIsNone(s.ping_view(run_id="r", at=510))
                self.assertEqual(
                    self.next_account(s, 520)["anchor_response_id"], "ping"
                )

    def test_miss_caps_current_foreground_and_only_its_own_catalog_entries(self):
        s = self.start("sole-publisher")
        with self.assertLogs("compache.optimizer.cache_breaks", "WARNING"):
            s.acknowledge_ping(run_id="r", at=255, response=self.response(reads=0))
        plan = self.next_plan(s, 270)
        self.assertEqual(plan.reuse.eligible_tokens, 0)
        self.assertEqual(
            plan.decision["header"]["planned_input_cost"]["components"][
                "system_prompt"
            ]["read_tokens"],
            0,
        )
        self.assertEqual(
            plan.decision["input_accounting"]["anchor_response_id"], "ping"
        )
        s = self.start("shared-header")
        self.start("independent-publisher")
        with self.assertLogs("compache.optimizer.cache_breaks", "WARNING"):
            s.acknowledge_ping(run_id="r", at=255, response=self.response(reads=0))
        plan = self.next_plan(s, 270)
        self.assertEqual(plan.reuse.eligible_tokens, 256)
        self.assertTrue(plan.reuse.cross_thread)

    def test_output_overrun_refreshes_then_compares_remaining_costs(self):
        for tid, output, expected_decision in (
            ("cheap", 4, "continue"),
            ("expensive", 900, "stop"),
        ):
            with self.subTest(tid=tid):
                self.policy.catalog = SharedPrefixCatalog()
                s = self.start(tid)
                meter_calls = tuple(self.meter_calls)
                response = self.response(output=output)
                response["response"]["usage"]["input_tokens_details"][
                    "cache_write_tokens"
                ] = 128
                self.assertTrue(
                    s.acknowledge_ping(run_id="r", at=255, response=response)
                )
                self.assertEqual(tuple(self.meter_calls), meter_calls)
                receipt = s.last_ping_receipt
                self.assertTrue(receipt["refresh_confirmed"])
                self.assertTrue(receipt["output_budget_exceeded"])
                self.assertFalse(receipt["output_within_budget"])
                self.assertEqual(receipt["refreshed_at"], 255)
                self.assertEqual(receipt["refreshed_reusable_tokens"], 2176)
                self.assertAlmostEqual(
                    receipt["receipt_cost_input_equivalents"],
                    2048 * 0.1 + 128 * 1.25 + (2359 - 2048 - 128) + output * 4,
                )
                next_ping = receipt["continuation"]
                uncached_rate = (183 + 128 * 1.25) / 311
                self.assertAlmostEqual(next_ping["uncached_input_rate"], uncached_rate)
                self.assertEqual(
                    next_ping["uncached_input_rate_source"], "receipt_write_fraction"
                )
                self.assertAlmostEqual(
                    next_ping["estimated_next_ping_cost_input_equivalents"],
                    2176 * 0.1 + 183 * uncached_rate + output * 4,
                )
                self.assertEqual(next_ping["decision"], expected_decision)
                self.assertEqual(next_ping["estimated_ping_output_tokens"], output)
                self.assertEqual(
                    next_ping["expected_cost_if_continued"]
                    < next_ping["expected_cost_if_stopped"],
                    expected_decision == "continue",
                )
                self.assertEqual(
                    s.next_ping_at, 510 if expected_decision == "continue" else None
                )
                with CacheBreakChecker(self.db) as other:
                    self.assertTrue(other.can_ping(s.scope))
                    self.assertEqual(
                        other.can_ping(s.scope, run_id="r", at=510),
                        expected_decision == "continue",
                    )
                # Both a canceled chain and an active chain retain the completed
                # ping's input, prefix, and TTL for the foreground return.
                plan = self.next_plan(s, 523)
                self.assertEqual(plan.reuse.eligible_tokens, 256)
                self.assertEqual(plan.reuse.age_seconds, 523 - 255)
                self.assertEqual(
                    plan.decision["input_accounting"]["anchor_response_id"], "ping"
                )
                self.assertIsNone(s.next_ping_at)
                self.assertIsNone(s.ping_view(run_id="r", at=524))
                done = s.complete(
                    plan,
                    Source("answer2", "output", 2, "a"),
                    ended_at=525,
                    input_tokens=2400,
                    cached_input_tokens=2304,
                )
                self.assertEqual(done["cache_event"]["expected_reusable_tokens"], 2176)
                self.assertEqual(
                    done["cache_event"]["outcome"], "expected_reuse_observed"
                )

    def test_observed_write_coverage_changes_repeat_capacity_and_cost(self):
        def decision(tid, reads=2048, writes=None):
            s = self.start(tid)
            self.assertTrue(
                s.acknowledge_ping(
                    run_id="r",
                    at=255,
                    input_tokens=2359,
                    cached_input_tokens=reads,
                    cache_write_input_tokens=writes,
                    output_tokens=4,
                )
            )
            return s.last_ping_receipt["continuation"]

        modeled_writes = decision("missing-writes")
        all_fresh = decision("zero-writes", writes=0)
        self.assertEqual(modeled_writes["uncached_input_rate"], 1.25)
        self.assertEqual(all_fresh["uncached_input_rate"], 1.0)
        self.assertEqual(
            all_fresh["expected_cost_if_stopped"],
            modeled_writes["expected_cost_if_stopped"],
        )
        # With reported zero writes, only the 2,048 reads were reusable.
        self.assertAlmostEqual(
            all_fresh["estimated_next_ping_cost_input_equivalents"],
            2048 * 0.1 + (2359 - 2048) + 4 * 4,
        )
        self.assertAlmostEqual(
            modeled_writes["estimated_next_ping_cost_input_equivalents"],
            2304 * 0.1 + (2359 - 2304) * 1.25 + 4 * 4,
        )
        entirely_cached = decision("all-read", reads=2359, writes=0)
        self.assertEqual(entirely_cached["uncached_input_rate"], 1.25)
        self.assertEqual(entirely_cached["uncached_input_rate_source"], "model_spec")

    def test_delayed_overrun_ack_uses_dispatch_for_ttl_and_skips_missed_timers(self):
        s = self.start()
        self.assertTrue(
            s.acknowledge_ping(
                run_id="r", at=255, completed_at=515, response=self.response(output=4)
            )
        )
        self.assertEqual(s.last_ping_receipt["refreshed_at"], 255)
        self.assertEqual(
            s.last_ping_receipt["continuation"]["reason"], "schedule_exhausted"
        )
        self.assertIsNone(s.next_ping_at)
        self.assertIsNone(self.next_plan(s, 556).reuse.age_seconds)

    def test_second_overrun_reprices_remaining_deadline_from_latest_receipt(self):
        self.policy = self.policy_for(940, 3)
        s = self.start(due=(255, 510, 765))
        self.assertTrue(
            s.acknowledge_ping(run_id="r", at=255, response=self.response(output=4))
        )
        self.assertEqual(s.last_ping_receipt["continuation"]["due_times"], [510, 765])
        self.assertTrue(
            s.acknowledge_ping(
                run_id="r",
                at=510,
                input_tokens=2359,
                cached_input_tokens=2304,
                output_tokens=900,
            )
        )
        receipt = s.last_ping_receipt
        self.assertEqual(receipt["refreshed_at"], 510)
        self.assertEqual(receipt["continuation"]["reason"], "no_marginal_savings")
        self.assertEqual(receipt["continuation"]["estimated_ping_output_tokens"], 900)
        self.assertIsNone(s.next_ping_at)
        self.assertIsNone(s.ping_view(run_id="r", at=765))

    def test_foreground_arrival_during_repricing_keeps_old_timers_closed(self):
        s = self.start()
        with CacheBreakChecker(self.db) as other:

            def return_during_reprice(*args, **kwargs):
                self.assertFalse(other.can_ping(s.scope, run_id="r", at=256))
                self.assertTrue(
                    other.begin_return(s.scope, run_id="next", started_at=257)
                )
                return plan_distribution_idle(*args, **kwargs)

            with patch(
                "compache.optimizer.session.plan_distribution_idle",
                side_effect=return_during_reprice,
            ):
                self.assertTrue(
                    s.acknowledge_ping(
                        run_id="r", at=255, response=self.response(output=4)
                    )
                )
            self.assertEqual(
                s.last_ping_receipt["continuation"]["reason"], "superseded"
            )
            self.assertTrue(other.can_ping(s.scope))
        self.assertIsNone(s.next_ping_at)
        self.assertIsNone(s.ping_view(run_id="r", at=510))

    def test_overrun_exceeding_repeat_context_budget_preserves_warmth(self):
        s = self.start()
        self.assertTrue(
            s.acknowledge_ping(run_id="r", at=255, response=self.response(output=8000))
        )
        self.assertTrue(s.last_ping_receipt["refresh_confirmed"])
        self.assertEqual(
            s.last_ping_receipt["continuation"]["reason"], "repeat_not_admissible"
        )
        self.assertIsNone(s.next_ping_at)

    def test_modeled_overrun_uses_the_scenario_without_anchoring_provider_input(self):
        s = self.start()
        self.assertTrue(
            s.acknowledge_ping(
                run_id="r",
                at=255,
                modeled_cached_input_tokens=2048,
                modeled_cache_write_input_tokens=10,
                modeled_output_tokens=4,
            )
        )
        receipt = s.last_ping_receipt
        self.assertTrue(receipt["output_budget_exceeded"])
        self.assertEqual(receipt["continuation"]["decision"], "continue")
        self.assertEqual(receipt["evidence_source"], "modeled")
        self.assertEqual(receipt["modeled_cache_write_input_tokens"], 10)
        self.assertIsNone(receipt["cache_write_input_tokens"])
        self.assertEqual(self.next_account(s, 450)["anchor_response_id"], "answer")

    def test_failed_incomplete_or_wrong_model_cannot_anchor_or_refresh(self):
        for status in ("failed", "incomplete", "model_mismatch"):
            with self.subTest(status=status):
                s = self.start(status)
                if status == "model_mismatch":
                    response = self.response()
                    response["response"]["model"] = "other"
                else:
                    response = self.response(status=status)
                self.assertFalse(
                    s.acknowledge_ping(run_id="r", at=255, response=response)
                )
                self.assertEqual(
                    s.last_ping_receipt["outcome"], "response_unsuccessful"
                )
                self.assertEqual(s.last_ping_receipt["response_status"], status)
                self.assertIsNone(s.last_ping_receipt["verified_private_read_tokens"])
                self.assertIsNone(s.ping_view(run_id="r", at=510))
                self.assertEqual(
                    self.next_account(s, 520)["anchor_response_id"], "answer"
                )

    def test_opaque_ack_and_missing_total_do_not_promote_local_estimates(self):
        for tid, counters, outcome in (
            ("opaque", {}, "unknown"),
            (
                "read-only",
                {"cached_input_tokens": 2048, "output_tokens": 0},
                "input_unknown",
            ),
        ):
            s = self.start(tid)
            self.assertFalse(s.acknowledge_ping(run_id="r", at=255, **counters))
            self.assertEqual(s.last_ping_receipt["outcome"], outcome)
            self.assertFalse(s.last_ping_receipt["occupancy_anchored"])
            self.assertIsNone(s.last_ping_receipt["input_tokens"])
            self.assertIsNone(s.ping_view(run_id="r", at=510))
            self.assertEqual(self.next_account(s, 520)["anchor_response_id"], "answer")

    def test_modeled_receipt_is_disjoint_from_provider_evidence_and_occupancy(self):
        s = self.start()
        self.assertTrue(
            s.acknowledge_ping(
                run_id="r",
                at=255,
                modeled_cached_input_tokens=2048,
                modeled_output_tokens=3,
            )
        )
        receipt = s.last_ping_receipt
        self.assertEqual(receipt["evidence_source"], "modeled")
        self.assertEqual(receipt["modeled_cached_tokens"], 2048)
        for key in (
            "observed_cached_tokens",
            "input_tokens",
            "output_tokens",
            "verified_private_read_tokens",
        ):
            self.assertIsNone(receipt[key])
        self.assertEqual(self.next_account(s, 450)["anchor_response_id"], "answer")

    def test_stale_and_malformed_receipts_do_not_apply_to_newer_send(self):
        s = self.start()
        self.assertFalse(
            s.acknowledge_ping(run_id="r", at=250, response=self.response())
        )
        with self.assertRaises(ContractError):
            s.acknowledge_ping(run_id="r", at=255, response=self.response(reads=3000))
        with self.assertRaises(ContractError):
            s.acknowledge_ping(
                run_id="r", at=255, response=self.response(), output_tokens=0
            )
        with self.assertRaises(ContractError):
            s.acknowledge_ping(
                run_id="r", at=255, response=self.response(), modeled_output_tokens=0
            )
        with CacheBreakChecker(self.db) as other:
            self.assertTrue(other.begin_return(s.scope, run_id="next", started_at=256))
        self.assertFalse(
            s.acknowledge_ping(run_id="r", at=257, response=self.response())
        )
        self.assertEqual(s.last_ping_receipt["outcome"], "stale_or_not_due")
        self.assertEqual(self.guard.events(s.scope), [])
