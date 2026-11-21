"""Known cache reuse, structural evidence, durable scope suppression and queued timers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from compache.optimizer.cache_breaks import CacheBreakChecker, CacheScope
from compache.errors import ContractError


class CacheBreaks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.db = Path(self.folder.name) / "cache.sqlite"
        self.checker = CacheBreakChecker(self.db)
        self.addCleanup(self.checker.close)
        self.scope = CacheScope(
            "private-tenant", "provider", "model", "private-thread", "private-route"
        )

    def observe(self, run_id, at, **changes):
        args = dict(
            run_id=run_id,
            started_at=at,
            system_prompt="private-instructions",
            tools=[{"name": "private-tool", "args": {"first": 1, "second": 2}}],
            ttl_seconds=300,
            reusable_input_tokens=8192,
            cached_input_tokens=8192,
        )
        args.update(changes)
        return self.checker.observe_return(self.scope, **args)

    def test_on_time_miss_latches_and_cancels_with_redacted_durable_warning(self):
        self.observe("private-run-1", 0, cached_input_tokens=0)
        self.assertTrue(
            self.checker.can_ping(self.scope, run_id="private-run-1", at=240)
        )
        with self.assertLogs(
            "compache.optimizer.cache_breaks", level="WARNING"
        ) as records:
            event = self.observe(
                "private-run-2",
                280,
                system_prompt="new-private-instructions",
                cached_input_tokens=0,
            )
        self.assertEqual(event["changed_components"], ["system_prompt"])
        self.assertEqual(event["outcome"], "on_time_cache_miss")
        self.assertEqual(event["observed_cached_tokens"], 0)
        self.assertTrue(event["newly_disabled"])
        self.assertEqual(records.records[0].compache_cache_break, event)
        self.assertNotIn("private-", json.dumps(event) + records.output[0])
        with CacheBreakChecker(self.db) as worker:
            self.assertEqual(worker.events(self.scope), [event])
            self.assertFalse(worker.can_ping(self.scope))
            self.assertFalse(
                worker.can_ping(self.scope, run_id="private-run-1", at=281)
            )
            self.assertFalse(
                worker.note_refresh(
                    self.scope, run_id="private-run-2", refreshed_at=282
                )
            )
        healthy = self.observe(
            "private-run-3", 300, system_prompt="new-private-instructions"
        )
        self.assertTrue(healthy["pings_disabled"])
        self.assertFalse(healthy["flagged"])
        self.assertNotIn(b"private-", self.db.read_bytes())

    def test_missing_counter_partial_hit_and_proven_partial_break(self):
        self.observe("a", 10)
        with self.assertLogs("compache.optimizer.cache_breaks"):
            changed = self.observe("b", 20, tools=[], cached_input_tokens=None)
        self.assertEqual(changed["outcome"], "header_change_unconfirmed")
        self.assertIsNone(changed["observed_cached_tokens"])
        self.assertFalse(changed["pings_disabled"])
        with self.assertLogs("compache.optimizer.cache_breaks"):
            ambiguous = self.observe("c", 30, tools=["x"], cached_input_tokens=4096)
        self.assertFalse(ambiguous["pings_disabled"])
        self.assertTrue(ambiguous["flagged"])
        with self.assertLogs("compache.optimizer.cache_breaks"):
            structural = self.observe(
                "d",
                40,
                tools=["new"],
                cached_input_tokens=None,
                shared_prefix_tokens=1024,
            )
        self.assertEqual(structural["outcome"], "on_time_prefix_invalidated")
        self.assertIsNone(structural["observed_cached_tokens"])
        self.assertTrue(structural["pings_disabled"])

    def test_known_full_hit_preserved_even_with_changes_and_alternative_prefix_evidence(
        self,
    ):
        self.observe("a", 0)
        event = self.observe(
            "b",
            10,
            system_prompt="changed",
            shared_prefix_tokens=0,
            cached_input_tokens=8192,
        )
        self.assertEqual(event["outcome"], "expected_reuse_observed")
        self.assertFalse(event["flagged"])
        self.assertTrue(self.checker.can_ping(self.scope))

    def test_exact_serialized_order_matters_only_at_the_cacheable_prefix(self):
        original_tools = [{"name": "tool", "params": {"a": 1, "b": 2}}]
        reordered = [{"name": "tool", "params": {"b": 2, "a": 1}}]
        self.observe("a", 0, system_prompt="foo", tools=original_tools)
        self.assertEqual(
            self.checker.prefix_changed(
                self.scope,
                system_prompt=b"foo",
                tools='[{"name":"tool","params":{"a":1,"b":2}}]',
            ),
            [],
        )
        self.assertEqual(
            self.checker.prefix_changed(
                self.scope, system_prompt="foo", tools=reordered
            ),
            ["tools"],
        )
        with self.assertLogs("compache.optimizer.cache_breaks"):
            e = self.observe(
                "b",
                10,
                system_prompt="foo",
                tools=reordered,
                cached_input_tokens=4096,
                shared_prefix_tokens=4096,
            )
        self.assertEqual(e["outcome"], "on_time_prefix_invalidated")
        self.assertEqual(e["changed_components"], ["tools"])
        self.assertEqual(e["observed_cached_tokens"], 4096)
        self.assertTrue(e["pings_disabled"])

    def test_surviving_static_read_with_exhausted_private_prefix(self):
        self.observe("a", 0)
        ambiguous = self.observe("b", 10, cached_input_tokens=2048)
        self.assertFalse(ambiguous["flagged"])
        with self.assertLogs("compache.optimizer.cache_breaks"):
            event = self.observe(
                "c", 20, cached_input_tokens=2048, verified_shared_static_tokens=2048
            )
        self.assertTrue(event["pings_disabled"])
        self.assertEqual(event["outcome"], "on_time_history_cache_miss")
        self.assertEqual(event["changed_components"], [])

    def test_on_time_zero_with_stable_headers_is_unattributed(self):
        self.observe("a", 0)
        with self.assertLogs("compache.optimizer.cache_breaks"):
            event = self.observe("b", 299.99, cached_input_tokens=0)
        self.assertEqual(event["changed_components"], [])
        self.assertEqual(event["outcome"], "on_time_cache_miss")
        self.assertTrue(event["pings_disabled"])

    def test_expected_cold_intentional_switch_ineligible_and_config_change(self):
        self.observe("a", 0, cached_input_tokens=0)
        for run, at, opts, outcome in [
            ("b", 300, {"system_prompt": "new"}, "expired"),
            ("c", 310, {"intentional_change": True}, "intentional_change"),
            ("d", 320, {"reusable_input_tokens": 0}, "below_cache_minimum"),
            ("e", 330, {}, "below_cache_minimum"),
            ("f", 340, {"ttl_seconds": 1800}, "cache_ttl_changed"),
        ]:
            with self.subTest(outcome=outcome):
                event = self.observe(run, at, cached_input_tokens=0, **opts)
                self.assertEqual(event["outcome"], outcome)
                self.assertFalse(event["pings_disabled"])
        self.assertEqual(self.checker.events(self.scope), [])

    def test_successful_ping_updates_ttl_only_for_a_current_eligible_prefix(self):
        self.assertTrue(self.checker.can_ping(self.scope))
        self.assertFalse(self.checker.can_ping(self.scope, run_id="a", at=10))
        self.observe("a", 10)
        self.assertFalse(self.checker.can_ping(self.scope, run_id="a", at=310))
        self.assertFalse(
            self.checker.note_refresh(self.scope, run_id="a", refreshed_at=310)
        )
        self.assertFalse(
            self.checker.note_refresh(self.scope, run_id="other", refreshed_at=250)
        )
        self.assertTrue(
            self.checker.note_refresh(self.scope, run_id="a", refreshed_at=250)
        )
        self.assertFalse(
            self.checker.note_refresh(self.scope, run_id="a", refreshed_at=200)
        )
        with self.assertLogs("compache.optimizer.cache_breaks"):
            e = self.observe("b", 400, cached_input_tokens=0)
        self.assertTrue(e["pings_disabled"])
        self.assertEqual(e["idle_since_refresh_seconds"], 150)

    def test_ping_updates_known_capacity_while_missing_count_preserves_it(self):
        self.observe("a", 0)
        self.assertTrue(
            self.checker.note_refresh(
                self.scope, run_id="a", refreshed_at=1, reusable_input_tokens=4096
            )
        )
        self.assertTrue(
            self.checker.note_refresh(self.scope, run_id="a", refreshed_at=2)
        )
        e = self.observe("b", 3, cached_input_tokens=4096)
        self.assertEqual(e["expected_reusable_tokens"], 4096)
        self.assertEqual(e["outcome"], "expected_reuse_observed")
        self.assertTrue(
            self.checker.note_refresh(
                self.scope, run_id="b", refreshed_at=4, reusable_input_tokens=0
            )
        )
        with CacheBreakChecker(self.db) as worker:
            self.assertFalse(worker.can_ping(self.scope, run_id="b", at=5))
            self.assertFalse(
                worker.note_refresh(
                    self.scope, run_id="b", refreshed_at=6, reusable_input_tokens=4096
                )
            )
        self.assertEqual(
            self.observe("c", 7, cached_input_tokens=0)["outcome"],
            "below_cache_minimum",
        )

    def test_isolated_scopes_and_reset_keep_audit(self):
        self.observe("a", 0)
        with self.assertLogs("compache.optimizer.cache_breaks"):
            self.observe("b", 10, cached_input_tokens=0)
        for k in ("installation", "provider", "model", "thread_id", "cache_key"):
            self.assertTrue(self.checker.can_ping(replace(self.scope, **{k: "other"})))
        self.checker.begin_return(self.scope, run_id="late", started_at=15)
        self.checker.reset(self.scope)
        self.assertFalse(
            self.checker.begin_return(self.scope, run_id="b", started_at=10)
        )
        self.assertEqual(self.observe("late", 15)["outcome"], "out_of_order_ignored")
        self.assertEqual(len(self.checker.events(self.scope)), 1)
        self.assertTrue(self.checker.can_ping(self.scope))
        self.assertFalse(self.checker.can_ping(self.scope, run_id="b", at=20))
        self.assertEqual(
            self.observe("c", 30, cached_input_tokens=0)["outcome"], "first_observation"
        )

    def test_pending_returns_revoke_and_ignore_out_of_order_success(self):
        self.observe("a", 0)
        self.assertTrue(
            self.checker.begin_return(self.scope, run_id="b", started_at=10)
        )
        with CacheBreakChecker(self.db) as worker:
            self.assertFalse(worker.can_ping(self.scope, run_id="a", at=11))
            self.assertFalse(
                worker.note_refresh(self.scope, run_id="a", refreshed_at=11)
            )
            self.assertTrue(worker.begin_return(self.scope, run_id="b", started_at=10))
            self.assertTrue(worker.begin_return(self.scope, run_id="b", started_at=12))
            self.assertEqual(self.observe("b", 10)["outcome"], "out_of_order_ignored")
            self.assertTrue(worker.begin_return(self.scope, run_id="c", started_at=20))
        self.assertEqual(
            self.observe("b", 10, cached_input_tokens=0)["outcome"],
            "out_of_order_ignored",
        )
        self.assertEqual(self.observe("c", 20)["outcome"], "expected_reuse_observed")
        self.assertFalse(self.checker.can_ping(self.scope, run_id="b", at=30))
        self.assertTrue(self.checker.can_ping(self.scope, run_id="c", at=30))
        self.assertEqual(self.checker.events(self.scope), [])

    def test_retry_from_two_connections_emits_once_and_cannot_rewrite_success(self):
        original = self.observe("a", 0)
        self.assertEqual(original, self.observe("a", 0))
        with self.assertRaises(ContractError):
            self.observe("a", 0, cached_input_tokens=0)

        def observe(_):
            with CacheBreakChecker(self.db) as other:
                return other.observe_return(
                    self.scope,
                    run_id="b",
                    started_at=10,
                    system_prompt="new",
                    tools=[],
                    reusable_input_tokens=8192,
                    cached_input_tokens=0,
                    ttl_seconds=300,
                )

        with self.assertLogs("compache.optimizer.cache_breaks") as capture:
            with ThreadPoolExecutor(max_workers=3) as pool:
                events = list(pool.map(observe, range(3)))
        self.assertEqual(len(capture.records), 1)
        self.assertEqual(events, [events[0]] * 3)
        self.assertEqual(self.checker.events(self.scope), events[:1])
