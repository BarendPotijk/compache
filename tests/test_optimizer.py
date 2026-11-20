import copy
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from compache.dependencies import build_packet, validate_label
from compache.optimizer import (
    Constraints,
    ModelSpec,
    Policy,
    Request,
    Source,
    ThreadState,
    end_run,
    expand,
    fit_profile,
    handle,
    optimize,
    prepare,
    replay,
    split_threads,
)
from compache.optimizer.contracts import digest
from compache.optimizer.fit import cdf, survival_curve
from compache.optimizer.replay import Cache, assess
from compache.optimizer.solve import candidates, feasible
from compache.traces import ContractError, Thread, load_threads


def thread(tid="t", company="a", n=3, gap=600, body_size=1000):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runs = []
    history = []
    for i in range(1, n + 1):
        rid = f"{tid}-r{i}"
        start = origin + timedelta(seconds=(i - 1) * gap)
        finish = start + timedelta(seconds=30)
        inp = {"part_id": rid + "-in", "text": "Describe this unrelated new item."}
        ret = {
            "part_id": rid + "-doc",
            "text": f"Unique document {i}. " + "x " * body_size,
            "document_id": f"doc-{i}",
            "version": "v1",
            "chunk_id": "one",
            "exposed_to_model": True,
        }
        out = {"part_id": rid + "-out", "text": "An observed final answer."}
        ids = history + [inp["part_id"], ret["part_id"]]
        runs.append(
            {
                "run_id": rid,
                "run_index": i,
                "started_at": start.isoformat(),
                "ended_at": finish.isoformat(),
                "history_policy": "full_canonical",
                "input": inp,
                "retrievals": [ret],
                "output": out,
                "generation": {
                    "observation_id": rid + "-g",
                    "model": "test",
                    "started_at": start.isoformat(),
                    "ended_at": finish.isoformat(),
                    "usage": None,
                    "request_part_ids": ids,
                },
            }
        )
        history = ids + [out["part_id"]]
    return Thread.from_dict({"thread_id": tid, "company_id": company, "runs": runs})


def labels_for(t, need_old=False):
    rows = []
    for r in t.runs:
        p = build_packet(t, r.run_id)
        src = next(
            s
            for s in p["request_parts"]
            if s["kind"] == "retrieval"
            and s["origin_run_index"]
            == (1 if need_old and r.run_index > 1 else r.run_index)
        )
        task = {
            "requirement_id": "t1",
            "description": "Required source content",
            "source_kind": "retrieval",
            "necessity": "inferred",
            "alternatives": [
                {"sources": [{"part_id": src["part_id"], "quote": src["text"]}]}
            ],
            "evidence_note": "Test source requirement",
            "question_quote": r.input.text,
            "fulfillment": "fulfilled",
            "historical_context_needed": "yes"
            if src["origin_run_index"] < r.run_index
            else "no",
        }
        rows.append(
            {
                "schema_version": "compache-dependency-label-v1",
                "thread_id": t.thread_id,
                "target_run_id": r.run_id,
                "status": "complete",
                "coverage_notes": [],
                "task_requirements": [task],
                "claims": [
                    {
                        "claim_id": "c1",
                        "target_quote": r.output.text,
                        "relation": "uncertain",
                        "requirements": [],
                    }
                ],
                "historical_dependencies": [],
                "unresolved": [],
            }
        )
    return rows


END = "2026-01-10T00:00:00+00:00"


class ContractAndFitTests(unittest.TestCase):
    def test_versions_finite_ranges_and_policy_relationships(self):
        for kwargs in (
            {"cache_ttl_seconds": float("nan")},
            {"read_multiplier": -1},
            {"context_tokens": True},
            {"write_multiplier": 0.5},
        ):
            with self.assertRaises(ContractError):
                ModelSpec("test", **kwargs)
        spec = ModelSpec("test")
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaises(ContractError):
            Policy(spec.to_dict(), "x", target=0.9)
        with self.assertRaises(ContractError):
            Policy(spec.to_dict(), "x", freshness_margin_seconds=45)
        with self.assertRaises(ContractError):
            Constraints(max_miss_rate=2)
        current = Constraints()
        self.assertEqual(current.min_historical_dependency_coverage, 0.95)
        self.assertEqual(Constraints.from_dict(current.to_dict()), current)
        legacy = Constraints.from_dict(
            {
                "schema_version": "compache-constraints-v1",
                "max_miss_rate": 0.0,
                "max_unknown_change_rate": 0.0,
            }
        )
        self.assertEqual(legacy.max_miss_rate, 0)
        self.assertEqual(legacy.min_historical_dependency_coverage, 0)

    def test_historical_coverage_respects_complete_alternatives_and_ignores_direct_lookup_denominator(
        self,
    ):
        historical = [
            {
                "requirement_id": f"h{i}",
                "historical_context_needed": "yes",
                "alternatives": [
                    {"sources": [{"part_id": f"a{i}"}, {"part_id": f"b{i}"}]},
                    {"sources": [{"part_id": f"c{i}"}]},
                ],
            }
            for i in range(20)
        ]
        direct = [
            {
                "requirement_id": f"d{i}",
                "historical_context_needed": "no",
                "alternatives": [{"sources": [{"part_id": "current"}]}],
            }
            for i in range(100)
        ]
        label = {"status": "complete", "task_requirements": historical + direct}
        kept = {"current", "a19"} | {f"c{i}" for i in range(19)}
        result = assess(label, kept)
        self.assertEqual(result["historical_requirements_assessed"], 20)
        self.assertEqual(result["historical_requirements_retained"], 19)
        summary = {
            "admission_failures": 0,
            "skipped_runs": 0,
            "miss_rate": 0.1,
            "unknown_change_rate": 0,
            "historical_requirements_assessed": 20,
            "historical_dependency_coverage": 0.95,
        }
        self.assertTrue(feasible(summary, Constraints()))
        summary["historical_dependency_coverage"] = 0.90
        self.assertFalse(feasible(summary, Constraints()))
        self.assertTrue(
            feasible(summary, Constraints(min_historical_dependency_coverage=0.90))
        )
        summary["unknown_change_rate"] = 0.01
        self.assertFalse(
            feasible(summary, Constraints(min_historical_dependency_coverage=0.90))
        )

    def test_search_uses_requested_targets_and_warm_thresholds(self):
        t = thread()
        spec = ModelSpec("test")
        profile = fit_profile([t], labels_for(t), spec, observation_end=END)
        policies = [
            p
            for p in candidates(profile, spec, max_pings=0)
            if p.retrieval_grace_runs is not None
        ]
        self.assertEqual({p.target for p in policies}, {0.10, 0.15, 0.20, 0.25})
        self.assertEqual({p.warm_trigger for p in policies}, {0.50, 0.60})

    def test_kaplan_meier_ties_censoring_and_horizon(self):
        curve = survival_curve([(10, True), (10, False), (20, True), (30, False)], 100)
        self.assertEqual(curve["cdf_points"], [[10, 0.25], [20, 0.625]])
        self.assertEqual(cdf(curve, 9), 0)
        self.assertEqual(cdf(curve, 10000), 0.625)
        self.assertEqual(curve["residual_survival"], 0.375)

    def test_profile_is_aggregate_only_and_uses_explicit_censor_cutoff(self):
        t = thread()
        spec = ModelSpec("test")
        labels = labels_for(t, True)
        profile = fit_profile(
            [t], labels, spec, observation_end=END, min_index_samples=2
        )
        self.assertNotIn("Unique document", json.dumps(profile))
        self.assertEqual(profile["cadence"]["pooled"]["returns"], 2)
        self.assertEqual(profile["cadence"]["pooled"]["censored"], 1)
        self.assertTrue(profile["cadence"]["by_run_index"]["1"]["fallback_to_pooled"])
        self.assertEqual(profile["dependency"]["assessable_runs"], 3)
        self.assertEqual(
            {
                x["distance"]
                for x in profile["dependency"][
                    "occurrences_in_history_needed_alternatives"
                ]
            },
            {1, 2},
        )
        with self.assertRaises(ContractError):
            fit_profile([t], labels, spec, observation_end="2025-01-01T00:00:00Z")

    def test_label_flag_must_match_alternatives(self):
        t = thread()
        row = labels_for(t, True)[-1]
        p = build_packet(t, t.runs[-1].run_id)
        self.assertEqual(validate_label(row, p), [])
        row["task_requirements"][0]["historical_context_needed"] = "no"
        self.assertTrue(any("current-source" in e for e in validate_label(row, p)))

    def test_or_and_sets_not_flattened(self):
        label = {
            "status": "complete",
            "task_requirements": [
                {
                    "requirement_id": "r",
                    "historical_context_needed": "yes",
                    "alternatives": [
                        {"sources": [{"part_id": "a"}, {"part_id": "b"}]},
                        {"sources": [{"part_id": "c"}]},
                    ],
                }
            ],
        }
        self.assertTrue(assess(label, {"a"})["missed_requirements"])
        self.assertFalse(assess(label, {"a", "b"})["missed_requirements"])
        self.assertFalse(assess(label, {"c"})["missed_requirements"])
        self.assertFalse(assess(None, set())["assessable"])


class GateTests(unittest.TestCase):
    def test_byte_pressure_compacts_even_below_token_target(self):
        spec = ModelSpec("test", context_tokens=131072, byte_limit=3000)
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        t = thread(body_size=500)
        result = prepare(Request.from_thread(t, "t-r3"), policy, ThreadState("t"), 0)
        self.assertTrue(result.admitted)
        self.assertEqual(result.removed_part_ids, {"t-r1-doc", "t-r2-doc"})
        self.assertLessEqual(sum(len(s.encode()) for s in result.wire), 3000)

    def setUp(self):
        self.spec = ModelSpec("test", context_tokens=2048)
        self.policy = Policy(
            self.spec.to_dict(),
            "x",
            retrieval_grace_runs=1,
            protect_query_matches=False,
        )

    def test_prompt_answer_current_source_and_grace_retained(self):
        t = thread()
        req = Request.from_thread(t, t.runs[-1].run_id)
        st = ThreadState("t")
        result = prepare(req, self.policy, st, 1200)
        self.assertTrue(result.admitted)
        self.assertIn("t-r1-doc", result.removed_part_ids)
        self.assertNotIn("t-r2-doc", result.removed_part_ids)
        self.assertNotIn("t-r3-doc", result.removed_part_ids)
        originals = {p.part_id: p.text for p in req.parts}
        for p in result.parts:
            if p.kind in ("input", "output") or p.run_index == 3:
                self.assertEqual(p.text, originals[p.part_id])

    def test_warm_projection_extends_and_cuts_do_not_return_cold(self):
        t = thread(n=4)
        st = ThreadState("t")
        self.policy = replace(self.policy, warm_trigger=0.9)
        a = prepare(Request.from_thread(t, "t-r3"), self.policy, st, 1200)
        b = prepare(Request.from_thread(t, "t-r4"), self.policy, st, 1210)
        self.assertEqual(b.wire[: len(a.wire)], a.wire)
        self.assertEqual(b.prefix_transition, "extended")
        c = prepare(
            Request.from_thread(t, "t-r4"),
            replace(self.policy, retrieval_grace_runs=None),
            st,
            2000,
        )
        self.assertIn("t-r1-doc", c.removed_part_ids)

    def test_crossing_warm_trigger_replans_once(self):
        spec = ModelSpec(
            "test", context_tokens=10000, bytes_per_token=1, part_overhead_tokens=0
        )
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        first = Request(
            "t",
            "r1",
            1,
            "test",
            (Source("u1", "input", 1, "q"), Source("d1", "retrieval", 1, "x" * 6000)),
        )
        req = Request(
            "t",
            "r2",
            2,
            "test",
            first.parts
            + (
                Source("a1", "output", 1, "a"),
                Source("u2", "input", 2, "q"),
                Source("d2", "retrieval", 2, "y" * 1000),
            ),
        )
        st = ThreadState("t")
        prepare(first, policy, st, 0)
        b = prepare(req, policy, st, 10)
        self.assertEqual(b.action, "replaced_warm")
        self.assertEqual(b.prefix_transition, "rewritten")
        self.assertLessEqual(b.tokens, policy.target * spec.context_tokens)
        c = prepare(req, policy, st, 11)
        self.assertEqual(c.wire, b.wire)
        self.assertEqual(c.action, "applied_warm")

    def test_warm_below_sixty_percent_waits_for_cold_return(self):
        spec = ModelSpec(
            "test", context_tokens=10000, bytes_per_token=1, part_overhead_tokens=0
        )
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        first = Request(
            "t",
            "r1",
            1,
            "test",
            (Source("u1", "input", 1, "q"), Source("d1", "retrieval", 1, "x" * 4000)),
        )
        req = Request(
            "t",
            "r2",
            2,
            "test",
            first.parts
            + (
                Source("a1", "output", 1, "a"),
                Source("u2", "input", 2, "q"),
                Source("d2", "retrieval", 2, "y" * 1000),
            ),
        )
        state = ThreadState("t")
        prior = prepare(first, policy, state, 0)
        warm = prepare(req, policy, state, 10)
        self.assertEqual(warm.wire[: len(prior.wire)], prior.wire)
        self.assertFalse(warm.removed_part_ids)
        cold = prepare(req, policy, state, 400)
        self.assertEqual(cold.window, "cold")
        self.assertEqual(cold.removed_part_ids, {"d1"})

    def test_unreachable_warm_target_preserves_prefix_but_cold_takes_partial_cut(self):
        spec = ModelSpec(
            "test", context_tokens=10000, bytes_per_token=1, part_overhead_tokens=0
        )
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        first = Request(
            "t",
            "r1",
            1,
            "test",
            (Source("u1", "input", 1, "q"), Source("d1", "retrieval", 1, "x" * 3000)),
        )
        req = Request(
            "t",
            "r2",
            2,
            "test",
            first.parts
            + (
                Source("a1", "output", 1, "a"),
                Source("u2", "input", 2, "q"),
                Source("d2", "retrieval", 2, "y" * 4000),
            ),
        )
        state = ThreadState("t")
        prior = prepare(first, policy, state, 0)
        warm = prepare(req, policy, state, 10)
        self.assertEqual(warm.action, "deferred_warm")
        self.assertEqual(warm.reason, "warm_compaction_target_unreachable")
        self.assertFalse(state.removed)
        self.assertEqual(warm.wire[: len(prior.wire)], prior.wire)
        again = prepare(req, policy, state, 11)
        self.assertEqual(again.wire, warm.wire)
        cold = prepare(req, policy, state, 400)
        self.assertTrue(cold.admitted)
        self.assertEqual(cold.removed_part_ids, {"d1"})
        self.assertGreater(cold.tokens, policy.target * spec.context_tokens)
        # A cold return is already uncached, so partial savings are useful.

    def test_admission_can_force_partial_warm_cut_without_dropping_current(self):
        spec = ModelSpec(
            "test", context_tokens=10000, bytes_per_token=1, part_overhead_tokens=0
        )
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        first = Request(
            "t",
            "r1",
            1,
            "test",
            (Source("u1", "input", 1, "q"), Source("d1", "retrieval", 1, "x" * 3000)),
        )
        req = Request(
            "t",
            "r2",
            2,
            "test",
            first.parts
            + (
                Source("a1", "output", 1, "a"),
                Source("u2", "input", 2, "q"),
                Source("d2", "retrieval", 2, "y" * 7000),
            ),
        )
        state = ThreadState("t")
        prepare(first, policy, state, 0)
        result = prepare(req, policy, state, 10)
        self.assertTrue(result.admitted)
        self.assertEqual(result.action, "replaced_warm_for_admission")
        self.assertEqual(result.removed_part_ids, {"d1"})
        self.assertIn("d2", result.full_part_ids)

    def test_new_policy_latches_only_when_cold(self):
        t = thread()
        st = ThreadState("t")
        r = Request.from_thread(t, "t-r1")
        first = prepare(r, self.policy, st, 0)
        new = replace(self.policy, warm_trigger=0.85)
        self.assertEqual(prepare(r, new, st, 10).policy_hash, first.policy_hash)
        self.assertEqual(prepare(r, new, st, 400).policy_hash, new.hash)

    def test_edited_or_reordered_prefix_is_not_reused(self):
        t = thread()
        st = ThreadState("t")
        r = Request.from_thread(t, "t-r3")
        prepare(r, self.policy, st, 0)
        parts = list(r.parts)
        parts[0], parts[1] = parts[1], parts[0]
        result = prepare(
            replace(r, parts=tuple(parts)),
            replace(self.policy, retrieval_grace_runs=None),
            st,
            10,
        )
        self.assertEqual(result.action, "planned_prefix_changed")
        self.assertEqual(result.prefix_transition, "rewritten")

    def test_recovery_is_bounded_scoped_and_independent_of_recipe(self):
        t = thread()
        req = Request.from_thread(t, "t-r3")
        p = next(p for p in req.parts if p.part_id == "t-r1-doc")
        source_handle = handle(req.thread_id, p)
        page = expand(req, source_handle, limit=10)
        self.assertEqual(page["text"], p.text[:10])
        self.assertEqual(page["next_offset"], 10)
        with self.assertRaises(ContractError):
            expand(replace(req, thread_id="other"), source_handle)
        with self.assertRaises(ContractError):
            expand(req, source_handle, offset=-1)

    def test_unknown_model_or_masked_source_is_observe_only(self):
        r = Request.from_thread(thread(), "t-r3")
        result = prepare(replace(r, model="unknown"), self.policy, ThreadState("t"), 0)
        self.assertEqual(result.action, "observe_only")
        self.assertFalse(result.removed_part_ids)
        parts = list(r.parts)
        parts[-1] = replace(parts[-1], text=None)
        result = prepare(
            replace(r, parts=tuple(parts)), self.policy, ThreadState("t"), 0
        )
        self.assertEqual(result.action, "observe_only")

    def test_admission_refuses_current_oversize_without_committing_state(self):
        spec = replace(self.spec, context_tokens=10)
        policy = replace(self.policy, model_spec=spec.to_dict())
        st = ThreadState("t")
        before = copy.deepcopy(st)
        result = prepare(Request.from_thread(thread(), "t-r1"), policy, st, 0)
        self.assertFalse(result.admitted)
        self.assertEqual(st, before)

    def test_no_future_answer_leak_in_request_or_decision(self):
        t = thread()
        changed = t.to_dict()
        changed["runs"][-1]["output"]["text"] = "A completely different future answer."
        self.assertEqual(
            Request.from_thread(t, "t-r1"),
            Request.from_thread(Thread.from_dict(changed), "t-r1"),
        )
        self.assertNotIn(
            t.runs[0].output.part_id,
            [p.part_id for p in Request.from_thread(t, "t-r1").parts],
        )


class PlannerAndCacheTests(unittest.TestCase):
    def profile(self):
        curve = {
            "observations": 100,
            "returns": 80,
            "censored": 20,
            "horizon_seconds": 604800,
            "interpolation": "linear",
            "cdf_points": [
                [300, 0.4],
                [600, 0.5],
                [1800, 0.6],
                [3600, 0.65],
                [86400, 0.75],
                [604800, 0.8],
            ],
            "residual_survival": 0.2,
        }
        return {
            "schema_version": "compache-profile-v1",
            "cadence": {
                "pooled": curve,
                "by_run_index": {},
                "later_calls": 3,
                "last_call_seconds": 30,
            },
        }

    def decision(self, read=0.05, c=100000, a=0, s=10000, v=100000):
        profile = self.profile()
        spec = ModelSpec("test", context_tokens=200000, read_multiplier=read)
        policy = Policy(spec.to_dict(), digest(profile), max_pings=6)
        return end_run(
            policy,
            profile,
            held_input=c,
            answer_tokens=a,
            cold_view=v,
            shared_tokens=s,
            run_index=1,
            request_started_at=0,
            ended_at=30,
        )

    def test_documented_retention_vectors(self):
        low = self.decision()
        high = self.decision(0.1)
        self.assertEqual(low.pings, 6)
        self.assertEqual(high.pings, 1)
        self.assertAlmostEqual(low.values[0], 8920, places=3)
        self.assertAlmostEqual(high.values[0], 4902.5, places=3)
        self.assertEqual(self.decision(c=50000, a=32000, v=82000).action, "lapse")
        self.assertEqual(self.decision(0.1, c=15000, v=15000).action, "lapse")
        self.assertEqual(self.decision(0.05, c=15000, v=15000).pings, 1)

    def test_no_curve_no_support_deadline_and_ceiling_lapse(self):
        profile = self.profile()
        spec = ModelSpec("test", context_tokens=1000)
        p = Policy(spec.to_dict(), digest(profile), max_pings=6)
        for prof, held, end in [
            (None, 100, 30),
            (profile, 900, 30),
            (profile, 100, 260),
        ]:
            self.assertEqual(
                end_run(
                    p,
                    prof,
                    held_input=held,
                    answer_tokens=0,
                    cold_view=held,
                    run_index=1,
                    request_started_at=0,
                    ended_at=end,
                ).action,
                "lapse",
            )
        p = replace(p, model_spec=replace(spec, cache_ttl_seconds=40).to_dict())
        self.assertEqual(
            end_run(
                p,
                profile,
                held_input=100,
                answer_tokens=0,
                cold_view=100,
                run_index=1,
                request_started_at=0,
                ended_at=1,
            ).action,
            "lapse",
        )

    def test_cache_ttl_refresh_partial_prefix_and_conservation(self):
        spec = ModelSpec("test", cache_min_tokens=1, cache_block_tokens=8)
        a = Source("a", "input", 1, "hello " * 100)
        b = Source("b", "retrieval", 1, "world " * 100)
        cache = Cache(spec)
        first = cache.charge((a, b), 0, 0)
        second = cache.charge((a, b), 290, 0)
        third = cache.charge((a, b), 580, 0)
        self.assertEqual(first["cache_read_tokens"], 0)
        self.assertGreater(second["cache_read_tokens"], 0)
        self.assertGreater(third["cache_read_tokens"], 0)
        expired = cache.charge((a, b), 880, 0)
        self.assertEqual(expired["cache_read_tokens"], 0)
        changed = cache.charge((a, replace(b, text="changed")), 881, 0)
        self.assertEqual(
            changed["cache_read_tokens"], spec.part_tokens(a.text) // 8 * 8
        )
        for row in (first, second, third, expired, changed):
            self.assertEqual(
                row["input_tokens_estimated"],
                row["fresh_input_tokens"]
                + row["cache_read_tokens"]
                + row["cache_write_tokens"],
            )

    def test_uncacheable_input_does_not_pay_write_premium(self):
        c = Cache(ModelSpec("test", cache_min_tokens=10000))
        p = Source("a", "input", 1, "short")
        row = c.charge((p,), 0, 0)
        self.assertEqual(row["cost_input_equivalents"], row["input_tokens_estimated"])
        self.assertEqual(row["cache_write_tokens"], 0)


class ReplayAndSolverTests(unittest.TestCase):
    def test_recovery_identity_comes_from_map_not_part_id_delimiters(self):
        t = thread("t/restored/name", body_size=500)
        labels = labels_for(t, True)
        spec = ModelSpec("test", context_tokens=2048)
        profile = fit_profile([t], labels, spec, observation_end=END)
        policy = Policy(
            spec.to_dict(),
            digest(profile),
            retrieval_grace_runs=0,
            target=0.01,
            cold_trigger=0.02,
            protect_query_matches=False,
        )
        result = replay([t], labels, policy, profile, observation_end=END)
        self.assertEqual(result["summary"]["expansion_round_trips"], 2)
        self.assertIn("t/restored/name-r1-doc", result["runs"][1]["recovered_part_ids"])
        self.assertIn("t/restored/name-r1-doc", result["runs"][2]["recovered_part_ids"])

    def test_solver_promotes_safe_compaction_on_redundant_history(self):
        ts = [
            thread("train", "a", n=4, body_size=500),
            thread("test", "b", n=4, body_size=500),
        ]
        labels = [row for t in ts for row in labels_for(t)]
        result = optimize(
            ts,
            labels,
            ModelSpec("test", cache_ttl_seconds=0),
            observation_end=END,
            heldout_company="b",
            max_pings=0,
        )
        self.assertTrue(result["result"]["candidate_promoted"])
        self.assertGreater(result["result"]["modeled_saving_vs_cache_only"], 0)
        self.assertGreater(result["heldout"]["summary"]["compacted_runs"], 0)
        self.assertEqual(result["heldout"]["summary"]["miss_runs"], 0)

    def test_cost_regression_retains_baseline_without_changing_candidate(self):
        ts = [
            thread(f"train-{i}", "a", n=3, gap=360, body_size=50) for i in range(6)
        ] + [thread("test", "b", n=3, gap=10000, body_size=50)]
        labels = [row for t in ts for row in labels_for(t)]
        result = optimize(
            ts, labels, ModelSpec("test"), observation_end=END, heldout_company="b"
        )
        self.assertGreater(result["policy"]["max_pings"], 0)
        self.assertLess(result["result"]["modeled_saving_vs_cache_only"], 0)
        self.assertFalse(result["result"]["candidate_promoted"])
        self.assertEqual(result["recommended_policy"]["max_pings"], 0)
        self.assertIsNone(result["recommended_policy"]["retrieval_grace_runs"])

    def test_unavailable_costs_are_explicit_and_mixed_models_rejected(self):
        raw = thread().to_dict()
        raw["runs"][0]["retrievals"][0].update(text=None, content_status="masked")
        t = Thread.from_dict(raw)
        spec = ModelSpec("test")
        profile = fit_profile([t], [], spec, observation_end=END)
        policy = Policy(spec.to_dict(), digest(profile))
        result = replay([t], [], policy, profile, observation_end=END)
        self.assertEqual(result["summary"]["runs_with_unavailable_text"], 3)
        self.assertEqual(result["summary"]["cost_coverage"], "visible_projection_only")
        with self.assertRaises(ContractError):
            fit_profile([t], [], replace(spec, model_id="other"), observation_end=END)
        with self.assertRaises(ContractError):
            replay(
                [t],
                [],
                replace(policy, model_spec=replace(spec, model_id="other").to_dict()),
                profile,
                observation_end=END,
            )

    def test_missing_required_source_has_priced_recovery_not_free_oracle(self):
        t = thread(body_size=500)
        labels = labels_for(t, True)
        spec = ModelSpec("test", context_tokens=2048)
        profile = fit_profile([t], labels, spec, observation_end=END)
        policy = Policy(
            spec.to_dict(),
            digest(profile),
            retrieval_grace_runs=0,
            target=0.01,
            cold_trigger=0.02,
            protect_query_matches=False,
        )
        result = replay([t], labels, policy, profile, observation_end=END)
        self.assertGreater(result["summary"]["miss_runs"], 0)
        self.assertGreater(result["summary"]["expansion_cost"], 0)
        self.assertEqual(result["summary"]["expansion_round_trips"], 2)
        self.assertIn("t-r1-doc", result["runs"][1]["recovered_part_ids"])
        # An old recovery copy is eligible for compaction; a later recovery
        # costs another call and does not erase the before-recovery miss.
        self.assertIn("t-r1-doc", result["runs"][-1]["recovered_part_ids"])
        self.assertTrue(result["runs"][-1]["missed_requirements"])

    def test_unknown_labels_never_become_zero_risk_negatives(self):
        t = thread()
        spec = ModelSpec("test", context_tokens=2048)
        profile = fit_profile([t], [], spec, observation_end=END)
        policy = Policy(
            spec.to_dict(),
            digest(profile),
            retrieval_grace_runs=0,
            protect_query_matches=False,
        )
        result = replay([t], [], policy, profile, observation_end=END)
        self.assertEqual(result["summary"]["assessable_runs"], 0)
        self.assertIsNone(result["summary"]["miss_rate"])
        self.assertGreater(result["summary"]["unknown_changed_runs"], 0)

    def test_whole_thread_split_determinism_and_no_overlap(self):
        ts = [thread(str(i), str(i % 2), body_size=10) for i in range(8)]
        a, b = split_threads(ts)
        c, d = split_threads(list(reversed(ts)))
        self.assertEqual([t.thread_id for t in a], [t.thread_id for t in c])
        self.assertEqual([t.thread_id for t in b], [t.thread_id for t in d])
        self.assertFalse({t.thread_id for t in a} & {t.thread_id for t in b})

    def test_heldout_answers_do_not_change_training_selection(self):
        ts = [
            thread("train", "a", n=2, body_size=10),
            thread("test", "b", n=2, body_size=10),
        ]
        spec = ModelSpec("test")
        a = optimize(
            ts, [], spec, observation_end=END, heldout_company="b", max_pings=0
        )
        edited = ts[1].to_dict()
        edited["runs"][-1]["output"]["text"] = "NEW HELDOUT OUTPUT " * 100
        b = optimize(
            [ts[0], Thread.from_dict(edited)],
            [],
            spec,
            observation_end=END,
            heldout_company="b",
            max_pings=0,
        )
        self.assertEqual(a["policy"], b["policy"])
        self.assertEqual(a["profile"], b["profile"])
        self.assertEqual(a["selection"], b["selection"])

    def test_shipped_replay_fixture_and_labels_are_compatible(self):
        root = Path(__file__).resolve().parents[1] / "examples/rag_workload"
        ts = load_threads(root / "threads.jsonl")
        labels = [
            json.loads(s) for s in (root / "labels.jsonl").read_text().splitlines()
        ]
        spec = ModelSpec.from_dict(
            json.loads((root / "pressure-model.json").read_text())
        )
        manifest = json.loads((root / "manifest.json").read_text())
        p = fit_profile(ts, labels, spec, observation_end=manifest["observation_end"])
        self.assertEqual(p["runs"], 207)
        self.assertEqual(p["dependency"]["unknown_runs"], 9)
        self.assertFalse(p["ruler"]["measured_provider_tokens_used"])

    def test_public_planner_rejects_mismatched_or_malformed_profiles(self):
        spec = ModelSpec("test")
        profile = fit_profile([thread()], [], spec, observation_end=END)
        policy = Policy(spec.to_dict(), digest(profile), max_pings=1)
        bad = copy.deepcopy(profile)
        bad["cadence"]["pooled"]["cdf_points"][0][1] = 2
        for p, prof in [
            (policy, bad),
            (replace(policy, profile_hash="wrong"), profile),
        ]:
            with self.assertRaises(ContractError):
                end_run(
                    p,
                    prof,
                    held_input=10,
                    answer_tokens=0,
                    cold_view=10,
                    run_index=1,
                    request_started_at=0,
                    ended_at=1,
                )


if __name__ == "__main__":
    unittest.main()
