import socket
import unittest
from unittest.mock import patch

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
from compache.traces import ContractError
from helpers import END, labels_for, thread


class WorkloadAccountingTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            "test", context_tokens=1600, cache_min_tokens=128, cache_block_tokens=128
        )
        tr = thread(tid="training", n=2, gap=310, body_size=2)
        labels = labels_for(tr)
        # One scored historical requirement: the previous user instruction.
        src = tr.runs[0].input
        labels[1]["task_requirements"][0].update(
            source_kind="input",
            historical_context_needed="yes",
            alternatives=[{"sources": [{"part_id": src.part_id, "quote": src.text}]}],
        )
        artifact = fit_workload(
            [tr],
            labels,
            self.spec,
            observation_end=END,
            recipes=[RetentionRecipe(None, None, 0)],
        )
        self.assertTrue(artifact["recipes"][0]["eligible"])
        self.policy = WorkloadPolicy(
            artifact, lookahead=1, max_pings=0, reserve_tokens=100
        )
        self.q = Source("question1", "input", 1, "Q" * 30)
        self.d = Source("retrieval1", "retrieval", 1, "D" * 1000)
        self.a = Source("answer1", "output", 1, "A" * 40)
        self.next_parts = (
            self.q,
            self.d,
            self.a,
            Source("question2", "input", 2, "q" * 20),
            Source("retrieval2", "retrieval", 2, "d" * 80),
        )
        self.calls, self.headers = [], []
        self.guard = CacheBreakChecker()
        self.addCleanup(self.guard.close)
        self.offline = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network call")
        )
        self.offline.start()
        self.addCleanup(self.offline.stop)

    def session(self, tid, policy=None):
        def measure(p):
            self.calls.append((p.part_id, p.text))
            return 12 if p.text.startswith("[Compacted passage") else len(p.text)

        def measure_header(payload):
            self.headers.append(payload)
            return len(payload)

        return WorkloadSession(
            policy or self.policy,
            CacheScope("company", "provider", "test", tid),
            self.guard,
            count_part_tokens=measure,
            count_tokens=measure_header,
        )

    def prepare(self, s, index, at, parts, header="0123456789"):
        req = Request(
            s.scope.thread_id, f"run-{index}", index, self.spec.model_id, tuple(parts)
        )
        return s.prepare(req, started_at=at, system_prompt=header, tools=b"")

    def test_unrounded_provider_count_drives_actual_warm_and_cold_cut(self):
        for tid, at in (("warm", 100), ("cold", 900)):
            with self.subTest(tid=tid):
                s = self.session(tid)
                first = self.prepare(s, 1, 0, (self.q, self.d))
                self.assertEqual(first.view.tokens, 1040)
                response = {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-1",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 1423,
                            "input_tokens_details": {"cached_tokens": 128},
                            "output_tokens": 900,
                        },
                    },
                }
                done = s.complete(first, self.a, ended_at=12, response=response)
                self.assertEqual(
                    done["input_reconciliation"]["actual_minus_estimated"], 383
                )
                self.assertEqual(
                    done["ping_input_accounting"]["estimated_input_tokens"], 1463
                )
                self.calls.clear()
                self.headers.clear()
                second = self.prepare(s, 2, at, self.next_parts)
                self.assertEqual(self.headers, [])
                self.assertNotIn((self.d.part_id, self.d.text), self.calls)
                rows = second.decision["candidate_input_accounting"]
                self.assertEqual(rows["keep"]["estimated_input_tokens"], 1563)
                self.assertEqual(rows["keep"]["local_tokens"], 1180)
                self.assertIsNone(second.decision["costs"]["keep"])
                cut = second.decision["input_accounting"]
                self.assertEqual(
                    (cut["anchor_response_id"], cut["anchor_input_tokens"]),
                    ("resp-1", 1423),
                )
                self.assertEqual(
                    (cut["added_tokens"], cut["removed_tokens"], cut["overhead_delta"]),
                    (152, 1000, 0),
                )
                self.assertEqual(
                    (second.decision["action"], second.view.tokens), ("cut-0", 575)
                )
                self.assertEqual(second.view.archived_ids, {"retrieval1"})
                self.assertEqual(
                    self.policy.spec.cache_block_tokens * (1423 // 128), 1408
                )
                done2 = s.complete(
                    second,
                    Source("answer2", "output", 2, "!"),
                    ended_at=at + 1,
                    input_tokens=590,
                )
                self.assertEqual(
                    done2["input_reconciliation"]["actual_minus_estimated"], 15
                )

    def test_reset_missing_counters_changed_header_and_failed_response(self):
        s = self.session("m")
        first = self.prepare(s, 1, 0, (self.q, self.d))
        with self.assertRaisesRegex(ContractError, "completed response"):
            s.complete(
                first,
                self.a,
                ended_at=1,
                response={"status": "incomplete", "usage": {"input_tokens": 1400}},
            )
        s.complete(
            first, self.a, ended_at=1, input_tokens=1200, answer_history_tokens=52
        )
        next_plan = self.prepare(s, 2, 20, self.next_parts, header="0123456789NEW")
        keep = next_plan.decision["candidate_input_accounting"]["keep"]
        self.assertEqual(
            (
                keep["added_tokens"],
                keep["overhead_delta"],
                keep["estimated_input_tokens"],
            ),
            (152, 3, 1355),
        )
        reply = Source("answer2", "output", 2, "ok")
        done = s.complete(next_plan, reply, ended_at=21)
        self.assertIsNone(done["input_reconciliation"]["actual_minus_estimated"])
        self.assertEqual(done["ping_input_accounting"]["anchor_run_id"], "run-1")
        s.invalidate_input_anchor()
        third = self.prepare(
            s,
            3,
            40,
            (*self.next_parts, reply, Source("q3", "input", 3, "?")),
            header="0123456789NEW",
        )
        self.assertEqual(
            third.decision["input_accounting"]["fallback_reason"], "no_anchor"
        )

    def test_wrong_meter_and_failed_answer_size_do_not_replace_state(self):
        session = self.session("stale")
        first = self.prepare(session, 1, 0, (self.q, self.d))
        with self.assertRaises(ContractError):
            session.complete(
                first,
                self.a,
                ended_at=1,
                input_tokens=15,
                cached_input_tokens=16,
                answer_history_tokens=700,
            )
        done = session.complete(
            first, self.a, ended_at=1, input_tokens=1200, response_id="only-once"
        )
        self.assertEqual(done["ping_input_accounting"]["estimated_input_tokens"], 1240)
        second = self.prepare(session, 2, 30, self.next_parts)
        result = Source("answer2", "output", 2, "OK")
        with self.assertRaisesRegex(ContractError, "already acknowledged"):
            session.complete(
                second, result, ended_at=31, input_tokens=1200, response_id="only-once"
            )
        accepted = session.complete(
            second,
            result,
            ended_at=31,
            input_tokens=second.view.tokens,
            response_id="new",
        )
        self.assertEqual(accepted["input_reconciliation"]["actual_minus_estimated"], 0)

    def test_learned_undercount_changes_admission_only_on_later_appends(self):
        train = thread(tid="no-cut", n=2, body_size=1)
        artifact = fit_workload(
            [train],
            labels_for(train),
            self.spec,
            observation_end=END,
            recipes=[RetentionRecipe()],
        )
        policy = WorkloadPolicy(artifact, lookahead=1, max_pings=0, reserve_tokens=100)
        session = self.session("headroom", policy)
        first = self.prepare(session, 1, 0, (self.q, self.d))
        session.complete(first, self.a, ended_at=1, input_tokens=1000)
        second = self.prepare(session, 2, 10, self.next_parts)
        self.assertEqual(second.view.tokens, 1140)
        self.assertEqual(
            second.decision["input_accounting"]["calibration_observations"], 0
        )
        out = Source("answer2", "output", 2, "x" * 5)
        session.complete(second, out, ended_at=11, input_tokens=1200)
        safe = self.prepare(
            session, 3, 20, (*self.next_parts, out, Source("q3", "input", 3, "x" * 100))
        )
        row = safe.decision["input_accounting"]
        self.assertEqual(
            (
                row["edit_class"],
                row["estimated_input_tokens"],
                row["admission_input_tokens"],
            ),
            ("append", 1305, 1365),
        )
        self.assertEqual(
            (row["calibration_observations"], row["admission_margin_tokens"]), (1, 60)
        )
        # 1460 fits the 1500 input cap; 1460 + the observed 60-token undercount does not.
        with self.assertRaisesRegex(ContractError, "no admissible"):
            self.prepare(
                session,
                3,
                21,
                (*self.next_parts, out, Source("q3", "input", 3, "x" * 255)),
            )

    def test_ping_reanchors_only_after_successful_current_ack(self):
        train = thread(tid="ping-train", n=2, gap=310)
        artifact = fit_workload(
            [train],
            labels_for(train),
            self.spec,
            observation_end=END,
            recipes=[RetentionRecipe()],
        )
        policy = WorkloadPolicy(artifact, lookahead=2, branches=1, max_pings=1)
        s = self.session("p", policy)
        plan = self.prepare(s, 1, 0, (self.q, self.d))
        result = s.complete(plan, self.a, ended_at=30, input_tokens=980)
        self.assertEqual(result["due_times"], (255,))
        ping = s.ping_view(run_id="run-1", at=255)
        self.assertEqual(ping.tokens, 1020)
        self.assertFalse(s.acknowledge_ping(run_id="older", at=255, input_tokens=50))
        self.assertTrue(
            s.acknowledge_ping(
                run_id="run-1",
                at=255,
                response_id="provider-ping",
                input_tokens=1031,
                cached_input_tokens=896,
                output_tokens=0,
            )
        )
        next_plan = self.prepare(s, 2, 310, self.next_parts)
        account = next_plan.decision["input_accounting"]
        self.assertEqual(account["anchor_response_id"], "provider-ping")
        self.assertEqual(account["added_tokens"], 100)
        self.assertEqual(next_plan.view.tokens, 1131)
