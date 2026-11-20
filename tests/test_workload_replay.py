import unittest
from datetime import datetime, timedelta

from test_optimizer import END, labels_for, thread

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.retention import RetentionRecipe
from compache.optimizer.workload import fit_workload
from compache.optimizer.workload_replay import replay_workload
from compache.traces import ContractError, Thread


def shifted(t, seconds):
    d = t.to_dict()
    for r in d["runs"]:
        for obj in (r, r["generation"]):
            for key in ("started_at", "ended_at"):
                obj[key] = (
                    datetime.fromisoformat(obj[key]) + timedelta(seconds=seconds)
                ).isoformat()
    return Thread.from_dict(d)


class WorkloadReplayTests(unittest.TestCase):
    def test_modeled_output_changes_only_future_timers_after_first_receipt(self):
        for ttl, gap in ((300, 630), (1800, 3780)):
            train = thread(tid="fit", n=2, gap=gap, body_size=16000)
            heldout = thread(tid="heldout", n=2, gap=gap, body_size=16000)
            spec = ModelSpec(
                "test",
                context_tokens=50000,
                cache_ttl_seconds=ttl,
                cache_min_tokens=1024,
                cache_block_tokens=128,
                ping_output_tokens=3,
            )
            artifact = fit_workload(
                [train], [], spec, observation_end=END, recipes=[RetentionRecipe()]
            )
            schedules = []
            contracts = []
            for actual, expected_count in ((3, 2), (9, 2), (4000, 1)):
                with self.subTest(ttl=ttl, actual=actual):
                    res = replay_workload(
                        [heldout],
                        [],
                        artifact,
                        system_prompt="",
                        default_tools=b"",
                        observation_end=END,
                        max_pings=2,
                        modeled_ping_output_tokens=actual,
                        include_host_contract=True,
                    )
                    self.assertEqual(
                        res["ping_output_scenario"]["modeled_receipt_tokens"], actual
                    )
                    contracts.append(res["host_contract"])
                    pings = [
                        row
                        for row in res["candidate"]["ledger"]
                        if row["role"] == "ping"
                    ]
                    self.assertEqual(len(pings), expected_count)
                    schedules.append([row["at"] for row in pings])
                    self.assertTrue(all(r["output_tokens"] == actual for r in pings))
                    self.assertTrue(
                        all(r["receipt"]["refresh_confirmed"] for r in pings)
                    )
                    self.assertEqual(res["baseline"]["pings"], 0)
            self.assertEqual(schedules[0], schedules[1])
            self.assertEqual(schedules[0][:1], schedules[2])
            self.assertEqual(contracts[0], contracts[1])
            self.assertEqual(contracts[0], contracts[2])

    def test_foreground_call_export_counts_complete_serial_runs_once(self):
        from compache.optimizer import count_foreground_calls

        def call(call_id, role, run="a", status="succeeded"):
            return {"call_id": call_id, "run_id": run, "role": role, "status": status}

        events = [
            call("1", "foreground_step"),
            call("1", "foreground_step"),
            call("2", "answer"),
            call("3", "ping"),
            call("4", "recovery"),
            call("5", "answer", status="failed"),
            call("6", "answer", run="partial"),
        ]
        self.assertEqual(
            count_foreground_calls(events, complete_run_ids=["a"]), {"a": 2}
        )
        with self.assertRaisesRegex(ContractError, "conflicting"):
            count_foreground_calls(
                [*events, call("1", "answer")], complete_run_ids=["a"]
            )
        with self.assertRaisesRegex(ContractError, "no successful"):
            count_foreground_calls(events, complete_run_ids=["missing"])
        for malformed in (
            [call("one", "foreground_step")],
            [call("one", "answer"), call("two", "answer")],
        ):
            with self.assertRaisesRegex(ContractError, "final answer"):
                count_foreground_calls(malformed, complete_run_ids=["a"])
        with self.assertRaisesRegex(ContractError, "different model"):
            count_foreground_calls(
                [
                    dict(call("one", "foreground_step"), model="a"),
                    dict(call("two", "answer"), model="b"),
                ],
                complete_run_ids=["a"],
            )

    def test_repeat_scoring_uses_heldout_counts_in_both_arms_after_decision(self):
        t = thread(tid="eval", n=2)
        known = t.runs[0].run_id
        observed = self.replay(
            [t], calls_per_run={known: 3}, include_host_contract=True
        )
        unknown = self.replay([t])
        for arm in ("candidate", "baseline"):
            rows = observed[arm]["ledger"]
            first, repeated, later = rows
            self.assertEqual(
                (first["foreground_calls"], later["call_count_source"]),
                (3, "assumed_one"),
            )
            self.assertEqual(
                (repeated["role"], repeated["calls"]), ("foreground_repeat", 2)
            )
            self.assertEqual(repeated["input_tokens"], 2 * first["input_tokens"])
            self.assertEqual(
                repeated["cache_read_tokens"], 2 * (first["input_tokens"] // 4 * 4)
            )
            self.assertEqual(repeated["output_tokens"], 0)
            self.assertEqual(observed[arm]["foreground_calls"], 4)
            self.assertEqual(observed[arm]["pings"], 0)
            self.assertEqual(
                [first["action"], later["action"]],
                [r["action"] for r in unknown[arm]["ledger"]],
            )
            self.assertAlmostEqual(
                observed[arm]["totals"]["cost_input_equivalents"]
                - unknown[arm]["totals"]["cost_input_equivalents"],
                repeated["cost_input_equivalents"],
            )
        self.assertEqual(
            observed["host_contract"]["foreground_calls_per_run"], {known: 3}
        )
        from compache.optimizer import compare_host_ledger

        a, repeated, b = observed["baseline"]["ledger"]
        events = [
            dict(row, call_id=str(i), status="succeeded")
            for i, row in enumerate((a, b))
        ]
        events[0]["at"] = repeated["at"]
        for i in range(2):
            events.append(
                dict(
                    run_id=known,
                    thread_id=t.thread_id,
                    role="foreground_step",
                    status="succeeded",
                    call_id=f"step-{i}",
                    at=a["at"] + i,
                    output_tokens=0,
                    **{
                        k: repeated[k] // 2
                        for k in (
                            "input_tokens",
                            "fresh_input_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                        )
                    },
                )
            )
        export = {
            "host_contract": observed["host_contract"],
            "policy_name": "fixture",
            "includes_all_calls": True,
            "deployment_call_count": 4,
            "events": events,
        }
        self.assertAlmostEqual(
            compare_host_ledger(observed, export)["savings_input_equivalents"], 0
        )
        self.assertIsNone(
            compare_host_ledger(
                observed, dict(export, events=events[:-1], deployment_call_count=3)
            )["savings_fraction"]
        )
        outside = [*events[:-1], dict(events[-1], at=b["at"])]
        with self.assertRaisesRegex(ContractError, "outside"):
            compare_host_ledger(observed, dict(export, events=outside))
        with self.assertRaisesRegex(ContractError, "foreground clock/output"):
            compare_host_ledger(
                observed,
                dict(
                    export,
                    events=[
                        dict(events[0], output_tokens=0),
                        events[1],
                        dict(events[2], output_tokens=a["output_tokens"]),
                        events[3],
                    ],
                ),
            )

    def replay(self, threads, cutoff=END, share=True, context=100000, **kwargs):
        train = thread(tid="train", n=2)
        spec = ModelSpec(
            "test",
            context_tokens=context,
            cache_ttl_seconds=300,
            cache_min_tokens=16,
            cache_block_tokens=4,
        )
        artifact = fit_workload(
            [train],
            labels_for(train),
            spec,
            observation_end=END,
            recipes=[RetentionRecipe()],
        )
        return replay_workload(
            threads,
            [l for t in threads for l in labels_for(t)],
            artifact,
            observation_end=cutoff,
            system_prompt="s" * 512,
            default_tools=[{"name": "retrieve", "description": "r" * 512}],
            share_headers=share,
            max_pings=0,
            lookahead=2,
            branches=1,
            **kwargs,
        )

    def test_first_turn_reuses_header_only_after_completed_neighbor_in_same_company(
        self,
    ):
        a = thread(tid="a", company="tenant", n=2, gap=600)
        b = shifted(
            thread(tid="b", company="tenant", n=1), 30
        )  # A completed just as B arrives.
        c = shifted(thread(tid="c", company="other", n=1), 50)
        d = shifted(
            thread(tid="d", company="tenant", n=1), 10
        )  # Prior writes still pending.
        res = self.replay([b, d, c, a])
        self.assertEqual(res["savings_fraction"], 0)
        for arm in ("baseline", "candidate"):
            s = res[arm]
            self.assertTrue(s["complete"])
            self.assertEqual(s["first_answers_with_shared_reads"], 1)
            self.assertGreater(s["first_system_read_tokens"], 0)
            self.assertGreater(s["first_default_tool_read_tokens"], 0)
            self.assertEqual(s["planned_answers"], 5)
            first = {
                r["thread_id"]: r["cache_read_tokens"]
                for r in s["ledger"]
                if r["run_index"] == 1
            }
            self.assertGreater(first["b"], 0)
            self.assertEqual([first[k] for k in "acd"], [0, 0, 0])
        self.assertEqual(
            self.replay([a, b], share=False)["candidate"][
                "first_answers_with_shared_reads"
            ],
            0,
        )

    def test_historical_usage_is_not_an_anchor_for_modeled_input(self):
        original = thread(tid="held-out", n=3)
        data = original.to_dict()
        for run in data["runs"]:
            # Reuse the exact output count and source IDs, but change the
            # captured input and cache meters in the historical envelope.
            out = ModelSpec("test").text_tokens(run["output"]["text"])
            run["generation"]["usage"] = {
                "input": 900_000,
                "output": out,
                "cache_read": 850_000,
                "cache_write": 50_000,
            }
        with_usage = self.replay([Thread.from_dict(data)])
        without_usage = self.replay([original])
        self.assertEqual(with_usage["input_accounting"]["historical_input_counters"], 3)
        self.assertEqual(with_usage["input_accounting"]["provider_anchors_used"], 0)
        for arm in ("candidate", "baseline"):
            self.assertTrue(with_usage[arm]["complete"])
            self.assertEqual(with_usage[arm]["ledger"], without_usage[arm]["ledger"])

    def test_generation_clock_uses_a_completed_neighbor_and_excludes_expired_prefix(
        self,
    ):
        lead = thread(tid="lead", company="tenant", n=1)
        wait = shifted(thread(tid="wait", company="tenant", n=1), 10).to_dict()
        at = datetime.fromisoformat(wait["runs"][0]["ended_at"])
        wait["runs"][0]["generation"]["started_at"] = at.isoformat()
        wait = Thread.from_dict(wait)
        expire = shifted(thread(tid="expire", company="tenant", n=1), 340)
        result = self.replay([lead, wait, expire])
        for arm in ("baseline", "candidate"):
            events = {r["thread_id"]: r for r in result[arm]["ledger"]}
            self.assertGreater(events["wait"]["cache_read_tokens"], 0)
            self.assertEqual(events["expire"]["cache_read_tokens"], 0)
            self.assertEqual(
                sum(result[arm]["first_each_default_tool_read_tokens"]),
                result[arm]["first_default_tool_read_tokens"],
            )
            self.assertGreater(
                result[arm]["first_input_cost_if_cold"],
                result[arm]["first_actual_input_cost"],
            )

    def test_tied_same_thread_provider_starts_report_incomplete_comparison(self):
        data = thread(tid="ties", n=2).to_dict()
        instant = data["runs"][0]["started_at"]
        for run in data["runs"]:
            for record in (run, run["generation"]):
                record.update(started_at=instant, ended_at=instant)
        result = self.replay([Thread.from_dict(data)])
        self.assertEqual(result["baseline"]["answers"], 2)
        self.assertEqual(result["candidate"]["answers"], 1)
        self.assertEqual(
            result["candidate"]["failures"][0]["reason"], "ambiguous_request_clock"
        )
        self.assertIsNone(result["savings_fraction"])

    def test_missing_or_too_large_input_has_no_workload_savings(self):
        r = self.replay([thread(tid="eval", n=2)], context=120)
        self.assertIsNone(r["savings_fraction"])
        self.assertEqual(r["candidate"]["answers"], 0)
        self.assertEqual(r["baseline"]["failures"][0]["reason"], "input_not_admissible")

    def test_installed_host_reference_reprices_all_calls_and_allows_negative_savings(
        self,
    ):
        from copy import deepcopy

        from compache.optimizer import compare_host_ledger

        res = self.replay([thread(tid="heldout", n=2)], include_host_contract=True)
        events = []
        for i, row in enumerate(res["baseline"]["ledger"]):
            events.append(dict(row, call_id=str(i), status="succeeded"))
        host = {
            "host_contract": res["host_contract"],
            "policy_name": "installed-policy-v3",
            "includes_all_calls": True,
            "deployment_call_count": 2,
            "events": events,
        }
        self.assertEqual(compare_host_ledger(res, host)["savings_input_equivalents"], 0)
        fields = {
            "input_tokens": 10,
            "fresh_input_tokens": 10,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
        }
        events[1].update(**fields)  # The host compacted the second input.
        events.append(
            dict(
                events[0],
                **fields,
                call_id="failed-reopen",
                status="failed",
                role="recovery",
                output_tokens=1,
            )
        )
        events.append(
            dict(events[0], **fields, call_id="ping", role="ping", output_tokens=1)
        )
        host["deployment_call_count"] = 4
        measured = compare_host_ledger(res, host)
        self.assertTrue(measured["complete"])
        self.assertLess(measured["savings_input_equivalents"], 0)
        self.assertEqual(measured["calls_by_role"]["ping"], 1)
        self.assertEqual(measured["failed_calls_charged"], 1)
        for change in (
            {"includes_all_calls": False},
            {"deployment_call_count": 5},
            {"events": [events[0], *events[2:]], "deployment_call_count": 3},
        ):
            self.assertIsNone(
                compare_host_ledger(res, dict(host, **change))[
                    "savings_input_equivalents"
                ]
            )
        for change in (
            {"host_contract": dict(host["host_contract"], observation_end=0)},
            {"events": events + [events[0]]},
            {"events": events + [dict(events[0], call_id="duplicate-answer")]},
            {
                "events": events
                + [
                    dict(
                        events[-1],
                        call_id="late",
                        at=host["host_contract"]["observation_end"] + 1,
                    )
                ]
            },
            {
                "events": events
                + [dict(events[-1], call_id="foreign", thread_id="other")]
            },
        ):
            with self.assertRaises(ContractError):
                compare_host_ledger(res, dict(host, **change))
        missing_usage = deepcopy(host)
        del missing_usage["events"][-1]["output_tokens"]
        with self.assertRaises(ContractError):
            compare_host_ledger(res, missing_usage)

    def test_overlap_and_early_cutoff_fail_before_comparison(self):
        with self.assertRaisesRegex(ContractError, "training thread"):
            self.replay([thread(tid="train", n=2)])
        with self.assertRaisesRegex(ContractError, "cutoff"):
            self.replay([thread(tid="test", n=1)], cutoff=0)


if __name__ == "__main__":
    unittest.main()
