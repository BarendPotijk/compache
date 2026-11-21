"""The top-level API: fit, replay, a live session, the price sheet and the profile."""

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import compache
from compache.models import model_spec, pricing
from compache.optimizer.contracts import ModelSpec
from compache.threads import Thread, write_threads

from helpers import labels_for, thread

ANSWER = "An observed final answer. " * 60


def threads(gap=400, count=4, n=6):
    out = []
    for i in range(count):
        data = thread(tid=f"t{i}", n=n, gap=gap).to_dict()
        for run in data["runs"]:
            run["output"]["text"] = ANSWER
        out.append(Thread.from_dict(data))
    return out


def labels(sample):
    # Every later turn needs the first retrieved document.
    return [row for t in sample for row in labels_for(t, need_old=True)]


SPEC = ModelSpec(
    "test",
    context_tokens=100_000,
    cache_ttl_seconds=300,
    cache_min_tokens=16,
    cache_block_tokens=8,
)


def fitted():
    sample = threads()
    return compache.fit(
        sample,
        labels(sample),
        model=SPEC,
        system_prompt="Answer from the supplied notes.",
        folds=2,
        share_headers=False,
    )


class FitTests(unittest.TestCase):
    def test_fit_reports_the_decision_and_saves_a_small_policy(self):
        policy = fitted()
        summary = policy.summary
        self.assertEqual(
            (summary["threads"], summary["turns"], summary["labelled_turns"]),
            (4, 24, 24),
        )
        self.assertEqual(summary["keep"]["answers"], 0)
        self.assertIsNone(summary["keep"]["retrieved"])
        self.assertGreater(summary["max_pings"], 0)
        self.assertGreater(summary["saving"], 0.3)
        self.assertEqual(
            summary["cost_per_1000_turns"]["unit"], "input-token equivalents"
        )
        text = policy.describe()
        self.assertIn(
            "Keep all questions, no earlier answers and all retrieved text.", text
        )
        self.assertIn("keep-warm ping", text)
        self.assertIn("cheaper than keeping everything", text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "policy.json"
            policy.save(path)
            self.assertLess(path.stat().st_size, 50_000)
            again = compache.Policy.load(path)
            self.assertEqual(again.summary, summary)
            self.assertEqual(again.recipe, policy.recipe)

    def test_without_labels_only_the_pings_are_fitted(self):
        policy = compache.fit(threads(), model="claude-sonnet-5-5", folds=2)
        self.assertIsNone(policy.recipe)
        self.assertIn("without labels only the pings are fitted", policy.describe())
        costs = policy.summary["cost_per_1000_turns"]
        self.assertEqual(costs["unit"], "USD")
        self.assertLess(costs["policy"], costs["keep_everything"])
        self.assertIn("$", policy.describe())
        # The recorded model name does not matter: the threads are priced on the sheet given.
        self.assertEqual(policy.spec.model_id, "claude-sonnet-5-5")

    def test_files_chat_logs_and_simple_labels(self):
        chats, rows = [], []
        for i in range(4):
            messages = [{"role": "system", "content": "Answer from the handbook."}]
            for k in range(5):
                start = 1_767_600_000 + i * 100_000 + k * 400
                messages += [
                    {"role": "user", "content": f"Question {k}?", "at": start},
                    {"role": "retrieval", "content": f"Passage {k}. " + "y " * 900},
                    {"role": "assistant", "content": ANSWER, "at": start + 20},
                ]
                if k:
                    rows.append(
                        {
                            "thread_id": f"c{i}",
                            "run": k + 1,
                            "needs": [[f"c{i}:1:retrieval:1"]],
                        }
                    )
            chats.append({"thread_id": f"c{i}", "messages": messages})
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "threads.jsonl").write_text(
                "".join(json.dumps(c) + "\n" for c in chats)
            )
            (base / "labels.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            policy = compache.fit(
                base / "threads.jsonl", base / "labels.jsonl", model=SPEC, folds=2
            )
        self.assertEqual(
            policy.artifact["header"]["system_prompt"], "Answer from the handbook."
        )
        self.assertEqual(policy.summary["labelled_turns"], 16)
        self.assertEqual(policy.summary["coverage"], 1.0)
        self.assertEqual(policy.recipe.output_runs, 0)

    def test_several_model_calls_in_a_turn_are_priced(self):
        once = threads()
        looped = []
        for thread_ in once:
            data = thread_.to_dict()
            for run in data["runs"]:
                run["calls"] = 4
            looped.append(data)
        plain = compache.fit(once, model=SPEC, folds=2, share_headers=False)
        loops = compache.fit(looped, model=SPEC, folds=2, share_headers=False)
        cost = lambda policy: policy.artifact["solve"]["keep_everything_cost"]  # noqa: E731
        # Three further calls a turn each read the turn's request from the cache.
        self.assertGreater(cost(loops), cost(plain))
        self.assertEqual(
            loops.artifact["demand"]["by_position"]["1"]["calls_per_run"], [4] * 4
        )
        self.assertEqual(
            plain.artifact["demand"]["by_position"]["1"]["calls_per_run"], [None] * 4
        )
        held = thread(tid="held", n=6, gap=400).to_dict()
        for run in held["runs"]:
            run["calls"] = 4
        report = compache.replay(loops, [held])
        self.assertEqual(report["detail"]["baseline"]["foreground_calls"], 24)
        self.assertGreater(report["detail"]["baseline"]["repeat_input_tokens"], 0)

    def test_too_little_data_is_refused(self):
        with self.assertRaises(compache.ContractError):
            compache.fit(threads(count=1), model=SPEC)
        with self.assertRaises(compache.ContractError):
            compache.Policy({"schema_version": "other"})


class ReplayTests(unittest.TestCase):
    def test_overflow_uses_each_arms_completed_turns_and_matched_costs(self):
        sample = threads()
        policy = compache.fit(
            sample, [r for t in sample for r in labels_for(t)], model=SPEC
        )
        held = thread(tid="held", n=32, gap=3600, body_size=2500)
        # Narrow the deployment window so the baseline fills it during the replay.
        from dataclasses import replace

        policy.artifact["model_spec"] = replace(SPEC, context_tokens=12_000).to_dict()
        report = compache.replay(policy, [held], labels_for(held, need_old=True))
        self.assertEqual(report["planned_turns"], 32)
        self.assertEqual(report["turns"], report["policy_turns"])
        self.assertEqual(report["policy_turns"], 32)
        self.assertLess(report["baseline_turns"], 32)
        self.assertIsNone(report["saving"])
        self.assertAlmostEqual(
            report["cost_per_1000_turns"]["policy"], report["policy_cost"] * 1000 / 32
        )
        matched = report["common_prefix"]
        self.assertEqual(matched["turns"], report["baseline_turns"])
        answers = [
            r for r in report["detail"]["candidate"]["ledger"] if r["role"] == "answer"
        ]
        self.assertGreater(report["recovery_cost"], 0)
        self.assertEqual(
            report["recovery_cost"], sum(r["projected_recovery_cost"] for r in answers)
        )
        self.assertEqual(
            report["coverage"], sum(r["historical_retained"] for r in answers) / 31
        )
        self.assertEqual(report["coverage_by_thread"]["held"]["required"], 31)
        paired = [
            r
            for r in report["detail"]["candidate"]["ledger"]
            if r["run_index"] <= matched["turns"]
        ]
        self.assertAlmostEqual(
            matched["cost_per_1000_turns"]["policy"],
            sum(
                r["cost_input_equivalents"] + r.get("projected_recovery_cost", 0)
                for r in paired
            )
            * 1000
            / matched["turns"],
        )
        too_large = thread(tid="oversized", n=1, body_size=50000)
        empty = compache.replay(policy, [too_large])
        self.assertEqual(empty["common_prefix"]["turns"], 0)
        self.assertIsNone(empty["cost_per_1000_turns"]["policy"])

    def test_deferred_cuts_keep_sources_that_an_every_turn_cut_would_lose(self):
        train = threads(gap=400)
        policy = compache.fit(
            train, [r for t in train for r in labels_for(t)], model=SPEC
        )
        self.assertEqual(policy.recipe.retrieval_runs, 0)
        held = thread(tid="held", n=6, gap=40)
        rows = labels_for(held)
        for label, earlier in zip(rows[1:], held.runs):
            source = earlier.retrievals[0]
            req = label["task_requirements"][0]
            req["historical_context_needed"] = "yes"
            req["alternatives"] = [
                {"sources": [{"part_id": source.part_id, "quote": source.text}]}
            ]
        report = compache.replay(policy, [held], rows)
        answers = [
            r for r in report["detail"]["candidate"]["ledger"] if r["role"] == "answer"
        ]
        self.assertGreater(report["coverage"], 0)
        self.assertLess(report["coverage"], 1)
        self.assertTrue(
            all(
                r["projected_recovery_cost"] == 0
                for r in answers
                if r["historical_retained"]
            )
        )
        self.assertAlmostEqual(
            report["recovery_cost"], sum(r["projected_recovery_cost"] for r in answers)
        )

    def test_replay_prices_unseen_threads(self):
        policy = fitted()
        held = [Thread.from_dict(thread(tid="held", n=6, gap=400).to_dict())]
        report = compache.replay(policy, held, labels(held), share_headers=False)
        self.assertEqual((report["threads"], report["turns"]), (1, 6))
        self.assertTrue(report["complete"])
        self.assertGreater(report["saving"], 0)
        self.assertEqual((report["needs_history"], report["coverage"]), (5, 1.0))
        self.assertGreater(report["pings"], 0)
        self.assertAlmostEqual(
            report["policy_cost"], report["replayed_cost"] + report["recovery_cost"]
        )
        self.assertIn("candidate", report["detail"])
        # The replay shares headers the way the fit did, and needs no labels to count tokens.
        self.assertFalse(policy.artifact["header"]["share_headers"])
        unlabelled = compache.replay(policy, held)
        self.assertEqual(unlabelled["replayed_cost"], report["replayed_cost"])
        self.assertEqual(
            (unlabelled["coverage"], unlabelled["needs_history"]), (None, None)
        )
        self.assertEqual(unlabelled["tokens_kept"], report["tokens_kept"])
        with self.assertRaisesRegex(compache.ContractError, "training thread"):
            compache.replay(policy, threads()[:1])


class SessionTests(unittest.TestCase):
    def test_a_conversation_under_the_policy(self):
        policy = fitted()
        session = policy.session("live")
        at, turns = 1_000_000.0, []
        for k in range(1, 5):
            turn = session.prepare(
                f"Describe item {k}.", [f"Unique document {k}. " + "x " * 1000], at=at
            )
            turns.append(turn)
            usage = {
                "input_tokens": 12,
                "output_tokens": 300,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": turn.tokens - 12,
            }
            due = session.complete(turn, ANSWER, at=at + 30, usage=usage)
            self.assertEqual(due, session.next_ping_at)
            self.assertIsNotNone(due)
            self.assertIsNone(session.ping(at=due - 5))
            ping = session.ping(at=due)
            self.assertEqual(ping.action, "ping")
            self.assertEqual(ping.parts[-1].kind, "output")
            read = {
                "input_tokens": 8,
                "output_tokens": 1,
                "cache_read_input_tokens": ping.tokens - 8,
                "cache_creation_input_tokens": 0,
            }
            self.assertTrue(session.ping_sent(at=due, usage=read))
            at += 400
        first, last = turns[0], turns[-1]
        self.assertEqual((first.action, len(first.archived_ids)), ("keep", 0))
        self.assertEqual([m["role"] for m in first.messages], ["user", "user"])
        # Earlier answers are archived: their place holds a notice with a handle.
        self.assertEqual(last.action, "cut")
        self.assertEqual(
            last.archived_ids, {"live:1:output", "live:2:output", "live:3:output"}
        )
        archived = next(p for p in last.parts if p.part_id == "live:1:output")
        self.assertIn("ctx:", archived.text)
        self.assertNotIn("observed", archived.text)
        self.assertEqual(last.messages[2]["role"], "assistant")
        self.assertLess(last.tokens, 4 * 530 + 3 * 400)
        reopened = session.reopen("live:1:output")
        self.assertEqual(reopened["status"], "ok")
        self.assertTrue(reopened["text"].startswith("An observed final answer."))
        self.assertEqual(session.reopen("no such thing")["status"], "not_found")

    def test_retry_missing_usage_and_misuse(self):
        policy = fitted()
        session = policy.session("other", system_prompt="A different prompt.")
        with self.assertRaises(compache.ContractError):
            session.reopen("anything")
        self.assertIsNone(session.ping(at=5.0))
        first = session.prepare(
            "Describe item 1.", ["Unique document 1. " + "x " * 1000], at=10.0
        )
        retry = session.prepare(
            "Describe item 1.",
            [{"text": "Unique document 1. " + "x " * 1000, "document_id": "doc-1"}],
            at=11.0,
        )
        self.assertEqual(
            [p.part_id for p in retry.parts], [p.part_id for p in first.parts]
        )
        self.assertEqual(retry.parts[1].document_id, "doc-1")
        # Without usage the request counts as cached, so a ping is still planned.
        self.assertIsNotNone(session.complete(retry, ANSWER, at=41.0))
        with self.assertRaisesRegex(compache.ContractError, "no prepared turn"):
            session.complete(retry, ANSWER, at=50.0)
        with self.assertRaises(compache.ContractError):
            session.prepare(
                "Question without text for a passage.", [{"document_id": "x"}], at=500.0
            )


class ModelTests(unittest.TestCase):
    def test_known_models_and_their_names(self):
        self.assertIn("claude-sonnet-5-5", compache.known_models())
        spec = model_spec("claude-sonnet-5-5")
        self.assertEqual(
            (spec.read_multiplier, spec.write_multiplier, spec.output_multiplier),
            (0.05, 1.25, 5.0),
        )
        self.assertEqual(
            (spec.cache_ttl_seconds, spec.cache_min_tokens, spec.context_tokens),
            (300, 512, 1_000_000),
        )
        self.assertEqual(model_spec("sonnet"), spec)
        self.assertEqual(model_spec("anthropic/claude-sonnet-5-5-20261001"), spec)
        luna = model_spec("gpt-6-luna")
        self.assertEqual(
            (luna.context_tokens, luna.long_input_threshold, luna.long_input_factor),
            (922_000, 272_000, 2.0),
        )
        self.assertEqual(pricing("opus")["input_price_per_million"], 4.0)
        self.assertIsNone(pricing("model.json"))

    def test_custom_specs(self):
        self.assertIs(model_spec(SPEC), SPEC)
        self.assertEqual(model_spec(SPEC.to_dict()), SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(SPEC.to_dict()))
            self.assertEqual(model_spec(str(path)), SPEC)
            self.assertEqual(model_spec(path), SPEC)
        partial = model_spec(
            {
                "model_id": "mine",
                "context_tokens": 200_000,
                "long_input_threshold": 100_000,
                "long_input_factor": 2.0,
            }
        )
        self.assertEqual(
            (partial.schema_version, partial.write_multiplier),
            ("compache-model-spec-v2", 1.25),
        )
        for bad in (
            "no-such-model",
            "",
            None,
            5,
            {"model_id": "x", "price": 1},
            {"context_tokens": 5},
        ):
            with self.subTest(bad=bad), self.assertRaises(compache.ContractError):
                model_spec(bad)


class ProfileTests(unittest.TestCase):
    def test_profile_measures_the_workload_and_links_the_calculator(self):
        sample = threads(gap=400)
        result = compache.profile(
            sample, labels(sample), model="claude-sonnet-5-5", system_prompt="x" * 400
        )
        measured, scenario = result["measured"], result["calculator"]["scenario"]
        self.assertEqual((measured["threads"], measured["turns"]), (4, 24))
        lengths = measured["turns_per_thread"]
        self.assertEqual(
            (lengths["median"], lengths["p95"], lengths["law"]), (6, 6, "lognormal")
        )
        self.assertEqual(
            (
                measured["labelled_turns"],
                measured["assessed_turns"],
                measured["turns_needing_history"],
            ),
            (24, 20, 20),
        )
        self.assertEqual(scenario["plannedTurns"], 6)
        self.assertEqual(scenario["systemTokens"], 100)
        self.assertAlmostEqual(scenario["gapMedianMinutes"], 370 / 60, places=2)
        # Each later turn needs the first document: one to five turns back.
        self.assertEqual(scenario["retrievalDependencyProbability"], 1.0)
        self.assertEqual(
            (
                scenario["retrievalDependencyTurnsBack"],
                scenario["retrievalDependencyTurnsBackP95"],
            ),
            (3, 5),
        )
        self.assertEqual(scenario["inputDependencyProbability"], 0.0)
        self.assertEqual(
            scenario["tokenMedian"], measured["tokens_per_turn"]["retrieval"]["median"]
        )
        self.assertEqual(result["calculator"]["model"], "sonnet")
        query = parse_qs(urlsplit(result["link"]).fragment)
        self.assertEqual(query["v"], ["6"])
        self.assertEqual(query["model"], ["sonnet"])
        self.assertEqual(json.loads(query["scenario"][0]), scenario)
        self.assertEqual(query["source"], ["4 threads, 24 turns"])

    def test_profile_without_labels_or_price_sheet(self):
        result = compache.profile(threads(count=2), model=SPEC)
        scenario = result["calculator"]["scenario"]
        self.assertNotIn("retrievalDependencyProbability", scenario)
        self.assertEqual(result["calculator"]["model"], "custom")
        self.assertEqual(scenario["model"]["cachedRate"], SPEC.read_multiplier)
        self.assertEqual(scenario["ttlMinutes"], 5)
        self.assertEqual(len(result["notes"]), 2)

    def test_profile_reads_files(self):
        sample = threads(count=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threads.jsonl.gz"
            write_threads(sample, path)
            result = compache.profile(path, model="gpt-6-luna")
        self.assertEqual(result["source"]["turns"], 12)
        self.assertEqual(result["calculator"]["model"], "luna")


if __name__ == "__main__":
    unittest.main()
