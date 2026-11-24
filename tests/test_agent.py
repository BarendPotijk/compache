"""The policy for agents: what a conversation holds, what it costs, what is fitted."""

from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import compache
from compache import agent
from compache.agent import AgentRecipe, Flow
from compache.cli import main
from compache.models import model_spec
from compache.quotes import Quotes, spoken

SPEC = compache.ModelSpec(
    "agent-model",
    context_tokens=200_000,
    cache_ttl_seconds=300,
    cache_min_tokens=0,
    read_multiplier=0.1,
    write_multiplier=1.25,
    output_multiplier=5.0,
)


def flow_of(recipe, rounds, tokens=1_000, spec=SPEC, at=0.0, every=5.0):
    """A conversation that has made `rounds` calls, one result each."""
    flow = Flow(spec, recipe)
    flow.add(2_000)
    for k in range(rounds):
        flow.rounds += 1
        flow.add(tokens, f"r{k}")
        flow.cut(at + k * every)
        flow.sent(at + k * every)
        flow.add(100)
    return flow


def runs(count, calls=40, seed=1, quotes=0.3, tokens=600, away=None):
    """Recorded agent runs whose replies quote earlier results, mostly recent ones."""
    rng, made = random.Random(seed), []
    for i in range(count):
        at, messages = 1_790_000_000.0 + i * 1e6, []
        messages.append({"role": "user", "content": f"Task {i}: fix the bug", "at": at})
        shown = []
        for k in range(calls):
            said = f"Step {k} of run {i}."
            if shown and rng.random() < quotes:
                back = min(len(shown) - 1, int(rng.expovariate(0.4)))
                said += "\n" + shown[-1 - back]
            at += 8
            messages.append(
                {
                    "role": "assistant",
                    "content": said,
                    "at": at,
                    "tool_calls": [
                        {
                            "id": f"c{i}-{k}",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls"}',
                            },
                        }
                    ],
                }
            )
            line = f"a line that only result {i}-{k} holds, word for word"
            shown.append(line)
            at += 3
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"c{i}-{k}",
                    "content": line + "\n" + "x" * 4 * tokens,
                    "at": at,
                }
            )
            if away and k == calls // 2:
                messages.append(
                    {"role": "assistant", "content": "Done so far.", "at": at + 4}
                )
                at += away
                messages.append(
                    {"role": "user", "content": "Now the next part", "at": at}
                )
        messages.append({"role": "assistant", "content": "Done.", "at": at + 8})
        made.append({"thread_id": f"run-{i}", "messages": messages})
    return made


class FlowTests(unittest.TestCase):
    def test_everything_stays_without_a_cut(self):
        flow = flow_of(AgentRecipe(), 30)
        self.assertEqual(flow.size, 2_000 + 30 * 1_100)
        self.assertFalse(flow.archived)

    def test_old_results_go_once_they_outweigh_what_follows(self):
        flow = flow_of(AgentRecipe(rounds=3, ratio=1.0), 30)
        self.assertTrue(flow.archived)
        for k in range(27, 30):
            self.assertFalse(flow.gone(f"r{k}"), "the newest rounds always stay")
        self.assertEqual(flow.size, sum(flow.shown))
        self.assertLess(flow.size, 2_000 + 30 * 1_100)

    def test_a_cut_waits_while_the_cache_is_alive_and_little_would_go(self):
        flow = flow_of(AgentRecipe(rounds=3, ratio=3.0), 6)
        self.assertEqual(flow.cut(30.0), [])
        self.assertTrue(flow.warm(30.0))

    def test_an_expired_cache_is_the_time_to_cut(self):
        flow = flow_of(AgentRecipe(rounds=3, ratio=None), 6)
        self.assertEqual(flow.cut(100.0), [])
        went = flow.cut(10_000.0)
        self.assertEqual([name for name, _ in went], ["r0", "r1", "r2"])

    def test_a_cut_is_made_where_this_very_call_is_cheaper_for_it(self):
        dear = replace(
            SPEC,
            schema_version="compache-model-spec-v2",
            long_input_threshold=20_000,
            long_input_factor=5.0,
            long_output_factor=5.0,
        )
        patient = AgentRecipe(rounds=8, ratio=None)
        flow = flow_of(patient, 30, spec=dear)
        self.assertTrue(flow.archived, "a request past the threshold costs five times")
        self.assertLessEqual(flow.size, 20_000)
        self.assertFalse(flow_of(patient, 30).archived)

    def test_a_result_shorter_than_its_notice_stays(self):
        flow = flow_of(AgentRecipe(rounds=1, ratio=0.0), 10, tokens=8)
        self.assertFalse(flow.archived)

    def test_what_is_cached_ahead_of_a_cut_is_still_read(self):
        flow = flow_of(AgentRecipe(rounds=2, ratio=None), 6)
        flow.recipe = AgentRecipe(rounds=2, ratio=0.0)
        self.assertTrue(flow.cut(30.0))
        self.assertEqual(flow.cached, 2_000)

    def test_a_request_that_would_not_fit_is_cut_whatever_it_costs(self):
        small = replace(SPEC, context_tokens=12_000)
        flow = flow_of(AgentRecipe(rounds=2, ratio=None), 9, spec=small)
        self.assertLessEqual(flow.size, 12_000)

    def test_what_was_fetched_back_stands_at_the_end_until_it_is_old(self):
        flow = flow_of(AgentRecipe(rounds=2, ratio=0.0), 8)
        self.assertTrue(flow.gone("r0"))
        size = flow.size
        flow.add(1_000, "r0", again=True)
        self.assertFalse(flow.gone("r0"))
        for k in range(8, 12):
            flow.rounds += 1
            flow.add(1_000, f"r{k}")
            flow.cut(10_000.0 * k)
        self.assertTrue(flow.gone("r0"))
        self.assertEqual(flow.size, sum(flow.shown))
        self.assertLessEqual(flow.size, size + 4 * 1_000)

    def test_the_size_is_always_what_is_shown(self):
        rng = random.Random(7)
        for _ in range(20):
            recipe = AgentRecipe(rng.randint(1, 6), rng.choice([0.0, 0.5, 2.0, None]))
            flow, at = Flow(SPEC, recipe), 0.0
            for k in range(60):
                flow.rounds += 1
                flow.add(rng.randint(1, 3_000), f"r{k}")
                gone = sorted(flow.archived - flow.back)
                if gone and rng.random() < 0.2:
                    flow.add(flow.sizes[gone[0]], gone[0], again=True)
                at += rng.choice([5.0, 5.0, 400.0])
                flow.cut(at)
                self.assertEqual(flow.size, sum(flow.shown))
                newest = {f"r{j}" for j in range(max(0, k - recipe.rounds + 1), k + 1)}
                self.assertFalse(newest & flow.archived)
                flow.sent(at)

    def test_a_copy_goes_its_own_way(self):
        flow = flow_of(AgentRecipe(rounds=2, ratio=None), 6)
        other = flow.copy()
        other.add(5_000, "more")
        other.rounds += 5
        other.cut(10_000.0)
        self.assertFalse(flow.archived)
        self.assertNotEqual(flow.size, other.size)
        self.assertEqual(flow.size, sum(flow.shown))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.threads = compache.load_threads(
            runs(16, quotes=0.5, tokens=2_500), model=SPEC.model_id
        )

    def test_turns_of_many_calls_are_an_agent_s(self):
        self.assertTrue(agent.is_agentic(self.threads))
        chat = [
            {
                "thread_id": "chat",
                "runs": [{"input": "q", "output": "a", "started_at": 1, "ended_at": 2}],
            }
        ]
        self.assertFalse(
            agent.is_agentic(compache.load_threads(chat, model=SPEC.model_id))
        )

    def test_the_fit_is_cheaper_and_keeps_what_replies_rely_on(self):
        fitted = agent.fit_agent(self.threads, SPEC)
        solved = fitted["solve"]
        self.assertLess(solved["chosen"]["cost"], solved["keep_everything_cost"])
        self.assertGreaterEqual(solved["chosen"]["coverage"], agent.FLOOR)
        self.assertGreater(solved["relying_calls"], 40)

    def test_where_little_is_relied_on_few_calls_may_miss(self):
        """Agents that rarely quote: what counts is how many of all calls would have to fetch back."""
        rows = runs(10, quotes=0.03, tokens=2_500, seed=4)
        threads = compache.load_threads(rows, model=SPEC.model_id)
        fitted = agent.fit_agent(threads, SPEC)
        solved = fitted["solve"]
        self.assertLess(solved["relying_calls"], 40)
        self.assertGreaterEqual(solved["known_calls"], 200)
        self.assertLess(fitted["recipe"]["rounds"], 8, "less is kept than the default")
        missed = solved["chosen"]["missed_calls"]
        self.assertLessEqual((missed + 1) / solved["known_calls"], agent.MOST_MISSED)

    def test_it_holds_on_runs_it_has_not_seen(self):
        train, test = self.threads[::2], self.threads[1::2]
        fitted = agent.fit_agent(train, SPEC)
        held = agent.replay_agent(fitted, test)
        self.assertGreater(held["saving"], 0.05)
        self.assertGreater(held["coverage"], 0.8)
        self.assertEqual(held["threads"], 8)

    def test_what_is_chosen_is_the_cheapest_that_may_be(self):
        fitted = agent.fit_agent(self.threads, SPEC)
        known = fitted["solve"]["known_calls"]
        allowed = [
            row
            for row in fitted["solve"]["candidates"]
            if row["recipe"]["rounds"] is None
            or (row["coverage"] or 0) >= agent.FLOOR
            or (row["missed_calls"] + 1) / known <= agent.MOST_MISSED
        ]
        self.assertEqual(
            fitted["solve"]["chosen"]["cost"], min(row["cost"] for row in allowed)
        )

    def test_the_policy_s_price_is_what_its_flow_costs(self):
        fitted = agent.fit_agent(self.threads, SPEC)
        recipe = AgentRecipe(**fitted["recipe"])
        conversations = agent._conversations(self.threads, SPEC, 0)
        again = agent.simulate(conversations, recipe, SPEC, 8192)
        self.assertAlmostEqual(again["cost"], fitted["solve"]["chosen"]["cost"])

    def test_too_little_evidence_leaves_the_cut_that_public_runs_taught(self):
        silent = compache.load_threads(runs(3, quotes=0.0), model=SPEC.model_id)
        fitted = agent.fit_agent(silent, SPEC)
        cuts = {
            (row["recipe"]["rounds"], row["recipe"]["ratio"])
            for row in fitted["solve"]["candidates"]
        }
        start = agent.default_recipe(SPEC)
        self.assertEqual(cuts, {(None, None), (start.rounds, start.ratio)})

    def test_pings_are_fitted_to_tools_that_take_longer_than_the_cache(self):
        rows = runs(6, quotes=0.0)
        for row in rows:
            at = row["messages"][0]["at"]
            for message in row["messages"][1:]:
                at += 8 if message["role"] == "assistant" else 420
                message["at"] = at
        threads = compache.load_threads(rows, model=SPEC.model_id)
        fitted = agent.fit_agent(threads, SPEC)
        self.assertGreater(fitted["recipe"]["tool_pings"], 0)

    def test_a_longer_lifetime_is_chosen_where_it_pays(self):
        # Coming back after a long time each step is what an hour's entries are for.
        hour = replace(SPEC, cache_ttl_seconds=3600, write_multiplier=2.0)
        rows = runs(6, quotes=0.0)
        for row in rows:
            at = row["messages"][0]["at"]
            for message in row["messages"][1:]:
                at += 8 if message["role"] == "assistant" else 1_500
                message["at"] = at
        threads = compache.load_threads(rows, model=SPEC.model_id)
        fitted = agent.fit_agent(threads, SPEC, lifetimes=[hour])
        self.assertEqual(fitted["model_spec"]["cache_ttl_seconds"], 3600)
        self.assertEqual(fitted["standard_spec"]["cache_ttl_seconds"], 300)

    def prompts(self, pause, count=10, header=0, tokens=3_000):
        """Sessions of short prompts with a user in between, behind instructions of some size."""
        rows = []
        for i in range(count):
            at, messages = 1_790_000_000.0 + 1e6 * i, []
            messages.append({"role": "system", "content": "s" * 4 * header})
            for p in range(6):
                messages.append({"role": "user", "content": f"Part {p}", "at": at})
                for k in range(8):
                    at += 8
                    call = {
                        "id": f"{p}-{k}",
                        "function": {"name": "r", "arguments": "{}"},
                    }
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [call],
                            "at": at,
                        }
                    )
                    at += 3
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"{p}-{k}",
                            "content": f"{i}-{p}-{k} " + "x" * 4 * tokens,
                            "at": at,
                        }
                    )
                at += 8
                messages.append({"role": "assistant", "content": "Done.", "at": at})
                at += pause
            rows.append({"thread_id": f"s{i}", "messages": messages})
        return rows

    def test_instructions_and_tools_may_be_cached_for_longer_than_the_rest(self):
        hour = replace(SPEC, cache_ttl_seconds=3600, write_multiplier=2.0)
        rows = self.prompts(900, header=20_000)
        threads = compache.load_threads(rows, model=SPEC.model_id)
        fitted = agent.fit_agent(threads, SPEC, lifetimes=[hour], header_tokens=20_000)
        solved = fitted["solve"]
        rows_ = solved["candidates"]
        both = [r for r in rows_ if r["header_ttl_seconds"] > r["cache_ttl_seconds"]]
        plain = [r for r in rows_ if r["header_ttl_seconds"] == r["cache_ttl_seconds"]]
        self.assertLess(min(r["cost"] for r in both), min(r["cost"] for r in plain))
        self.assertEqual(fitted["header_spec"]["cache_ttl_seconds"], 3600)
        self.assertEqual(fitted["model_spec"]["cache_ttl_seconds"], 300)
        held = agent.replay_agent(fitted, threads, header_tokens=20_000)
        self.assertAlmostEqual(held["policy_cost"], solved["chosen"]["cost"])

    def test_small_instructions_are_left_at_the_default(self):
        hour = replace(SPEC, cache_ttl_seconds=3600, write_multiplier=2.0)
        rows = self.prompts(20, tokens=500)
        threads = compache.load_threads(rows, model=SPEC.model_id)
        fitted = agent.fit_agent(threads, SPEC, lifetimes=[hour], header_tokens=500)
        self.assertNotIn("header_spec", fitted)

    def test_what_another_conversation_left_in_the_cache_is_read(self):
        talk = agent.Talk(
            [agent.Call(1_000.0, 1_001.0, 9_000, (), 10, frozenset(), False)]
        )
        talk.header = 8_000
        talk.times = [900.0, 1_000.0]
        warm = agent.simulate([talk], AgentRecipe(), SPEC)["cost"]
        talk.times = [100.0, 1_000.0]
        cold = agent.simulate([talk], AgentRecipe(), SPEC)["cost"]
        self.assertAlmostEqual(cold - warm, 8_000 * (1.25 - 0.1))

    def test_labels_say_what_a_turn_relied_on_too(self):
        silent = compache.load_threads(runs(1, quotes=0.0), model=SPEC.model_id)
        row = {
            "thread_id": "run-0",
            "run": 1,
            "needs": [["run-0:1:retrieval:1"]],
        }
        calls = agent._conversations(silent, SPEC, 0, [row])[0]
        self.assertEqual(calls[-1].needs, frozenset({"run-0:1:retrieval:1"}))

    def test_sizes_are_what_the_provider_counted(self):
        rows = runs(1, calls=6, quotes=0.0)
        held = 1_000
        for message in rows[0]["messages"]:
            if message["role"] == "assistant":
                # Twice as many tokens as four bytes a token would give.
                message["usage"] = {"input_tokens": held, "output_tokens": 30}
                held += 2 * 650 + 30
        threads = compache.load_threads(rows, model=SPEC.model_id)
        calls = agent.calls_of(threads[0], SPEC)
        totals, total = [], 0
        for call in calls:
            total += call.said + sum(tokens for _, tokens in call.found)
            totals.append(total)
        counted = [s.input_tokens for s in threads[0].runs[0].steps]
        for ours, theirs in zip(totals, counted):
            self.assertLess(abs(ours - theirs), 5)
        sizes = [tokens for call in calls for _, tokens in call.found]
        self.assertGreater(min(sizes), 1_200)
        # Without a count later on, sizes stay off by as much as they were.
        for message in rows[0]["messages"][6:]:
            message.pop("usage", None)
        threads = compache.load_threads(rows, model=SPEC.model_id)
        sizes = [t for c in agent.calls_of(threads[0], SPEC) for _, t in c.found]
        self.assertGreater(min(sizes), 1_100)

    def test_one_recorded_call_is_not_enough(self):
        one = [{"thread_id": "t", "runs": [{"input": "q", "output": "a", "at": 1}]}]
        one[0]["runs"][0] = {"input": "q", "output": "a", "started_at": 1}
        with self.assertRaises(compache.ContractError):
            agent.fit_agent(compache.load_threads(one, model=SPEC.model_id), SPEC)


class PolicyTests(unittest.TestCase):
    def test_fit_chooses_the_agent_s_policy_by_itself(self):
        policy = compache.fit(runs(8), model="claude-haiku-5-5")
        self.assertTrue(policy.agentic)
        self.assertEqual(policy.summary["calls"], 8 * 41)
        self.assertGreater(policy.summary["saving"], 0)
        self.assertEqual(policy.summary["cost_per_1000_calls"]["unit"], "USD")
        text = policy.describe()
        self.assertIn("rounds of tool results", text)
        self.assertIn("per 1,000 calls", text)

    def test_it_is_saved_loaded_and_replayed_like_any_other(self):
        policy = compache.fit(runs(8), model="claude-haiku-5-5")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            policy.save(path)
            again = compache.Policy.load(path)
            self.assertEqual(again.recipe, policy.recipe)
            report = compache.replay(path and again, runs(4, seed=9))
        self.assertGreater(report["saving"], 0)
        self.assertIn("policy", report["cost_per_1000_calls"])

    def test_the_header_s_lifetime_is_said_and_can_be_dropped(self):
        rows = FitTests().prompts(900, header=20_000)
        policy = compache.fit(rows, model="claude-haiku-5-5")
        self.assertEqual(policy.header_lifetime, 3600)
        self.assertIn("instructions and tools", policy.describe())
        plain = policy.at_default_lifetime()
        self.assertIsNone(plain.header_lifetime)
        self.assertEqual(plain.recipe, policy.recipe)

    def test_a_chat_s_policy_stays_what_it_was(self):
        rows = [
            {
                "thread_id": f"t{i}",
                "runs": [
                    {
                        "input": f"question {n}",
                        "retrievals": ["text " * 500],
                        "output": "answer " * 20,
                        "started_at": 1_790_000_000 + i * 1e5 + n * 400,
                    }
                    for n in range(6)
                ],
            }
            for i in range(4)
        ]
        self.assertFalse(compache.fit(rows, model="claude-sonnet-5-5").agentic)

    def test_the_default_without_a_cache_keeps_to_its_rounds(self):
        self.assertEqual(agent.default_recipe(model_spec("claude-haiku-5-5")).rounds, 8)
        none = agent.default_recipe(replace(SPEC, cache_ttl_seconds=0))
        self.assertEqual(none.ratio, 0.0)
        self.assertFalse(none.tool_pings)


class CommandTests(unittest.TestCase):
    """What the command line says of an agent's policy."""

    def run_it(self, *words):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main([str(w) for w in words])
        return code, out.getvalue()

    def test_fit_replay_and_default(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name, seed in (("train", 1), ("test", 2)):
                with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as stream:
                    for row in runs(8, seed=seed, quotes=0.5, tokens=2_000):
                        stream.write(json.dumps(row) + "\n")
            code, said = self.run_it(
                "fit",
                folder / "train.jsonl",
                "--model",
                "claude-haiku-5-5",
                "-o",
                folder / "policy.json",
            )
            self.assertEqual(code, 0, said)
            self.assertIn("rely on an earlier result", said)
            self.assertIn("rounds of tool results", said)
            code, said = self.run_it(
                "replay",
                folder / "test.jsonl",
                "--policy",
                folder / "policy.json",
                "-o",
                folder / "report.json",
            )
            self.assertEqual(code, 0, said)
            self.assertIn("per 1,000 calls", said)
            report = json.loads((folder / "report.json").read_text())
            self.assertGreater(report["saving"], 0)
            # The calculator prices chats, and says so.
            code, said = self.run_it(
                "profile",
                folder / "test.jsonl",
                "--model",
                "claude-haiku-5-5",
                "--policy",
                folder / "policy.json",
            )
            self.assertEqual(code, 1)
            self.assertIn("call by call", said)
            code, said = self.run_it(
                "default",
                "--agent",
                "--model",
                "claude-sonnet-5-5",
                "-o",
                folder / "default.json",
            )
            self.assertEqual(code, 0, said)
            self.assertIn("public coding-agent runs", said)
            self.assertTrue(compache.Policy.load(folder / "default.json").agentic)

    def test_the_default_for_agents_has_no_workload(self):
        policy = compache.default_policy("claude-sonnet-5-5", agent=True)
        self.assertEqual(policy.max_pings, 3)
        self.assertIsNone(policy.cache_lifetime)
        self.assertEqual(policy.header_lifetime, 3600)
        self.assertIsNone(
            compache.default_policy("gpt-6-luna", agent=True).header_lifetime
        )
        self.assertIsNone(policy.summary["saving"])
        with self.assertRaises(TypeError):
            compache.default_policy("claude-sonnet-5-5", agent=True, threads=3)


class SessionTests(unittest.TestCase):
    """The same policy driven by an application that assembles its own requests."""

    def policy(self, rounds=2, ratio=0.0, tool_pings=0, user_pings=0):
        return compache.Policy(
            {
                "schema_version": agent.SCHEMA,
                "model_spec": SPEC.to_dict(),
                "recipe": {
                    "rounds": rounds,
                    "ratio": ratio,
                    "tool_pings": tool_pings,
                    "user_pings": user_pings,
                },
                "reserve_tokens": 0,
                "training_threads": [],
            }
        )

    def test_a_loop_is_decided_on_round_by_round(self):
        session = self.policy().session("t")
        turn = session.prepare("Fix it", at=0.0)
        self.assertFalse(turn.archived_ids)
        for k in range(5):
            session.replied(turn, "looking", [{"path": f"f{k}"}], at=10.0 * k + 5)
            turn = session.extend(turn, ["content " * 500], at=10.0 * k + 10)
        self.assertEqual(turn.archived_ids, {f"t:1:retrieval:{k}" for k in range(1, 4)})
        shown = {p.part_id: p.text for p in turn.parts}
        self.assertIn("Compacted passage", shown["t:1:retrieval:1"])
        self.assertIn("content", shown["t:1:retrieval:5"])
        found = session.reopen("t:1:retrieval:1")
        self.assertIn("content content", found["text"])
        session.complete(turn, "fixed", at=70.0)
        again = session.prepare("And the tests?", at=80.0)
        self.assertEqual(again.archived_ids, turn.archived_ids)

    def test_a_ping_is_due_while_a_tool_runs_and_while_the_user_is_away(self):
        session = self.policy(None, None, tool_pings=2, user_pings=1).session("t")
        turn = session.prepare("Run the tests " * 400, at=0.0)
        usage = {
            "input_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 2_000,
            "output_tokens": 20,
        }
        self.assertEqual(session.replied(turn, "", [{}], at=4.0, usage=usage), 255.0)
        self.assertIsNone(session.ping(at=200.0), "not due yet")
        self.assertIsNotNone(session.ping(at=256.0))
        read = {**usage, "cache_read_input_tokens": 2_000}
        self.assertTrue(session.ping_sent(at=256.0, usage=read))
        self.assertEqual(session.next_ping_at, 256.0 + 255.0)
        self.assertIsNotNone(session.ping(at=512.0))
        self.assertTrue(session.ping_sent(at=512.0, usage=read))
        self.assertIsNone(session.next_ping_at, "two were planned")
        turn = session.extend(turn, ["passed"], at=600.0)
        self.assertEqual(session.complete(turn, "Passed.", at=603.0, usage=read), 855.0)

    def test_a_ping_that_would_come_too_late_is_not_sent(self):
        session = self.policy(None, None, tool_pings=1).session("t")
        turn = session.prepare("Run the tests " * 400, at=0.0)
        session.replied(turn, "", [{}], at=4.0)
        self.assertIsNone(session.ping(at=1_000.0))
        self.assertIsNone(session.next_ping_at)

    def test_a_ping_that_read_nothing_is_the_last(self):
        session = self.policy(None, None, tool_pings=3).session("t")
        turn = session.prepare("Run the tests " * 400, at=0.0)
        session.replied(turn, "", [{}], at=4.0)
        usage = {"input_tokens": 2_000, "cache_read_input_tokens": 0}
        self.assertFalse(session.ping_sent(at=256.0, usage=usage))
        self.assertIsNone(session.next_ping_at)

    def test_what_the_provider_counts_corrects_what_was_estimated(self):
        session = self.policy(None, None).session("t")
        turn = session.prepare("Fix it", at=0.0)
        usage = {"input_tokens": 9_000, "output_tokens": 50}
        session.replied(turn, "", [{}], at=4.0, usage=usage)
        flow = session._engine.flow
        self.assertEqual(flow.size, 9_050)
        turn = session.extend(turn, ["x" * 4_000], at=10.0)
        usage = {"input_tokens": 11_100, "output_tokens": 50}
        session.replied(turn, "", [{}], at=14.0, usage=usage)
        self.assertEqual(flow.size, 11_150)
        self.assertEqual(flow.sizes["t:1:retrieval:1"], 2_050)
        # What comes next is sized like what came before it.
        session.extend(turn, ["x" * 4_000], at=20.0)
        self.assertGreater(flow.sizes["t:1:retrieval:2"], 1_900)
        self.assertEqual(flow.size, sum(flow.shown))

    def test_a_session_can_branch(self):
        session = self.policy().session("t")
        turn = session.prepare("Fix it", at=0.0)
        session.complete(turn, "fixed", at=5.0)
        other = session.copy()
        turn = other.prepare("And this?", at=10.0)
        for k in range(4):
            turn = other.extend(turn, ["content " * 500], at=20.0 + k)
        self.assertTrue(turn.archived_ids)
        self.assertFalse(session.prepare("Something else", at=30.0).archived_ids)


class QuoteTests(unittest.TestCase):
    def test_a_line_that_only_a_tool_said_shows_what_a_reply_relied_on(self):
        quotes = Quotes()
        quotes.own("the user wrote this whole line themselves")
        quotes.result("R1.1", "   12\tdef compute_total_of(items):\n        pass")
        quotes.result("R1.2", "the user wrote this whole line themselves")
        said = spoken("", [{"old_string": "def compute_total_of(items):"}])
        self.assertEqual(quotes.said(said), ["R1.1"])
        # Said once, it is the model's own from then on.
        self.assertEqual(quotes.said(said), [])
        self.assertEqual(quotes.said("the user wrote this whole line themselves"), [])

    def test_a_line_seen_twice_counts_for_where_it_was_seen_last(self):
        quotes = Quotes()
        quotes.result("R1.1", "a line of some length that came up twice")
        quotes.result("R1.4", "a line of some length that came up twice")
        self.assertEqual(
            quotes.said("a line of some length that came up twice"), ["R1.4"]
        )

    def test_short_lines_and_numbers_say_nothing(self):
        quotes = Quotes()
        quotes.result("R1.1", "return None\n1234567890123456789012345678901234567890")
        self.assertEqual(
            quotes.said("return None\n1234567890123456789012345678901234567890"), []
        )


class StepTests(unittest.TestCase):
    def test_a_log_of_messages_gives_the_calls_of_a_turn(self):
        thread = compache.load_threads(runs(1, calls=3, quotes=1.0))[0]
        (run,) = thread.runs
        self.assertEqual(run.calls, 4)
        self.assertEqual([s.results for s in run.steps], [0, 1, 2, 3])
        self.assertTrue(any(s.needs for s in run.steps))
        starts = [s.started_at for s in run.steps]
        self.assertEqual(starts, sorted(starts))

    def test_steps_survive_a_round_trip(self):
        thread = compache.load_threads(runs(1, calls=3, quotes=1.0))[0]
        again = compache.load_threads([thread.to_dict()])[0]
        self.assertEqual(again.runs[0].steps, thread.runs[0].steps)

    def test_a_remark_beside_tool_results_is_no_new_turn(self):
        messages = [
            {"role": "user", "content": "Fix it", "at": 1},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking."},
                    {"type": "tool_use", "id": "a", "name": "read", "input": {}},
                ],
                "at": 2,
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "text"},
                    {"type": "text", "text": "A reminder from the harness."},
                ],
                "at": 3,
            },
            {"role": "assistant", "content": "Done.", "at": 4},
            {"role": "user", "content": "Thanks, and now?", "at": 5},
            {"role": "assistant", "content": "Nothing.", "at": 6},
        ]
        thread = compache.load_threads([{"thread_id": "t", "messages": messages}])[0]
        self.assertEqual(len(thread.runs), 2)
        self.assertIn("reminder", thread.runs[0].input.text)
        self.assertEqual(thread.runs[0].calls, 2)

    def test_steps_that_contradict_the_run_are_refused(self):
        run = {
            "input": "q",
            "output": "a",
            "started_at": 1,
            "ended_at": 9,
            "retrievals": ["one"],
            "steps": [
                {"started_at": 1, "ended_at": 2, "results": 0, "bytes": 10},
                {"started_at": 3, "ended_at": 9, "results": 2, "bytes": 10},
            ],
        }
        with self.assertRaises(compache.ContractError):
            compache.load_threads([{"thread_id": "t", "runs": [run]}])


if __name__ == "__main__":
    unittest.main()
