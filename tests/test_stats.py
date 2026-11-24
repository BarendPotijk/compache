"""What a record says was saved."""

from __future__ import annotations

import tempfile
import unittest

import compache
from compache import stats
from compache.learn import Learner
from compache.record import load_calls
from fakes import FakeModel
from test_agent_live import Agent, Claude, policy, worker
from test_live import Clock

MODEL = "claude-haiku-5-5"


class Haiku(Claude):
    model = MODEL


def row(thread, at, size, read=0, written=0, output=100, kind="answer", **more):
    usage = {
        "input": size,
        "output": output,
        "cache_read": read,
        "cache_write": written,
    }
    return {
        "v": 1,
        "thread": thread,
        "turn": 1,
        "kind": kind,
        "at": at,
        "model": MODEL,
        "usage": usage,
        **more,
    }


class SavingTests(unittest.TestCase):
    def test_what_was_left_out_would_have_been_read_from_the_cache(self):
        spec = compache.model_spec(MODEL)
        price = 0.1 / 1e6
        rows = [
            row("a", 0, 10_000, 0, 10_000, left_bytes=0, sent_bytes=40_000),
            row("a", 60, 6_000, 2_000, 4_000, left_bytes=20_000, sent_bytes=24_000),
        ]
        found = stats.savings(rows)
        # 24,000 bytes were 6,000 tokens, so 20,000 left out were 5,000.
        first = 10_000 * 1.25 + 100 * 5
        second = 10_000 * spec.read_multiplier + 1_000 * 1.25 + 100 * 5
        self.assertAlmostEqual(found["without"], (first + second) * price)
        paid = first + 2_000 * spec.read_multiplier + 4_000 * 1.25 + 100 * 5
        self.assertAlmostEqual(found["spent"], paid * price)
        self.assertEqual(found["models"][MODEL]["left_out_tokens"], 5_000)

    def test_after_a_pause_everything_would_have_been_written_again(self):
        rows = [
            row("a", 0, 50_000, 0, 50_000, left_bytes=0, sent_bytes=200_000),
            row("a", 4_000, 10_000, 0, 10_000, left_bytes=170_000, sent_bytes=40_000),
        ]
        self.assertGreater(stats.savings(rows)["saving"], 0.3)

    def test_fetching_back_and_pinging_are_paid_for(self):
        rows = [
            row("a", 0, 10_000, 0, 10_000, left_bytes=0, sent_bytes=40_000),
            row("a", 1, 10_100, 10_000, 100, kind="reopen"),
            row("a", 250, 10_100, 10_100, 0, 1, kind="ping"),
        ]
        found = stats.savings(rows)
        self.assertGreater(found["spent"], found["without"])
        self.assertEqual(found["models"][MODEL]["reopened"], 1)
        self.assertEqual(found["models"][MODEL]["pings"], 1)
        self.assertIn("more", "\n".join(stats.lines(found)))

    def test_what_was_left_alone_cost_what_it_cost(self):
        rows = [row("b", 0, 10_000, 0, 0), row("b", 9, 12_000, 0, 0)]
        found = stats.savings(rows)
        self.assertEqual(found["spent"], found["without"])
        self.assertEqual(found["left_alone"], 1)

    def test_an_hour_s_entries_are_paid_at_their_price(self):
        short = stats.savings([row("a", 0, 10_000, 0, 10_000)])["spent"]
        long = stats.savings([row("a", 0, 10_000, 0, 10_000, ttl=3600)])["spent"]
        self.assertGreater(long, short * 1.4)

    def test_a_model_without_prices_is_counted_apart(self):
        rows = [dict(row("a", 0, 10_000), model="my-model")]
        found = stats.savings(rows)
        self.assertEqual(found["spent"], 0)
        self.assertEqual(found["unpriced"]["my-model"]["calls"], 1)
        self.assertIn("no price sheet", "\n".join(stats.lines(found)))

    def test_an_empty_record_says_so(self):
        with tempfile.TemporaryDirectory() as folder:
            found = stats.savings(folder)
        self.assertEqual(stats.lines(found), ["No calls on record yet."])

    def test_a_loop_under_a_policy_is_cheaper_than_the_same_loop_without(self):
        """The estimate against the truth: the same agent, run both ways on the same provider."""
        spent = {}
        spec = compache.model_spec(MODEL)
        for name in ("policy", "nothing"):
            clock = Clock()
            with tempfile.TemporaryDirectory() as folder:
                model = FakeModel(worker(40), clock=lambda: clock.at)
                live = compache.Live(
                    policy(rounds=6, ratio=1.5, spec=spec)
                    if name == "policy"
                    else None,
                    record=folder,
                    clock=clock,
                )
                app = Agent(live, Haiku(), model, clock=clock)
                if name == "nothing":
                    # Without a policy nothing is marked, so the application marks for itself.
                    body = app.shape.body
                    app.shape.body = lambda stream: {
                        **body(stream),
                        "cache_control": {"type": "ephemeral"},
                    }
                app.run("Read forty files")
                spent[name] = stats.savings(load_calls(folder))
        self.assertLess(spent["policy"]["spent"], spent["nothing"]["spent"])
        ours, theirs = spent["policy"]["without"], spent["nothing"]["spent"]
        self.assertLess(abs(ours - theirs) / theirs, 0.1)

    def test_a_learned_record_reads_well(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            live = compache.Live(
                learn=Learner(folder, clock=clock, background=False), clock=clock
            )
            for k in range(10):
                clock.at += 4_000
                model = FakeModel(worker(40), clock=lambda: clock.at)
                Agent(live, Haiku(), model, clock=clock).run(f"Task {k}")
            said = "\n".join(stats.lines(stats.savings(folder)))
        self.assertIn("410 calls in 10 conversations", said)
        self.assertIn("left out", said)
        self.assertIn("1 conversation went out as written", said)


if __name__ == "__main__":
    unittest.main()
