"""What `compache profile` reads off threads: length, return delay, breaks, calls and the rest."""

import math
import random
import unittest
from datetime import datetime, timedelta, timezone

import compache
from compache.profile import _fit_delay, _fit_length, measure
from compache.threads import load_threads

ORIGIN = datetime(2026, 3, 1, tzinfo=timezone.utc)


def traffic(count=400, seed=7):
    """Threads with known behaviour: most are short, long ones keep going and return faster."""
    rng = random.Random(seed)
    rows = []
    for number in range(count):
        at = ORIGIN + timedelta(minutes=rng.uniform(0, 60 * 24 * 20))
        runs, turn = [], 1
        while True:
            end = at + timedelta(seconds=12)
            run = {
                "started_at": at.isoformat(),
                "ended_at": end.isoformat(),
                "input": f"Question {turn} of thread {number}?",
                "output": "A short answer.",
                "calls": 3 if turn % 2 else 1,
                "usage": {
                    "input": 900,
                    "output": 404,
                    "cache_read": 0,
                    "cache_write": 0,
                },
            }
            if rng.random() < 0.6:
                run["retrievals"] = ["passage " * 200]
            runs.append(run)
            chance = 0.95 - (0.95 - 0.5) * 0.5 ** ((turn - 1) / 3)
            if rng.random() >= chance or turn >= 60:
                break
            if rng.random() < 0.15:
                wait = 960 * math.exp(rng.gauss(0, 0.6))
            else:
                wait = 2 * 4 ** (0.5 ** ((turn - 1) / 2)) * math.exp(rng.gauss(0, 0.8))
            at = end + timedelta(minutes=wait)
            turn += 1
        rows.append({"thread_id": f"s{number}", "runs": runs})
    return rows


class MeasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = compache.model_spec("claude-sonnet-5-5")
        cls.rows = traffic()
        cls.measured = measure(
            load_threads(cls.rows, model=cls.spec.model_id), cls.spec
        )

    def test_the_chance_of_another_turn_rises_with_position(self):
        lengths = self.measured["turns_per_thread"]
        self.assertEqual(lengths["law"], "continuation")
        another = lengths["another_turn"]
        self.assertAlmostEqual(another["after_first"], 0.5, delta=0.08)
        self.assertAlmostEqual(another["in_long_threads"], 0.95, delta=0.03)
        self.assertIn(another["half_turn"], (2, 3, 5))
        self.assertAlmostEqual(lengths["one_turn_share"], 0.5, delta=0.08)

    def test_the_return_delay_shortens_and_long_breaks_stand_apart(self):
        delay = self.measured["idle_minutes"]
        self.assertAlmostEqual(delay["break_share"], 0.15, delta=0.04)
        self.assertAlmostEqual(delay["break_median"] / 960, 1, delta=0.3)
        self.assertAlmostEqual(delay["median"] / 2, 1, delta=0.3)
        self.assertAlmostEqual(delay["first_factor"] / 4, 1, delta=0.4)
        self.assertIn(delay["half_turn"], (1, 2, 3))
        self.assertGreater(delay["p95"], delay["median"])

    def test_calls_answer_time_retrieval_and_reasoning(self):
        m = self.measured
        self.assertEqual(m["answer_seconds"], 12)
        self.assertAlmostEqual(m["calls_per_turn"], 2.3, delta=0.25)
        self.assertAlmostEqual(m["turns_that_retrieve"], 0.6, delta=0.05)
        # 404 billed output tokens against a four-token answer.
        self.assertEqual(m["reasoning_tokens"], 400)
        self.assertEqual(m["tokens_per_turn"]["retrieval"]["median"], 406)
        self.assertGreater(m["prompt_warmed_by_others"], 0)

    def test_the_scenario_carries_what_was_measured(self):
        result = compache.profile(
            self.rows, model="claude-sonnet-5-5", system_prompt="s" * 4000
        )
        scenario = result["calculator"]["scenario"]
        self.assertEqual(scenario["lengthDistribution"], "continuation")
        for key in (
            "continueFirst",
            "continueLong",
            "continueHalfTurn",
            "gapFirstFactor",
            "gapHalfTurn",
            "breakShare",
            "breakMedianMinutes",
            "sharedPrefixWarm",
        ):
            self.assertIn(key, scenario)
        self.assertEqual(
            (scenario["answerSeconds"], scenario["reasoningTokens"]), (12.0, 400)
        )
        self.assertEqual(scenario["systemTokens"], 1000)
        self.assertGreater(scenario["callsPerTurn"], 2)
        self.assertLess(scenario["retrievalShare"], 0.7)
        self.assertEqual(scenario["outputTokens"], 10)

    def test_usage_says_how_much_of_a_live_cache_entry_was_read(self):
        def rows(read_share, resend=True, pieces=1):
            made = []
            for number in range(12):
                at, runs = ORIGIN + timedelta(days=number), []
                for turn in range(1, 7):
                    sent = 1010 * turn if resend else 1010
                    runs.append(
                        {
                            "started_at": at.isoformat(),
                            "ended_at": (at + timedelta(seconds=2)).isoformat(),
                            "input": "A question?",
                            "retrievals": ["passage " * (500 // pieces)] * pieces,
                            "output": "An answer.",
                            "usage": {
                                "input": sent,
                                "output": 3,
                                "cache_read": round(read_share * 1010 * (turn - 1))
                                if resend
                                else 0,
                                "cache_write": 0,
                            },
                        }
                    )
                    at += timedelta(minutes=1)
                made.append({"thread_id": f"u{number}-{read_share}", "runs": runs})
            return made

        def scenario(threads):
            return compache.profile(threads, model="claude-sonnet-5-5")["calculator"][
                "scenario"
            ]

        # Every return is inside the cache lifetime and reads six tenths of the last request.
        self.assertAlmostEqual(scenario(rows(0.6))["cacheHitShare"], 0.6, delta=0.01)
        self.assertEqual(scenario(rows(1.0))["cacheHitShare"], 1.0)
        # A record without any cache activity says nothing: caching may be switched off.
        self.assertNotIn("cacheHitShare", scenario(rows(0.0)))
        # One that writes to the cache and never reads it back does miss.
        missed = rows(0.0)
        for thread in missed:
            for run in thread["runs"]:
                run["usage"]["cache_write"] = run["usage"]["input"]
        self.assertEqual(scenario(missed)["cacheHitShare"], 0)
        absent = rows(0.0)
        for thread in absent:
            for run in thread["runs"]:
                run["usage"] = {
                    "input_tokens": run["usage"]["input"],
                    "output_tokens": 3,
                }
        self.assertNotIn("cacheHitShare", scenario(absent))
        # An app that sends one turn at a time has trimmed its history: the layout decides its reads.
        spec = compache.model_spec("claude-sonnet-5-5")
        trimmed = measure(
            load_threads(rows(1.0, resend=False), model=spec.model_id), spec
        )
        self.assertLess(trimmed["history_resent"], 0.5)
        self.assertIsNone(trimmed["cache_read_when_live"])
        whole = measure(load_threads(rows(1.0), model=spec.model_id), spec)
        self.assertGreater(whole["history_resent"], 0.95)
        # Archiving a turn's retrieved text leaves one 22-token notice for each part.
        self.assertEqual(scenario(rows(1.0))["archivedTokens"], 28)
        self.assertEqual(scenario(rows(1.0, pieces=4))["archivedTokens"], 112)

    def test_plain_traffic_keeps_the_plain_description(self):
        rng = random.Random(3)
        rows = []
        for number in range(40):
            at, runs = ORIGIN + timedelta(days=number), []
            for turn in range(8 + number % 5):
                end = at + timedelta(seconds=2)
                runs.append(
                    {
                        "started_at": at.isoformat(),
                        "ended_at": end.isoformat(),
                        "input": "q",
                        "retrievals": ["r " * 50],
                        "output": "a",
                    }
                )
                at = end + timedelta(minutes=4 * math.exp(rng.gauss(0, 1.2)))
            rows.append({"thread_id": f"p{number}", "runs": runs})
        scenario = compache.profile(rows, model="claude-sonnet-5-5")["calculator"][
            "scenario"
        ]
        self.assertEqual(scenario["lengthDistribution"], "lognormal")
        for key in (
            "continueFirst",
            "gapFirstFactor",
            "breakShare",
            "callsPerTurn",
            "retrievalShare",
            "reasoningTokens",
            "sharedPrefixWarm",
        ):
            self.assertNotIn(key, scenario)
        self.assertEqual(scenario["answerSeconds"], 2.0)
        self.assertAlmostEqual(scenario["gapMedianMinutes"] / 4, 1, delta=0.3)


class FitTests(unittest.TestCase):
    def test_a_few_threads_or_one_length_stay_lognormal(self):
        self.assertEqual(_fit_length([10, 26, 10, 25])["law"], "lognormal")
        fixed = _fit_length([12] * 50)
        self.assertEqual(
            (fixed["law"], fixed["median"], fixed["p95"]), ("lognormal", 12, 12)
        )

    def test_too_few_delays_report_only_their_spread(self):
        found = _fit_delay([(1, 3.0), (2, 5.0), (3, 4.0)])
        self.assertEqual((found["median"], found["p95"]), (4.0, 5.0))
        self.assertEqual(
            (found["first_factor"], found["half_turn"], found["break_share"]),
            (1.0, None, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
