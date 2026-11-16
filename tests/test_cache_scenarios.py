import unittest
from compache.optimizer.cache_scenarios import reprice
from compache.traces import ContractError


def row(seq, time, role="answer", usage=None, ok=True, arm="a", replicate=1):
    return {
        "arm": arm,
        "replicate": replicate,
        "sequence": seq,
        "logical_at": f"2026-01-01T00:{time}:00Z",
        "role": role,
        "ok": ok,
        "usage": usage
        if usage is not None
        else {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10},
    }


class ScenariosTests(unittest.TestCase):
    def test_ttl_prices_and_overhead(self):
        rows = [
            row(1, "00"),
            row(2, "05", "ping"),
            row(3, "15", "summary"),
            row(4, "20", "recovery"),
            row(5, "21"),
        ]
        result = reprice(rows)["replicates"]
        short, long = result
        self.assertEqual(
            [c["warm"] for c in short["calls"]], [False, False, False, False, True]
        )
        self.assertEqual(
            [c["warm"] for c in long["calls"]], [False, True, True, True, True]
        )
        self.assertEqual(short["answer_warm_count"], 1)
        self.assertAlmostEqual(
            short["calls"][0]["scenario_credits"], (100 * 2.5 + 10 * 12.5) / 1e6
        )
        self.assertAlmostEqual(
            long["calls"][1]["scenario_credits"],
            (20 * 2.5 + 80 * 0.25 + 10 * 12.5) / 1e6,
        )
        self.assertGreater(short["scenario_credits"], long["scenario_credits"])
        without = reprice(rows, exclude_recovery=True)["replicates"][1]
        self.assertAlmostEqual(
            long["scenario_credits"] - without["scenario_credits"],
            long["calls"][3]["scenario_credits"],
        )
        self.assertEqual(long["calls"][4]["warm"], without["calls"][4]["warm"])

    def test_missing_usage_and_bad_counts(self):
        r = row(1, "00")
        r["usage"] = None
        result = reprice([r])["replicates"][0]
        self.assertIsNone(result["scenario_credits"])
        self.assertIsNone(result["observed_credits"])
        r["usage"] = {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 1}
        with self.assertRaises(ContractError):
            reprice([r])

    def test_cold_ping_refresh_and_isolation(self):
        empty = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 1}
        rows = [row(1, "00", "ping", empty), row(2, "01"), row(3, "02", arm="b")]
        reps = reprice(rows)["replicates"]
        self.assertFalse(reps[0]["calls"][1]["warm"])
        self.assertFalse(reps[2]["calls"][0]["warm"])
        rows[0]["usage"]["input_tokens"] = 1
        self.assertTrue(reprice(rows)["replicates"][0]["calls"][1]["warm"])

    def test_api_cold_write_conservation_and_long_context(self):
        usage = {
            "input_tokens": 300000,
            "cached_input_tokens": 200000,
            "cache_write_input_tokens": 50000,
            "output_tokens": 100,
        }
        rows = [row(1, "00", usage=usage), row(2, "10", role="summary", usage=usage)]
        short, long = reprice(rows, pricing="openai_usd")["replicates"]
        self.assertEqual(short["cost_unit"], "USD")
        self.assertNotIn("scenario_credits", short)
        self.assertEqual(short["calls"][1]["modeled_cached_input_tokens"], 0)
        self.assertEqual(short["calls"][1]["modeled_cache_write_input_tokens"], 250000)
        self.assertEqual(long["calls"][1]["modeled_cached_input_tokens"], 200000)
        self.assertEqual(long["calls"][1]["modeled_cache_write_input_tokens"], 50000)
        self.assertAlmostEqual(
            short["calls"][1]["scenario_cost"],
            (50000 * 0.2 + 250000 * 0.25 + 100 * 0.75) / 1e6,
        )
        self.assertAlmostEqual(
            long["calls"][1]["scenario_cost"],
            (50000 * 0.2 + 200000 * 0.02 + 50000 * 0.25 + 100 * 0.75) / 1e6,
        )
        self.assertEqual(
            long["by_role"]["summary"]["scenario_cost"],
            long["calls"][1]["scenario_cost"],
        )

    def test_api_missing_writes_and_failed_call(self):
        usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10}
        self.assertIsNone(
            reprice([row(1, "00", usage=usage)], pricing="openai_usd")["replicates"][0][
                "scenario_cost"
            ]
        )
        usage["cache_write_input_tokens"] = 0
        self.assertIsNone(
            reprice([row(1, "00", usage=usage, ok=False)], pricing="openai_usd")[
                "replicates"
            ][0]["scenario_cost"]
        )


if __name__ == "__main__":
    unittest.main()


class PrefixTests(unittest.TestCase):
    def test_prefix_branches_expiry_and_rewrite(self):
        from compache.optimizer.cache_scenarios import reprice_prefix

        a, b, c, x = (ch * 64 for ch in "abcd")
        usage = {"input_tokens": 4096, "cached_input_tokens": 0, "output_tokens": 10}
        rows = [
            row(1, "00", usage=usage),
            row(2, "01", role="summary", usage=usage),
            row(3, "02", usage=usage),
            row(4, "07", usage=usage),
            row(5, "08", usage=usage),
        ]
        for r, blocks in zip(rows, ([a], [x], [a, b], [a, b, c], [x, c])):
            r["cache_blocks"] = blocks
        short, long = reprice_prefix(rows)["replicates"]
        self.assertEqual(short["calls"][2]["modeled_cached_input_tokens"], 3968)
        self.assertEqual(short["calls"][2]["observed_cached_input_tokens"], 0)
        self.assertEqual(short["calls"][3]["cache_status"], "expiry")
        self.assertEqual(long["calls"][3]["cache_status"], "warm")
        self.assertEqual(short["calls"][4]["cache_status"], "expiry")
        new = row(6, "09", usage=usage)
        new["cache_blocks"] = [c]
        self.assertEqual(
            reprice_prefix(rows + [new])["replicates"][0]["calls"][-1]["cache_status"],
            "rewrite",
        )
        self.assertEqual(short["calls"][0]["cache_status"], "first")

    def test_missing_and_failure(self):
        from compache.optimizer.cache_scenarios import reprice_prefix

        r = row(1, "00")
        r["cache_blocks"] = ["a" * 64]
        r["usage"] = None
        self.assertIsNone(reprice_prefix([r])["replicates"][0]["scenario_cost"])
        r["usage"] = {
            "input_tokens": 5000,
            "cached_input_tokens": 0,
            "output_tokens": 1,
        }
        r["ok"] = False
        self.assertIsNone(reprice_prefix([r])["replicates"][0]["scenario_cost"])
