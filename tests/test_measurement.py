import unittest
from compache.optimizer.measurement import aggregate, ARMS, ROLES


def fixture():
    counts = {
        (a, rep, role): int(role == "answer")
        for a in ARMS
        for rep in range(1, 4)
        for role in ROLES
    }
    rows = [
        {
            "arm": a,
            "replicate": rep,
            "role": "answer",
            "ok": True,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 10,
                "reasoning_output_tokens": 7,
            },
        }
        for a in ARMS
        for rep in range(1, 4)
    ]
    return rows, counts


class MeasurementTests(unittest.TestCase):
    def test_counters_and_price_no_double_count(self):
        rows, counts = fixture()
        result = aggregate(rows, model="gpt-6-luna", expected_counts=counts)
        total = result["aggregate"]["opencode"]["actual"]
        self.assertEqual(
            total["usage"],
            {"input_tokens": 300, "cached_input_tokens": 120, "output_tokens": 30},
        )
        self.assertAlmostEqual(
            total["standard_rate_credit_equivalent"],
            3 * (60 * 2.5e-6 + 40 * 0.25e-6 + 10 * 12.5e-6),
        )
        self.assertEqual(total["cache_buckets"], {"partial": 3})

    def test_missing_and_failed_not_zero(self):
        rows, counts = fixture()
        rows[0]["usage"] = None
        result = aggregate(rows, model="gpt-6-luna", expected_counts=counts)
        self.assertIsNone(
            result["aggregate"]["opencode"]["actual"]["standard_rate_credit_equivalent"]
        )
        self.assertIsNone(result["paired_savings"][0]["savings"])
        rows, counts = fixture()
        rows.pop()
        self.assertFalse(
            aggregate(rows, model="gpt-6-luna", expected_counts=counts)["aggregate"][
                "compache"
            ]["actual"]["complete"]
        )

    def test_recovery_exclusion_and_negative_savings(self):
        rows, counts = fixture()
        rows.append(
            {
                "arm": "compache",
                "replicate": 1,
                "role": "recovery",
                "ok": True,
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 10,
                },
            }
        )
        counts["compache", 1, "recovery"] = 1
        result = aggregate(rows, model="gpt-6-luna", expected_counts=counts)
        self.assertLess(result["aggregate_savings"], 0)
        self.assertGreater(
            result["aggregate"]["compache"]["actual"][
                "standard_rate_credit_equivalent"
            ],
            result["aggregate"]["compache"]["zero_recovery_direct_exclusion"][
                "standard_rate_credit_equivalent"
            ],
        )

    def test_unknown_model_and_counts_rejected(self):
        rows, counts = fixture()
        with self.assertRaises(ValueError):
            aggregate(rows, model="unknown", expected_counts=counts)
        counts.popitem()
        with self.assertRaises(ValueError):
            aggregate(rows, model="gpt-6-luna", expected_counts=counts)


if __name__ == "__main__":
    unittest.main()
