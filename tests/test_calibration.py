import copy
import json
import unittest

from compache.optimizer.calibration import OnlineTokenCalibrator
from compache.traces import ContractError


class CalibrationTests(unittest.TestCase):
    def test_online_estimates_are_model_specific_and_causal(self):
        ruler = OnlineTokenCalibrator(100, safety_margin=0.1)
        self.assertEqual(ruler.estimate_total("a", 1000), 1200)
        self.assertTrue(
            ruler.observe(
                "a",
                1000,
                {"input_tokens": 1300, "cached_input_tokens": 200},
                successful=True,
                provider_observed=True,
            )
        )
        self.assertEqual(ruler.estimate_total("a", 1000), 1420)
        self.assertEqual(ruler.estimate_total("b", 1000), 1200)
        ruler.observe(
            "a", 1000, {"input_tokens": 900}, successful=True, provider_observed=True
        )
        self.assertEqual(ruler.estimate_total("a", 1000), 1420)
        self.assertEqual(ruler.estimate_total("a", 500), 760)
        self.assertEqual(ruler.estimate_extra_overhead("a", 500), 260)

    def test_fixed_allowance_and_zero_margin_cover_observed_point(self):
        ruler = OnlineTokenCalibrator(16000, safety_margin=0)
        ruler.observe(
            "model",
            742402,
            {"input_tokens": 798187},
            successful=True,
            provider_observed=True,
        )
        self.assertEqual(ruler.estimate_total("model", 742402), 798187)
        self.assertEqual(ruler.estimate_total("model", 0), 16000)

    def test_resume_round_trip_and_later_observation_match_continuous_execution(self):
        live = OnlineTokenCalibrator(100)
        live.observe(
            "a", 10, {"input_tokens": 150}, successful=True, provider_observed=True
        )
        restored = OnlineTokenCalibrator.from_dict(
            json.loads(json.dumps(live.to_dict()))
        )
        for ruler in (live, restored):
            ruler.observe(
                "a", 100, {"input_tokens": 700}, successful=True, provider_observed=True
            )
            ruler.observe(
                "b", 200, {"input_tokens": 310}, successful=True, provider_observed=True
            )
        self.assertEqual(live.to_dict(), restored.to_dict())
        self.assertEqual(
            live.estimate_total("a", 500), restored.estimate_total("a", 500)
        )

    def test_unavailable_failed_or_synthetic_usage_never_updates(self):
        ruler = OnlineTokenCalibrator(0)
        for kwargs, usage in [
            ({"successful": False, "provider_observed": True}, {"input_tokens": 999}),
            ({"successful": True, "provider_observed": False}, {"input_tokens": 999}),
            ({"successful": True, "provider_observed": True}, None),
        ]:
            self.assertFalse(ruler.observe("a", 10, usage, **kwargs))
        self.assertEqual(ruler.to_dict()["models"], {})

    def test_invalid_observations_rejected(self):
        for payload, usage in [
            (True, {"input_tokens": 1}),
            (0, {"input_tokens": 1}),
            (1, {"input_tokens": True}),
            (1, {"input_tokens": -1}),
            (1, {"input_tokens": 0}),
            (1, {}),
            (1, {"input_tokens": 5, "cached_input_tokens": 6}),
            (1, {"input_tokens": 5, "cached_input_tokens": -1}),
            (1, {"input_tokens": 5, "cached_input_tokens": None}),
            (1, {"input_tokens": 5, "output_tokens": -1}),
        ]:
            with (
                self.subTest(payload=payload, usage=usage),
                self.assertRaises(ContractError),
            ):
                OnlineTokenCalibrator(0).observe(
                    "m", payload, usage, successful=True, provider_observed=True
                )

    def test_invalid_persisted_state_rejected(self):
        ruler = OnlineTokenCalibrator(0)
        ruler.observe(
            "a", 2, {"input_tokens": 3}, successful=True, provider_observed=True
        )
        state = ruler.to_dict()
        for key, value in [
            ("ratio_denominator", 0),
            ("ratio_numerator", 0),
            ("observations", 0),
        ]:
            bad = copy.deepcopy(state)
            bad["models"]["a"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                OnlineTokenCalibrator.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
