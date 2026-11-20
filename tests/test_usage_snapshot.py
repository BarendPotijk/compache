import unittest
from types import SimpleNamespace

from compache.optimizer.usage import response_usage


class ResponseMeterTests(unittest.TestCase):
    def test_sdk_final_event_and_unknown_optional_fields(self):
        response = SimpleNamespace(
            id="r",
            model="test",
            status="completed",
            usage=SimpleNamespace(
                input_tokens=1031,
                output_tokens=12,
                input_tokens_details=SimpleNamespace(cached_tokens=256),
            ),
        )
        meter = response_usage(
            SimpleNamespace(type="response.completed", response=response)
        )
        self.assertEqual(meter["response_id"], "r")
        self.assertEqual(meter["input_tokens"], 1031)
        self.assertEqual(meter["cached_input_tokens"], 256)
        self.assertIsNone(meter["cache_write_input_tokens"])
        self.assertIsNone(meter["reasoning_output_tokens"])
        self.assertEqual(response_usage(response), meter)

    def test_zero_is_observed_missing_is_unknown(self):
        meter = response_usage(
            {
                "usage": {
                    "input_tokens": 32,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": True,
                }
            }
        )
        self.assertEqual(meter["cached_input_tokens"], 0)
        self.assertEqual(meter["cache_write_input_tokens"], 0)
        self.assertIsNone(meter["output_tokens"])
        self.assertIsNone(response_usage(None)["input_tokens"])
