from dataclasses import replace
import unittest

from compache.optimizer.cache_breaks import _price_ping_receipt
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.distribution import repeat_ping_charge, request_charge
from compache.traces import ContractError


class RequestTariffTests(unittest.TestCase):
    def setUp(self):
        self.flat = ModelSpec("synthetic", cache_min_tokens=1, output_multiplier=5)
        self.tier = replace(self.flat, schema_version="compache-model-spec-v2",
                            long_input_threshold=272_000, long_input_factor=2, long_output_factor=1.5)

    def test_flat_and_tiered_artifact_roundtrips(self):
        legacy = self.flat.to_dict()
        self.assertNotIn("long_input_threshold", legacy)
        self.assertEqual(ModelSpec.from_dict(legacy).to_dict(), legacy)
        self.assertEqual(ModelSpec.from_dict(self.tier.to_dict()), self.tier)
        with self.assertRaises(ContractError):
            ModelSpec.from_dict(dict(legacy, long_input_threshold=100))
        with self.assertRaises(ContractError):
            replace(self.tier, long_input_threshold=0)

    def test_threshold_for_foreground_and_ping_prices(self):
        for size, input_factor, output_rate in [(272_000, 1, 5), (272_001, 2, 7.5)]:
            with self.subTest(size=size):
                reads, writes, output = size - 11, 8, 5
                expected = (3 + 1.25 * writes + .1 * reads) * input_factor + output * output_rate
                self.assertAlmostEqual(_price_ping_receipt(self.tier, size, reads, writes, output)[0], expected)
                same_mix = (3 + writes * 1.25) / 11
                self.assertAlmostEqual(repeat_ping_charge(self.tier, size, reads, output,
                                                          uncached_input_rate=same_mix), expected)
                self.assertAlmostEqual(request_charge(self.tier, size, reads, output),
                                       (11 * 1.25 + reads * .1) * input_factor + output * output_rate)


if __name__ == "__main__":
    unittest.main()
