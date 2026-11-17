"""Causal, model-specific planning estimates from reported provider input.

The caller chooses the fixed allowance for request framing and runtime overhead.
The remaining payload ruler is calibrated against earlier successful provider
calls only. Neither the allowance nor the margin is a provider token counter;
novel text, framing changes, and unseen tokenizer behavior can exceed the bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

from compache.traces import ContractError


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")


def _model(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractError("model_id must be nonempty")


class OnlineTokenCalibrator:
    """Conservative empirical ruler, scoped by model and fixed overhead.

    For each model retain the largest observed (input - fixed) / payload ratio,
    bounded below by one. Apply an explicitly assumed fractional safety margin
    to the payload, even before the first observation. The fixed allowance must
    cover runtime framing and rendered wrappers not included in payload. This
    is an empirical planning estimate, not a guarantee or native tokenization.
    Rebuild with observations in call order or save and restore ``to_dict``.
    """

    def __init__(self, fixed_overhead: int, safety_margin: float = 0.10):
        _nonnegative_integer(fixed_overhead, "fixed_overhead")
        if (
            type(safety_margin) not in (int, float)
            or not math.isfinite(safety_margin)
            or safety_margin < 0
            or safety_margin > 10
        ):
            raise ContractError("safety_margin must be finite and between 0 and 10")
        self.fixed_overhead = fixed_overhead
        self.safety_margin = Fraction(str(safety_margin))
        self._models: dict[str, tuple[Fraction, int]] = {}

    def estimate_total(self, model_id: str, payload_estimate: int) -> int:
        _model(model_id)
        _nonnegative_integer(payload_estimate, "payload_estimate")
        ratio, _ = self._models.get(model_id, (Fraction(1), 0))
        return self.fixed_overhead + math.ceil(
            payload_estimate * ratio * (1 + self.safety_margin)
        )

    def estimate_extra_overhead(self, model_id: str, payload_estimate: int) -> int:
        """Return the additive amount to add to this *current* projected payload."""
        return self.estimate_total(model_id, payload_estimate) - payload_estimate

    def observe(
        self,
        model_id: str,
        payload_estimate: int,
        usage: Mapping | None,
        *,
        successful: bool,
        provider_observed: bool,
    ) -> bool:
        """Update after a completed call, never before its admission decision.

        ``provider_observed`` must be true only for actual provider counters,
        not simulated usage or a local estimate. Missing usage or unsuccessful
        calls are skipped. Malformed supplied counters raise ContractError.
        """
        _model(model_id)
        _nonnegative_integer(payload_estimate, "payload_estimate")
        if type(successful) is not bool or type(provider_observed) is not bool:
            raise ContractError("successful and provider_observed must be booleans")
        if not successful or not provider_observed or usage is None:
            return False
        if not isinstance(usage, Mapping):
            raise ContractError("usage must be a mapping")
        actual = usage.get("input_tokens")
        _nonnegative_integer(actual, "input_tokens")
        if payload_estimate == 0 or actual == 0:
            raise ContractError(
                "calibration requires positive payload and input tokens"
            )
        if "cached_input_tokens" in usage:
            cached = usage["cached_input_tokens"]
            _nonnegative_integer(cached, "cached_input_tokens")
            if cached > actual:
                raise ContractError("cached input exceeds total input")
        if "output_tokens" in usage:
            _nonnegative_integer(usage["output_tokens"], "output_tokens")
        ratio, count = self._models.get(model_id, (Fraction(1), 0))
        observed = Fraction(max(0, actual - self.fixed_overhead), payload_estimate)
        self._models[model_id] = (max(ratio, observed), count + 1)
        return True

    def to_dict(self) -> dict:
        return {
            "schema_version": "compache-token-calibration-v1",
            "fixed_overhead": self.fixed_overhead,
            "safety_margin": {
                "numerator": self.safety_margin.numerator,
                "denominator": self.safety_margin.denominator,
            },
            "models": {
                model: {
                    "ratio_numerator": ratio.numerator,
                    "ratio_denominator": ratio.denominator,
                    "observations": count,
                }
                for model, (ratio, count) in sorted(self._models.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> OnlineTokenCalibrator:
        if (
            not isinstance(data, dict)
            or set(data)
            != {"schema_version", "fixed_overhead", "safety_margin", "models"}
            or data["schema_version"] != "compache-token-calibration-v1"
        ):
            raise ContractError("invalid token calibration state")
        margin = data["safety_margin"]
        if not isinstance(margin, dict) or set(margin) != {"numerator", "denominator"}:
            raise ContractError("invalid safety margin")
        _nonnegative_integer(margin["numerator"], "margin numerator")
        _nonnegative_integer(margin["denominator"], "margin denominator")
        if (
            not margin["denominator"]
            or margin["numerator"] > 10 * margin["denominator"]
        ):
            raise ContractError("invalid safety margin")
        result = cls(data["fixed_overhead"], 0)
        result.safety_margin = Fraction(margin["numerator"], margin["denominator"])
        if not isinstance(data["models"], dict):
            raise ContractError("invalid calibration models")
        for model, row in data["models"].items():
            _model(model)
            if not isinstance(row, dict) or set(row) != {
                "ratio_numerator",
                "ratio_denominator",
                "observations",
            }:
                raise ContractError("invalid model calibration")
            for key in row:
                _nonnegative_integer(row[key], key)
            if (
                not row["ratio_denominator"]
                or not row["observations"]
                or row["ratio_numerator"] < row["ratio_denominator"]
            ):
                raise ContractError("invalid model calibration")
            result._models[model] = (
                Fraction(row["ratio_numerator"], row["ratio_denominator"]),
                row["observations"],
            )
        return result
