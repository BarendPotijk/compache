"""Versioned, deterministic optimizer artifacts. Prices are caller assumptions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime

from compache.errors import ContractError


DEFAULT_MIN_RETENTION = 0.8


def require_held_out(threads, training_ids):
    """Reject a replay cohort containing any conversation used to fit the policy."""
    if not threads:
        raise ContractError("held-out evaluation requires at least one thread")
    seen = set(training_ids)
    if any(digest(t.thread_id) in seen for t in threads):
        raise ContractError("held-out evaluation includes a training thread")


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class FrozenDict(dict):
    """Immutable JSON object with shareable nested arrays for repeated gate calls."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("immutable fitted artifact")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = _refuse

    def __deepcopy__(self, memo):
        return self


def freeze_json(value):
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict((key, freeze_json(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    return value


def seconds(value: str) -> float:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError()
        return dt.timestamp()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError("expected a timezone-aware ISO timestamp") from exc


def number(value, name, low=0, high=math.inf, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ContractError(
            f"{name}: expected {'integer' if integer else 'finite number'} in [{low}, {high}]"
        )


class Artifact:
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ContractError(
                f"{cls.__name__}: fields must match the versioned contract"
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise ContractError(f"invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class ModelSpec(Artifact):
    model_id: str
    context_tokens: int = 4096
    cache_ttl_seconds: float = 300
    cache_min_tokens: int = 0
    cache_block_tokens: int = 1
    read_multiplier: float = 0.1
    write_multiplier: float = 1.25
    output_multiplier: float = 4.0
    ping_output_tokens: int = 0
    expansion_output_tokens: int = 16
    bytes_per_token: float = 4.0
    part_overhead_tokens: int = 6
    byte_limit: int = 16_000_000
    assumption_note: str = "Caller-supplied illustrative economics; not verified provider prices or billing."
    schema_version: str = "compache-model-spec-v1"
    long_input_threshold: int | None = None
    long_input_factor: float = 1.0
    long_output_factor: float = 1.0

    def to_dict(self):
        result = super().to_dict()
        if self.schema_version == "compache-model-spec-v1":
            for key in (
                "long_input_threshold",
                "long_input_factor",
                "long_output_factor",
            ):
                del result[key]
        return result

    @classmethod
    def from_dict(cls, data):
        if (
            isinstance(data, dict)
            and data.get("schema_version") == "compache-model-spec-v1"
        ):
            tier = dict(
                long_input_threshold=None, long_input_factor=1.0, long_output_factor=1.0
            )
            if tier.keys() & data.keys():
                raise ContractError("tiered rates require compache-model-spec-v2")
            data = {**data, **tier}
        return super().from_dict(data)

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ContractError("model_id must be nonempty")
        if (
            self.schema_version
            not in ("compache-model-spec-v1", "compache-model-spec-v2")
            or not isinstance(self.assumption_note, str)
            or not self.assumption_note.strip()
        ):
            raise ContractError("invalid model spec version or missing assumptions")
        for name in ("context_tokens", "cache_block_tokens", "byte_limit"):
            number(getattr(self, name), name, 1, integer=True)
        for name in (
            "cache_min_tokens",
            "ping_output_tokens",
            "expansion_output_tokens",
            "part_overhead_tokens",
        ):
            number(getattr(self, name), name, integer=True)
        number(self.bytes_per_token, "bytes_per_token", 0.1, 16)
        number(self.cache_ttl_seconds, "cache_ttl_seconds")
        number(self.read_multiplier, "read_multiplier", 0, 1)
        number(self.write_multiplier, "write_multiplier", 1)
        number(self.output_multiplier, "output_multiplier")
        if self.schema_version == "compache-model-spec-v2":
            number(self.long_input_threshold, "long_input_threshold", 1, integer=True)
            number(self.long_input_factor, "long_input_factor", 1)
            number(self.long_output_factor, "long_output_factor", 1)
        elif (
            self.long_input_threshold is not None
            or self.long_input_factor != 1
            or self.long_output_factor != 1
        ):
            raise ContractError("tiered rates require compache-model-spec-v2")

    def input_price_factor(self, size):
        return (
            self.long_input_factor
            if self.long_input_threshold is not None
            and size > self.long_input_threshold
            else 1.0
        )

    def output_rate(self, size):
        if self.long_input_threshold is not None and size > self.long_input_threshold:
            return self.output_multiplier * self.long_output_factor
        return self.output_multiplier

    def token_charge(self, fresh, writes, reads, output):
        size = fresh + writes + reads
        input_cost = (
            fresh + writes * self.write_multiplier + reads * self.read_multiplier
        )
        return input_cost * self.input_price_factor(size) + output * self.output_rate(
            size
        )

    def text_tokens(self, text: str | None) -> int:
        return math.ceil(len((text or "").encode()) / self.bytes_per_token)

    def part_tokens(self, text: str | None) -> int:
        return self.text_tokens(text) + self.part_overhead_tokens
