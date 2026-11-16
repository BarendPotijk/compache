"""Versioned, deterministic optimizer artifacts. Prices are caller assumptions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime

from compache.traces import ContractError


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

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ContractError("model_id must be nonempty")
        if (
            self.schema_version != "compache-model-spec-v1"
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

    def text_tokens(self, text: str | None) -> int:
        return math.ceil(len((text or "").encode()) / self.bytes_per_token)

    def part_tokens(self, text: str | None) -> int:
        return self.text_tokens(text) + self.part_overhead_tokens


@dataclass(frozen=True)
class Policy(Artifact):
    model_spec: dict
    profile_hash: str
    retrieval_grace_runs: int | None = None
    cold_trigger: float = 0.17
    target: float = 0.15
    warm_trigger: float = 0.65
    reserve: float = 0.05
    ping_lead_seconds: float = 45
    freshness_margin_seconds: float = 20
    max_pings: int = 0
    protect_query_matches: bool = True
    schema_version: str = "compache-policy-v1"

    def __post_init__(self):
        ModelSpec.from_dict(self.model_spec)
        if self.schema_version != "compache-policy-v1" or not isinstance(
            self.profile_hash, str
        ):
            raise ContractError("invalid policy version/profile hash")
        if self.retrieval_grace_runs is not None:
            number(self.retrieval_grace_runs, "retrieval_grace_runs", integer=True)
        for name in ("cold_trigger", "target", "warm_trigger", "reserve"):
            number(getattr(self, name), name, 0, 1)
        if not self.target < self.cold_trigger < self.warm_trigger <= 1 - self.reserve:
            raise ContractError(
                "require target < cold_trigger < warm_trigger <= 1-reserve"
            )
        number(self.max_pings, "max_pings", 0, 100, integer=True)
        number(self.ping_lead_seconds, "ping_lead_seconds", 0.001)
        number(self.freshness_margin_seconds, "freshness_margin_seconds")
        if self.freshness_margin_seconds >= self.ping_lead_seconds:
            raise ContractError("freshness margin must be smaller than ping lead")
        if type(self.protect_query_matches) is not bool:
            raise ContractError("protect_query_matches must be boolean")

    @property
    def spec(self):
        return ModelSpec.from_dict(self.model_spec)

    @property
    def hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class Constraints(Artifact):
    max_miss_rate: float = 0.0
    max_unknown_change_rate: float = 0.0
    schema_version: str = "compache-constraints-v1"

    def __post_init__(self):
        if self.schema_version != "compache-constraints-v1":
            raise ContractError("invalid constraints version")
        number(self.max_miss_rate, "max_miss_rate", 0, 1)
        number(self.max_unknown_change_rate, "max_unknown_change_rate", 0, 1)
