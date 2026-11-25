"""Price sheets and prompt-cache rules of known models.

`models.json` holds list prices and each provider's default cache rules. A
policy is fitted for one such sheet. Anything not listed there can be given as
a dict or JSON file of `ModelSpec` fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .errors import ContractError
from .optimizer.contracts import ModelSpec

__all__ = [
    "binds_reasoning",
    "cache_lifetimes",
    "known_models",
    "model_spec",
    "pricing",
]

# A keep-warm ping and a fetch-back request each ask the model for a few tokens.
PING_OUTPUT_TOKENS = 16
EXPANSION_OUTPUT_TOKENS = 32


@lru_cache(maxsize=1)
def _sheet() -> dict:
    text = resources.files(__package__).joinpath("models.json").read_text("utf-8")
    return json.loads(text)


def known_models() -> tuple[str, ...]:
    """IDs of the models with a built-in price sheet."""
    return tuple(_sheet()["models"])


def _lookup(name: str):
    models = _sheet()["models"]
    if name in models:
        return name, models[name]
    for model_id, entry in models.items():
        if entry.get("preset") == name:
            return model_id, entry
    # A dated or routed name of a known model, such as "anthropic/claude-sonnet-5-5-20261001",
    # or one that a gateway writes with a dot, such as "anthropic/claude-haiku-5.5".
    tail = name.rsplit("/", 1)[-1]
    for spelled in (tail, tail.replace(".", "-")):
        matches = [
            m
            for m in models
            if spelled.startswith(m) or spelled.startswith(m.replace(".", "-"))
        ]
        if matches:
            model_id = max(matches, key=len)
            return model_id, models[model_id]
    return None


def _from_sheet(model_id: str, entry: dict) -> ModelSpec:
    tiered = entry.get("long_input_threshold") is not None
    fields = dict(
        model_id=model_id,
        context_tokens=entry["max_input_tokens"],
        cache_ttl_seconds=entry["cache_ttl_seconds"],
        cache_min_tokens=entry["cache_min_tokens"],
        cache_block_tokens=entry["cache_block_tokens"],
        read_multiplier=entry["cache_read_price"] / entry["input_price"],
        write_multiplier=entry["cache_write_multiplier"],
        output_multiplier=entry["output_price"] / entry["input_price"],
        ping_output_tokens=PING_OUTPUT_TOKENS,
        expansion_output_tokens=EXPANSION_OUTPUT_TOKENS,
        assumption_note=f"List prices checked {_sheet()['checked']}: {entry['source']}",
    )
    if tiered:
        fields.update(
            schema_version="compache-model-spec-v2",
            long_input_threshold=entry["long_input_threshold"],
            long_input_factor=entry["long_input_factor"],
            long_output_factor=entry["long_output_factor"],
        )
    return ModelSpec(**fields)


def _from_fields(fields) -> ModelSpec:
    """A spec from some or all of its fields. The rest take `ModelSpec`'s defaults."""
    if not isinstance(fields, dict):
        raise ContractError("a model spec is a JSON object of its fields")
    fields = dict(fields)
    if fields.get("long_input_threshold") is not None:
        fields.setdefault("schema_version", "compache-model-spec-v2")
    try:
        return ModelSpec(**fields)
    except TypeError as exc:
        raise ContractError(f"invalid model spec: {exc}") from exc


def model_spec(model) -> ModelSpec:
    """A model spec from a known model name, a spec, a dict of its fields, or a JSON file of them."""
    if isinstance(model, ModelSpec):
        return model
    if isinstance(model, dict):
        return _from_fields(model)
    if isinstance(model, Path) or (isinstance(model, str) and model.endswith(".json")):
        return _from_fields(json.loads(Path(model).read_text("utf-8")))
    if isinstance(model, str) and model.strip():
        found = _lookup(model.strip())
        if found is not None:
            return _from_sheet(*found)
        raise ContractError(
            f"no price sheet for model {model!r}. Known: {', '.join(known_models())}. "
            "Pass a JSON file of model spec fields for any other model."
        )
    raise ContractError("model: expected a model name, a spec, or a path to one")


def cache_lifetimes(model) -> tuple[ModelSpec, ...]:
    """A known model under each other cache lifetime its provider can be asked for.

    Empty for a model given as a spec: its lifetime is the caller's to state.
    """
    if not isinstance(model, str) or model.endswith(".json"):
        return ()
    found = _lookup(model.strip())
    if found is None:
        return ()
    spec = _from_sheet(*found)
    return tuple(
        replace(
            spec,
            cache_ttl_seconds=lifetime["seconds"],
            write_multiplier=lifetime["write_multiplier"],
        )
        for lifetime in found[1].get("cache_lifetimes", ())
    )


_CLAUDE = re.compile(r"claude|opus|sonnet|haiku|fable|mythos")
_CLAUDE_VERSION = re.compile(
    r"(?:opus|sonnet|haiku|fable|mythos)[-.]?(\d{1,2})(?:[-.](\d{1,2}))?(?!\d)"
)
_OLD_CLAUDE = re.compile(r"claude-(?:instant|v?[0-4])")


def binds_reasoning(model) -> bool:
    """Whether a model's reasoning is only good behind what stood ahead of it when it was made.

    So it is with Claude from 5.1 on: where an earlier message, the instructions or
    the tools have changed since, the provider drops the reasoning or refuses the
    request. https://platform.claude.com/docs/en/build-with-claude/preserved-thinking
    """
    name = str(model or "").lower()
    if not _CLAUDE.search(name):
        return False
    version = _CLAUDE_VERSION.search(name)
    if version is not None:
        return (int(version[1]), int(version[2] or 0)) >= (5, 1)
    return not _OLD_CLAUDE.search(name)


def pricing(model) -> dict | None:
    """List price behind a known model name, for reports in dollars. None for a custom spec."""
    if not isinstance(model, str) or model.endswith(".json"):
        return None
    found = _lookup(model.strip())
    if found is None:
        return None
    model_id, entry = found
    return {
        "model_id": model_id,
        "name": entry["name"],
        "preset": entry.get("preset"),
        "currency": "USD",
        "input_price_per_million": entry["input_price"],
        "cache_read_price_per_million": entry["cache_read_price"],
        "output_price_per_million": entry["output_price"],
        "context_tokens": entry["context_tokens"],
        "max_input_tokens": entry["max_input_tokens"],
        "checked": _sheet()["checked"],
    }
