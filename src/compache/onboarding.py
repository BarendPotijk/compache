"""Local helpers for a host that elects to export count-only run metadata."""

from __future__ import annotations

import hmac
from hashlib import sha256

from .setup_metadata import validate_metadata_row
from .traces import ContractError


def pseudonymize_thread_id(identifier: str, *, key: bytes) -> str:
    """Create a stable join key. Keep the random secret in the host's custody."""
    if not isinstance(key, bytes) or len(key) < 16 or not isinstance(identifier, str) or not identifier:
        raise ContractError("supply a thread identity and at least 16 secret key bytes")
    return hmac.new(key, identifier.encode(), sha256).hexdigest()


def make_metadata_row(*, thread_id: str, route_id: str, model_id: str, run_index: int,
                      started_at: str, ended_at: str, added_tokens: int, output_tokens: int,
                      visible_output_tokens: int | None = None, calls_per_run: int | None = None,
                      retrieved_tokens: int | None = None) -> dict:
    """Build one allowlisted row from existing host counters. Performs no I/O."""
    row = dict(thread_id=thread_id, route_id=route_id, model_id=model_id, run_index=run_index,
               started_at=started_at, ended_at=ended_at, added_tokens=added_tokens,
               output_tokens=output_tokens)
    for name, value in (("visible_output_tokens", visible_output_tokens), ("calls_per_run", calls_per_run),
                        ("retrieved_tokens", retrieved_tokens)):
        if value is not None:
            row[name] = value
    validate_metadata_row(row)
    return row
