"""Pinned Headroom library adapter. No provider requests or learned-policy mutation."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from compache.traces import ContractError

HEADROOM_VERSION = "0.44.0"
HEADROOM_SOURCE_REVIEW = "18067b1d19f6606bae3b4d6b50efeeaa48dc477e"


class HeadroomLossless:
    """Execute Headroom's verified lossless agent mode, keeping prior bytes stable.

    This is a named library arm, not a reproduction of Headroom's full proxy,
    default CCR/ML mode, cold-prefix hook, or opt-in net-cost policy.
    """

    def __init__(self, tokenizer_model="gpt-4o"):
        if os.environ.get("HEADROOM_OFFLINE") != "1":
            raise ContractError("Set HEADROOM_OFFLINE=1 before running the Headroom comparison")
        try:
            installed = version("headroom-ai")
        except PackageNotFoundError as exc:
            raise ContractError(
                f"Optional baseline missing: install headroom-ai=={HEADROOM_VERSION} "
                "in a separate benchmark environment; see docs/comparisons.md"
            ) from exc
        if installed != HEADROOM_VERSION:
            raise ContractError(f"Headroom must be pinned to {HEADROOM_VERSION}; found {installed}")
        from headroom import compress

        self._compress = compress
        self.tokenizer_model = tokenizer_model
        source = inspect.getsourcefile(compress)
        if source is None:
            raise ContractError("Cannot fingerprint installed Headroom implementation")
        self.provenance = {
            "package": "headroom-ai", "version": installed,
            "reviewed_upstream_commit": HEADROOM_SOURCE_REVIEW,
            "installed_compress_sha256": hashlib.sha256(
                Path(source).read_bytes()
            ).hexdigest(),
            "tokenizer_model": tokenizer_model,
            "configuration": {"mode": "agent", "verify_lossless": True,
                              "kompress_model": "disabled", "compress_user_messages": False,
                              "compress_system_messages": False},
            "scope": "Native lossless library compression; caller-managed frozen prefix. Not the full proxy.",
        }

    def compress(self, messages, *, frozen, context_tokens):
        original = copy.deepcopy(messages)
        started = time.perf_counter()
        result = self._compress(
            copy.deepcopy(messages), model=self.tokenizer_model, model_limit=context_tokens,
            mode="agent", verify_lossless=True, kompress_model="disabled",
            compress_user_messages=False, compress_system_messages=False,
            frozen_message_count=frozen,
        )
        elapsed = time.perf_counter() - started
        if result.lossless is not True:
            raise ContractError("Headroom did not verify lossless output; comparison stopped")
        output = result.messages
        if len(output) != len(original):
            raise ContractError("Headroom changed message count; source alignment is unsupported")
        for before, after in zip(original, output):
            if not isinstance(after.get("content"), str):
                raise ContractError("Headroom returned non-text content; projection is unsupported")
            if {k: v for k, v in before.items() if k != "content"} != {
                k: v for k, v in after.items() if k != "content"
            }:
                raise ContractError("Headroom changed message identity/role; projection is unsupported")
        if output[:frozen] != original[:frozen]:
            raise ContractError("Headroom rewrote the frozen prefix; comparison stopped")
        return output, {
            "compression_seconds_measured": elapsed,
            "native_tokens_before": result.tokens_before,
            "native_tokens_after": result.tokens_after,
            "transforms": list(result.transforms_applied),
            "lossless_verified_by_headroom": True,
        }
