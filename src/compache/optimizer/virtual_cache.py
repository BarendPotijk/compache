"""Replay native Responses requests on a deterministic logical cache clock.

Use one VirtualCache per TTL and trajectory. Physical counters are evidence for
eligible capacity; the replay only changes how that capacity is split between
reads and writes. A forced warm lookup can reuse a previously written, matching
endpoint even after expiry. A forced cold lookup evicts its matching endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from compache.traces import ContractError

from .contracts import number
from .shared_prefix import _wire

DEFAULT_TTLS = (300, 1800)
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
)
# Pin the benchmark's rendering and cache namespace. Prewarm and output length
# may vary without changing the native input prefix.
_PREFIX_SETTINGS = (
    "model",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "instructions",
    "reasoning",
    "text",
    "context_management",
    "prompt_cache_key",
)


def _kind(item):
    tag = item.get("type", "message")
    if tag == "message":
        return item.get("role")
    if isinstance(tag, str) and (
        tag.endswith("_call_output") or tag == "tool_search_output"
    ):
        return "tool"
    return None


@dataclass(frozen=True)
class CacheRequest:
    """Content-free namespace and ordered (item count, prefix identity) pairs.

    A host with verified boundary identities can construct this record directly.
    Each identity must attest to the entire prefix, including message order.
    Supply only eligible lookups, ending with the one publishable endpoint.
    """

    partition: str
    boundaries: tuple[tuple[int, str], ...]

    def __post_init__(self):
        if (
            not isinstance(self.partition, str)
            or not self.partition
            or not isinstance(self.boundaries, tuple)
        ):
            raise ContractError(
                "provide a pinned partition and tuple of eligible boundaries"
            )
        previous = 0
        for boundary in self.boundaries:
            if not isinstance(boundary, tuple) or len(boundary) != 2:
                raise ContractError(
                    "each boundary requires (item count, prefix identity)"
                )
            end, key = boundary
            if (
                type(end) is not int
                or end <= previous
                or not isinstance(key, str)
                or not key
            ):
                raise ContractError(
                    "eligible boundaries require increasing item counts and nonempty identities"
                )
            previous = end

    @classmethod
    def from_payload(cls, payload, *, scope: str):
        """Fingerprint a fully materialized, implicit-only native Responses input.

        In implicit mode lookup considers the latest eligible ending, up to 20
        earlier eligible endings, and the endpoint of the initial developer group.
        Only the final eligible endpoint is eligible for a new write. Explicit
        block markers require per-breakpoint capacities unavailable in one receipt.
        """
        if not isinstance(scope, str) or not scope or not isinstance(payload, Mapping):
            raise ContractError("provide a cache scope and a native Responses payload")
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ContractError("request model is required")
        if payload.get("previous_response_id") or payload.get("conversation"):
            raise ContractError("supply the fully materialized request history")
        options = payload.get("prompt_cache_options", {})
        if (
            not isinstance(options, Mapping)
            or options.get("mode", "implicit") != "implicit"
        ):
            raise ContractError("virtual cache replay requires implicit caching")
        items = payload.get("input")
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        if not isinstance(items, list) or not all(
            isinstance(item, dict) for item in items
        ):
            raise ContractError(
                "request input must be text or an ordered array of native items"
            )
        settings = [[key, payload[key]] for key in _PREFIX_SETTINGS if key in payload]
        partition = sha256(_wire([scope, settings])).hexdigest()
        prefix = sha256()
        kinds = [_kind(item) for item in items]
        eligible = []
        initial_developer = None
        initial_group = True
        for i, (kind, item) in enumerate(zip(kinds, items)):
            content = item.get("content")
            if "prompt_cache_breakpoint" in item or (
                isinstance(content, list)
                and any(
                    isinstance(block, dict) and "prompt_cache_breakpoint" in block
                    for block in content
                )
            ):
                raise ContractError(
                    "explicit cache breakpoints require separate capacity evidence"
                )
            prefix.update(_wire(item))
            prefix.update(b"\n")
            next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
            initial_group = initial_group and kind == "developer"
            developer_end = initial_group and next_kind != "developer"
            if (
                developer_end
                or kind == "user"
                or (kind == "tool" and next_kind != "tool")
            ):
                boundary = (i + 1, prefix.hexdigest())
                eligible.append(boundary)
                if developer_end:
                    initial_developer = boundary
        boundaries = eligible[-21:]
        if initial_developer is not None and initial_developer not in boundaries:
            boundaries.insert(0, initial_developer)
        return cls(partition, tuple(boundaries))

    def published_endpoint_reachable_in(self, later: CacheRequest) -> bool | None:
        """Can the later lookup address this request's final publication?

        This describes the request layout. The receipt and elapsed time still
        determine whether any tokens were reusable. Different partitions can
        partially overlap at the provider, so their coverage is unknown here.
        """
        if not isinstance(later, CacheRequest):
            raise ContractError("provide a later CacheRequest")
        if self.partition != later.partition:
            return None
        return bool(self.boundaries) and self.boundaries[-1] in later.boundaries


class VirtualCache:
    """One serial logical timeline, with independent entries per pinned namespace."""

    def __init__(self, ttl_seconds=DEFAULT_TTLS[0]):
        number(ttl_seconds, "ttl_seconds", 1)
        self.ttl_seconds = ttl_seconds
        self._partitions = {}
        self._last_at = None

    def _lookup(self, request, at, state):
        if not isinstance(request, CacheRequest) or state not in (
            "auto",
            "warm",
            "cold",
        ):
            raise ContractError("provide a CacheRequest and auto, warm or cold state")
        number(at, "at", -1e20)
        if self._last_at is not None and at < self._last_at:
            raise ContractError("logical request times must be nondecreasing")
        entries = self._partitions.get(request.partition, {})
        matches = [
            (end, entries[key]) for end, key in request.boundaries if key in entries
        ]
        if state == "cold":
            return 0, None, "forced_cold"
        for end, (capacity, refreshed_at) in reversed(matches):
            if state == "warm" or at - refreshed_at < self.ttl_seconds:
                return capacity, end, "warm"
        return 0, None, "expired" if matches else "rewrite" if entries else "first"

    def available(self, request, *, at, state="auto"):
        """Available previously written prefix before bounding by this call's receipt."""
        capacity, _, _ = self._lookup(request, at, state)
        return capacity

    def apply(self, request, *, at, usage, state="auto"):
        """Replay a metered call, including a generation stopped at its output cap."""
        if not isinstance(usage, Mapping) or any(
            key not in usage for key in _USAGE_FIELDS
        ):
            raise ContractError(
                "a completed receipt with input, read, write and output is required"
            )
        for key in _USAGE_FIELDS:
            number(usage[key], key, integer=True)
        capped = (
            usage.get("status") == "incomplete"
            and usage.get("incomplete_reason") == "max_output_tokens"
        )
        if usage.get("status") not in (None, "completed") and not capped:
            raise ContractError(
                "cache replay requires a completed or output-capped response"
            )
        if usage.get("mixed_cache_write_ttls"):
            raise ContractError(
                "mixed cache lifetimes cannot be attributed to one implicit write"
            )
        return self._apply(
            request,
            at,
            state,
            usage["input_tokens"],
            usage["cached_input_tokens"] + usage["cache_write_input_tokens"],
            usage["output_tokens"],
            {key: usage[key] for key in _USAGE_FIELDS},
        )

    def project(
        self,
        request,
        *,
        at,
        input_tokens,
        cacheable_input_tokens,
        output_tokens,
        state="auto",
    ):
        """Commit a modeled call with supplied token assumptions and no observed usage."""
        for name, value in (
            ("input_tokens", input_tokens),
            ("cacheable_input_tokens", cacheable_input_tokens),
            ("output_tokens", output_tokens),
        ):
            number(value, name, integer=True)
        return self._apply(
            request,
            at,
            state,
            input_tokens,
            cacheable_input_tokens,
            output_tokens,
            None,
        )

    project_ping = project

    def _apply(self, request, at, state, total, capacity, output, observed):
        if capacity > total:
            raise ContractError("cacheable input exceeds total input tokens")
        previous, matched_end, status = self._lookup(request, at, state)
        if capacity and not request.boundaries:
            raise ContractError(
                "positive cache capacity with no eligible native input endpoint"
            )
        reads = min(previous, capacity)
        entries = self._partitions.setdefault(request.partition, {})
        for end, key in request.boundaries:
            if key not in entries:
                continue
            if state == "cold":
                del entries[key]
            elif reads and end <= matched_end:
                # A read covers the previously written, matching ancestors too.
                # It cannot refresh an unwritten endpoint or an unrelated sibling.
                size, refreshed_at = entries[key]
                if size <= reads and (
                    state == "warm" or at - refreshed_at < self.ttl_seconds
                ):
                    entries[key] = (size, at)
        if capacity:
            _, write_key = request.boundaries[-1]
            entries[write_key] = (capacity, at)
        self._last_at = at
        return {
            "ttl_seconds": self.ttl_seconds,
            "state": state,
            "logical_at": at,
            "cache_status": status if capacity else "uncacheable",
            "matched_item_count": matched_end if reads else None,
            "eligible_item_count": request.boundaries[-1][0]
            if request.boundaries
            else None,
            "cacheable_input_tokens": capacity,
            "usage_basis": "provider_receipt"
            if observed is not None
            else "supplied_estimates",
            "modeled_usage": {
                "input_tokens": total,
                "cached_input_tokens": reads,
                "cache_write_input_tokens": capacity - reads,
                "output_tokens": output,
            },
            "observed_usage": observed,
        }
