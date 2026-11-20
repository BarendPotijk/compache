"""Model reuse of provider-visible static system/tool prefixes across conversations.

Each catalog represents one host cache observation stream. A recorded successful
request is a modeled cache opportunity. Provider read counters cover the whole
request and cannot independently identify which component was served from cache.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import secrets
from threading import RLock

from compache.traces import ContractError
from .cache_breaks import CacheScope
from .contracts import ModelSpec, digest, number


COMPONENTS = ("system_prompt", "default_tools")


def _wire(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "provide ordered JSON values or exact serialized wire fragments"
        ) from exc


def _tool_parts(tools, fragments):
    if fragments is not None:
        if not isinstance(fragments, (tuple, list)) or any(
            not isinstance(v, (bytes, str)) for v in fragments
        ):
            raise ContractError(
                "tool_fragments must be a sequence of exact wire byte/text fragments"
            )
        parts = [_wire(v) for v in fragments]
        if tools is not None and b"".join(parts) != _wire(tools):
            raise ContractError(
                "tool_fragments must concatenate to the actual serialized tools"
            )
        return parts, True
    if isinstance(tools, (list, tuple)) and tools:
        parts = []
        for idx, tool in enumerate(tools):
            raw = json.dumps(
                tool, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            parts.append(
                (b"[" if idx == 0 else b",")
                + raw
                + (b"]" if idx == len(tools) - 1 else b"")
            )
        return parts, True
    return [_wire(tools)], False


def _common(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


@dataclass(frozen=True)
class SharedPrefix:
    wire_bytes: int
    system_tokens: int
    default_tool_tokens: int
    dynamic_tool_tokens: int
    tool_breakdown: tuple[int, ...]
    component_order: tuple[str, str]
    source: str
    cache_min_tokens: int
    block_tokens: int
    ttl_seconds: float
    _catalog_id: str = field(repr=False)
    _partition: str = field(repr=False)
    _thread: str = field(repr=False)
    _static_digest: bytes = field(repr=False)
    _blocks: tuple[bytes, ...] = field(repr=False)
    _boundaries: tuple[tuple[bytes, int], ...] = field(repr=False)

    @property
    def static_tokens(self):
        return self.system_tokens + self.default_tool_tokens

    @property
    def total_tokens(self):
        return self.static_tokens + self.dynamic_tool_tokens

    @property
    def unattributed_tool_tokens(self):
        return self.default_tool_tokens - sum(self.tool_breakdown)


@dataclass(frozen=True)
class SharedReuse:
    eligible_tokens: int
    system_tokens: int
    default_tool_tokens: int
    status: str
    source: str
    age_seconds: float | None = None
    cross_thread: bool = False


@dataclass(frozen=True)
class _Entry:
    prefix: SharedPrefix
    at: float
    expires_at: float
    request_tokens: int


class SharedPrefixCatalog:
    """Bounded in-process catalog shared by sessions on a worker (or injected by a host)."""

    def __init__(self, max_entries=1024):
        number(max_entries, "max_entries", 1, integer=True)
        self.max_entries = max_entries
        self._secret = secrets.token_bytes(32)
        self._identity = hashlib.sha256(self._secret).hexdigest()
        self._entries = OrderedDict()
        self._lock = RLock()

    def _hash(self, data):
        return hashlib.blake2b(data, digest_size=24, key=self._secret).digest()

    def measure(
        self,
        scope: CacheScope,
        *,
        system_prompt,
        default_tools,
        spec: ModelSpec,
        tokenize=None,
        count_tokens=None,
        component_order=COMPONENTS,
        dynamic_tools=None,
        share=True,
        tool_fragments=None,
        tokenizer_id=None,
    ):
        """Measure the exact supplied bytes in provider order, with no invented separators.

        Supply bytes/text containing each component's actual provider framing.
        Ordered lists/dicts are serialized as compact JSON. List wrappers count
        toward the first/last tool. Opaque tools have an unallocated total.
        Callbacks receive the concatenated bytes. Token-ID callbacks provide
        exact common-token counting; count-only matches are component-boundary
        approximations. Include any provider-controlled wrapper in the adapter.
        Dynamic tools form a final, thread-local segment. `share=False` isolates
        this catalog key per thread. Set `tokenizer_id` if the namespace uses
        multiple versions of the same model's tokenizer.
        """
        if (
            not isinstance(scope, CacheScope)
            or not isinstance(spec, ModelSpec)
            or spec.model_id != scope.model
        ):
            raise ContractError(
                "cache scope and ModelSpec must refer to the same model"
            )
        if type(share) is not bool or (
            tokenizer_id is not None and not isinstance(tokenizer_id, str)
        ):
            raise ContractError("invalid shared prefix namespace options")
        order = tuple(component_order)
        if order not in (COMPONENTS, COMPONENTS[::-1]):
            raise ContractError(
                "component_order must contain system_prompt and default_tools once"
            )
        if tokenize is not None and count_tokens is not None:
            raise ContractError("provide one tokenizer or token-count callback")
        tools, itemized = _tool_parts(default_tools, tool_fragments)
        sections = {"system_prompt": [_wire(system_prompt)], "default_tools": tools}
        pieces = [(name, part) for name in order for part in sections[name]]
        static_piece_count = len(pieces)
        pieces.append(("dynamic_tools", _wire(dynamic_tools)))
        wire = b"".join(part for _, part in pieces)
        if tokenize is not None:

            def encode(data):
                ids = tuple(tokenize(data)) if data else ()
                if any(type(i) is not int or i < 0 for i in ids):
                    raise ContractError(
                        "tokenize must return nonnegative integer token IDs"
                    )
                return ids

            full = encode(wire)
            total = len(full)
            source = "native_token_ids"
        elif count_tokens is not None:

            def count(data):
                n = count_tokens(data) if data else 0
                number(n, "provider token count", integer=True)
                return n

            full = None
            total = count(wire)
            source = "host_token_count"
        else:

            def count(data):
                return math.ceil(len(data) / spec.bytes_per_token)

            full = None
            total = count(wire)
            source = "estimated_from_bytes"
        position = 0
        ends = []
        fingerprints = []
        for _, part in pieces:
            position += len(part)
            prefix = wire[:position]
            fingerprints.append(self._hash(prefix))
            if position == len(wire):
                end = total
            elif full is not None:
                end = _common(encode(prefix), full)
            else:
                end = count(prefix)
            ends.append(end)
        # Appending bytes can retokenize a trailing word, reducing token counts.
        # Attribute boundary-spanning tokens to the later component. Clipping
        # also preserves nonnegative additive counts for count-only callbacks.
        for idx in range(len(ends) - 2, -1, -1):
            ends[idx] = min(ends[idx], ends[idx + 1])
        sizes = [end - start for start, end in zip([0] + ends[:-1], ends)]
        by_type = {
            name: sum(n for (kind, _), n in zip(pieces, sizes) if kind == name)
            for name in ("system_prompt", "default_tools", "dynamic_tools")
        }
        breakdown = (
            tuple(n for (kind, _), n in zip(pieces, sizes) if kind == "default_tools")
            if itemized
            else ()
        )
        static_len = ends[static_piece_count - 1]
        block = spec.cache_block_tokens
        blocks = (
            ()
            if full is None
            else tuple(
                self._hash(
                    json.dumps(full[i : i + block], separators=(",", ":")).encode(
                        "ascii"
                    )
                )
                for i in range(0, static_len - block + 1, block)
            )
        )
        thread = digest([scope.installation, scope.thread_id])
        partition = digest(
            [
                scope.installation,
                scope.provider,
                scope.model,
                scope.cache_key,
                order,
                source,
                spec.bytes_per_token if source == "estimated_from_bytes" else None,
                block,
                spec.cache_min_tokens,
                tokenizer_id,
                None if share else thread,
            ]
        )
        boundaries = tuple(
            zip(fingerprints[:static_piece_count], ends[:static_piece_count])
        )
        return SharedPrefix(
            len(wire),
            by_type["system_prompt"],
            by_type["default_tools"],
            by_type["dynamic_tools"],
            breakdown,
            order,
            source,
            spec.cache_min_tokens,
            block,
            spec.cache_ttl_seconds,
            self._identity,
            partition,
            thread,
            boundaries[-1][0],
            blocks,
            boundaries,
        )

    def _limits(self, prefix, at, ttl_seconds, request_tokens):
        if not isinstance(prefix, SharedPrefix) or prefix._catalog_id != self._identity:
            raise ContractError("prefix must be measured by this catalog")
        number(at, "at", -1e20)
        total = prefix.total_tokens if request_tokens is None else request_tokens
        number(total, "request_tokens", integer=True)
        ttl = prefix.ttl_seconds if ttl_seconds is None else ttl_seconds
        number(ttl, "ttl_seconds")
        return ttl, total

    def record(
        self,
        prefix,
        *,
        at,
        request_tokens=None,
        cached_input_tokens=None,
        ttl_seconds=None,
        successful=True,
    ):
        """Register only successful, cache-eligible requests. Read counters are aggregate."""
        ttl, total = self._limits(prefix, at, ttl_seconds, request_tokens)
        if type(successful) is not bool:
            raise ContractError("successful must be a boolean")
        if cached_input_tokens is not None:
            number(cached_input_tokens, "cached_input_tokens", 0, total, integer=True)
        capacity = (
            min(total, prefix.static_tokens)
            // prefix.block_tokens
            * prefix.block_tokens
        )
        if not successful or ttl <= 0 or capacity < max(1, prefix.cache_min_tokens):
            return False
        with self._lock:
            key = (
                prefix._partition,
                prefix._thread,
                prefix._static_digest,
                prefix._blocks,
                prefix.static_tokens,
                capacity,
            )
            prior = self._entries.get(key)
            if prior is not None and prior.at > at:
                return False
            self._entries[key] = _Entry(prefix, at, at + ttl, total)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return True

    def limit_publisher_reuse(self, prefix, *, at, read_tokens):
        """Cap older opportunities from a publisher after its ping missed eligible blocks."""
        self._limits(prefix, at, None, read_tokens)
        with self._lock:
            for key, entry in list(self._entries.items()):
                if (
                    entry.prefix._partition != prefix._partition
                    or entry.prefix._thread != prefix._thread
                    or entry.at > at
                    or entry.request_tokens <= read_tokens
                ):
                    continue
                if read_tokens < max(1, prefix.cache_min_tokens):
                    del self._entries[key]
                else:
                    self._entries[key] = replace(entry, request_tokens=read_tokens)

    def lookup(
        self,
        prefix,
        *,
        at,
        request_tokens=None,
        ttl_seconds=None,
        same_thread_upper_bound=None,
    ):
        """Bound static overlap by both requests and round to cache blocks.

        A host-verified upper bound on a thread's own matching prefix applies
        to that publisher only. Other threads in the namespace can have an
        independent, newer or longer matching header.
        """
        ttl, total = self._limits(prefix, at, ttl_seconds, request_tokens)
        if same_thread_upper_bound is not None:
            number(same_thread_upper_bound, "same_thread_upper_bound", integer=True)
        source = "modeled_" + prefix.source

        def empty(status):
            return SharedReuse(0, 0, 0, status, source)

        if ttl <= 0 or total < prefix.cache_min_tokens:
            return empty("ineligible_request")
        if min(
            total, prefix.static_tokens
        ) // prefix.block_tokens * prefix.block_tokens < max(
            1, prefix.cache_min_tokens
        ):
            return empty("static_prefix_below_minimum")
        best = 0
        winner = None
        any_expired = False
        any_active = False
        with self._lock:
            entries = tuple(self._entries.values())
        current = dict(prefix._boundaries)
        for entry in entries:
            prev = entry.prefix
            if prev._partition != prefix._partition or at < entry.at:
                continue
            if at >= entry.expires_at or at - entry.at >= ttl:
                any_expired = True
                continue
            any_active = True
            if prefix.source == "native_token_ids":
                matched = _common(prefix._blocks, prev._blocks) * prefix.block_tokens
            else:
                matched = max(
                    (min(n, current[h]) for h, n in prev._boundaries if h in current),
                    default=0,
                )
            if prev._thread == prefix._thread and same_thread_upper_bound is not None:
                matched = min(matched, same_thread_upper_bound)
            matched = (
                min(matched, total, entry.request_tokens)
                // prefix.block_tokens
                * prefix.block_tokens
            )
            if matched >= max(1, prefix.cache_min_tokens) and (
                matched > best or (matched == best and entry.at > winner.at)
            ):
                best, winner = matched, entry
        if winner is None:
            return empty(
                "no_matching_prefix"
                if any_active
                else "expired"
                if any_expired
                else "cold"
            )
        if prefix.component_order[0] == "system_prompt":
            system = min(best, prefix.system_tokens)
        else:
            system = max(0, best - prefix.default_tool_tokens)
        capacity = (
            min(total, prefix.static_tokens)
            // prefix.block_tokens
            * prefix.block_tokens
        )
        return SharedReuse(
            best,
            system,
            best - system,
            "full_static_hit" if best == capacity else "partial_static_hit",
            source,
            at - winner.at,
            prefix._thread != winner.prefix._thread,
        )
