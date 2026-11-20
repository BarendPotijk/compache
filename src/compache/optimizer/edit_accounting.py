"""Per-session local measurements and edit arithmetic for a known sent view."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from compache.traces import ContractError

from .contracts import ModelSpec, number
from .parts import Source, placeholder, wire_part

if TYPE_CHECKING:
    from .retention import RecipeView


@dataclass(frozen=True)
class _Block:
    source: Source
    tokens: int
    wire: str
    wire_bytes: int


class PartMeter:
    """Remember the sent projection and a bounded number of alternate blocks.

    A host counter must measure one complete rendered Source with the same
    deterministic local ruler for the session's lifetime. The default uses
    ModelSpec.part_tokens. Keys use part identity for lookup, then validate
    content and metadata before reusing the previous measurement.
    """

    def __init__(self, spec: ModelSpec, count_part_tokens=None, *, max_entries=2048):
        number(max_entries, "max_entries", 1, integer=True)
        if count_part_tokens is not None and not callable(count_part_tokens):
            raise ContractError("count_part_tokens must be a callable")
        self.spec = spec
        self._count = count_part_tokens
        self._limit = max_entries
        self._retained = {}
        self._recent = OrderedDict()
        # Host-measured blocks awaiting inclusion in the next sent view must
        # survive render passes over unrelated candidate/placeholder blocks.
        self._observed = OrderedDict()

    def measure(self, source: Source, *, archived_in=None):
        """Return the full or archived block, including its wire and byte count."""
        observed = self._observed.get(source.part_id)
        if observed is not None:
            if observed.source != source:
                del self._observed[source.part_id]
            elif archived_in is None:
                self._observed.move_to_end(source.part_id)
                return observed
        key = (source.part_id, archived_in)
        hit = self._recent.get(key)
        if hit is not None and hit[0] == source:
            self._recent.move_to_end(key)
            return hit[1]
        rendered = source
        if archived_in is not None:
            rendered = replace(source, text=placeholder(archived_in, source))
        block = self._retained.get(source.part_id)
        if block is None or block.source != rendered:
            if rendered.text is None:
                raise ContractError("measuring a block requires visible text")
            if self._count is None:
                tokens = self.spec.part_tokens(rendered.text)
            else:
                tokens = self._count(rendered)
                number(tokens, "count_part_tokens result", integer=True)
            wire = wire_part(rendered)
            block = _Block(rendered, tokens, wire, len(wire.encode("utf-8")))
        self._recent[key] = (source, block)
        self._recent.move_to_end(key)
        if len(self._recent) > self._limit:
            self._recent.popitem(last=False)
        return block

    def count(self, source: Source) -> int:
        return self.measure(source).tokens

    def observe_count(self, source: Source, tokens: int) -> None:
        """Remember the host's size of this exact full rendered source.

        This is the size replayed into a subsequent input. The billed output of
        a generation may include private reasoning and uses its own counter.
        """
        number(tokens, "observed block tokens", integer=True)
        if source.text is None:
            raise ContractError("observing a block size requires visible text")
        # Replace any speculative full measurement of the same source. Archived
        # replacements remain separate, since their actual wire is different.
        self._recent.pop((source.part_id, None), None)
        block = self._retained.get(source.part_id)
        if block is not None and block.source == source:
            measured = replace(block, tokens=tokens)
            self._retained[source.part_id] = measured
        else:
            wire = wire_part(source)
            measured = _Block(source, tokens, wire, len(wire.encode("utf-8")))
        self._observed[source.part_id] = measured
        self._observed.move_to_end(source.part_id)
        if len(self._observed) > self._limit:
            self._observed.popitem(last=False)

    def retain(self, view: RecipeView) -> None:
        """Pin every block of the successful sent view without re-tokenizing it."""
        if view.model != self.spec.model_id:
            raise ContractError("cannot retain a view from a different model")
        sizes = view.part_wire_bytes or tuple(len(s.encode("utf-8")) for s in view.wire)
        self._retained = {
            p.part_id: _Block(p, tokens, wire, size)
            for p, tokens, wire, size in zip(
                view.parts, view.part_tokens, view.wire, sizes, strict=True
            )
        }


@dataclass(frozen=True)
class InputAnchor:
    """The provider's unrounded input count for one particular sent wire view."""

    view: RecipeView = field(repr=False)
    reported_input_tokens: int
    response_id: str | None = None
    local_tokens: int = field(init=False)
    _blocks: MappingProxyType = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        number(self.reported_input_tokens, "reported_input_tokens", integer=True)
        if self.response_id is not None and (
            not isinstance(self.response_id, str) or not self.response_id.strip()
        ):
            raise ContractError("anchor response_id must be a nonempty string")
        view = self.view
        blocks = {
            p.part_id: (p, wire, size)
            for p, wire, size in zip(
                view.parts, view.wire, view.part_tokens, strict=True
            )
        }
        if len(blocks) != len(view.parts):
            raise ContractError("anchor must have unique source IDs")
        object.__setattr__(self, "_blocks", MappingProxyType(blocks))
        object.__setattr__(
            self, "local_tokens", view.overhead_tokens + sum(view.part_tokens)
        )


@dataclass(frozen=True)
class InputEstimate:
    source: str
    local_tokens: int
    tokens: int
    anchor_input_tokens: int | None = None
    anchor_local_tokens: int | None = None
    added_tokens: int | None = None
    removed_tokens: int | None = None
    overhead_delta: int | None = None
    fallback_reason: str | None = None
    anchor_run_id: str | None = None
    anchor_response_id: str | None = None
    body_edit_kind: str | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "estimated_input_tokens": self.tokens,
            "local_tokens": self.local_tokens,
            "anchor_run_id": self.anchor_run_id,
            "anchor_response_id": self.anchor_response_id,
            "anchor_input_tokens": self.anchor_input_tokens,
            "anchor_local_tokens": self.anchor_local_tokens,
            "added_tokens": self.added_tokens,
            "removed_tokens": self.removed_tokens,
            "overhead_delta": self.overhead_delta,
            "fallback_reason": self.fallback_reason,
            "body_edit_kind": self.body_edit_kind,
        }


def estimate_input(
    view: RecipeView, anchor: InputAnchor | None = None
) -> InputEstimate:
    """Apply typed body edits and the header/framing delta to a measured input.

    The local ruler is additive over blocks. Matching moved blocks keep their
    size in this calculation. Cache reads use a separate ordered-prefix match.
    """
    body = sum(view.part_tokens)
    local = view.overhead_tokens + body
    if anchor is None:
        return InputEstimate(
            "local_estimate", local, local, fallback_reason="no_anchor"
        )
    prior = anchor.view
    if (prior.thread_id, prior.model) != (view.thread_id, view.model):
        return InputEstimate(
            "local_estimate", local, local, fallback_reason="incompatible_scope"
        )

    # Scanning IDs/references costs no tokenization. A changed block, including
    # a newly inserted reopen handle, is one removal and one addition.
    retained_before = retained_after = 0
    prior_is_prefix = len(prior.parts) <= len(view.parts)
    for index, (p, wire, size) in enumerate(
        zip(view.parts, view.wire, view.part_tokens, strict=True)
    ):
        old = anchor._blocks.get(p.part_id)
        matches = old is not None and old[0] == p and old[1] == wire
        if prior_is_prefix and index < len(prior.parts):
            prior_is_prefix = matches and prior.parts[index].part_id == p.part_id
        if matches:
            # A host may refine an unchanged block's count after dispatch.
            # The refined count itself does not add input to the wire.
            retained_before += old[2]
            retained_after += size
    added = body - retained_after
    removed = anchor.local_tokens - prior.overhead_tokens - retained_before
    header_delta = view.overhead_tokens - prior.overhead_tokens
    remaining = anchor.reported_input_tokens - removed + min(0, header_delta)
    consumed = removed + max(0, -header_delta)
    exhausted_with_survivors = (
        consumed > 0
        and remaining == 0
        and (
            retained_before > 0
            or retained_after > 0
            or min(prior.overhead_tokens, view.overhead_tokens) > 0
        )
    )
    if remaining < 0 or exhausted_with_survivors:
        # The edit ruler removed more tokens than the provider ever observed.
        # Use a fresh local size instead of admitting an implausibly small view.
        source, tokens = "local_estimate", local
        reason = (
            "removed_more_than_reported"
            if remaining < 0
            else "exhausted_anchor_with_surviving_blocks"
        )
    else:
        source, tokens, reason = (
            "reported_input_plus_edits",
            anchor.reported_input_tokens + added - removed + header_delta,
            None,
        )
    return InputEstimate(
        source,
        local,
        tokens,
        anchor.reported_input_tokens,
        anchor.local_tokens,
        added,
        removed,
        header_delta,
        reason,
        prior.run_id,
        anchor.response_id,
        (
            "rewrite"
            if not prior_is_prefix
            else "unchanged"
            if len(prior.parts) == len(view.parts)
            else "append"
        ),
    )


class EditErrorHistory:
    """Recent one-sided errors on anchored predictions, separated by edit type.

    Only observe the forecast for the actual sent view once its model request
    has reported usage. The empirical p95 is an admission allowance in tokens.
    It does not alter measured input, prefix reads or request cost.
    """

    def __init__(self):
        self._samples = {
            kind: deque(maxlen=64)
            for kind in ("unchanged", "append", "rewrite", "header_change")
        }

    @staticmethod
    def classify(estimate, header_changed=False) -> str:
        if estimate is None or estimate.source != "reported_input_plus_edits":
            return "local"
        if header_changed or estimate.overhead_delta:
            return "header_change"
        if estimate.body_edit_kind is not None:
            return estimate.body_edit_kind
        if estimate.removed_tokens:
            return "rewrite"
        return "append" if estimate.added_tokens else "unchanged"

    def allowance(self, kind) -> tuple[int, int]:
        """Return sample count and nearest-rank p95 of positive undercounts."""
        if kind == "local":
            return 0, 0
        if kind not in self._samples:
            raise ContractError("unknown edit-error class")
        samples = self._samples[kind]
        n = len(samples)
        if not n:
            return 0, 0
        # Zero-based nearest-rank index, ceil(0.95*n)-1.
        return n, sorted(samples)[(95 * n - 1) // 100]

    def observe(self, kind, estimate, reported_input_tokens) -> None:
        if (
            reported_input_tokens is None
            or estimate is None
            or estimate.source != "reported_input_plus_edits"
        ):
            return
        if kind not in self._samples:
            if kind == "local":
                return
            raise ContractError("unknown edit-error class")
        number(reported_input_tokens, "reported_input_tokens", integer=True)
        self._samples[kind].append(max(0, reported_input_tokens - estimate.tokens))
