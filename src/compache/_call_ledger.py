"""Physical dispatch counts and provider usage receipts for the live request path."""

from threading import Lock

from . import notes as notes_
from .record import left_out
from .threads import canonical_usage, long_writes


def _position(part_id: str) -> tuple:
    """Where a part sits in its conversation: its turn, then question, retrieved text, answer."""
    head, _, tail = part_id.rpartition(":")
    if tail.isdigit():
        return int(head.rsplit(":", 2)[-2]), 1, int(tail)
    return int(head.rsplit(":", 1)[-1]), 0 if tail == "input" else 2, 0


class CallLedger:
    def __init__(self, stats):
        self.stats = stats
        self._lock = Lock()
        self.tokens = dict.fromkeys(
            ("input", "cache_read", "cache_write", "output", "left_out"), 0
        )

    def send(self, transport, payload):
        """Count each upstream dispatch, including refusals that have no token receipt."""
        with self._lock:
            self.stats["physical_calls"] += 1
        return transport(payload)

    def receipt(
        self, recorder, clock, conversation, kind, source, turn=None, pending=None
    ) -> None:
        """Count one model call. `source` has the `usage`, `cost` and `model` of its reply."""
        pending = pending or conversation.pending
        conversation.answered = source.model or conversation.answered
        if pending is not None and kind in ("answer", "reopen", "retry"):
            if pending.first_usage is None:
                pending.first_usage = source.usage
            pending.last_usage = source.usage
            pending.model = source.model or pending.model
        used = canonical_usage(source.usage)
        omitted = 0
        if used and pending is not None and pending.held and kind == "answer":
            out = left_out(pending.left, pending.held, used["input"], pending.media)
            omitted = max(0, out + pending.left_more)
        with self._lock:
            for name, count in (used or {}).items():
                self.tokens[name] += count
            self.tokens["left_out"] += omitted
        if recorder is None:
            return
        archived, left = None, None
        if turn is None:
            turn = pending.index if pending is not None else conversation.turns
            if pending is not None and pending.turn is not None and kind != "ping":
                if not conversation.agentic:
                    gone = sorted(pending.turn.archived_ids, key=_position)
                    archived = notes_.labels_of(notes_.Note(needs=tuple(gone)))
                if pending.held and kind == "answer":
                    left = (pending.left, pending.held)
        recorder.call(
            conversation.record_id,
            turn,
            kind,
            at=clock() if conversation.when is None else conversation.when,
            model=source.model,
            requested=conversation.requested,
            usage=used,
            cost=source.cost,
            archived=archived,
            left=left,
            media=left is not None and pending.media,
            left_more=pending.left_more if left is not None else 0,
            lifetime=conversation.lifetime,
            header_lifetime=conversation.header_lifetime,
            long_write=long_writes(source.usage),
            left_alone=conversation.left_alone,
        )
