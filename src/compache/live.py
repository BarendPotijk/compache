"""compache in the request path.

`Live` sits between an application and its model provider and sees every model
call as the JSON that goes over the wire. For each call it can do three things:

- apply a fitted policy: replace archived parts by their notice, mark the
  conversation cacheable, answer the model's `reopen` calls unseen, and keep
  the cache warm with pings;
- record the conversation without keeping any of its text: sizes, times and the
  provider's usage, as `compache.record` describes;
- ask the model, on a sample of turns, which earlier parts its reply needed:
  in a short request of its own once the reply is in, or in the reply itself,
  where the note is taken out before the application sees it.

With no policy it changes nothing in the requests and only records. That is how
a workload is learned going forward when past traces cannot be read.

The application never has to notice. A request shape this module does not know
is passed through untouched. A request the provider refuses after it was
changed is sent again as the application wrote it. A note is taken out of a
reply whatever else goes wrong.

`compache.wrap` puts a `Live` inside a provider's SDK client and
`compache proxy` behind a base URL. `Live.exchange` is the one entry they share.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from . import notes as notes_
from . import wire
from .api import Policy
from .optimizer.recovery import REOPEN_TOOL
from .record import Recorder
from .threads import cache_read_reported, canonical_usage

__all__ = ["Live", "Result"]

logger = logging.getLogger(__name__)

_REOPEN = REOPEN_TOOL["function"]["name"]
# The tool as a model in a live conversation is told about it: what a notice is, and what to do with one.
_REOPEN_TOOL = {
    "type": "function",
    "function": {
        "name": _REOPEN,
        "description": (
            "Read text of this conversation that was archived to save space. An archived part "
            "shows a short notice with a reference such as ctx:1a2b3c in place of its text. Give "
            "that reference to read the part, or give a short distinctive phrase to search every "
            "archived part for it. Use it whenever what you need is no longer visible, and before "
            "telling the user that you do not have something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": (
                        "The ctx: reference from a notice, or a few distinctive words to search "
                        "the archived parts for."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Leave out at first. For a long part, the next_offset of the last result.",
                },
            },
            "required": ["reference"],
            "additionalProperties": False,
        },
    },
}
_REOPEN_HOW = (
    "Earlier parts of this conversation may be archived to save space. An archived part shows a "
    'notice like [Archived to save space. Call reopen("ctx:...") to read this part.] in place of '
    "its text. Nothing is lost: an archived part is still part of the conversation, and the "
    "reopen tool reads it. When a question needs something that is not visible, and always "
    "before you tell the user that you do not have or cannot find something, call reopen: with "
    "the reference from a notice to read that part, or with a few distinctive words to search "
    "all archived parts. Then answer from what it returns. Do not reopen parts you do not need, "
    "and never mention archiving, notices or the reopen tool to the user."
)
_NOT_FOUND = (
    "No archived part matches that. Try other or fewer distinctive words from the text you "
    "need, or the reference of a notice."
)
# What a model is shown of a fetched part: its text and where it goes on, not the names parts go by here.
_SHOWN = (
    "status",
    "handle",
    "offset",
    "length",
    "text",
    "next_offset",
    "next_reference",
    "total",
    "matches",
    "message",
)
_HANDLE = re.compile(r"ctx:[0-9a-f]{24}")
_REFUSED = (400, 422)
_ROUNDS = 3
# A note is one line. The room is for a model that reasons before it writes: on Claude Haiku
# 5.5 that took a median of some 500 tokens, and more than 1,000 for one note in twelve.
_NOTE_TOKENS = 2048
# What stands in for the further pieces of a part whose first piece carries the notice.
_GONE = "…"


@dataclass
class Result:
    """What came back from a provider: a status, headers, and a JSON body or a stream.

    `events` is an iterator of bytes for a streamed reply. `raw` is the body as
    it arrived, and is None once `json` no longer is what arrived. `close`
    releases the connection of a stream that is abandoned.
    """

    status: int
    headers: Any = None
    json: Any = None
    events: Iterator[bytes] | None = None
    raw: bytes | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class _Said:
    """A reply's text, standing where the next request will carry it as a slot."""

    text: str


@dataclass
class _Found:
    """One retrieved part: the pieces of text of one tool result."""

    slots: list

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.slots)


@dataclass
class _Segment:
    """One turn of a request: the slots of its question, its retrieved parts, the slots of its answer."""

    inputs: list = field(default_factory=list)
    retrievals: list = field(default_factory=list)
    outputs: list = field(default_factory=list)

    @property
    def question(self) -> str:
        return "\n\n".join(s.text for s in self.inputs)

    @property
    def answer(self) -> str:
        return "\n\n".join(s.text for s in self.outputs if s.text)


@dataclass
class _Pending:
    """The turn that is being answered."""

    index: int
    started_at: float
    ended_at: float = 0.0
    turn: Any = None
    retrievals: int = 0
    calls: int = 0
    first_usage: Any = None
    last_usage: Any = None
    model: str | None = None
    asked: bool = False
    refs: list | None = None
    older: bool = False
    cited: tuple = ()
    # Parts the model fetched back while answering.
    reopened: list = field(default_factory=list)


@dataclass
class _Conversation:
    # The name its policy session and its parts go by, and the name it is recorded under.
    thread_id: str
    chain: str
    record_id: str = ""
    session: Any = None
    turns: int = 0
    pending: _Pending | None = None
    lock: Any = field(default_factory=threading.RLock)
    # What a ping is built from: the dialect, the last request without mark or note, its answer, how to send.
    last: tuple | None = None
    pings_off: bool = False
    # Whether the model has the `reopen` tool: from the first archived part on.
    reopens: bool = False
    # How often a turn had to be asked again as the application wrote it.
    fell_back: int = 0
    system_bytes: int = 0
    tools_bytes: int = 0
    # The latest time this conversation has seen.
    at: float = float("-inf")


@dataclass
class _Call:
    """One model call on its way: the conversation it belongs to and the forms of its request."""

    conversation: _Conversation
    dialect: Any
    turns: list
    chains: list
    send: Callable[[Any], Result]
    body: Any
    plain: Any
    marked: Any
    outgoing: Any
    asked: bool = False
    carrier: str | None = None
    # Ask for the note in a request of its own, once the reply is in.
    apart: bool = False


def _segments(slots) -> list[_Segment]:
    """Turns of a request: a question opens one, and everything up to the next question is its."""
    turns: list[_Segment] = []
    waiting = []
    for slot in slots:
        if not slot.text and not slot.closes:
            continue
        if slot.kind == "input":
            if not turns or turns[-1].outputs:
                turns.append(_Segment(retrievals=waiting))
                waiting = []
            turns[-1].inputs.append(slot)
        elif slot.kind == "retrieval":
            found = turns[-1].retrievals if turns else waiting
            if (
                found
                and slot.group is not None
                and found[-1].slots[-1].group == slot.group
            ):
                # Another piece of the same tool result.
                found[-1].slots.append(slot)
            else:
                found.append(_Found([slot]))
        elif turns:
            turns[-1].outputs.append(slot)
    return turns


def _link(chain: str, segment: _Segment) -> str:
    said = [segment.question, [s.text for s in segment.retrievals], segment.answer]
    return hashlib.sha256((chain + json.dumps(said)).encode()).hexdigest()


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _notice(placeholder: str) -> str:
    """The session's notice in words a model acts on: the tool to call and what to give it.

    It is exactly as long as the notice it stands for, so the sizes the policy counts on hold.
    """
    found = _HANDLE.search(placeholder)
    if found is None:
        return placeholder
    return (
        f'[Archived to save space. Call reopen("{found.group()}") to read this part.]'
    )


def _for_model(found: dict) -> dict:
    """The answer to a `reopen` call as the model reads it."""
    if found.get("status") == "not_found":
        return {"status": "not_found", "hint": _NOT_FOUND}
    shown = {k: v for k, v in found.items() if k in _SHOWN}
    if isinstance(shown.get("matches"), list):
        shown["matches"] = [
            {k: m[k] for k in ("handle", "offset", "excerpt") if k in m}
            for m in shown["matches"]
        ]
    return shown


def _position(part_id: str) -> tuple:
    """Where a part sits in its conversation: its turn, then question, retrieved text, answer."""
    head, _, tail = part_id.rpartition(":")
    if tail.isdigit():
        return int(head.rsplit(":", 2)[-2]), 1, int(tail)
    return int(head.rsplit(":", 1)[-1]), 0 if tail == "input" else 2, 0


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _turn_usage(first, last) -> dict | None:
    """One usage for a turn: the last call's input, all of its output, what the first call found cached."""
    first, last = canonical_usage(first), canonical_usage(last)
    if first is None or last is None or first == last:
        return first or last
    left = min(last["cache_read"] + last["cache_write"], last["input"])
    return {
        "input": last["input"],
        "output": first["output"] + last["output"],
        "cache_read": first["cache_read"],
        "cache_write": max(0, left - first["cache_read"]) if last["cache_write"] else 0,
    }


class Live:
    """A policy applied, a record kept and notes asked, on every model call that passes.

    `policy` is a `Policy`, a path to one, or None to change nothing and only
    record. `record` is a folder for a record without text, or a `Recorder`.
    `sample` is the share of turns whose reply is asked what it needed, and
    `notes` says how: `"apart"` in a short request of its own after the reply,
    which leaves the reply untouched by construction, or `"inline"` in the
    reply itself, which saves that request and lets the note weigh on the
    answer. Notes need a record to go to.
    """

    def __init__(
        self,
        policy=None,
        *,
        record=None,
        sample: float = 0.0,
        notes: str = "apart",
        salt: str = "",
        clock: Callable[[], float] = time.time,
        max_conversations: int = 4096,
    ):
        if isinstance(policy, (str, Path)):
            policy = Policy.load(policy)
        if notes not in ("apart", "inline"):
            raise ValueError("notes: expected 'apart' or 'inline'")
        if not 0 <= sample <= 1:
            raise ValueError("sample: expected a share between 0 and 1")
        self.policy = policy
        self.recorder = Recorder(record) if isinstance(record, (str, Path)) else record
        if sample and self.recorder is None:
            raise ValueError(
                "sample: notes need a record to go to. Give `record` a folder"
            )
        self.sample = sample
        self.notes, self.salt, self.clock = notes, salt, clock
        recipe = policy.recipe if policy is not None else None
        if recipe is not None and recipe.cited_answer_runs != 0 and not sample:
            # Here an answer is known to cite a text only through its note.
            logger.warning(
                "compache: this policy keeps the text that answers cite, and no reply is asked "
                "what it cited. Give `sample` the rate at which the policy's record was made."
            )
        self._limit = max_conversations
        self._lock = threading.RLock()
        self._by_chain: OrderedDict[str, _Conversation] = OrderedDict()
        self._names: dict[str, int] = {}
        self._bad_schemas: set[str] = set()
        self._refusals = 0
        self._notes_off = False
        self._apart: list[threading.Thread] = []
        # The thread that sends the keep-warm pings, once `compache.wrap.start_pings` made one.
        self.pinger = None
        # What happened so far, for a caller that wants to know.
        self.stats = dict.fromkeys(
            (
                "calls",
                "passed_through",
                "turns",
                "asked",
                "noted",
                "leaked",
                "retried",
                "reopened",
                "pings",
                "policy_refused",
                "fell_back",
                "stood_down",
            ),
            0,
        )

    # -- conversations -------------------------------------------------------

    def _conversation(
        self, thread: str | None, turns: list[_Segment]
    ) -> tuple[_Conversation, list[str]]:
        """The conversation a request continues, found by what it has said so far."""
        seed = "thread:" + thread if thread else "auto"
        chains = [hashlib.sha256(seed.encode()).hexdigest()]
        for segment in turns[:-1]:
            chains.append(_link(chains[-1], segment))
        done = len(turns) - 1
        with self._lock:
            found = self._by_chain.get(chains[-1]) if thread or done else None
            if found is None and done:
                # Its last turn may have been answered without the answer passing here.
                earlier = self._by_chain.get(chains[-2]) if thread or done > 1 else None
                if (
                    earlier is not None
                    and earlier.pending is not None
                    and earlier.pending.index == done
                ):
                    found = earlier
            if found is None:
                name = record = thread or "c-" + uuid.uuid4().hex[:16]
                if thread:
                    # A name can be met again: after a restart, or shared by several conversations.
                    count = self._names[thread] = self._names.get(thread, 0) + 1
                    if count > 1:
                        name = f"{thread}#{count}"
                        record = thread if done else f"{thread}~{count}"
                found = _Conversation(name, chains[0], record)
                if self.policy is not None:
                    found.session = self.policy.session(name)
                if thread:
                    # A named conversation is found again from its first call. An unnamed one only
                    # once it has an answer, since any two may open with the same words.
                    self._index(found, chains[0])
            else:
                self._by_chain.move_to_end(found.chain)
            return found, chains

    def _now(self, conversation: _Conversation) -> float:
        """The time, and never one this conversation has already passed: a clock can step back."""
        conversation.at = max(self.clock(), conversation.at + 1e-3)
        return conversation.at

    def _index(self, conversation: _Conversation, chain: str) -> None:
        with self._lock:
            if self._by_chain.get(conversation.chain) is conversation:
                del self._by_chain[conversation.chain]
            conversation.chain = chain
            self._by_chain[chain] = conversation
            while len(self._by_chain) > self._limit:
                self._by_chain.popitem(last=False)

    # -- one model call ------------------------------------------------------

    def exchange(
        self,
        path: str,
        body: Any,
        send: Callable[[Any], Result],
        *,
        thread: str | None = None,
    ) -> Result:
        """Carry one model call: `send(body)` performs it and returns a `Result`.

        `path` is the request path, which says which API it is. `thread` names
        the conversation. Without a name the conversation is recognized by what
        it has said so far.
        """
        self.stats["calls"] += 1
        dialect = wire.dialect_for(path)
        try:
            scan = (
                dialect.scan(body)
                if dialect is not None and isinstance(body, dict)
                else None
            )
            turns = _segments(scan.slots) if scan is not None else []
            if not turns or not scan.awaiting:
                raise ValueError("nothing here to decide on")
            scan.stream = scan.stream or wire.streams(path)
            conversation, chains = self._conversation(thread, turns)
        except Exception:
            # A shape this does not know: the application's request goes out as it is.
            self.stats["passed_through"] += 1
            return send(body)
        with conversation.lock:
            try:
                call = self._prepare(
                    conversation, chains, dialect, body, scan, turns, send
                )
            except Exception:
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                self.stats["passed_through"] += 1
                return send(body)
            return self._carry(call, scan)

    def _prepare(self, conversation, chains, dialect, body, scan, turns, send) -> _Call:
        """Bring the conversation up to this request and work out what to send for it."""
        now, number, current = self._now(conversation), len(turns), turns[-1]
        conversation.system_bytes = _size(scan.system)
        conversation.tools_bytes = _size(json.dumps(scan.tools)) if scan.tools else 0
        plain = body
        try:
            self._catch_up(conversation, chains, turns, now)
            pending, found = (
                conversation.pending,
                [slot.text for slot in current.retrievals],
            )
            if pending is None or pending.index != number:
                conversation.last = None
                pending = conversation.pending = _Pending(
                    number, now, retrievals=len(found)
                )
                if conversation.session is not None:
                    pending.turn = conversation.session.prepare(
                        current.question,
                        found,
                        at=now,
                        system_prompt=scan.system,
                        tools=scan.tools,
                    )
            elif len(found) > pending.retrievals:
                if conversation.session is not None:
                    pending.turn = conversation.session.extend(
                        pending.turn, found[pending.retrievals :]
                    )
                pending.retrievals = len(found)
            if conversation.session is not None:
                plain = self._apply(conversation, dialect, body, scan, turns)
        except Exception:
            # The policy cannot be applied here. From now on this conversation is only recorded.
            logger.warning(
                "compache: policy not applied to a conversation", exc_info=True
            )
            self._stand_down(conversation, number, now)
            plain = body
        marked = (
            dialect.with_cache_mark(plain)
            if conversation.session is not None
            else plain
        )
        call = _Call(
            conversation, dialect, turns, chains, send, body, plain, marked, marked
        )
        try:
            call.outgoing, call.asked, call.carrier = self._with_note(call, scan)
        except Exception:
            logger.warning("compache: a note was not asked", exc_info=True)
        return call

    def _carry(self, call: _Call, scan) -> Result:
        """Send the request, fall back to plainer forms of it if it is refused, and read the reply."""
        conversation, pending = call.conversation, call.conversation.pending
        result = call.send(call.outgoing)
        if result.status in _REFUSED and call.outgoing is not call.marked:
            # Refused with the note in it: send it without.
            self._note_refused(call)
            self.stats["retried"] += 1
            call.outgoing, call.asked, call.carrier = call.marked, False, None
            result = call.send(call.outgoing)
        if result.status in _REFUSED and call.outgoing is not call.body:
            # Refused with the policy applied: send what the application wrote.
            self.stats["retried"] += 1
            self.stats["policy_refused"] += 1
            self._stand_down(conversation, pending.index, pending.started_at)
            call.plain = call.marked = call.outgoing = call.body
            call.asked, call.carrier = False, None
            result = call.send(call.body)
        pending = conversation.pending
        pending.asked = pending.asked or call.asked
        if result.status >= 400 or (result.json is None and result.events is None):
            return result
        if result.events is not None:
            return replace(result, events=self._stream(call, result))
        return self._settle(call, result)

    def _catch_up(self, conversation, chains, turns, now) -> None:
        """Bring a conversation's state up to the turns a request shows as answered."""
        done = len(turns) - 1
        if conversation.turns == done:
            return
        pending = conversation.pending
        if pending is not None and pending.index == done == conversation.turns + 1:
            # Answered without the answer passing here: take it from the request.
            last = turns[done - 1]
            found = [slot.text for slot in last.retrievals]
            if conversation.session is not None and len(found) > pending.retrievals:
                pending.turn = conversation.session.extend(
                    pending.turn, found[pending.retrievals :]
                )
            self._finish(conversation, chains[done], last, now)
            return
        # A conversation met in the middle: walk the policy through what was said.
        conversation.pending, conversation.turns = None, 0
        if self.policy is not None:
            conversation.session = self.policy.session(conversation.thread_id)
            for k, segment in enumerate(turns[:done]):
                # When it was said is not known: just before now, each turn after the one before.
                at = now - (done - k) * 1e-3
                turn = conversation.session.prepare(
                    segment.question, [s.text for s in segment.retrievals], at=at
                )
                conversation.session.complete(turn, segment.answer, at=at)
        conversation.turns = done
        self._index(conversation, chains[done])

    def _stand_down(self, conversation, number, now) -> None:
        """Stop applying the policy to a conversation. It is still recorded."""
        if conversation.session is not None:
            self.stats["stood_down"] += 1
        conversation.session = None
        if conversation.pending is None or conversation.pending.index != number:
            conversation.pending = _Pending(number, now)
        conversation.pending.turn = None

    def _apply(self, conversation, dialect, body, scan, turns) -> Any:
        """The request with every archived part replaced by its notice, and the tool to fetch one back."""
        turn, name = conversation.pending.turn, conversation.thread_id
        out = body
        if turn.archived_ids:
            notice = {
                p.part_id: _notice(p.text)
                for p in turn.parts
                if p.part_id in turn.archived_ids
            }
            for k, segment in enumerate(turns, 1):
                pieces = [
                    (f"{name}:{k}:input", segment.inputs),
                    (f"{name}:{k}:output", segment.outputs),
                ]
                pieces += [
                    (f"{name}:{k}:retrieval:{j}", found.slots)
                    for j, found in enumerate(segment.retrievals, 1)
                ]
                for part_id, slots in pieces:
                    if part_id in notice:
                        for i, slot in enumerate(slots):
                            out = wire.set_text(
                                dialect, out, slot.path, _GONE if i else notice[part_id]
                            )
            conversation.reopens = True
        can_answer = not scan.stream or dialect.Stream is not wire.TextStream
        if (
            conversation.reopens
            and can_answer
            and _REOPEN not in dialect.tool_names(body)
        ):
            # A model that is offered the tool while nothing is archived calls it for nothing. So it
            # comes with the first notice, and stays: the first cut rewrites the request from near
            # its start anyway, and after that the tool never changes what is cached.
            out = dialect.with_tool(out, _REOPEN_TOOL, _REOPEN_HOW)
        return out

    # -- notes ---------------------------------------------------------------

    def _with_note(self, call: _Call, scan):
        """The request as it goes out, whether its reply is asked for a note, and how the note travels.

        A note is asked apart, once the reply is in: the reply is then what it
        would have been. With `notes="inline"` it rides in a reply that can be
        read in full before the application sees it, as a last line of text or
        as a property of a JSON schema, and a reply it spoiled is asked again
        without. A streamed reply cannot be taken back once it is shown, so it
        is always asked apart.
        """
        dialect, marked, conversation = call.dialect, call.marked, call.conversation
        if not self.sample or self._notes_off:
            return marked, False, None
        sampled = notes_.sampled(
            conversation.record_id, len(call.turns), self.sample, self.salt
        )
        schema = dialect.schema(call.body)
        extended = notes_.extend_schema(schema) if schema is not None else None
        if (
            self.notes == "apart"
            or scan.stream
            # A schema that cannot take the property, or that the provider refused with it.
            or (schema is not None and extended is None)
            or (schema is not None and _digest(schema) in self._bad_schemas)
        ):
            call.apart = sampled
            return marked, False, None
        if schema is not None:
            # On every turn, asked or not, so that the schema the provider caches stays one.
            outgoing, carrier = dialect.with_schema(marked, extended), "json"
        else:
            outgoing, carrier = marked, "text"
        if not sampled:
            return outgoing, False, carrier
        listing = self._listing(conversation, call.turns)
        return (
            dialect.with_note(outgoing, notes_.instruction(listing, carrier=carrier)),
            True,
            carrier,
        )

    def _listing(self, conversation, turns, pending=None) -> list[dict]:
        """The conversation as a reply may name it. An archived part is listed without words.

        `pending` is the turn the reply belongs to, for a note that is asked
        once the conversation may have moved on.
        """
        pending = pending or conversation.pending
        turn = pending.turn if pending is not None else None
        gone = turn.archived_ids if turn is not None else ()
        name, listing = conversation.thread_id, []
        for k, segment in enumerate(turns, 1):
            entry = {
                "input": None if f"{name}:{k}:input" in gone else segment.question,
                "retrievals": [
                    None if f"{name}:{k}:retrieval:{j}" in gone else slot.text
                    for j, slot in enumerate(segment.retrievals, 1)
                ],
            }
            if k < len(turns):
                entry["output"] = (
                    None if f"{name}:{k}:output" in gone else segment.answer
                )
            listing.append(entry)
        return listing

    def _note_refused(self, call: _Call) -> None:
        schema = call.dialect.schema(call.body)
        if schema is not None:
            self._bad_schemas.add(_digest(schema))
        self._refusals += 1
        if self._refusals >= 3 and not self._notes_off:
            self._notes_off = True
            logger.warning("compache: notes switched off after three refused requests")

    def _take_note(self, call: _Call, content, pending=None) -> None:
        """Keep what a reply named, as labels."""
        pending = pending or call.conversation.pending
        if pending is None:
            return
        listing = self._listing(call.conversation, call.turns, pending)
        note = notes_.resolve(content, listing, call.conversation.thread_id)
        pending.refs, pending.older, pending.cited = (
            notes_.labels_of(note),
            note.older,
            note.cited,
        )
        self.stats["noted"] += 1
        self._refusals = 0

    def _without_note(self, call: _Call, data):
        """A whole reply with its note taken out, and the note. Touches no state, so it cannot fail halfway."""
        dialect, found = call.dialect, None
        reply = dialect.reply(data)
        if call.carrier == "text":
            for path in reply.paths:
                clean, got = notes_.read(wire.get(data, path))
                if got is not None:
                    data, found = wire.put(data, path, clean), got
            if found is not None:
                data = dialect.prune(data)
        elif call.carrier == "json" and reply.paths:
            clean, labels = notes_.split_json(wire.get(data, reply.paths[0]))
            if labels is not None:
                data, found = wire.put(data, reply.paths[0], clean), labels
        return data, found

    # -- replies -------------------------------------------------------------

    def _settle(self, call: _Call, result: Result, kind: str = "answer") -> Result:
        """Read a whole reply: answer `reopen` calls, take the note out, and keep the books."""
        conversation, dialect = call.conversation, call.dialect
        data, sent, rounds = result.json, call.outgoing, 0
        try:
            reply = dialect.reply(data)
            self._ledger(conversation, kind, reply)
            while (
                conversation.session is not None
                and reply.calls
                and all(c["name"] == _REOPEN for c in reply.calls)
            ):
                rounds += 1
                if rounds > _ROUNDS:
                    break
                sent = dialect.after_tools(
                    sent, data, self._reopen(conversation, reply.calls)
                )
                again = call.send(sent)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                self._ledger(conversation, "reopen", reply)
            # A reply asked for a note that says nothing is asked again without the note first.
            if call.outgoing is not call.body and (
                rounds > _ROUNDS or not (call.asked or reply.text or reply.calls)
            ):
                # The model goes on fetching, or says nothing: what it was shown does not do for
                # this turn. It is asked once more, exactly as the application wrote it.
                again = self._as_written(call)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                self._ledger(conversation, "retry", reply)
            if any(c["name"] == _REOPEN for c in reply.calls):
                # Called beside the application's own tools: the application never sees ours.
                data = dialect.without_tool(data, _REOPEN)
        except Exception:
            logger.warning("compache: a reply was not read", exc_info=True)
        try:
            data, found = self._without_note(call, data)
        except Exception:
            logger.warning("compache: a reply was passed on as it came", exc_info=True)
            return result
        try:
            reply = dialect.reply(data)
            if call.outgoing is not call.marked and self._spoiled(call, reply, found):
                # The note got in the way of the answer: ask again without it.
                self.stats["retried"] += 1
                again = call.send(call.marked)
                if again.status < 400 and again.json is not None:
                    plain = replace(
                        call, outgoing=call.marked, asked=False, carrier=None
                    )
                    return self._settle(plain, again, "retry")
            if found is not None and call.asked:
                self._take_note(call, found)
            self._after(call, reply.text, reply.calls)
        except Exception:
            logger.warning("compache: a reply was not recorded", exc_info=True)
        # `raw` stays with a body that was not touched, so that it can go back byte for byte.
        return result if data is result.json else replace(result, json=data, raw=None)

    def _spoiled(self, call: _Call, reply, found) -> bool:
        """Whether a reply that was asked for a note is not what it would have been without.

        A label where the reader would see it, a property that could not be
        cut out, nothing left once the note is out, or an answer the output
        limit ended: the note took room the answer may have needed.
        """
        if call.carrier == "json":
            return f'"{notes_.KEY}"' in reply.text or (call.asked and reply.cut)
        if not call.asked:
            return False
        if found is not None and notes_.leaked(reply.text):
            self.stats["leaked"] += 1
            return True
        return reply.cut or not (reply.text or reply.calls)

    def _stream(self, call: _Call, result: Result) -> Iterator[bytes]:
        """A streamed reply passed on as it arrives, with our own tool calls answered on the way."""
        conversation, dialect = call.conversation, call.dialect
        tool = _REOPEN if conversation.session is not None else None
        editor = dialect.Stream(dialect, tool)
        sent, rounds, events, closers = call.outgoing, 0, result.events, [result.close]
        kind, counted = "answer", False
        try:
            while True:
                for raw, name, data in wire.sse_events(events):
                    yield from editor.feed(raw, name, data)
                if not editor.again or kind == "retry":
                    break
                rounds += 1
                with conversation.lock:
                    self._ledger(conversation, kind, editor)
                    if rounds > _ROUNDS:
                        # The model goes on fetching: it is asked as the application wrote it.
                        again, kind = self._as_written(call), "retry"
                    else:
                        answers = self._reopen(conversation, editor.ours)
                        sent = dialect.after_tools(sent, editor.said, answers)
                        again, kind = call.send(sent), "reopen"
                closers.append(again.close)
                if again.status >= 400 or again.events is None:
                    # No further round came: the one before it is already in the books.
                    counted = True
                    break
                events = again.events
                editor.next_round()
            yield from editor.end()
        finally:
            for close in closers:
                if close is not None:
                    close()
            with conversation.lock:
                try:
                    if not counted:
                        self._ledger(conversation, kind, editor)
                    if editor.finished:
                        self._after(call, editor.text, editor.calls)
                except Exception:
                    logger.warning(
                        "compache: a streamed reply was not recorded", exc_info=True
                    )

    def _as_written(self, call: _Call) -> Result:
        """Send the application's own request for a turn the policy's view did not do for.

        A conversation that needs this twice is left alone from then on: its
        users should not wait for fetches that lead nowhere.
        """
        conversation = call.conversation
        self.stats["retried"] += 1
        self.stats["fell_back"] += 1
        conversation.fell_back += 1
        call.asked, call.carrier = False, None
        result = call.send(call.body)
        if conversation.fell_back >= 2 and conversation.session is not None:
            pending = conversation.pending
            self._stand_down(conversation, pending.index, pending.started_at)
        return result

    def _reopen(self, conversation, calls) -> dict:
        """Answers to the model's `reopen` calls, by call ID."""
        answers = {}
        for call in calls:
            self.stats["reopened"] += 1
            arguments = call.get("arguments") or {}
            try:
                # An application may paste what it retrieved into its questions: search those too.
                found = conversation.session.reopen(
                    str(arguments.get("reference", "")),
                    int(arguments.get("offset") or 0),
                    search=("retrieval", "input", "output"),
                )
            except Exception as error:
                found = {"status": "error", "message": str(error)}
            pending = conversation.pending
            if pending is not None and found.get("status") == "ok":
                part = found.get("part_id")
                if part and part not in pending.reopened:
                    pending.reopened.append(part)
            answers[call["id"]] = json.dumps(_for_model(found), ensure_ascii=False)
        return answers

    def _after(self, call: _Call, text: str, calls) -> None:
        """A reply has been read: the turn goes on with tool calls, or ends with this answer."""
        conversation, pending = call.conversation, call.conversation.pending
        if pending is None:
            return
        pending.calls += 1
        if calls:
            # The application runs its tools and comes back with their results.
            return
        current = call.turns[-1]
        closed = replace(
            current,
            outputs=[*current.outputs, _Said(text)] if text else current.outputs,
        )
        if pending.asked:
            self.stats["asked"] += 1
        self._finish(
            conversation,
            _link(call.chains[-1], closed),
            closed,
            self._now(conversation),
        )
        # A ping is built from the last request, which is kept only while one is planned.
        planned = conversation.session is not None and conversation.session.next_ping_at
        conversation.last = (
            (call.dialect, call.plain, text, call.send) if planned else None
        )
        if call.apart and not self._notes_off:
            self.stats["asked"] += 1
            worker = threading.Thread(
                target=self._ask_apart, args=(call, closed, pending), daemon=True
            )
            self._apart = [t for t in self._apart if t.is_alive()] + [worker]
            worker.start()

    def _ask_apart(self, call: _Call, closed: _Segment, pending: _Pending) -> None:
        """Ask in a request of its own what the reply needed, and add the answer to the record."""
        conversation, dialect = call.conversation, call.dialect
        try:
            listing = self._listing(conversation, call.turns, pending)
            text = closed.outputs[-1].text if closed.outputs else ""
            # In plain text, whatever form the application asked its own answer in.
            body = dialect.follow_up(
                dialect.without_schema(call.marked),
                text,
                notes_.instruction(listing, carrier="apart"),
                _NOTE_TOKENS,
            )
            if body is None:
                return
            reply = self._whole(dialect, call.send(body))
            if reply is None:
                self._note_refused(call)
                return
            _, found = notes_.read(reply.text)
            with conversation.lock:
                self._ledger(conversation, "note", reply, pending.index)
                if found is None:
                    return
                pending.asked = True
                self._take_note(call, found, pending)
                self._write_turn(conversation, pending, closed, pending.ended_at)
        except Exception:
            logger.warning("compache: a note was not asked", exc_info=True)

    @staticmethod
    def _whole(dialect, result: Result):
        """The reply to a request of our own, read in full. None if the provider refused it.

        It comes as JSON, or as a stream where the path always streams: that is read to its end.
        """
        if result.status >= 400:
            return None
        if result.json is not None:
            return dialect.reply(result.json)
        if result.events is None:
            return None
        editor = dialect.Stream(dialect)
        try:
            for raw, name, data in wire.sse_events(result.events):
                editor.feed(raw, name, data)
        finally:
            if result.close is not None:
                result.close()
        return editor

    def drain(self, timeout: float | None = None) -> None:
        """Wait for notes that are being asked apart."""
        for worker in self._apart:
            worker.join(timeout)

    def _finish(self, conversation, chain, segment, now) -> None:
        """Close the pending turn: tell the policy, write the record, and index what was said."""
        pending = conversation.pending
        pending.ended_at = max(now, pending.started_at)
        if conversation.session is not None and pending.turn is not None:
            visible = {
                p.part_id
                for p in pending.turn.parts
                if p.kind == "retrieval" and p.part_id not in pending.turn.archived_ids
            }
            try:
                conversation.session.complete(
                    pending.turn,
                    segment.answer,
                    at=pending.ended_at,
                    usage=pending.first_usage,
                    last_usage=pending.last_usage if pending.calls > 1 else None,
                    cited_part_ids=[i for i in pending.cited if i in visible],
                )
            except Exception:
                logger.warning(
                    "compache: a turn was not taken by the policy", exc_info=True
                )
                self.stats["stood_down"] += 1
                conversation.session = None
        self._write_turn(conversation, pending, segment, pending.ended_at)
        self.stats["turns"] += 1
        conversation.turns, conversation.pending = pending.index, None
        self._index(conversation, chain)

    def _write_turn(self, conversation, pending, segment, ended_at) -> None:
        if self.recorder is None:
            return
        self.recorder.turn(
            conversation.record_id,
            pending.index,
            started_at=pending.started_at,
            ended_at=ended_at,
            model=pending.model,
            sizes={
                "input": _size(segment.question),
                "retrievals": [_size(s.text) for s in segment.retrievals],
                "output": _size(segment.answer),
                "system": conversation.system_bytes,
                "tools": conversation.tools_bytes,
            },
            usage=_turn_usage(pending.first_usage, pending.last_usage),
            read_reported=cache_read_reported(pending.first_usage),
            calls=max(1, pending.calls),
            asked=pending.asked,
            refs=self._named(pending),
            older=pending.older,
        )

    @staticmethod
    def _named(pending) -> list | None:
        """What a turn's note named, and what its reply fetched back: it needed that too."""
        if pending.refs is None or not pending.reopened:
            return pending.refs
        fetched = notes_.labels_of(notes_.Note(needs=tuple(pending.reopened)))
        return list(dict.fromkeys([*pending.refs, *fetched]))

    def _ledger(self, conversation, kind, source, turn=None) -> None:
        """Count one model call. `source` has the `usage`, `cost` and `model` of its reply."""
        pending = conversation.pending
        if pending is not None and kind in ("answer", "reopen", "retry"):
            if pending.first_usage is None:
                pending.first_usage = source.usage
            pending.last_usage = source.usage
            pending.model = source.model or pending.model
        if self.recorder is None:
            return
        archived = None
        if turn is None:
            turn = pending.index if pending is not None else conversation.turns
            if pending is not None and pending.turn is not None and kind != "ping":
                gone = sorted(pending.turn.archived_ids, key=_position)
                archived = notes_.labels_of(notes_.Note(needs=tuple(gone)))
        self.recorder.call(
            conversation.record_id,
            turn,
            kind,
            at=self.clock(),
            model=source.model,
            usage=canonical_usage(source.usage),
            cost=source.cost,
            archived=archived,
        )

    # -- pings ---------------------------------------------------------------

    def next_ping_at(self) -> float | None:
        """When the earliest keep-warm ping of any conversation is due, or None."""
        with self._lock:
            waiting = [c for c in self._by_chain.values() if self._may_ping(c)]
        due = [c.session.next_ping_at for c in waiting]
        return min((at for at in due if at is not None), default=None)

    @staticmethod
    def _may_ping(conversation) -> bool:
        return (
            conversation.session is not None
            and conversation.last is not None
            and conversation.pending is None
            and not conversation.pings_off
        )

    def tick(self, now: float | None = None) -> int:
        """Send the keep-warm pings that are due. Returns how many went out."""
        now = self.clock() if now is None else now
        with self._lock:
            candidates = [c for c in self._by_chain.values() if self._may_ping(c)]
        sent = 0
        for conversation in candidates:
            if not conversation.lock.acquire(blocking=False):
                continue
            try:
                sent += self._ping(conversation, now)
            except Exception:
                logger.warning("compache: a keep-warm ping failed", exc_info=True)
                conversation.pings_off = True
            finally:
                conversation.lock.release()
        return sent

    def _ping(self, conversation, now) -> int:
        session = conversation.session
        if not self._may_ping(conversation):
            return 0
        due = session.next_ping_at
        if due is None or due > now or session.ping(at=now) is None:
            return 0
        dialect, plain, answer, send = conversation.last
        body = dialect.follow_up(plain, answer, ".", 1)
        if body is None:
            conversation.pings_off = True
            return 0
        reply, usage = self._whole(dialect, send(body)), None
        if reply is not None:
            usage = reply.usage
            self._ledger(conversation, "ping", reply)
        else:
            conversation.pings_off = True
        session.ping_sent(at=now, usage=usage, completed_at=max(now, self.clock()))
        self.stats["pings"] += 1
        return 1
