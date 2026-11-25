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
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

from . import notes as notes_
from . import wire
from .api import Policy
from .learn import Learner
from .models import binds_reasoning
from .optimizer.parts import explained, notice
from .optimizer.recovery import REOPEN_TOOL
from .quotes import Quotes, spoken
from .record import Recorder, left_out
from .threads import (
    cache_read_reported,
    canonical_usage,
    long_writes,
    reasoning_tokens,
    tagged,
)

__all__ = ["Live", "Result", "Shown"]

logger = logging.getLogger(__name__)

_REOPEN = REOPEN_TOOL["function"]["name"]
# The tool as a model in a live conversation is told about it: what a notice is, and what to do with one.
_REOPEN_TOOL = {
    "type": "function",
    "function": {
        "name": _REOPEN,
        "description": (
            "Read text of this conversation again that was cleared to save space. A cleared part "
            "shows a short notice with a reference such as ctx:1a2b3c in place of its text. Give "
            "that reference to read the part, or give a short distinctive phrase to search every "
            "cleared part for it. Use it when a question needs something that is no longer visible, "
            "and before telling the user that you do not have something. What you answered before "
            "came from the full text: do not read it again to check that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": (
                        "The ctx: reference from a notice, or a few distinctive words to search "
                        "the cleared parts for."
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
    "Earlier parts of this conversation are cleared after a while to save space. You read each "
    "of them in full when it came in, and what you answered then came from it: rely on your "
    "earlier answers, and do not take them back or check them because the text is no longer "
    f"shown. A cleared part shows a notice like {notice('ctx:...')} in place of its text, and the "
    "reopen tool shows the text again. When a question needs something that is not visible, and "
    "always before you tell the user that you do not have or cannot find something, call reopen: "
    "with the reference from a notice to read that part, or with a few distinctive words to "
    "search all cleared parts. Then answer from what it returns. Never mention notices, cleared "
    "text or the reopen tool to the user."
)
_NOT_FOUND = (
    "No cleared part matches that. Try other or fewer distinctive words from the text you "
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
# The same for an agent, which has it from its first call on and knows what a tool is for.
_AGENT_TOOL = {
    "type": "function",
    "function": {
        "name": _REOPEN,
        "description": (
            "Old tool results are cleared from this conversation to save space, some steps after "
            "you read them. In the place of a cleared result you see a notice with a reference "
            "such as ctx:1a2b3c. You did read that result in full, and what you wrote right after "
            "it comes from it: rely on that, and do not reopen results to check it. Call this "
            "tool with the reference only when you need something from a result that you did "
            "not write down, rather than running the other tool again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "The ctx: reference of a notice, or a few distinctive words to search for.",
                },
                "offset": {
                    "type": "integer",
                    "description": "For a long result, the next_offset of the last answer.",
                },
            },
            "required": ["reference"],
            "additionalProperties": False,
        },
    },
}
_HANDLE = re.compile(r"ctx:[0-9a-f]{24}")
_REFUSED = (400, 422)
# What Claude's cache marks ask for where they name no lifetime.
_FIVE_MINUTES = 300
# How often in a row the model may fetch something back within one call of the application's.
_ROUNDS = 3
_AGENT_ROUNDS = 8
# How many of the newest tool results are looked at to find the loop a request goes on with.
_LOOK_BACK = 64
# The ID of a tool call that is found once. Others count, and two conversations may have the same.
_ONE_OF_A_KIND = re.compile(
    r"(?=[^\W_]*[0-9])(?=[^\W_]*[A-Za-z])(?<![A-Za-z0-9])[A-Za-z0-9]{16,}"
    r"|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)
# How far a message may have moved in a conversation for what was put in behind it to follow.
_MOVED = 4
# How many conversations that stand at the same point, or share a call's ID, are told apart.
_MOST_AT_ONE_POINT = 16
# How many turns back a conversation is looked for that another model went on from.
_BACK = 8
# A note is one line. The room is for a model that reasons before it writes: on Claude Haiku
# 5.5 that took a median of some 500 tokens, and more than 1,000 for one note in twelve.
_NOTE_TOKENS = 2048
# What stands in for the further pieces of a part whose first piece carries the notice.
_GONE = "…"
# What stands for the text of a result that holds pictures and says nothing.
_WORDLESS = "(no text)"


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
    def taken(self) -> Any:
        """As the policy's session takes it."""
        first = self.slots[0]
        if first.stays or first.whole is not None:
            more = {"stays": first.stays, "more": first.whole is not None}
            return {"text": self.text, **more}
        return self.text

    @property
    def text(self) -> str:
        said = "\n\n".join(s.text for s in self.slots)
        return said or (_WORDLESS if self.slots[0].whole is not None else said)

    @property
    def answers(self) -> bool:
        """Whether a tool returned it to the model that called it."""
        return self.slots[0].answers


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


@dataclass(frozen=True)
class _Insert:
    """What the model fetched back while answering, as messages of ours in the conversation."""

    # Behind which of the application's messages it stands: its place, and what it is known by.
    at: int
    behind: str | None
    # The API its messages are written for.
    dialect: str
    messages: list
    # What it goes by in the policy's session, and the turn it was fetched in.
    names: tuple
    turn: int


@dataclass(frozen=True)
class _Point:
    """How a conversation stood when a request came, for another that goes on from there."""

    chain: str
    # Its session and the turn under way as they were then. Both are copied before they are used.
    session: Any
    pending: Any
    inserts: tuple
    reopens: bool
    # The model its requests named, and the size of what opens them.
    requested: str | None
    opens: tuple


@dataclass
class _Pending:
    """The turn that is being answered."""

    index: int
    started_at: float
    ended_at: float = 0.0
    turn: Any = None
    question: str = ""
    # What its tools have returned so far, each by a hash of its text.
    seen: list = field(default_factory=list)
    calls: int = 0
    # Requests of it that went out and have no reply yet, and when the last one went out.
    in_flight: int = 0
    sent_at: float = 0.0
    # Whether the latest of them has been answered, and the tool calls that answer made.
    answered: bool = False
    called: tuple = ()
    # Whether the latest was answered to the application's own request, with everything in view.
    whole: bool = False
    # Its calls one by one, for the record.
    steps: list = field(default_factory=list)
    # Bytes of text compache left out of its last request, less what it added, and what it held.
    left: int = 0
    held: int = 0
    # Whether its last request holds pictures or the like, whose tokens have no text, and how many
    # tokens of them were left out of it.
    media: bool = False
    left_more: int = 0
    # Whether anything was out of view in any of its requests.
    cut: bool = False
    # The message that will carry the reply, where that holds reasoning.
    place: int | None = None
    # Where in its last request a cache mark belongs: behind what stays as it is.
    marks: tuple = ()
    first_usage: Any = None
    last_usage: Any = None
    model: str | None = None
    asked: bool = False
    refs: list | None = None
    older: bool = False
    cited: tuple = ()
    # Parts the model fetched back while answering.
    reopened: list = field(default_factory=list)

    def copy(self) -> _Pending:
        return replace(
            self,
            seen=list(self.seen),
            steps=list(self.steps),
            reopened=list(self.reopened),
            in_flight=0,
        )

    def goes_on_from(self, earlier: _Pending) -> None:
        """Take over what an earlier state of the same turn has counted."""
        for name in (
            "started_at",
            "calls",
            "steps",
            "first_usage",
            "last_usage",
            "cut",
        ):
            setattr(self, name, getattr(earlier, name))


@dataclass
class _Conversation:
    # The name its policy session and its parts go by, and the name it is recorded under.
    thread_id: str
    chain: str
    record_id: str = ""
    # The name its requests gave, where they gave one.
    named: str | None = None
    session: Any = None
    # The cache lifetime its requests ask for, where the policy was fitted for another than the default.
    lifetime: float | None = None
    # The same for its instructions and tools alone.
    header_lifetime: float | None = None
    # Whether it is planned under a cache that lives five minutes.
    five_minutes: bool = False
    # Whether it can only be planned under the default, and whether its session still has to be made anew for that.
    default_lifetime: bool = False
    stale: bool = False
    turns: int = 0
    pending: _Pending | None = None
    lock: Any = field(default_factory=threading.RLock)
    # What a ping is built from: the dialect, the last request without mark or note, its answer, how to send.
    last: tuple | None = None
    pings_off: bool = False
    # Whether the model has the `reopen` tool: from the first archived part on.
    reopens: bool = False
    # Whether its policy decides call by call.
    agentic: bool = False
    # The model its requests name.
    requested: str | None = None
    # The model it began with and the policy that gave it, where policies are learned.
    model: str | None = None
    policy: Any = None
    # Whether it is one of those the learner leaves as they are.
    left_alone: bool = False
    # What was fetched back and still stands in the conversation, in order.
    inserts: list = field(default_factory=list)
    # What was ever fetched back, and from where on.
    fetched: set = field(default_factory=set)
    # How it stood before its latest turn and at its latest request, for one that branches off there.
    base: _Point | None = None
    here: _Point | None = None
    # Whether another went on from where it stands, so that nothing is paid to keep it warm.
    superseded: bool = False
    # What went out of view with the request that is being made, and what goes first with the next cut.
    went: list = field(default_factory=list)
    holds: tuple | None = None
    # The lines its tools returned, where replies are read for what they quote.
    quotes: Quotes | None = None
    # Whether its requests carry what the model reasoned before.
    carries: bool = False
    # Whether the model binds its reasoning to what stood ahead of it. What the latest request held
    # that it reasoned on, as hashes, and the reasoning that was made behind something else and stays out.
    binds: bool = False
    sent_as: tuple | None = None
    planned: tuple | None = None
    unbound: set = field(default_factory=set)
    # The model its replies name.
    answered: str | None = None
    # How often a turn had to be asked again as the application wrote it.
    fell_back: int = 0
    system_bytes: int = 0
    tools_bytes: int = 0
    # The latest time this conversation has seen.
    at: float = float("-inf")
    # When a call that was made elsewhere really happened: its start while it is prepared, then its end.
    when: float | None = None
    # When this process last sent the conversation this one takes up again, whole.
    sent_whole: float | None = None


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
    # Whether a note can be asked of this call at all.
    asks: bool = True
    # The turn it belongs to. Another may have begun by the time a stream has ended.
    pending: Any = None
    # What went out of it last, as long as the reply to that may hold reasoning.
    sent: Any = None
    # What reasoning had been given up before it and what had gone out, and the session, where that
    # gave up more for it.
    before: tuple = (frozenset(), None)
    kept: Any = None


class _Unsent(Exception):
    """A request of the application's own could not be sent: what is wrong is for it to hear."""


def _said_wrong(result: Result) -> dict:
    """What a provider says is wrong, as an object with a message."""
    found = result.json
    if found is None and result.raw:
        try:
            found = json.loads(result.raw)
        except ValueError:
            found = None
    if isinstance(found, dict) and isinstance(found.get("error"), dict):
        return found["error"]
    said = found.get("error") if isinstance(found, dict) else None
    return {"message": str(said or f"the provider answered {result.status}")}


class _Later:
    """A thread for what must not be done while something is collected.

    That happens in whatever thread comes by, which may hold a lock, or be the
    event loop that would have to do it.
    """

    def __init__(self):
        self._waiting: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    def ready(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="compache-later", daemon=True
            )
            self._thread.start()

    def do(self, action: Callable[[], None]) -> None:
        self._waiting.put(action)

    def _run(self) -> None:
        while True:
            action = self._waiting.get()
            # One may have to wait for its conversation, and the others not for that.
            made = threading.Thread(target=self._try, args=(action,), daemon=True)
            del action
            try:
                made.start()
            except RuntimeError:
                made.run()
            del made

    @staticmethod
    def _try(action: Callable[[], None]) -> None:
        try:
            action()
        except Exception:
            logger.debug("compache: a stream was not closed", exc_info=True)


_LATER = _Later()


@dataclass
class _Reading:
    """How far a streamed reply is: whether its events have begun, and whether somebody closes them."""

    begun: bool = False
    asked: bool = False


class _Passing:
    """A streamed reply on its way. One that is closed before it was read lets go of what it holds too."""

    def __init__(self, events, release: Callable[[], None], reading: _Reading):
        self._events, self._release, self._reading = events, release, reading
        self._wanted = self._over = False

    def __iter__(self) -> _Passing:
        return self

    def __next__(self) -> bytes:
        try:
            found = next(self._events)
        except BaseException:
            self._over = True
            raise
        if self._wanted:
            # Another thread closed it while this one was reading.
            self.close()
            raise StopIteration
        return found

    @staticmethod
    def _close(events, release, reading) -> bool:
        reading.asked = True
        try:
            events.close()
        except ValueError:
            # It is being read at this moment, in another thread.
            reading.asked = False
            return False
        if not reading.begun:
            # Once, whatever becomes of it.
            reading.begun = True
            release()
        return True

    def close(self) -> None:
        self._wanted = True
        if not self._over:
            # What is being read is closed by its reader, once that comes back.
            self._over = self._close(self._events, self._release, self._reading)

    def __del__(self):
        if not self._over:
            _LATER.do(partial(self._close, self._events, self._release, self._reading))


class Shown:
    """A request as the policy would send it, for a host that sends it itself and tells how it went."""

    def __init__(self, live, path, body, conversation=None, pending=None):
        self.body = body
        self._live, self._path = live, path
        self._conversation, self._pending = conversation, pending
        self._waits = False

    def begin(self) -> None:
        """It goes out now."""
        if self._conversation is not None:
            with self._conversation.lock:
                self._waits = True
                self._pending.in_flight += 1

    def over(self) -> None:
        """Its reply is in, or none will come."""
        if self._waits:
            with self._conversation.lock:
                self._waits = False
                self._pending.in_flight -= 1

    def heard(self, usage, *, model=None, said=None, calls=()) -> None:
        """What it was billed and what its reply said. `calls` are its tool calls, with `id` and `arguments`."""
        self.over()
        if self._conversation is not None:
            self._live._told(
                self._conversation, self._pending, usage, model, said, list(calls)
            )

    def conclude(self, body, *, thread: str) -> None:
        """Close the turn: `body` holds the conversation up to and including the answer."""
        self._live.conclude(self._path, body, thread=thread)


def _segments(slots, remarks: bool = True) -> list[_Segment]:
    """Turns of a request: a question opens one, and everything up to the next question is its.

    What a harness puts beside the results of tools, in a tag of its own, is a
    remark in the same turn, unless `remarks` is False.
    """
    turns: list[_Segment] = []
    waiting = []
    results = None
    for slot in slots:
        if slot.answers:
            results = slot.message
        if not slot.text and not slot.closes and slot.whole is None:
            continue
        if slot.kind == "input":
            remark = remarks and slot.message == results and tagged(slot.text)
            if not turns or (turns[-1].outputs and not remark):
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


def _rounds(found: list) -> list[list]:
    """Retrieved parts by the round of tool calls that returned them."""
    rounds: list[list] = []
    last = None
    for part in found:
        at = part.slots[0].message
        if last is None or at - last > 1:
            rounds.append([])
        rounds[-1].append(part)
        last = part.slots[-1].message
    return rounds


def _hashes(found: list) -> list[int]:
    return [hash(part.text) for part in found]


def _link(chain: str, segment: _Segment) -> str:
    # An application keeps an answer that streamed in as many pieces as it likes.
    answer = "".join("".join(s.text.split()) for s in segment.outputs if s.text)
    said = [segment.question, [s.text for s in segment.retrievals], answer]
    return hashlib.sha256((chain + json.dumps(said)).encode()).hexdigest()


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


# What our tool returned for a notice, as the model reads it.
_FETCHED = re.compile(
    r'\{"status": "ok", .{0,300}?"handle": "(ctx:[0-9a-f]{24})"', re.S
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
    # A policy that cut while the turn went on left less than the first call read.
    read = min(first["cache_read"], last["input"])
    return {
        "input": last["input"],
        "output": first["output"] + last["output"],
        "cache_read": read,
        "cache_write": max(0, left - read) if last["cache_write"] else 0,
    }


class Live:
    """A policy applied, a record kept and notes asked, on every model call that passes.

    `policy` is a `Policy`, a path to one, or None to change nothing and only
    record. `record` is a folder for a record without text, or a `Recorder`.
    `learn` is a folder instead of both: what passes is recorded there, and the
    policy follows from it, as `compache.learn` describes.
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
        learn=None,
        sample: float = 0.0,
        notes: str = "apart",
        salt: str = "",
        clock: Callable[[], float] = time.time,
        max_conversations: int = 4096,
    ):
        if learn is not None and not (policy is None and record is None):
            raise ValueError(
                "learn: the folder holds the record, and the policy follows from it"
            )
        if isinstance(policy, (str, Path)):
            policy = Policy.load(policy)
        if notes not in ("apart", "inline"):
            raise ValueError("notes: expected 'apart' or 'inline'")
        if not 0 <= sample <= 1:
            raise ValueError("sample: expected a share between 0 and 1")
        self.policy = policy
        self.learner = None
        if learn is not None:
            self.learner = (
                learn if isinstance(learn, Learner) else Learner(learn, clock=clock)
            )
            record = self.learner.folder
        self.recorder = Recorder(record) if isinstance(record, (str, Path)) else record
        if sample and self.recorder is None:
            raise ValueError(
                "sample: notes need a record to go to. Give `record` a folder"
            )
        self.sample = sample
        self.notes, self.salt, self.clock = notes, salt, clock
        recipe = policy.recipe if policy is not None and not policy.agentic else None
        if recipe is not None and recipe.cited_answer_runs != 0 and not sample:
            # Here an answer is known to cite a text only through its note.
            logger.warning(
                "compache: this policy keeps the text that answers cite, and no reply is asked "
                "what it cited. Give `sample` the rate at which the policy's record was made."
            )
        self._limit = max_conversations
        self._lock = threading.RLock()
        # Those that stand at the same point, the newest last.
        self._by_chain: OrderedDict[str, list[_Conversation]] = OrderedDict()
        # The same by how a conversation stood before its latest turn, and by its tool calls, whose
        # IDs need not be unique.
        self._by_base: OrderedDict[str, _Conversation] = OrderedDict()
        self._by_call: OrderedDict[str, list[_Conversation]] = OrderedDict()
        # The conversations that a keep-warm ping is planned for.
        self._warm: dict[int, _Conversation] = {}
        self._names: dict[str, int] = {}
        # The conversation a name was last seen in, for a host that tells us about it by name.
        self._latest: OrderedDict[str, _Conversation] = OrderedDict()
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
        # What the provider counted so far, and what was left out of it, in tokens.
        counted = ("input", "cache_read", "cache_write", "output", "left_out")
        self.tokens = dict.fromkeys(counted, 0)

    # -- conversations -------------------------------------------------------

    def _taken(self, thread: str | None, turns: list[_Segment], scan, closing=False):
        """The conversation a request continues, with its lock held, and the chain of what it has said."""
        for last in (False, False, False, True):
            found, chains = self._conversation(thread, turns, scan, closing)
            found.lock.acquire()
            try:
                # Another request may have come first while this one waited.
                if last or self._stands(found, chains, turns, scan, closing):
                    return found, chains
            except BaseException:
                found.lock.release()
                raise
            found.lock.release()
        raise AssertionError("not reached")

    def _conversation(
        self, thread: str | None, turns: list[_Segment], scan, closing: bool = False
    ) -> tuple[_Conversation, list[str]]:
        """The conversation a request continues, found by what it has said so far.

        `closing` says that the request is no new one: it ends on its answer.
        """
        seed = "thread:" + thread if thread else "auto"
        chains = [hashlib.sha256(seed.encode()).hexdigest()]
        for segment in turns[:-1]:
            chains.append(_link(chains[-1], segment))
        done = len(turns) - 1
        with self._lock:
            found = self._in_loop(scan.results, chains, turns)
            if found is not None and found.chain != chains[-1]:
                # It made those calls, but another may have given the answer that the request shows.
                here = self._by_chain.get(chains[-1], ())
                if any(self._own(c, turns, scan, closing) for c in here):
                    found = None
            others = [] if found is None else [found]
            if not others and (thread or done):
                others = self._by_chain.get(chains[-1], ())[::-1]
            own = [c for c in others if self._own(c, turns, scan, closing)]
            # One that is in the middle of something else comes last.
            own.sort(key=lambda c: not self._fits(c.pending, turns))
            found = own[0] if own else None
            for other in others if found is None else ():
                found = found or self._branch(other, chains[-1], turns, scan, thread)
            if found is None and done and (thread or done > 1):
                # Its last turn may have been answered without the answer passing here.
                for one in reversed(self._by_chain.get(chains[-2], ())):
                    pending = one.pending
                    if pending is not None and pending.index == done:
                        if self._goes_on(pending, turns[-2]):
                            found = found or one
            if found is None and done and chains[-1] in self._by_base:
                # It goes on from where another request, such as a host's own, branched off.
                other = self._by_base[chains[-1]]
                found = self._branch(other, chains[-1], turns, scan, thread)
            if found is None and thread:
                found = self._resumed(chains, scan)
            if found is None:
                name = record = thread or "c-" + uuid.uuid4().hex[:16]
                if thread:
                    # A name can be met again: after a restart, or shared by several conversations.
                    count = self._names[thread] = self._names.get(thread, 0) + 1
                    if count > 1:
                        name = f"{thread}#{count}"
                        record = thread if done else f"{thread}~{count}"
                found = _Conversation(name, chains[0], record, thread)
                earlier = self._latest.get(thread) if thread else None
                if earlier is not None and not earlier.reopens:
                    found.sent_whole = earlier.at
                self._start(found, name, scan)
                if thread:
                    # A named conversation is found again from its first call. An unnamed one only
                    # once it has an answer, since any two may open with the same words.
                    self._index(found, chains[0])
            elif found.chain in self._by_chain:
                self._by_chain.move_to_end(found.chain)
            if thread:
                self._latest[thread] = found
                self._latest.move_to_end(thread)
                while len(self._latest) > self._limit:
                    self._latest.popitem(last=False)
            return found, chains

    def _in_loop(self, results, chains, turns) -> _Conversation | None:
        """The conversation whose reply made the tool calls that a request answers."""
        done, near = len(turns) - 1, []
        for call_id in reversed(results[-_LOOK_BACK:]):
            for found in reversed(self._by_call.get(call_id, ()) if call_id else ()):
                # IDs need not be unique, so what was said has to agree.
                pending = found.pending
                if pending is None:
                    continue
                if pending.index == done + 1 and found.chain == chains[-1]:
                    if self._goes_on(pending, turns[-1]):
                        return found
                    if call_id in pending.called and found not in near:
                        # What was asked has been rendered anew. An ID that counts says little then.
                        if _ONE_OF_A_KIND.search(call_id):
                            near.append(found)
                elif pending.index == done and done and found.chain == chains[-2]:
                    if self._goes_on(pending, turns[-2]):
                        return found
        if len(near) == 1:
            # It answers the calls of this one and of no other: the application renders its turn anew.
            return near[0]
        if not any(results[-_LOOK_BACK:]):
            # An API without IDs: found by what its tools had returned when the reply came.
            asked, seen = turns[-1].question, _hashes(turns[-1].retrievals)
            for k in range(len(seen) - 1, -1, -1):
                key = self._loop(chains[-1], asked, seen[:k])
                for found in reversed(self._by_call.get(key, ())):
                    pending = found.pending
                    if pending is not None and found.chain == chains[-1]:
                        if pending.index == done + 1 and self._goes_on(
                            pending, turns[-1]
                        ):
                            return found
        return None

    @staticmethod
    def _loop(chain: str, question: str, seen: list) -> str:
        said = json.dumps([chain, question, seen]).encode()
        return "loop:" + hashlib.sha256(said).hexdigest()

    def _fits(self, pending: _Pending | None, turns: list[_Segment]) -> bool:
        """Whether nothing speaks against a turn being the one a request goes on with."""
        if pending is None or pending.index != len(turns):
            return True
        return self._goes_on(pending, turns[-1])

    @staticmethod
    def _goes_on(pending: _Pending, segment: _Segment) -> bool:
        """Whether a turn of a request is the one that is being answered, as far as that had come."""
        seen = _hashes(segment.retrievals[: len(pending.seen)])
        return seen == pending.seen and segment.question.startswith(pending.question)

    @staticmethod
    def _several(conversation: _Conversation) -> bool:
        """Whether several conversations may be under way where this one is.

        So it is with agents. A chat has one line, whatever its application
        renders anew from call to call.
        """
        return conversation.session is None or conversation.agentic

    def _own(self, conversation, turns, scan, closing=False) -> bool:
        """Whether a request that meets a conversation at this point is its next one."""
        pending, current = conversation.pending, turns[-1]
        if pending is None and (conversation.agentic or conversation.left_alone):
            # What another model is asked beside is no turn of this one, whose cache is kept.
            return not scan.model or conversation.requested in (None, scan.model)
        if pending is None or pending.index != len(turns):
            return True
        # Whether this answers the calls that were made last.
        answers = set(pending.called) & set(
            scan.results[-len(pending.called) or None :]
        )
        if not self._goes_on(pending, current):
            # What it had said was rendered anew, or it is somebody else's.
            return any(map(_ONE_OF_A_KIND.search, answers))
        if not (pending.calls or pending.in_flight):
            # Nothing has come of it yet.
            return True
        if closing or answers or len(current.retrievals) > len(pending.seen):
            return True
        # The same request again: for want of an answer, or of another that goes on from here.
        return not self._several(conversation) or not (
            pending.answered or pending.in_flight
        )

    def _stands(self, conversation, chains, turns, scan, closing) -> bool:
        """Whether a conversation still is where a request found it."""
        pending, done = conversation.pending, len(turns) - 1
        if pending is not None and pending.index == done:
            return conversation.chain == chains[-2] and self._goes_on(
                pending, turns[-2]
            )
        if conversation.chain == chains[-1]:
            return self._own(conversation, turns, scan, closing)
        if pending is None and 0 < conversation.turns < done:
            # One that is resumed.
            return conversation.chain == chains[conversation.turns]
        # One that is met for the first time, perhaps in the middle.
        return pending is None and not conversation.turns

    @staticmethod
    def _point(conversation, chain, pending=None) -> _Point | None:
        """How a conversation stands, for one that branches off. Only an agent's can be copied."""
        session = conversation.session
        if not (conversation.left_alone or (conversation.agentic and session)):
            return None
        return _Point(
            chain,
            session.copy() if session is not None else None,
            pending.copy() if pending is not None else None,
            tuple(conversation.inserts),
            conversation.reopens,
            conversation.requested,
            (conversation.system_bytes, conversation.tools_bytes),
        )

    @staticmethod
    def _opens(scan) -> tuple:
        tools = _size(json.dumps(scan.tools)) if scan.tools else 0
        return _size(scan.system), tools

    def _branch(
        self, other: _Conversation, chain: str, turns: list, scan, thread: str | None
    ) -> _Conversation | None:
        """A conversation that goes on from where another stood. None where that is not known."""
        done, point = len(turns) - 1, other.here
        if (
            point is None
            or point.pending is None
            or point.pending.index != len(turns)
            or not self._goes_on(point.pending, turns[-1])
            or point.opens != self._opens(scan)
        ):
            point = other.base
        if point is None or point.chain != chain:
            return None
        record = other.record_id
        if not done and thread:
            # Nothing was said before it: it is another conversation of the same name.
            count = self._names[thread] = self._names.get(thread, 0) + 1
            record = f"{thread}~{count}"
        made = _Conversation(other.thread_id, chain, record, other.named)
        made.agentic, made.left_alone = other.agentic, other.left_alone
        made.inserts, made.reopens = list(point.inserts), point.reopens
        made.fetched = set(other.fetched)
        made.lifetime, made.five_minutes = other.lifetime, other.five_minutes
        made.header_lifetime = other.header_lifetime
        made.default_lifetime = other.default_lifetime
        made.model, made.policy = other.model, other.policy
        made.binds, made.sent_as = other.binds, other.sent_as
        made.unbound, made.answered = set(other.unbound), other.answered
        made.requested = point.requested
        made.turns, made.at = done, other.at
        if point.session is not None and other.session is not None:
            made.session = point.session.copy()
            if other.requested == point.requested:
                made.session.follows(other.session)
            if scan.model and point.requested not in (None, scan.model):
                # It has a cache of its own, and nothing of this is in it.
                made.session.lost()
        made.requested = scan.model or point.requested
        if other.pending is None and scan.model in (None, "", other.requested):
            # What they share is kept warm for the newest of them. Another model has a cache of its own.
            other.superseded = True
        self._remember(made, turns[:done])
        if point.pending is not None:
            made.pending = point.pending.copy()
            made.base, made.here = other.base, point
            found = turns[-1].retrievals[: len(point.pending.seen)]
            self._heard(made, len(turns), point.pending.question, found)
        if thread or done:
            self._index(made, chain)
        return made

    def _resumed(self, chains: list[str], scan) -> _Conversation | None:
        """An agent's conversation as a model left it, for a request that comes back to that model.

        Another went on from there, and what it made of it is not in this one's cache.
        """
        for done in range(len(chains) - 2, max(0, len(chains) - 2 - _BACK), -1):
            for other in reversed(self._by_chain.get(chains[done], ())):
                if (
                    not (other.agentic and scan.model)
                    or other.requested != scan.model
                    or not other.lock.acquire(blocking=False)
                ):
                    continue
                try:
                    here = other.chain == chains[done] and other.turns == done
                    if here and other.pending is None and other.session is not None:
                        return self._left_off(other)
                finally:
                    other.lock.release()
        return None

    @staticmethod
    def _left_off(other: _Conversation) -> _Conversation:
        """A conversation like another, between its turns. Its lock is held."""
        made = replace(
            other,
            lock=threading.RLock(),
            session=other.session.copy(),
            inserts=list(other.inserts),
            fetched=set(other.fetched),
            unbound=set(other.unbound),
            went=[],
            quotes=None,
            last=None,
            base=None,
            here=None,
            when=None,
        )
        made.superseded, other.superseded = False, True
        return made

    def conversations(self) -> list[_Conversation]:
        """Those that are known here, each once."""
        found: dict[int, _Conversation] = {}
        with self._lock:
            for some in (*self._by_chain.values(), *self._by_call.values()):
                found.update((id(c), c) for c in some)
        return list(found.values())

    def _now(self, conversation: _Conversation) -> float:
        """The time, and never one this conversation has already passed: a clock can step back."""
        now = self.clock() if conversation.when is None else conversation.when
        conversation.at = max(now, conversation.at + 1e-3)
        return conversation.at

    def _index(self, conversation: _Conversation, chain: str) -> None:
        with self._lock:
            left = [
                c
                for c in self._by_chain.pop(conversation.chain, ())
                if c is not conversation
            ]
            if left:
                self._by_chain[conversation.chain] = left
            conversation.chain = chain
            others = [c for c in self._by_chain.pop(chain, ()) if c is not conversation]
            self._by_chain[chain] = [*others[1 - _MOST_AT_ONE_POINT :], conversation]
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
        return self._exchange(path, body, send, thread)

    def observe(
        self,
        path: str,
        body: Any,
        reply: Any,
        *,
        thread: str | None = None,
        at: tuple[float, float] | None = None,
        ask: Callable[[Any], Result] | None = None,
    ) -> None:
        """Keep the books on a model call that something else made.

        `body` is the request and `reply` the response, both as JSON. `at` is
        when the call went out and when its reply was in, as timestamps. `ask`
        performs a request of our own, such as a note's, and returns a
        `Result`; without it no note is asked. There is nothing to apply a
        policy to, so a `Live` that observes has none.
        """
        if self.policy is not None or self.learner is not None:
            raise ValueError(
                "observe: a call that was already made has no request to change"
            )

        def send(payload):
            if payload is body:
                return Result(200, json=reply)
            return ask(payload) if ask is not None else Result(501)

        self._exchange(path, body, send, thread, at, ask is not None)

    # -- a host that makes the requests itself ---------------------------------

    def view(self, path: str, body: Any, *, thread: str) -> Any:
        """The request as the policy would send it, for a host that sends it itself.

        Such a host, an agent's context engine for one, gives the application a say
        in what a request holds and in nothing else. Parts of earlier turns the
        policy archived come back as their notice, and how to fetch them back joins
        the system prompt. `heard` tells what the provider billed, `reopen` answers
        the model's calls, and `conclude` closes the turn. `thread` names the
        conversation. The request itself is left as it was.
        """
        return self._view(path, body, thread).body

    def show(self, path: str, body: Any, *, thread: str) -> Shown:
        """The same for a host with several agents under one name: what comes back is told of its own reply."""
        shown = self._view(path, body, thread)
        shown.begin()
        return shown

    def _view(self, path: str, body: Any, thread: str) -> Shown:
        if not isinstance(body, dict):
            return Shown(self, path, body)
        dialect = wire.dialect_for(path)
        try:
            scan = dialect.scan(body)
            turns = self._turns(scan)
            if not turns or not scan.awaiting:
                return Shown(self, path, body)
            conversation, chains = self._taken(thread, turns, scan)
        except Exception:
            self.stats["passed_through"] += 1
            return Shown(self, path, body)
        try:
            try:
                call = self._prepare(
                    conversation, chains, dialect, body, scan, turns, None, asks=False
                )
            except Exception:
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                self.stats["passed_through"] += 1
                return Shown(self, path, body)
            self.stats["calls"] += 1
            if conversation.binds and conversation.session is not None:
                # The host does not say whether its reply holds reasoning.
                conversation.sent_as = conversation.planned or wire.shape(call.plain)
            return Shown(self, path, call.plain, conversation, conversation.pending)
        finally:
            conversation.lock.release()

    def heard(
        self,
        thread: str,
        usage: Any,
        *,
        model: str | None = None,
        said: str | None = None,
        calls=(),
    ) -> None:
        """What one model call of a conversation shown to `view` was billed, as the provider counts it.

        `said` is the reply's text and `calls` the arguments of its tool calls,
        where the host has them at hand.
        """
        conversation = self._latest.get(thread)
        if conversation is not None:
            made = [{"arguments": arguments} for arguments in calls]
            self._told(conversation, conversation.pending, usage, model, said, made)

    def _told(self, conversation, pending, usage, model, said, calls) -> None:
        """Keep the books on a call that a host made itself."""
        with conversation.lock:
            if pending is None or pending is not conversation.pending:
                return
            pending.calls += 1
            pending.answered = True
            try:
                self._ledger(
                    conversation, "answer", wire.Reply(usage=usage, model=model)
                )
                self._step(conversation, pending, said, calls)
                keys = tuple(c["id"] for c in calls if c.get("id"))
                self._register(conversation, keys)
                pending.called = keys
                if conversation.session is not None and pending.turn is not None:
                    conversation.session.replied(
                        pending.turn,
                        said or "",
                        [c.get("arguments") for c in calls],
                        at=self._now(conversation),
                        usage=usage,
                    )
            except Exception:
                logger.warning("compache: a call was not recorded", exc_info=True)

    def reopen(
        self, thread: str, reference: str, offset: Any = 0, *, call: str | None = None
    ) -> dict:
        """The answer to the model's `reopen` call in a conversation shown to `view`, as it is to be read.

        `call` is the ID of that call, where several conversations go by one name.
        """
        with self._lock:
            made = [c for c in self._by_call.get(call or "", ()) if c.named == thread]
            latest = self._latest.get(thread)
        if not made and call:
            # It is not known whose call that is.
            made = [c for c in self.conversations() if c.named == thread]
        made = made or ([latest] if latest is not None else [])
        made = [c for c in made if c.session is not None]
        conversation = made[-1] if len(made) == 1 else None
        if len(made) > 1 and reference.strip().startswith("ctx:"):
            # An ID that counts is that of several. A notice tells whose it is, and words do not.
            conversation = next(
                (c for c in reversed(made) if self._archived(c, reference)), None
            )
        if conversation is None:
            said = "this does not say which notice is meant: give its ctx: reference"
            if not made:
                said = "there is nothing archived in this conversation"
            return {"status": "error", "message": said}
        with conversation.lock:
            answers = self._reopen(
                conversation,
                [
                    {
                        "id": "reopen",
                        "arguments": {"reference": reference, "offset": offset},
                    }
                ],
            )
        return json.loads(answers["reopen"])

    @staticmethod
    def _archived(conversation, reference: str) -> bool:
        """Whether a notice's reference is one of a conversation's."""
        with conversation.lock:
            try:
                found = conversation.session.reopen(
                    reference, search=("retrieval", "input", "output")
                )
            except Exception:
                return False
            return found.get("status") == "ok"

    def conclude(self, path: str, body: Any, *, thread: str) -> None:
        """Close the turn that `body` ends on, where the host's own reply ended it.

        `body` holds the conversation up to and including that answer. A
        conversation that was never shown to `view` is not recorded, and a
        turn that still waits for a tool goes on.
        """
        try:
            dialect = wire.dialect_for(path)
            scan = dialect.scan(body)
            turns = self._turns(scan)
            if not turns or not turns[-1].outputs or dialect.waits_for_tools(body):
                return
            conversation, chains = self._taken(thread, turns, scan, True)
            try:
                now = self._now(conversation)
                self._catch_up(conversation, chains, turns, now, scan)
                pending = conversation.pending
                if pending is not None and pending.index == len(turns):
                    self._finish(
                        conversation, _link(chains[-1], turns[-1]), turns[-1], now
                    )
            finally:
                conversation.lock.release()
        except Exception:
            logger.warning("compache: a turn was not recorded", exc_info=True)

    # -- the cache lifetime --------------------------------------------------

    def _start(self, conversation, name, scan) -> Any:
        """Apply the policy to a conversation from here on. Returns it, or None where there is none."""
        policy = self.policy
        if self.learner is not None:
            # Chosen once. One that was left alone stays alone.
            if conversation.model is None:
                conversation.model = scan.model or ""
                conversation.policy, conversation.left_alone = self.learner.choose(
                    scan.model, bool(scan.tools)
                )
            policy = conversation.policy
        if policy is None:
            return None
        if conversation.default_lifetime:
            policy = policy.at_default_lifetime()
        conversation.session = policy.session(name)
        conversation.session.binds(conversation.binds)
        conversation.agentic = policy.agentic
        conversation.lifetime = policy.cache_lifetime
        conversation.header_lifetime = policy.header_lifetime
        conversation.five_minutes = policy.spec.cache_ttl_seconds == _FIVE_MINUTES
        return policy

    @staticmethod
    def _asking(conversation, dialect, body):
        """A request made here, asking for the cache lifetime the conversation is planned under."""
        if conversation.session is None:
            return body
        if conversation.lifetime is not None:
            return wire.with_lifetime(body, conversation.lifetime)
        if conversation.five_minutes:
            body = wire.with_default_lifetime(body)
        if conversation.header_lifetime is not None:
            body = dialect.with_header_lifetime(body, conversation.header_lifetime)
        return body

    @staticmethod
    def _plan_default(conversation) -> None:
        """Plan a conversation under the provider's default lifetime, from its next turn on."""
        conversation.default_lifetime = conversation.stale = True
        conversation.lifetime = conversation.header_lifetime = None

    def _lifetime_refused(self, conversation) -> None:
        """The provider takes a request only under its default lifetime: so is everything planned from now on."""
        logger.warning(
            "compache: cache entries of %g seconds are refused, "
            "planning on the provider's default from now on",
            conversation.lifetime or conversation.header_lifetime,
        )
        self._plan_default(conversation)
        if self.learner is not None:
            self.learner.default_lifetime = True
        elif self.policy is not None:
            self.policy = self.policy.at_default_lifetime()

    def _exchange(self, path, body, send, thread, at=None, asks=True) -> Result:
        self.stats["calls"] += 1
        dialect = wire.dialect_for(path)
        try:
            scan = (
                dialect.scan(body)
                if dialect is not None and isinstance(body, dict)
                else None
            )
            turns = self._turns(scan) if scan is not None else []
            if not turns or not scan.awaiting:
                raise ValueError("nothing here to decide on")
            scan.stream = scan.stream or wire.streams(path)
            scan.model = scan.model or wire.model_of(path)
            conversation, chains = self._taken(thread, turns, scan)
        except Exception:
            # A shape this does not know: the application's request goes out as it is.
            self.stats["passed_through"] += 1
            return send(body)
        try:
            conversation.when = at[0] if at is not None else None
            try:
                if conversation.lifetime is not None and not wire.has_cache_mark(
                    dialect.with_cache_mark(body)
                ):
                    # Nothing in this request could ask for it.
                    self._plan_default(conversation)
                call = self._prepare(
                    conversation, chains, dialect, body, scan, turns, send, asks
                )
            except Exception:
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                self.stats["passed_through"] += 1
                return send(body)
            if at is not None:
                conversation.when = at[1]
            return self._carry(call, scan)
        finally:
            conversation.when = None
            conversation.lock.release()

    def _turns(self, scan) -> list[_Segment]:
        """The turns of a request. What a harness says beside its results is a remark only to an agent."""
        policy = self.policy
        return _segments(scan.slots, policy is None or policy.agentic)

    def _prepare(
        self, conversation, chains, dialect, body, scan, turns, send, asks=True
    ) -> _Call:
        """Bring the conversation up to this request and work out what to send for it."""
        now, number, current = self._now(conversation), len(turns), turns[-1]
        conversation.system_bytes, conversation.tools_bytes = self._opens(scan)
        conversation.superseded = False
        plain, kept = body, None
        before = frozenset(conversation.unbound), conversation.sent_as
        try:
            conversation.went = []
            session = conversation.session
            same = not scan.model or conversation.requested in (None, scan.model)
            warm = same and session is not None and session.warm(now)
            conversation.binds = self._binds(
                dialect, body, scan.model, conversation.answered
            )
            if session is not None:
                session.binds(conversation.binds)
                if conversation.agentic:
                    session.carries(scan.thoughts)
            self._catch_up(conversation, chains, turns, now, scan)
            pending, found = conversation.pending, current.retrievals
            earlier = None
            if (
                pending is not None
                and pending.index == number
                and self._several(conversation)
                and _hashes(found[: len(pending.seen)]) != pending.seen
            ):
                # The application rewrote what its tools had returned: start over from what it says now.
                earlier, pending = pending, None
                if conversation.session is not None:
                    self._anew(conversation, chains, turns, now, scan)
            if pending is None or pending.index != number:
                pending = self._begin(conversation, chains, current, number, now, scan)
                if earlier is not None:
                    pending.goes_on_from(earlier)
            else:
                self._asks(conversation, scan)
                if len(found) > len(pending.seen):
                    late = found[len(pending.seen) :]
                    self._feed(conversation, pending, late, now, scan)
                # A chat may render what was found before anew.
                pending.seen = _hashes(found[: len(pending.seen)])
            if (
                send is not None
                and conversation.session is not None
                and _REOPEN in dialect.tool_names(body)
            ):
                raise ValueError(
                    f"the application has a tool of its own named {_REOPEN}"
                )
            conversation.last = None
            conversation.carries = conversation.carries or bool(scan.thoughts)
            pending.sent_at, pending.answered, pending.whole = now, False, False
            if conversation.session is not None:
                if self.learner is not None and conversation.agentic:
                    if not warm:
                        # Nothing is there to be read, so what opens it may as well be kept for longer.
                        conversation.header_lifetime = self.learner.header_lifetime(
                            conversation.model, now
                        )
                    self.learner.opened(conversation.model, now)
                plain = self._apply(conversation, dialect, body, scan, turns)
                conversation.planned = None
                holds = scan.thoughts or conversation.inserts
                if holds and self._unbind(conversation, dialect, body, plain):
                    held = dialect.said_so_far(body)
                    kept = conversation.session.copy()
                    conversation.session.unbound(
                        [
                            place
                            for place in scan.thoughts
                            if dialect.reasoned(held[place]) in conversation.unbound
                        ]
                    )
                    conversation.holds = conversation.session.holds
                    plain = self._apply(conversation, dialect, body, scan, turns)
                    self._unbind(conversation, dialect, body, plain)
            conversation.here = self._point(conversation, conversation.chain, pending)
        except Exception:
            # The policy cannot be applied here. From now on this conversation is only recorded.
            logger.warning(
                "compache: policy not applied to a conversation", exc_info=True
            )
            self._stand_down(conversation, number, now)
        if conversation.session is None:
            body = plain = self._written(conversation, dialect, body)
        marked = plain
        if conversation.session is not None:
            marked = dialect.with_cache_mark(plain)
            marked = dialect.with_boundaries(marked, conversation.pending.marks)
            marked = self._asking(conversation, dialect, marked)
        call = _Call(
            conversation,
            dialect,
            turns,
            chains,
            send,
            body,
            plain,
            marked,
            marked,
            asks=asks,
            pending=conversation.pending,
            before=before,
            kept=kept,
        )
        try:
            call.outgoing, call.asked, call.carrier = self._with_note(call, scan)
        except Exception:
            logger.warning("compache: a note was not asked", exc_info=True)
        return call

    # -- what reasoning is bound to ----------------------------------------------

    @staticmethod
    def _binds(dialect, body, *models) -> bool:
        """Whether a model only takes its reasoning back behind what stood ahead of it when it was made."""
        if not hasattr(dialect, "reasoned"):
            return False
        asked = body.get("thinking")
        budget = isinstance(asked, dict) and asked.get("type") == "enabled"
        return any(binds_reasoning(name, budget) for name in models)

    @staticmethod
    def _unbind(conversation, dialect, written, out) -> bool:
        """Find the reasoning in a request that was made behind something else than it holds now.

        The provider drops it, or refuses the request for it. Returns whether there
        is any. `written` is the request as the application wrote it, which its
        first reasoning was made behind.
        """
        if not conversation.binds:
            return False
        conversation.planned = now = wire.shape(out)
        before = conversation.sent_as or wire.shape(written)
        first = wire.changed(before, now)
        if first is None:
            # Nothing has. The reply to what went out last comes next, and it is bound to all of that.
            first = len(before)
        found = {dialect.reasoned(m) for m in dialect.said_so_far(out)[first:]}
        found -= {None, *conversation.unbound}
        conversation.unbound |= found
        return bool(found)

    def _written(self, conversation, dialect, body) -> Any:
        """A request as the application wrote it, without reasoning that was made behind something else."""
        if not conversation.binds or conversation.sent_as is None:
            return body
        said, conversation.planned = dialect.said_so_far(body), None
        if not any(map(dialect.reasoned, said)):
            return body
        out = body
        for _ in range(2):
            kept = self._bound(conversation, dialect, said)
            out = body if kept is said else {**body, "messages": kept}
            if not self._unbind(conversation, dialect, body, out):
                break
        return out

    @staticmethod
    def _goes(call: _Call, body) -> None:
        """Remember what goes out: that is what the model reasons behind, if it does."""
        conversation = call.conversation
        if conversation.session or conversation.sent_as:
            call.sent = body
            planned = conversation.planned if conversation.binds else None
            if body is call.marked and planned is not None:
                conversation.sent_as, call.sent = planned, None

    @staticmethod
    def _reasoned(call: _Call, reply) -> None:
        """It did: what stood ahead of it is worth its hashes."""
        conversation = call.conversation
        if reply.reasons and call.sent is not None:
            # A gateway may have a name of its own for the model, which is known once it has replied.
            if conversation.binds or Live._binds(
                call.dialect, call.body, conversation.answered
            ):
                conversation.sent_as, call.sent = wire.shape(call.sent), None

    def _carry(self, call: _Call, scan) -> Result:
        """Send the request, fall back to plainer forms of it if it is refused, and read the reply."""
        pending = call.conversation.pending
        pending.in_flight += 1
        try:
            result = self._sent(call, pending)
        except BaseException:
            pending.in_flight -= 1
            raise
        if result.events is None:
            pending.in_flight -= 1
        return result

    def _sent(self, call: _Call, pending: _Pending) -> Result:
        conversation = call.conversation
        try:
            result = self._taken_as(call, pending)
            unknown = conversation.binds and call.before[1] is None
            if result.status in _REFUSED and unknown:
                result = self._without_any(call, result)
        except BaseException:
            self._not_taken(call)
            raise
        if result.status >= 400:
            self._not_taken(call)
            return result
        pending.asked = pending.asked or call.asked
        if result.json is None and result.events is None:
            return result
        if result.events is not None:

            def release():
                with conversation.lock:
                    pending.in_flight -= 1
                if result.close is not None:
                    result.close()

            _LATER.ready()
            reading = _Reading()
            events = self._stream(call, result, pending, reading)
            return replace(result, events=_Passing(events, release, reading))
        return self._settle(call, result)

    @staticmethod
    def _not_taken(call: _Call) -> None:
        """Nothing is reasoned behind a request that failed, and none is given up for it."""
        conversation = call.conversation
        conversation.unbound = set(call.before[0])
        conversation.sent_as = call.before[1]
        if call.kept is not None and conversation.session is not None:
            conversation.session = call.kept

    def _taken_as(self, call: _Call, pending: _Pending) -> Result:
        """Send a request, and plainer forms of it where it is refused."""
        conversation = call.conversation
        self._goes(call, call.outgoing)
        result = call.send(call.outgoing)
        result = self._under_default(call, result)
        if result.status in _REFUSED and call.outgoing is not call.marked:
            # Refused with the note in it: send it without.
            self._note_refused(call)
            self.stats["retried"] += 1
            call.outgoing, call.asked, call.carrier = call.marked, False, None
            self._goes(call, call.outgoing)
            result = self._under_default(call, call.send(call.outgoing))
        if result.status in _REFUSED and call.outgoing is not call.body:
            # Refused with the policy applied: send what the application wrote, and all that was
            # reasoned before this request.
            conversation.unbound = set(call.before[0])
            conversation.sent_as = call.before[1]
            result = self._send_written(call)
            if result.status < 400:
                # So it was the policy. What is refused either way is none of its doing.
                self.stats["policy_refused"] += 1
                self._stand_down(conversation, pending.index, pending.started_at)
                call.plain = call.marked = call.outgoing = call.body
                call.asked, call.carrier = False, None
        return result

    def _send_written(self, call: _Call) -> Result:
        """Send what the application wrote. What reasoning it costs is only given up once that is taken."""
        conversation = call.conversation
        before = set(conversation.unbound), conversation.sent_as
        self.stats["retried"] += 1
        try:
            call.body = self._written(conversation, call.dialect, call.body)
            self._goes(call, call.body)
            result = call.send(call.body)
        except BaseException:
            conversation.unbound, conversation.sent_as = before
            raise
        if result.status >= 400:
            conversation.unbound, conversation.sent_as = before
        return result

    def _without_any(self, call: _Call, result: Result) -> Result:
        """A refused request once more, without its reasoning: what that was made behind is not known here."""
        conversation, dialect = call.conversation, call.dialect
        said = dialect.said_so_far(call.body)
        found = {dialect.reasoned(m) for m in said} - {None}
        if not found:
            return result
        bare = {**call.body, "messages": self._bare(dialect, said, found)}
        self.stats["retried"] += 1
        again = call.send(bare)
        if again.status >= 400:
            return result
        conversation.unbound |= found
        conversation.sent_as, call.sent = wire.shape(bare), None
        call.body = call.plain = call.marked = call.outgoing = bare
        call.asked, call.carrier = False, None
        if conversation.session is not None and call.pending is not None:
            call.pending.whole = True
        return again

    def _under_default(self, call: _Call, result: Result) -> Result:
        """What became of a request. One that was refused goes out again under the default lifetime.

        Only that tells whether the lifetime was why: what a provider says about it differs.
        """
        conversation = call.conversation
        asked = conversation.lifetime or conversation.header_lifetime
        if result.status not in _REFUSED or asked is None:
            return result
        self.stats["retried"] += 1
        marked = wire.without_lifetime(call.marked)
        noted = call.outgoing is not call.marked
        outgoing = wire.without_lifetime(call.outgoing) if noted else marked
        again = call.send(outgoing)
        if again.status in _REFUSED:
            return result
        if again.status < 400:
            call.marked, call.outgoing = marked, outgoing
            self._lifetime_refused(conversation)
        return again

    def _catch_up(self, conversation, chains, turns, now, scan) -> None:
        """Bring a conversation's state up to the turns a request shows as answered."""
        done = len(turns) - 1
        pending = conversation.pending
        renew = conversation.stale and pending is None
        if conversation.turns == done and not renew:
            return
        if pending is not None and pending.index == done == conversation.turns + 1:
            last = turns[done - 1]
            if _hashes(last.retrievals[: len(pending.seen)]) == pending.seen:
                # Answered without the answer passing here: take it from the request.
                late = last.retrievals[len(pending.seen) :]
                if late:
                    self._feed(conversation, pending, late, now, scan)
                # An agent's policy has taken the usage of every call that passed.
                counted = not conversation.agentic
                self._finish(conversation, chains[done], last, now, counted)
                return
        k = conversation.turns
        if pending is None and 0 < k < done and not renew and conversation.agentic:
            if conversation.session is not None and conversation.chain == chains[k]:
                return self._go_on(conversation, chains, turns, now, scan)
        self._walk(conversation, chains, turns, now, scan)

    def _go_on(self, conversation, chains, turns, now, scan) -> None:
        """Take in turns that went to another model, behind those that went to this one."""
        k, done = conversation.turns, len(turns) - 1
        for number, segment in enumerate(turns[k:done], k + 1):
            turn = self._shown(conversation, segment, now, scan)
            conversation.session.elsewhere(turn, segment.answer)
            if self.recorder is not None:
                self._heard(conversation, number, segment.question, segment.retrievals)
                self._quotes(conversation).said(segment.answer)
        conversation.turns = done
        self._index(conversation, chains[done])

    def _anew(self, conversation, chains, turns, now, scan) -> None:
        """Take a conversation back to where its latest turn began."""
        base, session = conversation.base, conversation.session
        if base is None or base.session is None or base.chain != chains[-1]:
            return self._walk(conversation, chains, turns, now, scan)
        conversation.pending, conversation.went = None, []
        conversation.session = base.session.copy()
        conversation.session.follows(session)
        conversation.inserts = list(base.inserts)
        # Its results go by the same names, and may be others now.
        conversation.fetched = set()

    def _walk(self, conversation, chains, turns, now, scan) -> None:
        """Walk the policy through a conversation that is met in the middle, up to its last question."""
        done = len(turns) - 1
        conversation.pending, conversation.turns, conversation.stale = None, 0, False
        conversation.inserts, conversation.went = [], []
        conversation.fetched = set()
        conversation.base = conversation.here = conversation.quotes = None
        policy = self._start(conversation, conversation.thread_id, scan)
        if policy is not None:
            # When it was said is not known, and a process that meets a conversation in the middle
            # has usually just been started: the provider's cache is taken to have expired. It has
            # not where this process sent the conversation itself, before a turn was asked again.
            before = now - policy.spec.cache_ttl_seconds - 60
            if conversation.sent_whole is not None:
                before = max(before, min(conversation.sent_whole, now - 1e-3))
            for k, segment in enumerate(turns[:done]):
                at = before - (done - k) * 1e-3
                # Behind the same instructions and tools as now, or they would look changed.
                turn = self._shown(conversation, segment, at, scan)
                conversation.session.complete(turn, segment.answer, at=at)
        self._remember(conversation, turns[:done])
        conversation.turns = done
        self._index(conversation, chains[done])

    @staticmethod
    def _thought(conversation, scan, before: int) -> None:
        """Take in the reasoning that stands ahead of a message and was not seen as it was made."""
        if conversation.agentic:
            for place, tokens in scan.thoughts.items():
                if place < before:
                    conversation.session.thought(place, tokens)

    def _shown(self, conversation, segment, at, scan):
        """A turn's question and what was retrieved for it, as the policy's session takes them."""
        session, found = conversation.session, segment.retrievals
        if segment.inputs:
            self._thought(conversation, scan, segment.inputs[0].message)
        ahead = len(found)
        if conversation.agentic:
            # What came before the question is there at once. What its tools returned came round by round.
            ahead = next((i for i, part in enumerate(found) if part.answers), ahead)
        turn = session.prepare(
            segment.question,
            [part.taken for part in found[:ahead]],
            at=at,
            system_prompt=scan.system,
            tools=scan.tools,
        )
        self._went(conversation, turn)
        return self._extended(conversation, turn, found[ahead:], at, scan)

    def _extended(self, conversation, turn, found, at, scan):
        """A turn with more that its tools returned, one round of tool calls at a time."""
        for part in (
            _rounds(found) if conversation.agentic else [found] if found else []
        ):
            self._thought(conversation, scan, part[0].slots[0].message)
            turn = conversation.session.extend(turn, [f.taken for f in part], at=at)
            self._went(conversation, turn)
        return turn

    @staticmethod
    def _went(conversation, turn) -> None:
        if conversation.agentic:
            conversation.went += turn.plan.decision["went"]
            conversation.holds = turn.plan.decision["holds"]

    def _begin(self, conversation, chains, current, number, now, scan) -> _Pending:
        """Open the turn a request asks about."""
        session, found = conversation.session, current.retrievals
        conversation.base = self._point(conversation, chains[-1])
        if conversation.base is not None and number > 1:
            with self._lock:
                self._by_base[chains[-1]] = conversation
                while len(self._by_base) > self._limit:
                    self._by_base.popitem(last=False)
        self._asks(conversation, scan)
        pending = conversation.pending = _Pending(
            number, now, question=current.question, seen=_hashes(found)
        )
        if session is not None:
            pending.turn = self._shown(conversation, current, now, scan)
        self._heard(conversation, number, current.question, found)
        return pending

    @staticmethod
    def _asks(conversation, scan) -> None:
        """Take note of the model a request names. Another has a cache of its own, and nothing of this is in it."""
        if scan.model and conversation.requested not in (None, scan.model):
            if conversation.session is not None:
                conversation.session.lost()
        conversation.requested = scan.model or conversation.requested

    def _feed(self, conversation, pending, found, now, scan) -> None:
        """Take in what the tools of a turn have returned since its last request."""
        first = len(pending.seen) + 1
        pending.seen += _hashes(found)
        if conversation.session is not None:
            pending.turn = self._extended(conversation, pending.turn, found, now, scan)
        self._heard(conversation, pending.index, "", found, first)

    # -- what replies quote ----------------------------------------------------

    @staticmethod
    def _quotes(conversation) -> Quotes:
        if conversation.quotes is None:
            conversation.quotes = Quotes()
        return conversation.quotes

    def _heard(self, conversation, number, question, found, first=1) -> None:
        """Remember the lines a turn's question and results hold, as hashes, to see them quoted."""
        if self.recorder is None:
            return
        quotes = self._quotes(conversation)
        quotes.own(question)
        for j, part in enumerate(found, first):
            quotes.result(f"R{number}.{j}", part.text)

    def _remember(self, conversation, turns) -> None:
        """The same for turns that were answered before."""
        if self.recorder is None:
            return
        for k, segment in enumerate(turns, 1):
            self._heard(conversation, k, segment.question, segment.retrievals)
            self._quotes(conversation).said(segment.answer)

    def _stand_down(self, conversation, number, now) -> None:
        """Stop applying the policy to a conversation. It is still recorded."""
        if conversation.session is not None:
            self.stats["stood_down"] += 1
        conversation.session = conversation.lifetime = None
        conversation.header_lifetime = None
        conversation.stale = False
        if conversation.pending is None or conversation.pending.index != number:
            conversation.pending = _Pending(number, now)
        conversation.pending.turn = None

    def _apply(self, conversation, dialect, body, scan, turns) -> Any:
        """The request with every archived part replaced by its notice, and the tool to fetch one back."""
        pending, name = conversation.pending, conversation.thread_id
        turn, out = pending.turn, body
        if turn.archived_ids:
            shown = {
                p.part_id: p.text for p in turn.parts if p.part_id in turn.archived_ids
            }
            # Where the host runs our tool, what it returned stands among the results of the others.
            hosted = _REOPEN in dialect.tool_names(body)
            known = {m.group() for m in map(_HANDLE.search, shown.values()) if m}
            first = conversation.agentic
            for k, segment in enumerate(turns, 1):
                pieces = [
                    (f"{name}:{k}:input", segment.inputs),
                    (f"{name}:{k}:output", segment.outputs),
                ]
                for j, found in enumerate(segment.retrievals, 1):
                    part_id = f"{name}:{k}:retrieval:{j}"
                    pieces.append((part_id, found.slots))
                    of = _FETCHED.match(found.text) if hosted else None
                    if of and part_id in shown and of[1] in known:
                        # A copy is no text of its own: the notice names what it is a copy of,
                        # as long as that is there.
                        shown[part_id] = notice(of[1])
                    if first and part_id in shown and found.slots:
                        # Results are archived from the oldest on, so this one stays the first, and
                        # what is cached of it is never written again.
                        at = _HANDLE.search(shown[part_id])
                        shown[part_id] = explained(at.group()) if at else shown[part_id]
                        first = False
                for part_id, slots in pieces:
                    if part_id in shown and slots and slots[0].whole is not None:
                        out = wire.put(out, slots[0].whole, shown[part_id])
                    elif part_id in shown:
                        for i, slot in enumerate(slots):
                            text = _GONE if i else shown[part_id]
                            out = wire.set_text(dialect, out, slot.path, text)
            conversation.reopens = True
        said = dialect.said_so_far(body)
        places = self._places(conversation, turns, scan)
        conversation.inserts = [
            placed
            for placed in (
                self._placed(conversation, dialect, said, insert, pending.index)
                for insert in conversation.inserts
            )
            if placed is not None
        ]
        gone: list = []
        left_out: set = set()
        if conversation.agentic and scan.thoughts:
            left_out.update(conversation.session.without(scan.thoughts))
        if conversation.unbound:
            unbound = conversation.unbound
            left_out.update(
                p for p in scan.thoughts if dialect.reasoned(said[p]) in unbound
            )
        if left_out:
            out, gone = dialect.without_thoughts(out, sorted(left_out))
        # From the last to the first, so that the places of the others hold.
        for insert in reversed(conversation.inserts):
            at = insert.at - sum(place < insert.at for place in gone)
            extra = self._bound(conversation, dialect, insert.messages)
            out = dialect.with_inserted(out, at, extra)
        put_in = sum(len(insert.messages) for insert in conversation.inserts)
        if conversation.agentic and scan.tools:
            # An agent works its way to a first cut soon enough. A tool that joins only then would
            # change what comes ahead of everything else that is cached.
            conversation.reopens = True
        if conversation.reopens and _REOPEN not in dialect.tool_names(body):
            # A model that is offered the tool while nothing is archived calls it for nothing. So in
            # a chat it comes with the first notice, and stays: the first cut rewrites the request
            # from near its start anyway, and after that the tool never changes what is cached.
            if conversation.agentic:
                out = dialect.with_tool(out, _AGENT_TOOL)
            else:
                out = dialect.with_tool(out, _REOPEN_TOOL, _REOPEN_HOW)
        if self.recorder is not None:
            pending.held = wire.volume(out)
            pending.left = wire.volume(body) - pending.held
            # What is reasoned has tokens and no text that says how many.
            pending.media = bool(scan.thoughts.keys() - left_out) or wire.holds_media(
                dialect.said_so_far(out)
            )
            pending.left_more = conversation.session.left_more
            if conversation.session.counts_thoughts:
                pending.left_more += sum(scan.thoughts.get(p, 0) for p in left_out)
        pending.cut = pending.cut or bool(turn.archived_ids)
        # Where those are in the request as it goes out: behind messages of ours.
        shift = len(dialect.said_so_far(out)) - len(said) - put_in + len(gone)
        pending.marks = tuple(
            at
            + shift
            + sum(len(i.messages) for i in conversation.inserts if i.at <= at)
            - sum(place < at for place in gone)
            for at in places
        )
        return out

    @staticmethod
    def _bound(conversation, dialect, messages) -> list:
        """Messages without reasoning that was made behind something else."""
        if not conversation.unbound or not hasattr(dialect, "reasoned"):
            return messages
        return Live._bare(dialect, messages, conversation.unbound)

    @staticmethod
    def _bare(dialect, messages, reasoned) -> list:
        """Messages without the reasoning that goes by those names."""
        places = [i for i, m in enumerate(messages) if dialect.reasoned(m) in reasoned]
        if not places:
            return messages
        return dialect.without_thoughts({"messages": messages}, places)[0]["messages"]

    @staticmethod
    def _placed(conversation, dialect, said, insert, turn) -> _Insert | None:
        """An insert where its place is in a request now. None where it is out of view, or its place is gone."""
        session = conversation.session
        if not conversation.agentic:
            if insert.turn != turn:
                return None
        elif not any(session.shows_again(name) for name in insert.names):
            return None
        if insert.dialect == dialect.name:
            # A harness may leave out a message it had put in, or put in others.
            near = range(insert.at, max(-1, insert.at - _MOVED - 1), -1)
            for at in (*near, *range(insert.at + 1, insert.at + _MOVED + 1)):
                if at > len(said):
                    continue
                behind = wire.landmark(said[at - 1]) if at else None
                if behind == insert.behind:
                    return replace(insert, at=at)
        for name in insert.names if conversation.agentic else ():
            session.dropped(name)
        return None

    @staticmethod
    def _places(conversation, turns, scan) -> list[int]:
        """Where a cut begins: the one that was made now, and the one that comes next.

        What lies ahead of them stays as it is, so a cache mark there lets the
        request that makes the cut read it. Given as the message that holds the
        first part to go, as the application counts its messages.
        """
        firsts = [*conversation.went[:1], conversation.holds]
        if firsts == [None]:
            return []
        at = {(("thought", p), True): p for p in scan.thoughts}
        at |= {(n, True): i.at for i in conversation.inserts for n in i.names}
        for k, segment in enumerate(turns, 1):
            for j, found in enumerate(segment.retrievals, 1):
                part_id = f"{conversation.thread_id}:{k}:retrieval:{j}"
                at[part_id, False] = found.slots[0].message
        return list(dict.fromkeys(at[first] for first in firsts if first in at))

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
        if not self.sample or self._notes_off or not call.asks:
            return marked, False, None
        sampled = notes_.sampled(
            conversation.record_id, len(call.turns), self.sample, self.salt
        )
        schema = dialect.schema(call.body)
        extended = notes_.extend_schema(schema) if schema is not None else None
        if (
            self.notes == "apart"
            or scan.stream
            # A call that was made elsewhere has gone out: only a note after it is possible.
            or conversation.when is not None
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
        most = _AGENT_ROUNDS if conversation.agentic else _ROUNDS
        written = False
        try:
            reply = dialect.reply(data)
            self._ledger(conversation, kind, reply, pending=call.pending)
            self._reasoned(call, reply)
            while (
                conversation.session is not None
                and reply.calls
                and all(c["name"] == _REOPEN for c in reply.calls)
            ):
                rounds += 1
                if rounds > most:
                    break
                answers = self._reopen(conversation, reply.calls, call, data, True)
                sent = self._asking(
                    conversation, dialect, dialect.after_tools(sent, data, answers)
                )
                self._goes(call, sent)
                again, written = self._followed(call, sent)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                how = "retry" if written else "reopen"
                self._ledger(conversation, how, reply, pending=call.pending)
                self._reasoned(call, reply)
                if written:
                    break
            # A reply asked for a note that says nothing is asked again without the note first.
            if (
                not written
                and self._changed(call)
                and (rounds > most or not (call.asked or reply.text or reply.calls))
            ):
                # The model goes on fetching, or says nothing: what it was shown does not do for
                # this turn. It is asked once more, exactly as the application wrote it.
                again = self._as_written(call)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                self._ledger(conversation, "retry", reply, pending=call.pending)
                self._reasoned(call, reply)
            ours = [c for c in reply.calls if c["name"] == _REOPEN]
            if ours and _REOPEN not in dialect.tool_names(call.body):
                # Called beside the application's own tools. The application never sees ours, and the
                # model finds its answer with theirs.
                if conversation.session is not None and len(ours) < len(reply.calls):
                    self._reopen(conversation, ours, call, data)
                data = dialect.without_tool(data, _REOPEN)
        except _Unsent as error:
            raise error.__cause__ from None
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
                self._goes(call, call.marked)
                again = call.send(call.marked)
                if again.status < 400 and again.json is not None:
                    plain = replace(
                        call, outgoing=call.marked, asked=False, carrier=None
                    )
                    return self._settle(plain, again, "retry")
            if found is not None and call.asked:
                self._take_note(call, found)
            self._after(call, reply.text, reply.calls, reply.reasons)
        except Exception:
            logger.warning("compache: a reply was not recorded", exc_info=True)
        # `raw` stays with a body that was not touched, so that it can go back byte for byte.
        return result if data is result.json else replace(result, json=data, raw=None)

    @staticmethod
    def _changed(call: _Call) -> bool:
        """Whether what the model was asked may be why a reply to it is of no use."""
        if not call.conversation.agentic or call.outgoing is not call.marked:
            return call.outgoing is not call.body
        pending = call.conversation.pending
        return pending is not None and bool(
            pending.turn is not None and pending.turn.archived_ids
        )

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

    def _stream(
        self, call: _Call, result: Result, pending, reading: _Reading
    ) -> Iterator[bytes]:
        """A streamed reply passed on as it arrives, with our own tool calls answered on the way."""
        reading.begun = True
        conversation, dialect = call.conversation, call.dialect
        most = _AGENT_ROUNDS if conversation.agentic else _ROUNDS
        tool = _REOPEN if conversation.session is not None else None
        editor = dialect.Stream(dialect, tool)
        sent, rounds, events, closers = call.outgoing, 0, result.events, [result.close]
        kind, counted, closed, failed = "answer", False, False, None
        dropped = False

        def let_go():
            for closing in closers:
                try:
                    if closing is not None:
                        closing()
                except Exception:
                    # The books are closed all the same.
                    logger.debug("compache: a response was not closed", exc_info=True)
            if not closed:
                if failed is not None:
                    editor.finished = False
                close()

        def close():
            """Close the books on the reply, before the application can come back to it."""
            nonlocal closed
            closed = True
            with conversation.lock:
                pending.in_flight -= 1
                try:
                    if not counted:
                        self._ledger(conversation, kind, editor, pending=pending)
                        self._reasoned(call, editor)
                    if editor.finished:
                        if editor.ours and editor.calls:
                            self._reopen(conversation, editor.ours, call, editor.said)
                        self._after(call, editor.text, editor.calls, editor.reasons)
                except Exception:
                    logger.warning(
                        "compache: a streamed reply was not recorded", exc_info=True
                    )

        try:
            while True:
                for raw, name, data in wire.sse_events(events):
                    shown = editor.feed(raw, name, data)
                    if editor.over and not editor.again and not closed:
                        close()
                    yield from shown
                if not editor.again or kind == "retry":
                    break
                rounds += 1
                with conversation.lock:
                    self._ledger(conversation, kind, editor, pending=pending)
                    self._reasoned(call, editor)
                    # The round that is over is in the books, whatever comes next.
                    counted = True
                    if rounds > most:
                        # The model goes on fetching: it is asked as the application wrote it.
                        again, kind = self._as_written(call), "retry"
                    else:
                        answers = self._reopen(
                            conversation, editor.ours, call, editor.said
                        )
                        sent = self._asking(
                            conversation,
                            dialect,
                            dialect.after_tools(sent, editor.said, answers),
                        )
                        self._goes(call, sent)
                        again, written = self._followed(call, sent)
                        kind = "retry" if written else "reopen"
                closers.append(again.close)
                if again.status >= 400 or again.events is None:
                    failed = again
                    break
                events, counted = again.events, False
                editor.next_round()
            if failed is not None:
                yield from editor.failed(failed.status, _said_wrong(failed))
            else:
                yield from editor.end()
        except _Unsent as error:
            editor.finished = False
            raise error.__cause__ from None
        except GeneratorExit:
            dropped = not reading.asked
            raise
        finally:
            if dropped:
                _LATER.do(let_go)
            else:
                let_go()

    def _followed(self, call: _Call, body) -> tuple[Result, bool]:
        """What follows a fetch, and whether it had to be asked as the application wrote it.

        Its answer is not in the application's request, so that goes out only
        where this one fails.
        """
        again = None
        try:
            again = call.send(body)
        except Exception:
            logger.warning("compache: a fetch was not followed up", exc_info=True)
        if again is not None and again.status < 400:
            return again, False
        refused = again is not None and again.status in _REFUSED
        return self._as_written(call, refused), True

    def _as_written(self, call: _Call, counts: bool = True) -> Result:
        """Send the application's own request for a turn the policy's view did not do for.

        A conversation that needs this twice is left alone from then on: its
        users should not wait for fetches that lead nowhere. `counts` says
        whether it was the policy's doing.
        """
        conversation = call.conversation
        call.asked, call.carrier = False, None
        try:
            result = self._send_written(call)
        except Exception as error:
            raise _Unsent() from error
        if result.status < 400:
            call.plain = call.marked = call.outgoing = call.body
            if call.pending is not None:
                call.pending.whole = True
            if counts:
                self.stats["fell_back"] += 1
                conversation.fell_back += 1
        if conversation.fell_back >= 2 and conversation.session is not None:
            pending = conversation.pending
            self._stand_down(conversation, pending.index, pending.started_at)
        return result

    def _reopen(
        self, conversation, calls, asked: _Call | None = None, said=None, alone=False
    ):
        """Answers to the model's `reopen` calls, by call ID.

        `asked` is the call whose reply `said` made them. The application never
        sees them, so they are put into its later requests for as long as what
        they returned is new. `alone` says that it sees nothing of that reply.
        """
        answers, names, before = {}, [], set(conversation.fetched)
        for call in calls:
            self.stats["reopened"] += 1
            part = None
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
            told = json.dumps(_for_model(found), ensure_ascii=False)
            answers[call["id"]] = self._with_pictures(asked, part, told)
            in_view = bool(part) and (
                conversation.session.shows_again(part)
                or (asked is None and self._copied(conversation, found))
            )
            if part and not in_view:
                # A model that works with more than stays in view would fetch for ever.
                piece = part, found.get("offset")
                if piece in before:
                    conversation.session.stay_longer()
                conversation.fetched.add(piece)
            if not part or in_view:
                part = f"reopened:{uuid.uuid4().hex[:12]}"
            names.append((part, told))
        if asked is not None and conversation.pending is not None:
            try:
                heard = asked.dialect.said_so_far(asked.body)
                at = len(heard)
                while at and wire.instructs(heard[at - 1]):
                    # It may be there for this one request.
                    at -= 1
                dialect = asked.dialect
                if alone and hasattr(dialect, "reasoned"):
                    # As it was asked on: what is reasoned in the next reply may be bound to that.
                    after = dialect.after_tools(asked.body, said, answers)
                    messages = dialect.said_so_far(after)[len(heard) :]
                else:
                    messages = dialect.fetched(said, answers)
                    if conversation.binds:
                        # What was reasoned in it is in the application's own message.
                        own = {dialect.reasoned(m) for m in messages} - {None}
                        messages = self._bare(dialect, messages, own)
                insert = _Insert(
                    at,
                    wire.landmark(heard[at - 1]) if at else None,
                    asked.dialect.name,
                    messages,
                    tuple(name for name, _ in names),
                    conversation.pending.index,
                )
                conversation.inserts.append(insert)
                for name, text in names:
                    conversation.session.reopened(name, text)
            except Exception:
                logger.warning("compache: a fetched part does not stay", exc_info=True)
        return answers

    @staticmethod
    def _copied(conversation, found) -> bool:
        """Whether what a host fetched back before still stands among the results of its tools."""
        turn = conversation.pending.turn if conversation.pending is not None else None
        where = f'"offset": {found.get("offset")},'
        for part in turn.parts if turn is not None else ():
            if part.part_id not in turn.archived_ids:
                of = _FETCHED.match(part.text or "")
                if of and of[1] == found.get("handle") and where in part.text[:400]:
                    return True
        return False

    @staticmethod
    def _with_pictures(asked: _Call | None, part: str | None, told: str) -> Any:
        """What a fetch is answered with: what it found, and the pictures that stood with it."""
        if asked is None or not part or ":retrieval:" not in part:
            return told
        _, turn, _, at = part.rsplit(":", 3)
        try:
            whole = asked.turns[int(turn) - 1].retrievals[int(at) - 1].slots[0].whole
            pictures = wire.media_in(wire.get(asked.body, whole)) if whole else []
        except (IndexError, KeyError, TypeError, ValueError):
            return told
        if not pictures:
            return told
        return [{"type": asked.dialect.text_block, "text": told}, *pictures]

    def _after(self, call: _Call, text: str, calls, reasons: bool = False) -> None:
        """A reply has been read: the turn goes on with tool calls, or ends with this answer."""
        conversation, pending = call.conversation, call.pending
        if pending is None or pending is not conversation.pending:
            return
        pending.calls += 1
        pending.answered = True
        # Where the application puts this reply, and with it what the model reasoned.
        pending.place = len(call.dialect.said_so_far(call.body)) if reasons else None
        self._step(conversation, pending, text, calls)
        if calls:
            # The application runs its tools and comes back with their results.
            self._wait_for_tools(call, text, calls)
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
        if conversation.agentic:
            # A further message would have the provider leave out what the model reasoned, and so
            # miss the cache: the same request goes out once more.
            last = (call.dialect, call.marked, None, call.send)
        else:
            last = (call.dialect, call.plain, text, call.send)
        self._plan_pings(conversation, last)
        if call.apart and not self._notes_off:
            self.stats["asked"] += 1
            worker = threading.Thread(
                target=self._ask_apart, args=(call, closed, pending), daemon=True
            )
            self._apart = [t for t in self._apart if t.is_alive()] + [worker]
            worker.start()

    def _wait_for_tools(self, call: _Call, text: str, calls) -> None:
        """A turn goes on once its tools are done. Until then its cache may have to be kept warm."""
        conversation, pending = call.conversation, call.conversation.pending
        # An API that gives its calls no IDs has their names here.
        keys = [c["id"] for c in calls if c.get("id") not in (None, c.get("name"))]
        if not keys:
            said = call.turns[-1].question
            keys = [self._loop(conversation.chain, said, pending.seen)]
        self._register(conversation, keys)
        pending.called = tuple(keys)
        if conversation.session is None or pending.turn is None:
            return
        conversation.session.replied(
            pending.turn,
            text,
            [made.get("arguments") for made in calls],
            at=self._now(conversation),
            # What was billed for everything says nothing of the view.
            usage=None if pending.whole else pending.last_usage,
            place=pending.place,
        )
        if pending.whole:
            conversation.session.lost()
        self._plan_pings(conversation, (call.dialect, call.marked, None, call.send))

    def _register(self, conversation, keys) -> None:
        """Remember whose tool calls these are."""
        with self._lock:
            for key in keys:
                others = self._by_call.pop(key, ())
                others = [c for c in others if c is not conversation]
                self._by_call[key] = [*others[1 - _MOST_AT_ONE_POINT :], conversation]
            while len(self._by_call) > 8 * self._limit:
                self._by_call.popitem(last=False)

    def _plan_pings(self, conversation, last: tuple) -> None:
        """Keep what a ping is built from, but only while one is planned."""
        planned = (
            conversation.session is not None
            and not conversation.stale
            and conversation.session.next_ping_at
        )
        conversation.last = last if planned else None
        with self._lock:
            if planned:
                self._warm[id(conversation)] = conversation
            else:
                self._warm.pop(id(conversation), None)

    def _step(self, conversation, pending, text, calls) -> None:
        """Keep the books on one call of a turn: when, how much it had been shown, what it quoted."""
        if self.recorder is None:
            return
        said = spoken(text or "", [made.get("arguments") for made in calls])
        counted = canonical_usage(pending.last_usage) or {}
        pending.steps.append(
            {
                "started_at": pending.sent_at,
                "ended_at": self._now(conversation),
                "results": len(pending.seen),
                "bytes": _size(said),
                # A request with something out of view does not say how large the conversation is.
                "input_tokens": None if pending.cut else counted.get("input"),
                "output_tokens": counted.get("output"),
                "thought_tokens": reasoning_tokens(pending.last_usage) or 0
                if pending.place is not None
                else 0,
                # Where a host does not pass on what was said, what it relied on is not known.
                "needs": None
                if text is None
                else self._quotes(conversation).said(said),
            }
        )

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
            asked = self._asking(conversation, dialect, body)
            reply = self._whole(dialect, call.send(asked))
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

    def _finish(self, conversation, chain, segment, now, counted=True) -> None:
        """Close the pending turn: tell the policy, write the record, and index what was said."""
        pending = conversation.pending
        pending.ended_at = max(now, pending.started_at)
        if conversation.session is not None and pending.turn is not None:
            visible = {
                p.part_id
                for p in pending.turn.parts
                if p.kind == "retrieval" and p.part_id not in pending.turn.archived_ids
            }
            counted = counted and not (pending.whole and conversation.agentic)
            try:
                conversation.session.complete(
                    pending.turn,
                    segment.answer,
                    at=pending.ended_at,
                    usage=pending.first_usage if counted else None,
                    # What the model fetched back stays in an agent's conversation.
                    last_usage=pending.last_usage
                    if counted and (pending.calls > 1 or conversation.agentic)
                    else None,
                    place=pending.place if counted else None,
                    cited_part_ids=[i for i in pending.cited if i in visible],
                )
                if pending.whole:
                    conversation.session.lost()
            except Exception:
                logger.warning(
                    "compache: a turn was not taken by the policy", exc_info=True
                )
                self.stats["stood_down"] += 1
                conversation.session = conversation.lifetime = None
        self._write_turn(conversation, pending, segment, pending.ended_at)
        self.stats["turns"] += 1
        if self.learner is not None:
            self.learner.heard(max(1, pending.calls))
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
            requested=conversation.requested,
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
            # A chat's turn of one call says it all. An agent's may quote what it had read before.
            steps=pending.steps
            if len(pending.steps) > 1 or conversation.tools_bytes
            else (),
            carries=conversation.carries,
            cut=pending.cut,
        )

    @staticmethod
    def _named(pending) -> list | None:
        """What a turn's note named, and what its reply fetched back: it needed that too."""
        if pending.refs is None or not pending.reopened:
            return pending.refs
        fetched = notes_.labels_of(notes_.Note(needs=tuple(pending.reopened)))
        return list(dict.fromkeys([*pending.refs, *fetched]))

    def _ledger(self, conversation, kind, source, turn=None, pending=None) -> None:
        """Count one model call. `source` has the `usage`, `cost` and `model` of its reply."""
        pending = pending or conversation.pending
        conversation.answered = source.model or conversation.answered
        if pending is not None and kind in ("answer", "reopen", "retry"):
            if pending.first_usage is None:
                pending.first_usage = source.usage
            pending.last_usage = source.usage
            pending.model = source.model or pending.model
        used = canonical_usage(source.usage)
        for name, count in (used or {}).items():
            self.tokens[name] += count
        if used and pending is not None and pending.held and kind == "answer":
            out = left_out(pending.left, pending.held, used["input"], pending.media)
            self.tokens["left_out"] += max(0, out + pending.left_more)
        if self.recorder is None:
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
        self.recorder.call(
            conversation.record_id,
            turn,
            kind,
            at=self.clock() if conversation.when is None else conversation.when,
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

    # -- pings ---------------------------------------------------------------

    def next_ping_at(self) -> float | None:
        """When the earliest keep-warm ping of any conversation is due, or None."""
        with self._lock:
            waiting = [c for c in self._warm.values() if self._may_ping(c)]
        due = [c.session.next_ping_at for c in waiting]
        return min((at for at in due if at is not None), default=None)

    @staticmethod
    def _may_ping(conversation) -> bool:
        pending = conversation.pending
        return (
            conversation.session is not None
            and conversation.last is not None
            and not conversation.pings_off
            and not conversation.superseded
            # Between two turns, or while the tools of an agent's turn run.
            and (
                pending is None
                or (conversation.agentic and pending.calls and not pending.in_flight)
            )
        )

    def tick(self, now: float | None = None) -> int:
        """Send the keep-warm pings that are due. Returns how many went out."""
        now = self.clock() if now is None else now
        with self._lock:
            candidates = [c for c in self._warm.values() if self._may_ping(c)]
        sent = 0
        for conversation in candidates:
            if not conversation.lock.acquire(blocking=False):
                continue
            try:
                sent += self._ping(conversation, now)
                if conversation.session.next_ping_at is None:
                    with self._lock:
                        self._warm.pop(id(conversation), None)
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
        if answer is None:
            # Its tools still run, so nothing can follow: the same request goes out once more.
            body = dialect.again(plain, 1)
        else:
            body = dialect.follow_up(plain, answer, ".", 1)
        if body is None:
            conversation.pings_off = True
            return 0
        reply, usage = (
            self._whole(dialect, send(self._asking(conversation, dialect, body))),
            None,
        )
        if reply is not None:
            usage = reply.usage
            self._ledger(conversation, "ping", reply)
        else:
            conversation.pings_off = True
        session.ping_sent(at=now, usage=usage, completed_at=max(now, self.clock()))
        self.stats["pings"] += 1
        return 1
