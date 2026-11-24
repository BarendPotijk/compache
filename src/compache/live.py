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
from .learn import Learner
from .optimizer.recovery import REOPEN_TOOL
from .quotes import Quotes, spoken
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
# What Claude's cache marks ask for where they name no lifetime.
_FIVE_MINUTES = 300
# How often in a row the model may fetch something back within one call of the application's.
_ROUNDS = 3
_AGENT_ROUNDS = 8
# How many of the newest tool results are looked at to find the loop a request goes on with.
_LOOK_BACK = 64
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


@dataclass
class _Insert:
    """What the model fetched back while answering, as messages of ours in the conversation."""

    # Where in the application's messages it stands.
    at: int
    messages: list
    # What it goes by in the policy's session, and the turn it was fetched in.
    names: tuple
    turn: int


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
    # Its calls one by one, for the record.
    steps: list = field(default_factory=list)
    # Bytes of text compache left out of its last request, less what it added, and what it held.
    left: int = 0
    held: int = 0
    # Whether anything was out of view in any of its requests.
    cut: bool = False
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
    session: Any = None
    # The cache lifetime its requests ask for, where the policy was fitted for another than the default.
    lifetime: float | None = None
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
    # The model it began with and the policy that gave it, where policies are learned.
    model: str | None = None
    policy: Any = None
    # What was fetched back and still stands in the conversation, in order.
    inserts: list = field(default_factory=list)
    # How it stood before its latest turn, for a request that branches off there.
    base: tuple | None = None
    # What went out of view with the request that is being made, and what goes first with the next cut.
    went: list = field(default_factory=list)
    holds: tuple | None = None
    # The lines its tools returned, where replies are read for what they quote.
    quotes: Quotes | None = None
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


def _segments(slots) -> list[_Segment]:
    """Turns of a request: a question opens one, and everything up to the next question is its.

    What is said along with the results of tools, before the model has answered
    to them, is a remark in the same turn.
    """
    turns: list[_Segment] = []
    waiting = []
    loops = False
    for slot in slots:
        loops = slot.answers or (loops and slot.kind == "input")
        if not slot.text and not slot.closes:
            continue
        if slot.kind == "input":
            if not turns or (turns[-1].outputs and not loops):
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
        self._by_chain: OrderedDict[str, _Conversation] = OrderedDict()
        # The same by how a conversation stood before its latest turn, and by its tool calls.
        self._by_base: OrderedDict[str, _Conversation] = OrderedDict()
        self._by_call: OrderedDict[str, _Conversation] = OrderedDict()
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

    # -- conversations -------------------------------------------------------

    def _conversation(
        self, thread: str | None, turns: list[_Segment], scan
    ) -> tuple[_Conversation, list[str]]:
        """The conversation a request continues, found by what it has said so far."""
        seed = "thread:" + thread if thread else "auto"
        chains = [hashlib.sha256(seed.encode()).hexdigest()]
        for segment in turns[:-1]:
            chains.append(_link(chains[-1], segment))
        done = len(turns) - 1
        with self._lock:
            found = self._in_loop(scan.results, chains, turns)
            if found is None and (thread or done):
                found = self._by_chain.get(chains[-1])
                if found is not None and self._elsewhere(found, turns):
                    found = self._branch(found, chains[-1], done, thread)
            if found is None and done:
                # Its last turn may have been answered without the answer passing here.
                earlier = self._by_chain.get(chains[-2]) if thread or done > 1 else None
                if (
                    earlier is not None
                    and earlier.pending is not None
                    and earlier.pending.index == done
                ):
                    found = earlier
            if found is None and done and chains[-1] in self._by_base:
                # It goes on from where another request, such as a host's own, branched off.
                found = self._branch(
                    self._by_base[chains[-1]], chains[-1], done, thread
                )
            if found is None:
                name = record = thread or "c-" + uuid.uuid4().hex[:16]
                if thread:
                    # A name can be met again: after a restart, or shared by several conversations.
                    count = self._names[thread] = self._names.get(thread, 0) + 1
                    if count > 1:
                        name = f"{thread}#{count}"
                        record = thread if done else f"{thread}~{count}"
                found = _Conversation(name, chains[0], record)
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
        done = len(turns) - 1
        for call_id in reversed(results[-_LOOK_BACK:]):
            found = self._by_call.get(call_id) if call_id else None
            if found is None:
                continue
            pending = found.pending
            if pending is None:
                return None
            if pending.index == done + 1 and found.chain == chains[-1]:
                # IDs need not be unique, so what was said has to agree.
                if self._goes_on(pending, turns[-1]):
                    return found
            elif pending.index == done and done and found.chain == chains[-2]:
                if self._goes_on(pending, turns[-2]):
                    return found
            return None
        if not any(results[-_LOOK_BACK:]):
            # An API without IDs: found by what its tools had returned when the reply came.
            asked, seen = turns[-1].question, _hashes(turns[-1].retrievals)
            for k in range(len(seen) - 1, -1, -1):
                found = self._by_call.get(self._loop(chains[-1], asked, seen[:k]))
                if found is not None and found.pending is not None:
                    if found.pending.index == done + 1 and found.chain == chains[-1]:
                        return found
        return None

    @staticmethod
    def _loop(chain: str, question: str, seen: list) -> str:
        said = json.dumps([chain, question, seen]).encode()
        return "loop:" + hashlib.sha256(said).hexdigest()

    @staticmethod
    def _goes_on(pending: _Pending, segment: _Segment) -> bool:
        """Whether a turn of a request is the one that is being answered, as far as that had come."""
        seen = _hashes(segment.retrievals[: len(pending.seen)])
        return seen == pending.seen and segment.question.startswith(pending.question)

    @staticmethod
    def _elsewhere(conversation: _Conversation, turns) -> bool:
        """Whether a conversation is busy with another question than the one a request asks at this point."""
        pending = conversation.pending
        return (
            pending is not None
            and pending.index == len(turns)
            and bool(pending.calls or pending.in_flight)
            and not turns[-1].question.startswith(pending.question)
        )

    def _branch(
        self, other: _Conversation, chain: str, done: int, thread: str | None
    ) -> _Conversation | None:
        """A conversation that goes on from where another stood. None where that is not known."""
        if other.base is None or other.base[0] != chain:
            return None
        _, session, inserts, reopens = other.base
        record = other.record_id
        if not done and thread:
            # Nothing was said before it: it is another conversation of the same name.
            count = self._names[thread] = self._names.get(thread, 0) + 1
            record = f"{thread}~{count}"
        made = _Conversation(other.thread_id, chain, record)
        made.session, made.agentic = session.copy(), True
        made.inserts, made.reopens = list(inserts), reopens
        made.lifetime, made.five_minutes = other.lifetime, other.five_minutes
        made.default_lifetime = other.default_lifetime
        made.model, made.policy = other.model, other.policy
        made.turns, made.at = done, other.at
        made.session.follows(other.session)
        # What they share is kept warm for the newest of them.
        other.pings_off = True
        if thread or done:
            self._index(made, chain)
        return made

    def _now(self, conversation: _Conversation) -> float:
        """The time, and never one this conversation has already passed: a clock can step back."""
        now = self.clock() if conversation.when is None else conversation.when
        conversation.at = max(now, conversation.at + 1e-3)
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
        if not isinstance(body, dict):
            return body
        dialect = wire.dialect_for(path)
        try:
            scan = dialect.scan(body)
            turns = _segments(scan.slots)
            if not turns or not scan.awaiting:
                return body
            conversation, chains = self._conversation(thread, turns, scan)
        except Exception:
            self.stats["passed_through"] += 1
            return body
        with conversation.lock:
            try:
                call = self._prepare(
                    conversation, chains, dialect, body, scan, turns, None, asks=False
                )
            except Exception:
                logger.warning(
                    "compache: a request was passed on unchanged", exc_info=True
                )
                self.stats["passed_through"] += 1
                return body
            self.stats["calls"] += 1
            return call.plain

    def heard(self, thread: str, usage: Any, *, model: str | None = None) -> None:
        """What one model call of a conversation shown to `view` was billed, as the provider counts it."""
        conversation = self._latest.get(thread)
        if conversation is None:
            return
        with conversation.lock:
            pending = conversation.pending
            if pending is None:
                return
            pending.calls += 1
            try:
                self._ledger(
                    conversation, "answer", wire.Reply(usage=usage, model=model)
                )
                if conversation.session is not None and pending.turn is not None:
                    conversation.session.replied(
                        pending.turn, at=self._now(conversation), usage=usage
                    )
            except Exception:
                logger.warning("compache: a call was not recorded", exc_info=True)

    def reopen(self, thread: str, reference: str, offset: Any = 0) -> dict:
        """The answer to the model's `reopen` call in a conversation shown to `view`, as it is to be read."""
        conversation = self._latest.get(thread)
        if conversation is None or conversation.session is None:
            return {
                "status": "error",
                "message": "there is nothing archived in this conversation",
            }
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

    def conclude(self, path: str, body: Any, *, thread: str) -> None:
        """Close the turn that `body` ends on, where the host's own reply ended it.

        `body` holds the conversation up to and including that answer. A
        conversation that was never shown to `view` is not recorded, and a
        turn that still waits for a tool goes on.
        """
        try:
            dialect = wire.dialect_for(path)
            scan = dialect.scan(body)
            turns = _segments(scan.slots)
            if not turns or not turns[-1].outputs or dialect.waits_for_tools(body):
                return
            conversation, chains = self._conversation(thread, turns, scan)
            with conversation.lock:
                now = self._now(conversation)
                self._catch_up(conversation, chains, turns, now, scan)
                pending = conversation.pending
                if pending is not None and pending.index == len(turns):
                    self._finish(
                        conversation, _link(chains[-1], turns[-1]), turns[-1], now
                    )
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
                conversation.policy = self.learner.policy(scan.model, bool(scan.tools))
            policy = conversation.policy
        if policy is None:
            return None
        if conversation.default_lifetime:
            policy = policy.at_default_lifetime()
        conversation.session = policy.session(name)
        conversation.agentic = policy.agentic
        conversation.lifetime = policy.cache_lifetime
        conversation.five_minutes = policy.spec.cache_ttl_seconds == _FIVE_MINUTES
        return policy

    @staticmethod
    def _asking(conversation, body):
        """A request made here, asking for the cache lifetime the conversation is planned under."""
        if conversation.session is None:
            return body
        if conversation.lifetime is not None:
            return wire.with_lifetime(body, conversation.lifetime)
        return wire.with_default_lifetime(body) if conversation.five_minutes else body

    @staticmethod
    def _plan_default(conversation) -> None:
        """Plan a conversation under the provider's default lifetime, from its next turn on."""
        conversation.default_lifetime = conversation.stale = True
        conversation.lifetime = None

    def _lifetime_refused(self, conversation) -> None:
        """The provider takes a request only under its default lifetime: so is everything planned from now on."""
        logger.warning(
            "compache: cache entries of %g seconds are refused, "
            "planning on the provider's default from now on",
            conversation.lifetime,
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
            turns = _segments(scan.slots) if scan is not None else []
            if not turns or not scan.awaiting:
                raise ValueError("nothing here to decide on")
            scan.stream = scan.stream or wire.streams(path)
            scan.model = scan.model or wire.model_of(path)
            conversation, chains = self._conversation(thread, turns, scan)
        except Exception:
            # A shape this does not know: the application's request goes out as it is.
            self.stats["passed_through"] += 1
            return send(body)
        with conversation.lock:
            try:
                conversation.when = at[0] if at is not None else None
                try:
                    if conversation.lifetime is not None and not wire.has_cache_mark(
                        dialect.with_cache_mark(body)
                    ):
                        # Nothing in this request could ask for it.
                        self._plan_default(conversation)
                    call = self._prepare(
                        conversation,
                        chains,
                        dialect,
                        body,
                        scan,
                        turns,
                        send,
                        asks,
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

    def _prepare(
        self, conversation, chains, dialect, body, scan, turns, send, asks=True
    ) -> _Call:
        """Bring the conversation up to this request and work out what to send for it."""
        now, number, current = self._now(conversation), len(turns), turns[-1]
        conversation.system_bytes = _size(scan.system)
        conversation.tools_bytes = _size(json.dumps(scan.tools)) if scan.tools else 0
        plain = body
        try:
            conversation.went = []
            self._catch_up(conversation, chains, turns, now, scan)
            pending, found = conversation.pending, current.retrievals
            earlier = None
            if (
                pending is not None
                and pending.index == number
                and _hashes(found[: len(pending.seen)]) != pending.seen
            ):
                # The application rewrote what its tools had returned: start over from what it says now.
                earlier, pending = pending, None
                if conversation.session is not None:
                    self._walk(conversation, chains, turns, now, scan)
            if pending is None or pending.index != number:
                pending = self._begin(conversation, chains, current, number, now, scan)
                if earlier is not None:
                    pending.goes_on_from(earlier)
            elif len(found) > len(pending.seen):
                self._feed(conversation, pending, found[len(pending.seen) :], now)
            conversation.last = None
            pending.sent_at = now
            if conversation.session is not None:
                plain = self._apply(conversation, dialect, body, scan, turns)
        except Exception:
            # The policy cannot be applied here. From now on this conversation is only recorded.
            logger.warning(
                "compache: policy not applied to a conversation", exc_info=True
            )
            self._stand_down(conversation, number, now)
            plain = body
        marked = plain
        if conversation.session is not None:
            marked = dialect.with_cache_mark(plain)
            for place in conversation.pending.marks:
                marked = dialect.with_boundary(marked, place)
            marked = self._asking(conversation, marked)
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
        )
        try:
            call.outgoing, call.asked, call.carrier = self._with_note(call, scan)
        except Exception:
            logger.warning("compache: a note was not asked", exc_info=True)
        return call

    def _carry(self, call: _Call, scan) -> Result:
        """Send the request, fall back to plainer forms of it if it is refused, and read the reply."""
        conversation, pending = call.conversation, call.conversation.pending
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
        result = call.send(call.outgoing)
        result = self._under_default(call, result)
        if result.status in _REFUSED and call.outgoing is not call.marked:
            # Refused with the note in it: send it without.
            self._note_refused(call)
            self.stats["retried"] += 1
            call.outgoing, call.asked, call.carrier = call.marked, False, None
            result = self._under_default(call, call.send(call.outgoing))
        if result.status in _REFUSED and call.outgoing is not call.body:
            # Refused with the policy applied: send what the application wrote.
            self.stats["retried"] += 1
            self.stats["policy_refused"] += 1
            self._stand_down(conversation, pending.index, pending.started_at)
            call.plain = call.marked = call.outgoing = call.body
            call.asked, call.carrier = False, None
            result = call.send(call.body)
        pending.asked = pending.asked or call.asked
        if result.status >= 400 or (result.json is None and result.events is None):
            return result
        if result.events is not None:
            return replace(result, events=self._stream(call, result, pending))
        return self._settle(call, result)

    def _under_default(self, call: _Call, result: Result) -> Result:
        """What became of a request. One that was refused goes out again under the default lifetime.

        Only that tells whether the lifetime was why: what a provider says about it differs.
        """
        conversation = call.conversation
        if result.status not in _REFUSED or conversation.lifetime is None:
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
                    self._feed(conversation, pending, late, now)
                # An agent's policy has taken the usage of every call that passed.
                counted = not conversation.agentic
                self._finish(conversation, chains[done], last, now, counted)
                return
        self._walk(conversation, chains, turns, now, scan)

    def _walk(self, conversation, chains, turns, now, scan) -> None:
        """Walk the policy through a conversation that is met in the middle, up to its last question."""
        done = len(turns) - 1
        conversation.pending, conversation.turns, conversation.stale = None, 0, False
        conversation.inserts, conversation.went = [], []
        conversation.base = conversation.quotes = None
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
        if self.recorder is not None:
            for k, segment in enumerate(turns[:done], 1):
                self._heard(conversation, k, segment.question, segment.retrievals)
                self._quotes(conversation).said(segment.answer)
        conversation.turns = done
        self._index(conversation, chains[done])

    def _shown(self, conversation, segment, at, scan):
        """A turn's question and what was retrieved for it, as the policy's session takes them."""
        session, found = conversation.session, segment.retrievals
        ahead = len(found)
        if conversation.agentic:
            # What came before the question is there at once. What its tools returned came round by round.
            ahead = next((i for i, part in enumerate(found) if part.answers), ahead)
        turn = session.prepare(
            segment.question,
            [part.text for part in found[:ahead]],
            at=at,
            system_prompt=scan.system,
            tools=scan.tools,
        )
        self._went(conversation, turn)
        return self._extended(conversation, turn, found[ahead:], at)

    def _extended(self, conversation, turn, found, at):
        """A turn with more that its tools returned, one round of tool calls at a time."""
        for part in (
            _rounds(found) if conversation.agentic else [found] if found else []
        ):
            turn = conversation.session.extend(turn, [f.text for f in part], at=at)
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
        if conversation.agentic and session is not None:
            conversation.base = (
                chains[-1],
                session.copy(),
                tuple(conversation.inserts),
                conversation.reopens,
            )
            if number > 1:
                self._by_base[chains[-1]] = conversation
                while len(self._by_base) > self._limit:
                    self._by_base.popitem(last=False)
        pending = conversation.pending = _Pending(
            number, now, question=current.question, seen=_hashes(found)
        )
        if session is not None:
            pending.turn = self._shown(conversation, current, now, scan)
        self._heard(conversation, number, current.question, found)
        return pending

    def _feed(self, conversation, pending, found, now) -> None:
        """Take in what the tools of a turn have returned since its last request."""
        first = len(pending.seen) + 1
        pending.seen += _hashes(found)
        if conversation.session is not None:
            pending.turn = self._extended(conversation, pending.turn, found, now)
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

    def _stand_down(self, conversation, number, now) -> None:
        """Stop applying the policy to a conversation. It is still recorded."""
        if conversation.session is not None:
            self.stats["stood_down"] += 1
        conversation.session = conversation.lifetime = None
        conversation.stale = False
        if conversation.pending is None or conversation.pending.index != number:
            conversation.pending = _Pending(number, now)
        conversation.pending.turn = None

    def _apply(self, conversation, dialect, body, scan, turns) -> Any:
        """The request with every archived part replaced by its notice, and the tool to fetch one back."""
        pending, name = conversation.pending, conversation.thread_id
        turn, out = pending.turn, body
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
                            shown = _GONE if i else notice[part_id]
                            out = wire.set_text(dialect, out, slot.path, shown)
            conversation.reopens = True
        said, places = dialect.said_so_far(body), self._places(conversation, turns)
        conversation.inserts = [
            insert
            for insert in conversation.inserts
            if insert.at <= len(said)
            and (
                any(conversation.session.shows_again(n) for n in insert.names)
                if conversation.agentic
                else insert.turn == pending.index
            )
        ]
        # From the last to the first, so that the places of the others hold.
        for insert in reversed(conversation.inserts):
            out = dialect.with_inserted(out, insert.at, insert.messages)
        put_in = sum(len(insert.messages) for insert in conversation.inserts)
        if conversation.agentic and scan.tools:
            # An agent works its way to a first cut soon enough. A tool that joins only then would
            # change what comes ahead of everything else that is cached.
            conversation.reopens = True
        if conversation.reopens and _REOPEN not in dialect.tool_names(body):
            # A model that is offered the tool while nothing is archived calls it for nothing. So in
            # a chat it comes with the first notice, and stays: the first cut rewrites the request
            # from near its start anyway, and after that the tool never changes what is cached.
            out = dialect.with_tool(out, _REOPEN_TOOL, _REOPEN_HOW)
        if self.recorder is not None:
            pending.held = wire.volume(out)
            pending.left = wire.volume(body) - pending.held
        pending.cut = pending.cut or bool(turn.archived_ids)
        # Where those are in the request as it goes out: behind messages of ours.
        shift = len(dialect.said_so_far(out)) - len(said) - put_in
        pending.marks = tuple(
            at
            + shift
            + sum(len(i.messages) for i in conversation.inserts if i.at <= at)
            for at in places
        )
        return out

    @staticmethod
    def _places(conversation, turns) -> list[int]:
        """Where a cut begins: the one that was made now, and the one that comes next.

        What lies ahead of them stays as it is, so a cache mark there lets the
        request that makes the cut read it. Given as the message that holds the
        first part to go, as the application counts its messages.
        """
        firsts = [*conversation.went[:1], conversation.holds]
        if firsts == [None]:
            return []
        at = {(n, True): i.at for i in conversation.inserts for n in i.names}
        for k, segment in enumerate(turns, 1):
            for j, found in enumerate(segment.retrievals, 1):
                part_id = f"{conversation.thread_id}:{k}:retrieval:{j}"
                at[part_id, False] = found.slots[0].message
        return sorted({at[first] for first in firsts if first in at})

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
        try:
            reply = dialect.reply(data)
            self._ledger(conversation, kind, reply, pending=call.pending)
            while (
                conversation.session is not None
                and reply.calls
                and all(c["name"] == _REOPEN for c in reply.calls)
            ):
                rounds += 1
                if rounds > most:
                    break
                answers = self._reopen(conversation, reply.calls, call, data)
                sent = self._asking(
                    conversation, dialect.after_tools(sent, data, answers)
                )
                again = call.send(sent)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                self._ledger(conversation, "reopen", reply, pending=call.pending)
            # A reply asked for a note that says nothing is asked again without the note first.
            if self._changed(call) and (
                rounds > most or not (call.asked or reply.text or reply.calls)
            ):
                # The model goes on fetching, or says nothing: what it was shown does not do for
                # this turn. It is asked once more, exactly as the application wrote it.
                again = self._as_written(call)
                if again.status >= 400 or again.json is None:
                    return again
                result, data = again, again.json
                reply = dialect.reply(data)
                self._ledger(conversation, "retry", reply, pending=call.pending)
            ours = [c for c in reply.calls if c["name"] == _REOPEN]
            if ours:
                # Called beside the application's own tools. The application never sees ours, and the
                # model finds its answer with theirs.
                if conversation.session is not None and len(ours) < len(reply.calls):
                    self._reopen(conversation, ours, call, data)
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

    def _stream(self, call: _Call, result: Result, pending) -> Iterator[bytes]:
        """A streamed reply passed on as it arrives, with our own tool calls answered on the way."""
        conversation, dialect = call.conversation, call.dialect
        most = _AGENT_ROUNDS if conversation.agentic else _ROUNDS
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
                    self._ledger(conversation, kind, editor, pending=pending)
                    if rounds > most:
                        # The model goes on fetching: it is asked as the application wrote it.
                        again, kind = self._as_written(call), "retry"
                    else:
                        answers = self._reopen(
                            conversation, editor.ours, call, editor.said
                        )
                        sent = self._asking(
                            conversation,
                            dialect.after_tools(sent, editor.said, answers),
                        )
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
                pending.in_flight -= 1
                try:
                    if not counted:
                        self._ledger(conversation, kind, editor, pending=pending)
                    if editor.finished:
                        if editor.ours and editor.calls:
                            self._reopen(conversation, editor.ours, call, editor.said)
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

    def _reopen(self, conversation, calls, asked: _Call | None = None, said=None):
        """Answers to the model's `reopen` calls, by call ID.

        `asked` is the call whose reply `said` made them. The application never
        sees them, so they are put into its later requests for as long as what
        they returned is new.
        """
        answers, names = {}, []
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
            answers[call["id"]] = json.dumps(_for_model(found), ensure_ascii=False)
            if not part or conversation.session.shows_again(part):
                part = f"reopened:{uuid.uuid4().hex[:12]}"
            names.append((part, answers[call["id"]]))
        if asked is not None and conversation.pending is not None:
            try:
                insert = _Insert(
                    len(asked.dialect.said_so_far(asked.body)),
                    asked.dialect.fetched(said, answers),
                    tuple(name for name, _ in names),
                    conversation.pending.index,
                )
                conversation.inserts.append(insert)
                for name, text in names:
                    conversation.session.reopened(name, text)
            except Exception:
                logger.warning("compache: a fetched part does not stay", exc_info=True)
        return answers

    def _after(self, call: _Call, text: str, calls) -> None:
        """A reply has been read: the turn goes on with tool calls, or ends with this answer."""
        conversation, pending = call.conversation, call.pending
        if pending is None or pending is not conversation.pending:
            return
        pending.calls += 1
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
        self._plan_pings(conversation, (call.dialect, call.plain, text, call.send))
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
        with self._lock:
            for key in keys:
                self._by_call[key] = conversation
                self._by_call.move_to_end(key)
            while len(self._by_call) > 8 * self._limit:
                self._by_call.popitem(last=False)
        if conversation.session is None or pending.turn is None:
            return
        conversation.session.replied(
            pending.turn,
            text,
            [made.get("arguments") for made in calls],
            at=self._now(conversation),
            usage=pending.last_usage,
        )
        self._plan_pings(conversation, (call.dialect, call.marked, None, call.send))

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
        said = spoken(text, [made.get("arguments") for made in calls])
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
                "needs": self._quotes(conversation).said(said),
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
            reply = self._whole(dialect, call.send(self._asking(conversation, body)))
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
            try:
                conversation.session.complete(
                    pending.turn,
                    segment.answer,
                    at=pending.ended_at,
                    usage=pending.first_usage if counted else None,
                    last_usage=pending.last_usage
                    if counted and pending.calls > 1
                    else None,
                    cited_part_ids=[i for i in pending.cited if i in visible],
                )
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
            steps=pending.steps if len(pending.steps) > 1 else (),
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
        if pending is not None and kind in ("answer", "reopen", "retry"):
            if pending.first_usage is None:
                pending.first_usage = source.usage
            pending.last_usage = source.usage
            pending.model = source.model or pending.model
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
            usage=canonical_usage(source.usage),
            cost=source.cost,
            archived=archived,
            left=left,
            lifetime=conversation.lifetime,
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
            self._whole(dialect, send(self._asking(conversation, body))),
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
