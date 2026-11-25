"""Policies for agents, whose turns are loops of many model calls.

A chatbot makes a call or two per message, and its history is decided on once
per message. An agent makes dozens, each one sending every tool result so far.
So here the unit is the call. Results are kept by age in rounds of tool calls:
the newest rounds always go out in full, older ones as a notice once that pays.

  - When the cache has expired, or a long request is billed dearer, leaving older
    results out is cheaper at once, and they go.
  - While it is alive, taking something out means writing what follows it
    again. That pays once what goes is large next to what stays behind it.
  - While a tool runs or a user is away, a ping keeps the cache if that costs
    less than losing it.
  - What opens every request, the instructions and the tools, may be cached for
    longer than the conversation behind it: that stays the same for a day.

How many rounds, how large, how many pings and which cache lifetimes is fitted
on recorded conversations. `Flow` holds what a conversation sends and what the
provider has cached of it. The fit, the replay and the runtime all run on it,
so what was priced is what happens.
"""

from __future__ import annotations

import json
from bisect import bisect_left, insort
from dataclasses import asdict, dataclass, replace

from .errors import ContractError
from .labels import normalize_label
from .models import binds_reasoning
from .optimizer.contracts import ModelSpec, digest, number, seconds
from .optimizer.distribution import _charge
from .optimizer.parts import NOTICE_BYTES, placeholder
from .quotes import spoken

__all__ = [
    "SCHEMA",
    "AgentEngine",
    "AgentRecipe",
    "Call",
    "Flow",
    "calls_of",
    "default_header",
    "default_recipe",
    "fit_agent",
    "is_agentic",
    "replay_agent",
    "sentences",
    "simulate",
]

SCHEMA = "compache-agent-v1"
# How long before a cache entry would expire a ping goes out.
PING_LEAD_SECONDS = 45.0
# A model's call for an archived result, in output tokens.
REOPEN_OUTPUT_TOKENS = 32
# Calls per turn above which a conversation is taken to be an agent's.
LOOP_CALLS = 4
# Least share of the calls that rely on an earlier result that find it in view.
FLOOR = 0.85
# Where few calls rely on one at all, the most of all calls that may miss one.
MOST_MISSED = 0.01
# So many relying calls make the share speak, and so many calls the count of misses.
_LEAST_RELYING, _LEAST_KNOWN = 40, 200
# The most rounds of results that stay because the model keeps fetching back.
_LONGEST = 100
# How far the provider's count of a text is taken to be from four bytes a token, at most.
_MOST_OFF = 4.0
_ROUNDS = (3, 5, 8, 12, 20, 30)
_RATIOS = (0.5, 1.0, 1.5, 2.0, 3.0, None)
_TOOL_PINGS = (0, 1, 3, 8, 11)
_USER_PINGS = (0, 1, 3)


@dataclass(frozen=True)
class AgentRecipe:
    """What an agent's conversation keeps. None keeps everything, or never cuts a live cache."""

    # Newest rounds of tool results that always go out in full.
    rounds: int | None = None
    # How large what goes has to be next to what would be written again, while the cache is alive.
    ratio: float | None = None
    # Most keep-warm pings while a tool runs, and while a user is away.
    tool_pings: int = 0
    user_pings: int = 0

    def __post_init__(self):
        if self.rounds is not None:
            number(self.rounds, "rounds", 1, integer=True)
        if self.ratio is not None:
            number(self.ratio, "ratio", 0)
        number(self.tool_pings, "tool_pings", 0, 60, integer=True)
        number(self.user_pings, "user_pings", 0, 60, integer=True)


@dataclass(frozen=True, slots=True)
class Call:
    """One model call: what had joined the conversation since the one before it, and what came back."""

    at: float
    end: float
    # Tokens of everything that is no tool result: the question, and what the reply before it said.
    said: int
    # The tool results that arrived, by ID and size.
    found: tuple[tuple[str, int], ...]
    output: int
    # Results of earlier rounds that the reply relies on. None where that is not known.
    needs: frozenset | None
    # Whether the reply waits for a tool. Otherwise it waits for the user.
    tool: bool
    # Reasoning of the reply before it that the request carries.
    thought: int = 0


class Talk(list):
    """The calls of one conversation, in order."""

    # Tokens of the instructions and tools that open every request.
    header: int = 0
    # When calls were made behind the same ones, in any conversation.
    times: list[float] = []

    def header_used(self, at: float) -> float | None:
        """When a call was last made behind these instructions and tools before `at`."""
        k = bisect_left(self.times, at)
        return self.times[k - 1] if k else None


class Flow:
    """A conversation as it goes out: what is in it, what went out as a notice, what is cached."""

    __slots__ = (
        "ahead",
        "archived",
        "at",
        "back",
        "binds",
        "cached",
        "cap",
        "ended",
        "freed",
        "held",
        "live",
        "longer",
        "names",
        "notice",
        "reasons",
        "recipe",
        "ripe",
        "round",
        "rounds",
        "shown",
        "size",
        "sizes",
        "spec",
        "whole",
    )

    def __init__(self, spec: ModelSpec, recipe: AgentRecipe, reserve: int = 0):
        self.spec, self.recipe = spec, recipe
        self.notice = spec.part_tokens("." * NOTICE_BYTES)
        self.cap = spec.context_tokens - reserve
        # One entry per block, in order: tokens as sent and in full, the result's ID, its round.
        self.shown: list[int] = []
        self.whole: list[int] = []
        self.names: list = []
        self.round: list[int] = []
        # Where the results still are that could become a notice, how many of them are old
        # enough, and what those would free.
        self.live: list[int] = []
        self.ripe = self.freed = 0
        # Reasoning of the turn that goes on, and all of it. Where the model binds it, what stands
        # behind a cut is lost with the cut.
        self.held: list[int] = []
        self.reasons: list[int] = []
        self.binds = binds_reasoning(spec.model_id)
        # The last block that went with the latest cut.
        self.ended: int | None = None
        # How many rounds results stay beyond what the recipe says.
        self.longer = 0
        # Up to which block the tokens ahead of it were counted last, and how many there are.
        self.ahead = (0, 0)
        self.sizes: dict = {}
        self.size = 0
        self.rounds = 0
        self.cached, self.at = 0, None
        self.archived: set = set()
        # Archived results that were fetched back and stand at the end again.
        self.back: set = set()

    def copy(self) -> Flow:
        made = Flow.__new__(Flow)
        for name in Flow.__slots__:
            value = getattr(self, name)
            held = isinstance(value, (list, set, dict))
            setattr(made, name, value.copy() if held else value)
        return made

    def add(
        self, tokens: int, name=None, *, again: bool = False, held: bool = False
    ) -> None:
        """Something joined the end. `again` marks what leaves no notice: a result that was fetched
        back, and the model's reasoning, which is `held` until `release` while its turn goes on.
        """
        self.shown.append(tokens)
        self.whole.append(tokens)
        self.names.append((name, again) if name is not None else None)
        self.round.append(self.rounds)
        self.size += tokens
        if name is None:
            return
        self.sizes.setdefault(name, tokens)
        if again:
            self.back.add(name)
        if held:
            self.held.append(len(self.shown) - 1)
            self.reasons.append(len(self.shown) - 1)
        elif again or tokens > self.notice:
            self.live.append(len(self.shown) - 1)

    def release(self) -> None:
        """The turn is over: its reasoning can go from here on."""
        if self.held:
            self.live = sorted(self.live + self.held)
            self.held, self.ripe, self.freed, self.ahead = [], 0, 0, (0, 0)

    def forget(self, i: int) -> None:
        """Block `i` is no longer sent, and what could go is counted again."""
        if self.shown[i] or self.whole[i]:
            self.size -= self.shown[i]
            self.shown[i] = self.whole[i] = 0
            for name in ("live", "held", "reasons"):
                setattr(self, name, [k for k in getattr(self, name) if k != i])
            self.ripe, self.freed, self.ahead = 0, 0, (0, 0)

    def settle(self, i: int, tokens: int) -> None:
        """Block `i`, which can no longer go, is taken to hold so many tokens more."""
        self.size += tokens
        self.shown[i] += tokens
        self.ahead = (0, 0)

    def resize(self, i: int, tokens: int) -> None:
        """What block `i`, which is in full view, turned out to hold. What could go is counted again."""
        self.size += tokens - self.shown[i]
        self.shown[i] = self.whole[i] = tokens
        if self.names[i] is not None:
            self.sizes[self.names[i][0]] = tokens
            if tokens > self.notice and i not in self.held and i not in self.live:
                # It is worth a notice after all.
                insort(self.live, i)
        self.ripe, self.freed, self.ahead = 0, 0, (0, 0)

    def stay_longer(self) -> None:
        """What went is wanted again, and not for the first time: results stay for twice as long."""
        if self.recipe.rounds is not None:
            self.longer = min(2 * self.longer + self.recipe.rounds, _LONGEST)
            self.ripe, self.freed, self.ahead = 0, 0, (0, 0)

    def gone(self, name) -> bool:
        """Whether a result is out of view."""
        return name in self.archived and name not in self.back

    def warm(self, at: float) -> bool:
        return self.at is not None and 0 <= at - self.at < self.spec.cache_ttl_seconds

    def _before(self, first: int) -> int:
        """How many tokens go out ahead of block `first`."""
        at, tokens = self.ahead
        if first != at:
            tokens += sum(self.shown[at:first])
            self.ahead = (first, tokens)
        return tokens

    def _lost(self, first: int) -> list[int]:
        """Reasoning that a cut at block `first` would take along: it was made behind what goes."""
        if not self.binds:
            return []
        goes = set(self.live[: self.ripe])
        return [i for i in self.reasons if i > first and i not in goes]

    def _pays(self, at: float, before: int, lost: list[int]) -> bool:
        spec, freed = self.spec, self.freed
        if self.size > self.cap:
            return True
        # Reasoning of the turn that goes on is only given up for as much again.
        going_on = set(self.held)
        worth = freed >= sum(self.shown[i] for i in lost if i in going_on)
        if not self.warm(at):
            return worth
        # What stays behind the cut would be written again.
        taken = sum(self.shown[i] for i in lost)
        rest = max(0, self.size - before - freed - taken)
        ratio = self.recipe.ratio
        if ratio is not None and worth and freed >= ratio * rest:
            return True
        # Or it is cheaper in this very call, as where a long request is billed dearer. What is lost
        # beside counts for no gain: it is taken to be read from the cache as before.
        held = min(self.cached, self.size)
        read, write = spec.read_multiplier, spec.write_multiplier
        if spec.long_input_threshold is None:
            again = max(0, held - min(held, before) - taken)
            return worth and (write - read) * again < write * freed
        after = self.size - freed - taken
        kept = read * taken * spec.input_price_factor(after)
        return _charge(spec, after, min(held, before, after), 0) + kept < (
            _charge(spec, self.size, held, 0)
        )

    def cut(self, at: float) -> list:
        """Turn older results into notices, if that pays now.

        Returns what went, oldest first: each by its name and whether it had been fetched back.
        """
        if self.recipe.rounds is None:
            return []
        if self.longer and self.size > (self.spec.long_input_threshold or self.cap):
            # There is no room for that, or it is paid for dearer.
            self.longer, self.ripe, self.freed, self.ahead = 0, 0, 0, (0, 0)
        limit, live = self.rounds - self.recipe.rounds - self.longer, self.live
        # They arrived in order, so the first that stays ends the search.
        while self.ripe < len(live) and self.round[live[self.ripe]] <= limit:
            i = live[self.ripe]
            self.freed += self.whole[i] - (0 if self.names[i][1] else self.notice)
            self.ripe += 1
        if not self.ripe:
            return []
        first = live[0]
        before, lost = self._before(first), self._lost(first)
        if not self._pays(at, before, lost):
            return []
        if self.warm(at):
            self.cached = min(self.cached, before)
        went = []
        for i in [*live[: self.ripe], *lost]:
            name, again = self.names[i]
            self.size -= self.shown[i] - (0 if again else self.notice)
            self.shown[i] = 0 if again else self.notice
            if again:
                self.back.discard(name)
            else:
                self.archived.add(name)
            went.append((name, again))
        self.ended = live[self.ripe - 1]
        self.live, self.ripe, self.freed = live[self.ripe :], 0, 0
        gone = {*live[: len(live) - len(self.live)], *lost}
        self.reasons = [i for i in self.reasons if i not in gone]
        if lost:
            self.live = [i for i in self.live if i not in gone]
            self.held = [i for i in self.held if i not in gone]
        return went

    def charge(self, at: float, output: int) -> float:
        """What the request costs as it stands, in fresh-input-token equivalents."""
        shared = min(self.cached, self.size) if self.warm(at) else 0
        return _charge(self.spec, self.size, shared, output)

    def sent(self, at: float, cached: int | None = None) -> None:
        self.cached = self.size if cached is None else cached
        self.at = at

    @property
    def interval(self) -> float:
        """How long after a request or ping the next ping is due."""
        ttl = self.spec.cache_ttl_seconds
        return ttl - min(PING_LEAD_SECONDS, ttl / 4)


# -- recorded conversations ------------------------------------------------------


def _steps(run, spec):
    """The calls of a run: start, end, results seen, reply tokens that join the history, output, needs,
    input, and reasoning that stays.
    """
    start, end = seconds(run.started_at), seconds(run.ended_at)
    total = len(run.retrievals)
    if run.steps:
        for step in run.steps:
            said = spec.text_tokens("." * step.bytes)
            yield (
                seconds(step.started_at),
                seconds(step.ended_at),
                step.results,
                said,
                said if step.output_tokens is None else step.output_tokens,
                step.needs,
                step.input_tokens,
                step.thought_tokens,
            )
        return
    # Where the calls were only counted, the results are taken to have come one round at a time.
    count = run.calls or (2 if total else 1)
    reply = spec.part_tokens(run.output.text or "")
    for k in range(count):
        at = start + (end - start) * k / count
        done = start + (end - start) * (k + 1) / count
        seen = total if count == 1 else round(total * k / (count - 1))
        last = k + 1 == count
        said = reply if last else 0
        yield at, done, seen, said, said, None, None, 0


def calls_of(thread, spec: ModelSpec, header_tokens: int = 0) -> Talk:
    """A thread as the calls it was made of.

    Where the provider's count of a request is on record, what had joined since
    the one before it is taken to hold what the count grew by. Elsewhere sizes
    are off by as much as they were there.
    """
    calls = Talk()
    calls.header = said = header_tokens
    held = known = measured = thought = 0
    for run in thread.runs:
        said += spec.part_tokens(run.input.text or "")
        sizes = [
            (p.part_id, spec.part_tokens(p.text or ""))
            for p in run.retrievals
            if p.exposed_to_model
        ]
        steps, seen = list(_steps(run, spec)), 0
        for k, step in enumerate(steps):
            at, end, results, reply, output, needs, counted, after = step
            results = max(seen, min(results, len(sizes)))
            found, seen = tuple(sizes[seen:results]), results
            fresh = {name for name, _ in found}
            ruled = said + sum(tokens for _, tokens in found)
            off = known / measured if measured else 1.0
            grown = counted - held - thought if counted is not None else 0
            if grown > 0 and found and ruled:
                # The reply and what surrounds a result are counted too: all of it grows alike.
                off = grown / ruled
                known, measured = known + grown, measured + ruled
            if found:
                off = min(_MOST_OFF, max(1 / _MOST_OFF, off))
                said = round(said * off)
                found = tuple((name, round(tokens * off)) for name, tokens in found)
            if grown > 0:
                # What no size accounts for, such as instructions the log leaves out, was said.
                counted_said = max(0, grown - sum(tokens for _, tokens in found))
                if not calls and said:
                    calls.header = min(
                        counted_said, round(header_tokens * counted_said / said)
                    )
                said = counted_said
            held += said + thought + sum(tokens for _, tokens in found)
            calls.append(
                Call(
                    at,
                    end,
                    said,
                    found,
                    output,
                    None
                    if needs is None
                    else frozenset(n for n in needs if n not in fresh),
                    k + 1 < len(steps),
                    thought,
                )
            )
            said, thought = reply, after
    return calls


def is_agentic(threads) -> bool:
    """Whether these conversations are loops of calls, so that a policy has to decide call by call."""
    counts = sorted(len(run.steps) or run.calls or 1 for t in threads for run in t.runs)
    # By the turn a call is most likely to be in: one long loop among short turns is what costs.
    if not counts:
        return False
    half, passed = sum(counts) / 2, 0
    for count in counts:
        passed += count
        if passed >= half:
            return count > LOOP_CALLS
    return False


# -- pricing ---------------------------------------------------------------------


def _priced(conversation, recipe, spec, reserve, header, heeded=True):
    """What one conversation costs under a recipe. `heeded` says whether what its calls need and is out
    of view is fetched back. None where a request would not fit.
    """
    cost = 0.0
    missed = pings = cuts = peak = 0
    lasting = header or spec
    flow = Flow(spec, recipe, reserve)
    opens = getattr(conversation, "header", 0)
    if opens < spec.cache_min_tokens or lasting.cache_ttl_seconds <= 0:
        opens = 0
    for i, call in enumerate(conversation):
        if call.thought:
            flow.add(call.thought, ("thought", i), again=True, held=True)
        if i and not conversation[i - 1].tool:
            # A new turn: what was reasoned in the ones before it can go.
            flow.release()
        flow.add(call.said)
        if call.found:
            flow.rounds += 1
            for name, tokens in call.found:
                flow.add(tokens, name)
        cuts += bool(flow.cut(call.at))
        if opens and not flow.warm(call.at):
            # The conversation has gone from the cache. What opens it may be there still.
            used = conversation.header_used(call.at)
            there = used is not None and call.at - used < lasting.cache_ttl_seconds
            flow.sent(call.at, opens if there else 0)
            if header is not None and not there:
                more = header.write_multiplier - spec.write_multiplier
                cost += opens * more * spec.input_price_factor(flow.size)
        if flow.size > flow.cap:
            return None
        peak = max(peak, flow.size)
        lost = [n for n in call.needs or () if flow.gone(n)] if heeded else ()
        if lost:
            # The model asks for them first: a call that reads the same request.
            missed += 1
            cost += flow.charge(call.at, REOPEN_OUTPUT_TOKENS)
            flow.sent(call.at)
            for name in lost:
                flow.add(flow.sizes[name], name, again=True)
        cost += flow.charge(call.at, call.output)
        flow.sent(call.at)
        most = recipe.tool_pings if call.tool else recipe.user_pings
        if most:
            back = conversation[i + 1].at if i + 1 < len(conversation) else None
            for n in range(1, most + 1):
                due = call.at + n * flow.interval
                if due < call.end or (back is not None and due >= back):
                    break
                if flow.size < spec.cache_min_tokens:
                    break
                cost += flow.charge(due, spec.ping_output_tokens)
                flow.sent(due)
                pings += 1
    return cost, missed, pings, cuts, peak


def simulate(
    conversations,
    recipe: AgentRecipe,
    spec: ModelSpec,
    reserve: int = 0,
    header: ModelSpec | None = None,
):
    """What conversations cost under a recipe, every call and ping by the provider's cache rules.

    `header` is the lifetime that instructions and tools are cached for, where
    that is another than the conversation's. What fetching back costs is seen
    on the calls whose needs are known, and taken to be as much on the others.
    """
    cost = fetched = 0.0
    calls = known = asked = missed = pings = cuts = peak = 0
    for conversation in conversations:
        found = _priced(conversation, recipe, spec, reserve, header)
        if found is None:
            return None
        whole = plain = found[0]
        if found[1]:
            # The same with nothing fetched back: the difference is what that costs, here and later.
            plain = _priced(conversation, recipe, spec, reserve, header, False)[0]
        cost, fetched = cost + plain, fetched + whole - plain
        missed, pings, cuts = missed + found[1], pings + found[2], cuts + found[3]
        peak = max(peak, found[4])
        calls += len(conversation)
        known += sum(call.needs is not None for call in conversation)
        asked += sum(bool(call.needs) for call in conversation)
    return {
        "cost": cost + fetched * (calls / known if known else 1),
        "calls": calls,
        "asked": asked,
        "missed": missed,
        "pings": pings,
        "cuts": cuts,
        "largest_request": peak,
        "coverage": 1 - missed / asked if asked else None,
    }


def _wanted(threads, labels) -> dict:
    """The earlier results that labelled turns needed, by thread and turn. Turns nobody could tell of are left out."""
    by_id = {t.thread_id: t for t in threads}
    owner = {r.run_id: t for t in threads for r in t.runs}
    wanted: dict = {}
    for row in labels or ():
        thread = by_id.get(row.get("thread_id")) if isinstance(row, dict) else None
        if thread is None and isinstance(row, dict):
            thread = owner.get(row.get("target_run_id", row.get("run_id")))
        if thread is None:
            raise ContractError(
                "label targets must be unique runs in the supplied traces"
            )
        label = normalize_label(row, thread)
        index = {r.run_id: r.run_index for r in thread.runs}
        turn = index.get(label.get("target_run_id"))
        if turn is None or label.get("status") == "unassessable":
            continue
        named = {
            source["part_id"]
            for requirement in label.get("task_requirements") or ()
            for way in (requirement.get("alternatives") or ())[:1]
            for source in way.get("sources") or ()
        }
        found = {p for p in named if thread.part_origin.get(p, ("",))[0] == "retrieval"}
        # That a first turn needs nothing from before it says nothing.
        if found or turn > 1:
            wanted.setdefault((thread.thread_id, turn), set()).update(found)
    return wanted


def _conversations(threads, spec, header_tokens, labels=()):
    """Threads as calls, with what their labels say a turn needed on the turn's last call.

    `header_tokens` is the size of instructions and tools, or one by thread ID.
    """
    wanted = _wanted(threads, labels)
    named = {thread_id for thread_id, _ in wanted}
    made: list[Talk] = []
    for thread in threads:
        opens = header_tokens
        if isinstance(header_tokens, dict):
            opens = header_tokens.get(thread.thread_id, 0)
        calls = calls_of(thread, spec, opens)
        if thread.thread_id in named:
            at = 0
            for run in thread.runs:
                at += len(list(_steps(run, spec)))
                more = wanted.get((thread.thread_id, run.run_index))
                last = calls[at - 1]
                # A call that nothing is known of stays that, unless its turn was only counted.
                if more is not None and (last.needs is not None or not run.steps):
                    needs = (last.needs or frozenset()) | frozenset(more)
                    calls[at - 1] = replace(last, needs=needs)
        made.append(calls)
    # Conversations that open with as many tokens are taken to open with the same.
    behind: dict[int, list[float]] = {}
    for calls in made:
        behind.setdefault(calls.header // 256, []).extend(c.at for c in calls)
    for times in behind.values():
        times.sort()
    for calls in made:
        calls.times = behind[calls.header // 256]
    return made


def _ages(conversations) -> list[int]:
    """How many rounds back each result lay that a call relied on, 1 for the newest round but one."""
    found = []
    for calls in conversations:
        arrived, rounds = {}, 0
        for call in calls:
            if call.found:
                rounds += 1
                arrived.update((name, rounds) for name, _ in call.found)
            found += [rounds - arrived[n] for n in call.needs or () if n in arrived]
    return sorted(found)


def _cuts(conversations, spec) -> list[AgentRecipe]:
    """The cuts worth pricing: a grid that follows how far back these conversations reach."""
    if sum(c.needs is not None for calls in conversations for c in calls) < (
        _LEAST_KNOWN
    ):
        # Too little says what can go: only the cut that public runs taught is considered.
        return [replace(default_recipe(spec), tool_pings=0, user_pings=0)]
    ages = _ages(conversations)
    reach = {
        max(_ROUNDS[0], ages[min(len(ages) - 1, int(q * len(ages)))] + 1)
        for q in (0.8, 0.9, 0.95)
        if ages
    }
    return [
        AgentRecipe(rounds, ratio)
        for rounds in sorted({*_ROUNDS} | reach)
        for ratio in _RATIOS
    ]


def fit_agent(
    threads,
    spec: ModelSpec,
    *,
    labels=(),
    lifetimes=(),
    header_tokens: int | dict = 0,
    floor: float = FLOOR,
    reserve: int = 8192,
    pings: bool = True,
) -> dict:
    """Fit what an agent's conversations keep, on recorded ones. Returns a policy artifact.

    `floor` is the least share of calls that rely on an earlier result and find
    it in view. Where calls rarely rely on one, the misses are held to a share
    of all calls instead. They are priced as fetching it back. Every lifetime
    is priced for the whole request, and the longer ones for its instructions
    and tools alone. `pings=False` leaves keep-warm pings out of it.
    """
    conversations = _conversations(threads, spec, header_tokens, labels)
    if sum(map(len, conversations)) < 2:
        raise ContractError("fitting an agent's policy needs recorded calls")
    specs = {s.cache_ttl_seconds: s for s in (spec, *lifetimes)}
    cuts = _cuts(conversations, spec)
    known = sum(c.needs is not None for calls in conversations for c in calls)
    relying = sum(bool(c.needs) for calls in conversations for c in calls)
    rows = []

    def price(recipe, asked, header=None):
        result = simulate(conversations, recipe, asked, reserve, header)
        if result is not None:
            rows.append(
                {
                    "recipe": asdict(recipe),
                    "cache_ttl_seconds": asked.cache_ttl_seconds,
                    "header_ttl_seconds": (header or asked).cache_ttl_seconds,
                    "cost": result["cost"],
                    "coverage": result["coverage"],
                    "missed_calls": result["missed"],
                    "pings": result["pings"],
                    "cuts": result["cuts"],
                }
            )
        return result

    def allowed(row):
        # A cut that public runs taught is not held to a floor that these say too little about.
        if row["recipe"]["rounds"] is None or len(cuts) == 1:
            return True
        missed = row["missed_calls"]
        if relying >= _LEAST_RELYING and row["coverage"] >= floor:
            return True
        return (missed + 1) / known <= MOST_MISSED

    for asked in specs.values():
        price(AgentRecipe(), asked)
        for recipe in cuts:
            price(recipe, asked)
    # Pings cost and save the same whatever else goes out, so only the best cuts are tried with them.
    best: dict = {}
    for row in sorted(filter(allowed, rows), key=lambda r: r["cost"]):
        best.setdefault(row["cache_ttl_seconds"], []).append(
            AgentRecipe(**row["recipe"])
        )
    longer = {
        ttl: [s for s in specs.values() if s.cache_ttl_seconds > ttl] for ttl in specs
    }
    for ttl, recipes in best.items():
        for recipe in {*recipes[:4], AgentRecipe()}:
            paid = {}
            for tool in _TOOL_PINGS[1:] if pings else ():
                result = price(replace(recipe, tool_pings=tool), specs[ttl])
                if result is not None:
                    paid[tool] = result["cost"]
            tool = min(paid, key=paid.get) if paid else 0
            for header in (None, *longer[ttl]):
                for some in sorted({0, tool}):
                    for user in _USER_PINGS if pings else (0,):
                        if header is not None or user:
                            pinged = replace(recipe, tool_pings=some, user_pings=user)
                            price(pinged, specs[ttl], header)

    def cheapest(some):
        return min(
            some,
            key=lambda r: (
                r["cost"],
                -(r["coverage"] or 0),
                r["cache_ttl_seconds"],
                r["header_ttl_seconds"],
                r["recipe"]["tool_pings"] + r["recipe"]["user_pings"],
            ),
        )

    usable = [r for r in rows if allowed(r)]
    if not usable:
        raise ContractError(
            "no candidate policy fits the context window on these conversations"
        )
    chosen = cheapest(usable)
    whole = [r for r in usable if r["recipe"]["rounds"] is None]
    plain = [
        r
        for r in whole
        if r["cache_ttl_seconds"] == spec.cache_ttl_seconds
        and r["header_ttl_seconds"] == spec.cache_ttl_seconds
        and not r["recipe"]["tool_pings"]
        and not r["recipe"]["user_pings"]
    ]
    keep = plain[0]["cost"] if plain else None
    asked = specs[chosen["cache_ttl_seconds"]]
    header = specs[chosen["header_ttl_seconds"]]
    return {
        "schema_version": SCHEMA,
        "model_spec": asked.to_dict(),
        **({} if asked is spec else {"standard_spec": spec.to_dict()}),
        **({} if header is asked else {"header_spec": header.to_dict()}),
        "recipe": chosen["recipe"],
        "reserve_tokens": reserve,
        "solve": {
            "unit": "fresh_input_token_equivalents",
            "calls": sum(map(len, conversations)),
            "turns": sum(len(t.runs) for t in threads),
            "relying_calls": relying,
            "known_calls": known,
            "floor": floor,
            "chosen": chosen,
            "keep_everything_cost": keep,
            "caching_alone": cheapest(whole) if whole else None,
            "saving_fraction": 1 - chosen["cost"] / keep if keep else None,
            "candidates": rows,
        },
        "training_threads": sorted(digest(t.thread_id) for t in threads),
    }


def replay_agent(
    artifact: dict, threads, *, labels=(), header_tokens: int | dict = 0
) -> dict:
    """What a policy would have cost on other conversations, next to keeping everything."""
    spec = ModelSpec.from_dict(artifact["model_spec"])
    standard = ModelSpec.from_dict(
        artifact.get("standard_spec", artifact["model_spec"])
    )
    conversations = _conversations(threads, spec, header_tokens, labels)
    reserve = artifact.get("reserve_tokens", 0)
    header = artifact.get("header_spec")
    ours = simulate(
        conversations,
        AgentRecipe(**artifact["recipe"]),
        spec,
        reserve,
        ModelSpec.from_dict(header) if header else None,
    )
    keep = simulate(conversations, AgentRecipe(), standard, reserve)
    if ours is None:
        raise ContractError("a conversation does not fit the context window")
    return {
        "threads": len(conversations),
        "turns": sum(len(t.runs) for t in threads),
        "calls": ours["calls"],
        "policy_cost": ours["cost"],
        "keep_everything_cost": keep["cost"] if keep else None,
        "saving": 1 - ours["cost"] / keep["cost"] if keep and keep["cost"] else None,
        "relying_calls": ours["asked"],
        "coverage": ours["coverage"],
        "missed_calls": ours["missed"],
        "pings": ours["pings"],
        "cuts": ours["cuts"],
    }


def default_recipe(spec: ModelSpec) -> AgentRecipe:
    """Where to start without recorded conversations.

    What does well over seven kinds of sessions made of 174 public coding-agent runs, from
    one task at a stretch to prompts of eight calls with a user in between, and over sessions
    of real issues with Claude Code and Codex (benchmarks/agents).
    """
    if spec.cache_ttl_seconds <= 0:
        return AgentRecipe(rounds=8, ratio=0.0)
    return AgentRecipe(rounds=8, ratio=2.0, tool_pings=3)


def default_header(spec: ModelSpec, lifetimes) -> ModelSpec | None:
    """The lifetime to start with for instructions and tools: the longest a provider can be asked for.

    They stay the same from session to session, so an entry that lives an hour is written once
    and read for as long as somebody is at work.
    """
    longer = [s for s in lifetimes if s.cache_ttl_seconds > spec.cache_ttl_seconds]
    return max(longer, key=lambda s: s.cache_ttl_seconds, default=None)


# -- in words ----------------------------------------------------------------------


def _count(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def sentences(recipe: AgentRecipe) -> list[str]:
    """What a recipe does, in sentences."""
    lines = []
    if recipe.rounds is None:
        lines.append("Keep every tool result.")
    else:
        which = (
            "the newest round of tool results"
            if recipe.rounds == 1
            else f"the newest {recipe.rounds} rounds of tool results"
        )
        lines.append(f"Keep {which} in full. Older ones become a notice")
        if recipe.ratio is None:
            lines[-1] += " once the cache has expired."
        else:
            lines[-1] += (
                " once the cache has expired, or once they are "
                f"{recipe.ratio:g} times as large as what follows them."
            )
    pings = []
    if recipe.tool_pings:
        pings.append(f"{_count(recipe.tool_pings, 'keep-warm ping')} while a tool runs")
    if recipe.user_pings:
        noun = "" if pings else " keep-warm ping" + "s" * (recipe.user_pings != 1)
        pings.append(f"{recipe.user_pings}{noun} while a user is away")
    lines.append(
        f"Send up to {' and '.join(pings)}." if pings else "Send no keep-warm pings."
    )
    return lines


# -- at runtime ----------------------------------------------------------------------


@dataclass(frozen=True)
class AgentView:
    parts: tuple
    archived_ids: frozenset
    tokens: int


@dataclass(frozen=True)
class AgentPlan:
    request: object
    view: AgentView
    decision: dict


class AgentEngine:
    """Decides on one conversation, call by call. It stands behind a `compache.Session`."""

    def __init__(self, artifact: dict, thread_id: str):
        self.spec = ModelSpec.from_dict(artifact["model_spec"])
        self.recipe = AgentRecipe(**artifact["recipe"])
        self.thread_id = thread_id
        self.flow = Flow(self.spec, self.recipe, artifact.get("reserve_tokens", 0))
        self._known: set = set()
        self._sent_at = None
        # Tokens of text the turn's replies have added so far.
        self._spoken = 0
        self._due: tuple = ()
        # What has joined since the provider last counted: the block, and its tokens by our ruler.
        self._fresh: list = []
        # What tool results came to by the provider's count and by our ruler, and the ratio of the two.
        self._counted = self._ruled = 0
        self._off = 1.0
        # Where reasoning of its replies stands: the block, by message. Whether the provider counts it
        # while its turn goes on, and once that is over, as far as that is known.
        self._thoughts: dict = {}
        # The messages whose reasoning is out of view for good, whatever it is counted as.
        self._out: set = set()
        self._counts_thoughts: bool | None = None
        self._keeps_thoughts: bool | None = None
        # Results of the request under way that hold more than text, and what the text of those that
        # can go is taken to hold.
        self._more: list = []
        self._pictures: dict = {}
        # Where the cut ended that the request under way was made with.
        self._just_cut: int | None = None
        # The turn that is being answered, and how everything stood before it.
        self._turn: tuple | None = None

    def copy(self) -> AgentEngine:
        made = AgentEngine.__new__(AgentEngine)
        made.__dict__.update(self.__dict__)
        made.flow, made._known = self.flow.copy(), set(self._known)
        made._fresh, made._thoughts = list(self._fresh), dict(self._thoughts)
        made._out, made._more = set(self._out), list(self._more)
        made._pictures = dict(self._pictures)
        return made

    def _begins(self, run_id) -> None:
        """A turn begins, or begins again: a retry starts from where the turn did."""
        if self._turn is not None and self._turn[0] == run_id:
            tried, before = self.copy(), self._turn
            self.__dict__.update(before[1].copy().__dict__)
            self._turn = before
            self.follows(tried)
        else:
            self._turn = None
            self._turn = (run_id, self.copy())

    def follows(self, other: AgentEngine) -> None:
        """Take over how warm the cache is from a conversation that went on from the same start."""
        mine, theirs = self.flow, other.flow
        if (
            mine.at is not None
            and theirs.at is not None
            and theirs.at > mine.at
            and theirs.archived == mine.archived
        ):
            mine.sent(theirs.at, min(mine.size, theirs.cached))

    # -- a request -------------------------------------------------------------

    def _arrived(self, parts) -> None:
        fresh = [p for p in parts if p.part_id not in self._known]
        if any(p.kind == "retrieval" for p in fresh):
            self.flow.rounds += 1
        for part in fresh:
            self._known.add(part.part_id)
            named = part.part_id if part.kind == "retrieval" else None
            ruled = self.spec.part_tokens(part.text)
            if part.stays or part.more:
                # What it holds besides its text has tokens that no ruler tells.
                self._more.append(len(self.flow.shown))
                if part.stays:
                    named = None
                else:
                    self._pictures[named] = round(ruled * self._off)
            self._joins(ruled, named)

    @property
    def left_more(self) -> int:
        """What pictures and the like that are out of view were counted as."""
        sizes = self.flow.sizes
        gone = (n for n in self._pictures if self.flow.gone(n))
        return sum(max(0, sizes[n] - self._pictures[n]) for n in gone)

    def _joins(self, ruled: int, name=None) -> None:
        self._fresh.append((len(self.flow.shown), ruled))
        self.flow.add(round(ruled * self._off), name)

    @property
    def holds(self):
        """What goes first with the next cut."""
        live = self.flow.live
        return self.flow.names[live[0]] if live else None

    def _plan(self, request, at) -> AgentPlan:
        self._due, self._sent_at = (), at
        went = self.flow.cut(at)
        self._just_cut = self.flow.ended if went else None
        archived = frozenset(self.flow.archived)
        parts = tuple(
            replace(p, text=placeholder(self.thread_id, p))
            if p.part_id in archived
            else p
            for p in request.parts
        )
        decision = {
            "action": "cut" if went else "keep",
            "went": tuple(went),
            "holds": self.holds,
        }
        return AgentPlan(
            request, AgentView(parts, archived, max(0, self.flow.size)), decision
        )

    def prepare(self, request, *, started_at, system_prompt, tools, **_) -> AgentPlan:
        self._begins(request.run_id)
        if not self._known:
            header = f"{system_prompt or ''}{json.dumps(tools) if tools else ''}"
            self._joins(self.spec.text_tokens(header))
        self._spoken = 0
        self.flow.release()
        if self._keeps_thoughts is False:
            self._drop(self._thoughts)
        self._arrived(request.parts)
        return self._plan(request, started_at)

    def extend(self, plan, parts, *, at=None) -> AgentPlan:
        """The results of a round of tool calls, and the request that carries them."""
        parts = tuple(parts)
        request = replace(plan.request, parts=plan.request.parts + parts)
        self._arrived(parts)
        return self._plan(request, self._sent_at if at is None else at)

    # -- its reply --------------------------------------------------------------

    def _seen(self, counters) -> None:
        """Take the provider's count for the request, and what it holds in its cache now."""
        reported = counters.get("input_tokens")
        cached = None
        if reported is not None:
            self._measure(reported)
            read = counters.get("cached_input_tokens")
            written = counters.get("cache_write_input_tokens")
            cached = reported
            if written:
                cached = min(reported, (read or 0) + written)
            elif read is not None and self.spec.write_multiplier > 1:
                # A model that bills writes wrote nothing: what was read is all that is cached.
                cached = read
            block = self.spec.cache_block_tokens
            cached = cached // block * block
            if cached < self.spec.cache_min_tokens:
                cached = 0
        self.flow.sent(self._sent_at, cached)

    # -- its reasoning -----------------------------------------------------------

    def thought(self, place, tokens: int) -> None:
        """Reasoning of a reply's joined the conversation, in the message at that place."""
        name = ("thought", place)
        if (
            tokens > 0
            and self._counts_thoughts is not False
            and name not in self._known
        ):
            self._known.add(name)
            self._thoughts[place] = len(self.flow.shown)
            self.flow.add(tokens, name, again=True, held=True)

    def carries(self, places) -> None:
        """Say which messages still hold reasoning: what an application leaves out itself is not sent."""
        for place in [p for p in self._thoughts if p not in places]:
            self.flow.forget(self._thoughts.pop(place))

    def without(self, places) -> list:
        """Those of the messages with reasoning whose reasoning has gone out of view."""
        names = self.flow.back
        return [
            p
            for p in places
            if p in self._out or (p in self._thoughts and ("thought", p) not in names)
        ]

    @property
    def counts_thoughts(self) -> bool:
        return self._counts_thoughts is not False and self._keeps_thoughts is not False

    def _drop(self, places) -> None:
        """The provider does not count the reasoning in those messages, so it takes none of the room."""
        for place in list(places):
            if ("thought", place) not in self.flow.back:
                self._out.add(place)
            self.flow.forget(self._thoughts.pop(place))

    def unbound(self, places) -> None:
        """Reasoning in those messages was made behind something else than is sent now: it stays out."""
        self._drop([p for p in places if p in self._thoughts])
        self._out.update(places)

    def _counted_thoughts(self, reported: int) -> None:
        """See whether the provider counts reasoning: each once, on a request that nothing else changed."""
        flow = self.flow
        held = {p: i for p, i in self._thoughts.items() if i in flow.held}
        over = {p: i for p, i in self._thoughts.items() if i not in flow.held}
        for name, places in (("_counts_thoughts", held), ("_keeps_thoughts", over)):
            tokens = sum(flow.shown[i] for i in places.values())
            if getattr(self, name) is None and tokens >= 64:
                counts = reported - flow.size >= -tokens / 2
                setattr(self, name, counts)
                if not counts:
                    self._drop(places)

    def _measure(self, reported: int) -> None:
        """Let what has joined hold what the provider's count of the request grew by."""
        flow = self.flow
        # What a cut has taken already is a notice, and that is as long as it is.
        fresh = [(i, r) for i, r in self._fresh if r and flow.shown[i] == flow.whole[i]]
        more = [i for i in self._more if flow.shown[i] == flow.whole[i]]
        self._fresh, self._more = [], []
        if self._just_cut is None:
            self._counted_thoughts(reported)
        hidden = reported - flow.size
        if more and hidden > 0:
            # A picture's tokens outweigh whatever else was not told by the ruler.
            for i in more:
                flow.resize(i, flow.shown[i] + hidden // len(more))
            flow.resize(more[-1], flow.shown[more[-1]] + hidden % len(more))
            return
        if self._just_cut is not None:
            # What went was larger, or smaller, than it was counted as.
            flow.settle(self._just_cut, hidden)
            return
        held = sum(flow.shown[i] for i, _ in fresh)
        if not held or held + hidden <= 0:
            if hidden > 0:
                flow.add(hidden)
            elif hidden and flow.shown:
                # Something has gone from it, and nothing says where: ahead of all that can go.
                flow.settle(0, hidden)
            return
        ruled = sum(r for _, r in fresh)
        if any(flow.names[i] is not None for i, _ in fresh):
            self._counted += held + hidden
            self._ruled += ruled
            off = self._counted / self._ruled
            self._off = min(_MOST_OFF, max(1 / _MOST_OFF, off))
            flow.notice = round(
                self.spec.part_tokens("." * NOTICE_BYTES) * max(1.0, self._off)
            )
        left = held + hidden
        for i, r in fresh[:-1]:
            share = (held + hidden) * r // ruled
            flow.resize(i, share)
            left -= share
        flow.resize(fresh[-1][0], left)

    def _wait(self, most, at) -> None:
        flow = self.flow
        if most and flow.cached and flow.size <= flow.cap:
            times = (self._sent_at + n * flow.interval for n in range(1, most + 1))
            self._due = tuple(due for due in times if due >= at)

    def replied(
        self,
        plan,
        *,
        at,
        text="",
        calls=(),
        output_tokens=None,
        thought=0,
        place=None,
        **counters,
    ):
        """A reply that calls tools is in. The next request comes when they are done."""
        self._seen(counters)
        said = self.spec.text_tokens(text) if text else 0
        self._spoken += said
        called = spoken("", calls)
        visible = said + (self.spec.text_tokens(called) if called else 0)
        if place is not None:
            self.thought(place, thought)
        # Reasoning that was billed and stays out of the conversation may not be told apart from what
        # was said, so the next count is taken to speak of the reply too.
        self._fresh.append((len(self.flow.shown), visible))
        self.flow.add(
            visible if output_tokens is None else max(0, output_tokens - thought)
        )
        self._wait(self.recipe.tool_pings, at)

    def complete(
        self, plan, answer, *, ended_at, thought=0, place=None, **counters
    ) -> None:
        """The reply that ends the turn. The next request comes from the user."""
        self._seen(counters)
        self._known.add(answer.part_id)
        if place is not None:
            self.thought(place, thought)
        # What was said in between its tool calls has joined already.
        said = self.spec.part_tokens(answer.text) - self._spoken
        self.flow.add(round(max(0, said) * self._off))
        self._spoken = 0
        self._wait(self.recipe.user_pings, ended_at)

    @property
    def archived_ids(self) -> frozenset:
        return frozenset(self.flow.archived)

    def reopened(self, name, tokens: int) -> None:
        """Something was fetched back: it stands at the end of the conversation from here on."""
        self._fresh.append((len(self.flow.shown), tokens))
        self.flow.add(round(tokens * self._off), name, again=True)

    def shows_again(self, name) -> bool:
        """Whether what was fetched back under `name` still stands where it was put."""
        return name in self.flow.back

    def dropped(self, name) -> None:
        """What was fetched back under `name` could not be put in its place any more."""
        self.flow.back.discard(name)
        for i in range(len(self.flow.names) - 1, -1, -1):
            if self.flow.names[i] == (name, True):
                self.flow.forget(i)
                break

    # -- pings -------------------------------------------------------------------

    @property
    def next_ping_at(self):
        return self._due[0] if self._due else None

    def ping_view(self, *, run_id, at):
        if not self._due or at < self._due[0]:
            return None
        if at >= self.flow.at + self.spec.cache_ttl_seconds:
            # Too late: what it would keep alive is gone, and it would pay to write it again.
            self._due = ()
            return None
        return AgentView((), frozenset(self.flow.archived), max(0, self.flow.size))

    def acknowledge_ping(
        self, *, run_id, at, completed_at=None, output_tokens=None, **counters
    ) -> bool:
        """Take a ping's usage. One that read nothing has kept nothing warm: no more are sent."""
        read = counters.get("cached_input_tokens")
        if not counters or (read is not None and read < self.flow.cached / 2):
            # It read nothing, or only what opens every request: the conversation is not kept by it.
            self._due = ()
            return False
        self.flow.sent(at, self.flow.cached)
        # The next ones follow this one, whenever it really went out.
        self._due = tuple(at + n * self.flow.interval for n in range(1, len(self._due)))
        return True
