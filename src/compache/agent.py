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

How many rounds, how large, how many pings and which cache lifetime is fitted
on recorded conversations. `Flow` holds what a conversation sends and what the
provider has cached of it. The fit, the replay and the runtime all run on it,
so what was priced is what happens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from .errors import ContractError
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
# Fewer such calls than this say too little about which results can go.
_LEAST_RELYING = 40
_ROUNDS = (1, 2, 3, 5, 8, 12, 20, 30)
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
    # Results of earlier rounds that the reply relies on.
    needs: frozenset
    # Whether the reply waits for a tool. Otherwise it waits for the user.
    tool: bool


class Flow:
    """A conversation as it goes out: what is in it, what went out as a notice, what is cached."""

    __slots__ = (
        "ahead",
        "archived",
        "at",
        "back",
        "cached",
        "cap",
        "freed",
        "live",
        "names",
        "notice",
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

    def add(self, tokens: int, name=None, *, again: bool = False) -> None:
        """Something joined the end. `again` marks a result that was fetched back."""
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
        if again or tokens > self.notice:
            self.live.append(len(self.shown) - 1)

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

    def _pays(self, at: float, before: int) -> bool:
        spec, freed = self.spec, self.freed
        if self.size > self.cap or not self.warm(at):
            return True
        ratio = self.recipe.ratio
        if ratio is not None and freed >= ratio * (self.size - before - freed):
            return True
        # Or it is cheaper in this very call, as where a long request is billed dearer.
        held = min(self.cached, self.size)
        if spec.long_input_threshold is None:
            read, write = spec.read_multiplier, spec.write_multiplier
            return (write - read) * (held - min(held, before)) < write * freed
        return _charge(spec, self.size - freed, min(held, before), 0) < (
            _charge(spec, self.size, held, 0)
        )

    def cut(self, at: float) -> list:
        """Turn older results into notices, if that pays now.

        Returns what went, oldest first: each by its name and whether it had been fetched back.
        """
        if self.recipe.rounds is None:
            return []
        limit, live = self.rounds - self.recipe.rounds, self.live
        # They arrived in order, so the first that stays ends the search.
        while self.ripe < len(live) and self.round[live[self.ripe]] <= limit:
            i = live[self.ripe]
            self.freed += self.whole[i] - (0 if self.names[i][1] else self.notice)
            self.ripe += 1
        if not self.ripe:
            return []
        first = live[0]
        before = self._before(first)
        if not self._pays(at, before):
            return []
        if self.warm(at):
            self.cached = min(self.cached, before)
        went = []
        for i in live[: self.ripe]:
            name, again = self.names[i]
            self.shown[i] = 0 if again else self.notice
            if again:
                self.back.discard(name)
            else:
                self.archived.add(name)
            went.append((name, again))
        self.size -= self.freed
        self.live, self.ripe, self.freed = live[self.ripe :], 0, 0
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
    """The calls of a run: start, end, results seen, reply tokens that join the history, output, needs."""
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
        yield at, done, seen, reply if last else 0, reply if last else 0, ()


def calls_of(thread, spec: ModelSpec, header_tokens: int = 0) -> list[Call]:
    """A thread as the calls it was made of."""
    calls: list[Call] = []
    said = header_tokens
    for run in thread.runs:
        said += spec.part_tokens(run.input.text or "")
        sizes = [
            (p.part_id, spec.part_tokens(p.text or ""))
            for p in run.retrievals
            if p.exposed_to_model
        ]
        steps, seen = list(_steps(run, spec)), 0
        for k, (at, end, results, reply, output, needs) in enumerate(steps):
            results = max(seen, min(results, len(sizes)))
            found, seen = tuple(sizes[seen:results]), results
            fresh = {name for name, _ in found}
            calls.append(
                Call(
                    at,
                    end,
                    said,
                    found,
                    output,
                    frozenset(n for n in needs if n not in fresh),
                    k + 1 < len(steps),
                )
            )
            said = reply
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


def simulate(conversations, recipe: AgentRecipe, spec: ModelSpec, reserve: int = 0):
    """What conversations cost under a recipe, every call and ping by the provider's cache rules."""
    cost = 0.0
    calls = asked = missed = pings = cuts = peak = 0
    for conversation in conversations:
        flow = Flow(spec, recipe, reserve)
        for i, call in enumerate(conversation):
            flow.add(call.said)
            if call.found:
                flow.rounds += 1
                for name, tokens in call.found:
                    flow.add(tokens, name)
            cuts += bool(flow.cut(call.at))
            if flow.size > flow.cap:
                return None
            peak = max(peak, flow.size)
            calls += 1
            asked += bool(call.needs)
            lost = [n for n in call.needs if flow.gone(n)]
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
    return {
        "cost": cost,
        "calls": calls,
        "asked": asked,
        "missed": missed,
        "pings": pings,
        "cuts": cuts,
        "largest_request": peak,
        "coverage": 1 - missed / asked if asked else None,
    }


def _conversations(threads, spec, header_tokens, labels=()):
    """Threads as calls, with what their labels say a turn needed on the turn's last call."""
    wanted: dict = {}
    for row in labels or ():
        needs = row.get("needs")
        if needs:
            first = [n for n in needs[0] if ":retrieval:" in n]
            if first:
                wanted[row["thread_id"], row["run"]] = first
    named = {thread_id for thread_id, _ in wanted}
    made = []
    for thread in threads:
        calls = calls_of(thread, spec, header_tokens)
        if thread.thread_id in named:
            at = 0
            for run in thread.runs:
                at += len(list(_steps(run, spec)))
                more = wanted.get((thread.thread_id, run.run_index))
                if more:
                    last = calls[at - 1]
                    calls[at - 1] = replace(last, needs=last.needs | frozenset(more))
        made.append(calls)
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
            found += [rounds - arrived[n] for n in call.needs if n in arrived]
    return sorted(found)


def _cuts(conversations, spec) -> list[AgentRecipe]:
    """The cuts worth pricing: a grid that follows how far back these conversations reach."""
    ages = _ages(conversations)
    if sum(bool(c.needs) for calls in conversations for c in calls) < _LEAST_RELYING:
        # Too little says what can go: only the cut that public runs taught is considered.
        return [replace(default_recipe(spec), tool_pings=0, user_pings=0)]
    reach = {ages[min(len(ages) - 1, int(q * len(ages)))] + 1 for q in (0.8, 0.9, 0.95)}
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
    header_tokens: int = 0,
    floor: float = FLOOR,
    reserve: int = 8192,
) -> dict:
    """Fit what an agent's conversations keep, on recorded ones. Returns a policy artifact.

    `floor` is the least share of calls that rely on an earlier result and find
    it in view. The rest are priced as fetching it back.
    """
    conversations = _conversations(threads, spec, header_tokens, labels)
    if sum(map(len, conversations)) < 2:
        raise ContractError("fitting an agent's policy needs recorded calls")
    specs = {s.cache_ttl_seconds: s for s in (spec, *lifetimes)}
    cuts = _cuts(conversations, spec)
    # A cut that public runs taught is not held to a floor that these say too little about.
    known = len(cuts) > 1
    rows = []

    def price(recipe, asked):
        result = simulate(conversations, recipe, asked, reserve)
        if result is not None:
            rows.append(
                {
                    "recipe": asdict(recipe),
                    "cache_ttl_seconds": asked.cache_ttl_seconds,
                    "cost": result["cost"],
                    "coverage": result["coverage"],
                    "missed_calls": result["missed"],
                    "pings": result["pings"],
                    "cuts": result["cuts"],
                }
            )
        return result

    def allowed(row):
        if row["recipe"]["rounds"] is None or not known:
            return True
        return row["coverage"] is not None and row["coverage"] >= floor

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
    for ttl, recipes in best.items():
        for recipe in {*recipes[:4], AgentRecipe()}:
            paid = {}
            for tool in _TOOL_PINGS[1:]:
                result = price(replace(recipe, tool_pings=tool), specs[ttl])
                if result is not None:
                    paid[tool] = result["cost"]
            tool = min(paid, key=paid.get) if paid else 0
            for user in _USER_PINGS[1:]:
                for some in {0, tool}:
                    price(replace(recipe, tool_pings=some, user_pings=user), specs[ttl])

    def cheapest(some):
        return min(
            some,
            key=lambda r: (
                r["cost"],
                -(r["coverage"] or 0),
                r["cache_ttl_seconds"],
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
        and not r["recipe"]["tool_pings"]
        and not r["recipe"]["user_pings"]
    ]
    keep = plain[0]["cost"] if plain else None
    asked = specs[chosen["cache_ttl_seconds"]]
    return {
        "schema_version": SCHEMA,
        "model_spec": asked.to_dict(),
        **({} if asked is spec else {"standard_spec": spec.to_dict()}),
        "recipe": chosen["recipe"],
        "reserve_tokens": reserve,
        "solve": {
            "unit": "fresh_input_token_equivalents",
            "calls": sum(map(len, conversations)),
            "turns": sum(len(t.runs) for t in threads),
            "relying_calls": sum(bool(x.needs) for c in conversations for x in c),
            "floor": floor,
            "chosen": chosen,
            "keep_everything_cost": keep,
            "caching_alone": cheapest(whole) if whole else None,
            "saving_fraction": 1 - chosen["cost"] / keep if keep else None,
            "candidates": rows,
        },
        "training_threads": sorted(digest(t.thread_id) for t in threads),
    }


def replay_agent(artifact: dict, threads, *, labels=(), header_tokens: int = 0) -> dict:
    """What a policy would have cost on other conversations, next to keeping everything."""
    spec = ModelSpec.from_dict(artifact["model_spec"])
    standard = ModelSpec.from_dict(
        artifact.get("standard_spec", artifact["model_spec"])
    )
    conversations = _conversations(threads, spec, header_tokens, labels)
    reserve = artifact.get("reserve_tokens", 0)
    ours = simulate(conversations, AgentRecipe(**artifact["recipe"]), spec, reserve)
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
    """Where to start without recorded conversations, by how cheap the model's cache reads are.

    Fitted on 174 public coding-agent runs of 43 to 100 calls, one task and three to a
    session: the cheaper a read, the less it is worth to write a conversation again for
    leaving something out.
    """
    if spec.cache_ttl_seconds <= 0:
        return AgentRecipe(rounds=12, ratio=0.0)
    cheap = spec.read_multiplier <= 0.06
    return AgentRecipe(
        rounds=8 if cheap else 12,
        ratio=2.0 if cheap else 1.5,
        tool_pings=3,
        user_pings=1,
    )


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

    def copy(self) -> AgentEngine:
        made = AgentEngine.__new__(AgentEngine)
        made.__dict__.update(self.__dict__)
        made.flow, made._known = self.flow.copy(), set(self._known)
        return made

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
            self.flow.add(self.spec.part_tokens(part.text), named)

    def _plan(self, request, at) -> AgentPlan:
        self._due, self._sent_at = (), at
        went = self.flow.cut(at)
        archived = frozenset(self.flow.archived)
        parts = tuple(
            replace(p, text=placeholder(self.thread_id, p))
            if p.part_id in archived
            else p
            for p in request.parts
        )
        live = self.flow.live
        decision = {
            "action": "cut" if went else "keep",
            # What went now, and what goes first with the next cut.
            "went": tuple(went),
            "holds": self.flow.names[live[0]] if live else None,
        }
        return AgentPlan(
            request, AgentView(parts, archived, max(0, self.flow.size)), decision
        )

    def prepare(self, request, *, started_at, system_prompt, tools, **_) -> AgentPlan:
        if not self._known:
            header = f"{system_prompt or ''}{json.dumps(tools) if tools else ''}"
            self.flow.add(self.spec.text_tokens(header))
        self._spoken = 0
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
            # Tool calls, reasoning and the like, and what our ruler is off by.
            hidden = reported - self.flow.size
            if hidden:
                self.flow.add(hidden)
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

    def _wait(self, most, at) -> None:
        flow = self.flow
        if most and flow.cached and flow.size <= flow.cap:
            times = (self._sent_at + n * flow.interval for n in range(1, most + 1))
            self._due = tuple(due for due in times if due >= at)

    def replied(self, plan, *, at, text="", calls=(), output_tokens=None, **counters):
        """A reply that calls tools is in. The next request comes when they are done."""
        self._seen(counters)
        said = self.spec.text_tokens(text) if text else 0
        self._spoken += said
        called = spoken("", calls)
        if output_tokens is None:
            output_tokens = said + (self.spec.text_tokens(called) if called else 0)
        self.flow.add(output_tokens)
        self._wait(self.recipe.tool_pings, at)

    def complete(self, plan, answer, *, ended_at, **counters) -> None:
        """The reply that ends the turn. The next request comes from the user."""
        self._seen(counters)
        self._known.add(answer.part_id)
        # What was said in between its tool calls has joined already.
        self.flow.add(max(0, self.spec.part_tokens(answer.text) - self._spoken))
        self._spoken = 0
        self._wait(self.recipe.user_pings, ended_at)

    @property
    def archived_ids(self) -> frozenset:
        return frozenset(self.flow.archived)

    def reopened(self, name, tokens: int) -> None:
        """Something was fetched back: it stands at the end of the conversation from here on."""
        self.flow.add(tokens, name, again=True)

    def shows_again(self, name) -> bool:
        """Whether what was fetched back under `name` still stands where it was put."""
        return name in self.flow.back

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
        if not counters or (read is not None and read <= 0):
            self._due = ()
            return False
        self.flow.sent(at, self.flow.cached)
        # The next ones follow this one, whenever it really went out.
        self._due = tuple(at + n * self.flow.interval for n in range(1, len(self._due)))
        return True
