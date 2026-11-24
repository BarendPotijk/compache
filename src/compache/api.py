"""Fit a policy on recorded threads, price it on others, and apply it to live ones."""

from __future__ import annotations

import json
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import agent
from .errors import ContractError
from .models import cache_lifetimes, model_spec, pricing
from .optimizer.cache_breaks import CacheBreakChecker, CacheScope
from .optimizer.contracts import ModelSpec, seconds
from .optimizer.parts import Request, Source
from .optimizer.recovery import reopen as _reopen
from .optimizer.replay import arm_costs, replay_workload
from .optimizer.retention import RetentionRecipe
from .optimizer.session import WorkloadPolicy, WorkloadSession
from .optimizer.solve import solve_workload
from .profile import header_size, measure, workload
from .record import SIZES_ONLY, is_record
from .record import load as load_record
from .threads import (
    Thread,
    cache_read_reported,
    canonical_usage,
    load_threads,
    read_jsonl,
    reasoning_tokens,
    recorded,
    system_prompt_of,
)
from .wire import TTLS, cache_mark

__all__ = ["Policy", "Session", "Turn", "fit", "replay"]

_HORIZON_SECONDS = 86400
_CHAT = "compache-workload-planner-v1"
# Characters an agent reads back at once: the whole of most tool results.
_AGENT_PAGE = 32_000


def _rows(source) -> list:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return list(read_jsonl(source))
    return list(source)


def _labels(labels, threads) -> list:
    """Label rows as given, or those a record folder of threads carries itself."""
    if labels is None and is_record(threads):
        return load_record(threads)[1]
    return _rows(labels)


def _threads(source, model=None):
    """Threads priced as one model, its spec, and what the rows record about themselves.

    Without `model` the threads are priced as the model most of them record.
    """
    rows = _rows(source)
    found = recorded(rows)
    if model is None:
        model = found["model"]
        if model is None:
            raise ContractError(
                "the threads do not record which model answered: name it with --model or model="
            )
    spec = model_spec(model)
    found["headers"] = _headers(rows, spec)
    return load_threads(rows, model=spec.model_id), spec, found, model


def _headers(rows, spec) -> dict:
    """Tokens of the instructions and tools that each thread says it opened its requests with."""
    sizes = {}
    for row in rows:
        row = row.to_dict() if isinstance(row, Thread) else row
        if not isinstance(row, dict) or not isinstance(row.get("thread_id"), str):
            continue
        tokens = sum(
            spec.text_tokens("." * row[key])
            for key in ("system_bytes", "tools_bytes")
            if type(row.get(key)) is int
        )
        if not tokens:
            prompt, tools = system_prompt_of(row), row.get("tools")
            tools = tools if isinstance(tools, list) else ()
            tokens = header_size(spec, prompt or "", tools)
        sizes[row["thread_id"].strip()] = tokens
    return sizes


def _calls(threads):
    """Model calls per turn where the record has them and any turn made more than one."""
    counts = {run.run_id: run.calls for t in threads for run in t.runs if run.calls}
    return counts if any(n > 1 for n in counts.values()) else None


def _observed_until(threads, given):
    """When the record ends. Unless told otherwise, every thread is taken to be over."""
    if given is not None:
        return given
    last = max(seconds(t.runs[-1].ended_at) for t in threads)
    end = datetime.fromtimestamp(last, timezone.utc) + timedelta(
        seconds=_HORIZON_SECONDS
    )
    return end.isoformat().replace("+00:00", "Z")


def fit(
    threads,
    labels=None,
    *,
    model=None,
    system_prompt=None,
    tools=None,
    observed_until=None,
    **options,
) -> Policy:
    """Fit what to keep in view, how many keep-warm pings to send and how long the cache is to live.

    `threads` and `labels` are paths to JSONL files (or folders of them), or
    lists of dicts. `model` names a known model or gives a model spec. Left
    out, it is the model the threads record, as are the system prompt, the
    tools and when the record was read. Without labels only the caching is
    fitted and the whole history stays in view. Where the threads record
    several model calls in a turn, those are priced.

    Every candidate is replayed against the provider's cache rules at the
    threads' own timestamps, each thread priced by a model of returns fitted on
    the others, and the cheapest wins after paying for what it would fetch back.
    Where a known model can be asked for another cache lifetime, as Claude can
    for an hour, every candidate is priced under that too.
    `observed_until` is when the record was cut, for a record that ends while
    conversations were still going. `options` go to the solver.

    Where turns are loops of many model calls, as an agent's are, the policy
    decides call by call instead: which tool results stay in view, and what
    keeps the cache warm while a tool runs. What replies quote from earlier
    results says which were still needed, so an agent's record needs no labels.
    """
    labels = _labels(labels, threads)
    threads, spec, found, model = _threads(threads, model)
    lifetimes = options.pop("lifetimes", None)
    if lifetimes is None:
        lifetimes = cache_lifetimes(model)
    for other in lifetimes:
        if other.cache_ttl_seconds not in TTLS:
            raise ContractError(
                f"no request can ask for cache entries of {other.cache_ttl_seconds:g} seconds"
            )
    opens = found["headers"] if system_prompt is None and tools is None else {}
    if system_prompt is None:
        system_prompt = found["system_prompt"] or ""
    tools = list(found["tools"] or ()) if tools is None else list(tools)
    header = {"system_prompt": system_prompt, "default_tools": tools}
    agentic = options.pop("agent", None)
    if agent.is_agentic(threads) if agentic is None else agentic:
        unknown = sorted(set(options) - {"floor", "reserve", "pings"})
        if unknown:
            raise ContractError(
                f"a policy that decides call by call is not fitted with {', '.join(unknown)}: "
                "give agent=False for one that decides per message"
            )
        artifact = agent.fit_agent(
            threads,
            spec,
            labels=labels,
            lifetimes=lifetimes,
            header_tokens=opens or header_size(spec, system_prompt, tools),
            **options,
        )
        return Policy({**artifact, "header": header, "pricing": pricing(model)})
    if all(t.to_dict().get("provenance") == SIZES_ONLY for t in threads):
        # No words to match a question against: see `solve_workload`.
        options.setdefault("match_questions", False)
    artifact = solve_workload(
        threads,
        labels,
        spec,
        observation_end=_observed_until(
            threads, observed_until or found["observed_until"]
        ),
        system_prompt=system_prompt,
        default_tools=tools,
        lifetimes=lifetimes,
        **{"calls_per_run": _calls(threads), **options},
    )
    scored = artifact["recipes"]
    artifact["fitted"] = {
        "threads": len(threads),
        "turns": artifact["solve"]["turns"],
        "labelled_turns": scored[0]["turns"] - scored[0]["unlabeled_runs"]
        if scored
        else 0,
        "unknown_requirements": scored[0]["unknown_requirements"] if scored else 0,
        "assessed_requirements": scored[0]["direct_required"]
        + scored[0]["historical_required"]
        if scored
        else 0,
        "unknown_turns": scored[0]["unknown_runs"] if scored else 0,
        "historical_requirements": scored[0]["historical_required"] if scored else 0,
        "candidate_cuts": len(scored),
    }
    # The policy file keeps the cut that won. The priced candidates stay under `solve`.
    artifact["recipes"] = [row for row in scored if row["eligible"]]
    artifact["header"] = {
        **header,
        "share_headers": options.get("share_headers", True),
    }
    artifact["pricing"] = pricing(model)
    # The measured workload, so that the calculator can show the policy at work on it.
    preset, scenario = workload(
        measure(threads, spec, labels),
        spec,
        model,
        header_size(spec, system_prompt, tools),
    )
    artifact["workload"] = {
        "model": preset,
        "scenario": scenario,
        "source": f"{len(threads):,} threads, {artifact['fitted']['turns']:,} turns",
    }
    return Policy(artifact)


def replay(policy, threads, labels=None, *, observed_until=None, **options) -> dict:
    """Price a fitted policy on threads it has not seen, against keeping everything.

    With labels the report also says how much of what those threads needed
    stays in view, and adds the cost of fetching the rest back.
    """
    if not isinstance(policy, Policy):
        policy = Policy(policy)
    labels = _labels(labels, threads)
    threads, spec, found, _ = _threads(threads, policy.spec)
    header = policy.artifact.get("header") or {}
    if policy.agentic:
        prompt, tools = header.get("system_prompt", ""), header.get("default_tools")
        opens = header_size(spec, prompt, tools or ())
        report = agent.replay_agent(
            policy.artifact,
            threads,
            labels=labels,
            header_tokens={k: v or opens for k, v in found["headers"].items()},
        )
        calls = report["calls"]
        costs = {"policy": report["policy_cost"]}
        if report["keep_everything_cost"] is not None:
            costs["keep_everything"] = report["keep_everything_cost"]
        return {**report, "cost_per_1000_calls": policy._per_1000(costs, calls)}
    # Threads share a cached system prompt in the replay if they did in the fit.
    options.setdefault("share_headers", header.get("share_headers", True))
    options.setdefault("calls_per_run", _calls(threads))
    detail = replay_workload(
        threads,
        labels,
        policy.artifact,
        observation_end=_observed_until(
            threads, observed_until or found["observed_until"]
        ),
        system_prompt=header.get("system_prompt", ""),
        default_tools=header.get("default_tools", []),
        **options,
    )
    candidate, baseline = detail["candidate"], detail["baseline"]
    turns, replayed, recovery = arm_costs(candidate["ledger"])
    cost = replayed + recovery
    baseline_turns, keep, _ = arm_costs(baseline["ledger"])
    complete = candidate["complete"] and baseline["complete"]
    answers = [r for r in candidate["ledger"] if r["role"] == "answer"]
    per_thread = {}
    for row in answers:
        counts = per_thread.setdefault(row["thread_id"], [0, 0])
        counts[0] += row["historical_retained"]
        counts[1] += row["historical_required"]
    full_tokens = sum(r["canonical_part_tokens"] for r in answers)
    common_runs = {r["run_id"] for r in answers} & {
        r["run_id"] for r in baseline["ledger"] if r["role"] == "answer"
    }
    paired = {}
    for name, arm in (("policy", candidate), ("keep_everything", baseline)):
        _, charges, extra = arm_costs(
            [r for r in arm["ledger"] if r["run_id"] in common_runs]
        )
        paired[name] = charges + extra
    return {
        "threads": len(threads),
        "turns": turns,
        "planned_turns": candidate["planned_answers"],
        "policy_turns": turns,
        "baseline_turns": baseline_turns,
        "complete": complete,
        "policy_cost": cost,
        "replayed_cost": replayed,
        "recovery_cost": recovery,
        "keep_everything_cost": keep,
        "saving": 1 - cost / keep if complete and keep else None,
        "common_prefix": {
            "turns": len(common_runs),
            "cost_per_1000_turns": policy._per_1000(paired, len(common_runs)),
            "saving": 1 - paired["policy"] / paired["keep_everything"]
            if paired["keep_everything"]
            else None,
        },
        "pings": candidate["pings"],
        "needs_history": candidate["historical_required"] if labels else None,
        "coverage": candidate["coverage"],
        "coverage_by_thread": {
            tid: {"retained": n, "required": d, "coverage": n / d if d else None}
            for tid, (n, d) in per_thread.items()
        },
        "unassessable_turns": candidate["unassessable_runs"],
        "unknown_requirements": sum(r["unknown_requirements"] for r in answers),
        "assessed_requirements": sum(r["assessed_requirements"] for r in answers),
        "tokens_kept": sum(r["sent_part_tokens"] for r in answers) / full_tokens
        if full_tokens
        else None,
        "cost_per_1000_turns": {
            **policy._per_1000({"policy": cost}, turns),
            **policy._per_1000({"keep_everything": keep}, baseline_turns),
        },
        "detail": detail,
    }


def _duration(seconds):
    """A cache lifetime as a word before a noun: one-hour, 12-hour."""
    hours = seconds / 3600
    return "one-hour" if hours == 1 else f"{hours:g}-hour"


def _window(count, noun):
    if count is None:
        return f"all {noun}"
    if count == 0:
        return f"no earlier {noun}"
    return f"{noun} from the last {'turn' if count == 1 else f'{count} turns'}"


class Policy:
    """A fitted policy: what stays in view, and how many pings keep the cache warm."""

    def __init__(self, artifact: dict):
        if not isinstance(artifact, dict) or artifact.get("schema_version") not in (
            _CHAT,
            agent.SCHEMA,
        ):
            raise ContractError("expected a fitted policy")
        self.artifact = artifact
        # Whether it decides call by call, for an agent, or once per user message.
        self.agentic = artifact["schema_version"] == agent.SCHEMA
        self._compiled = None
        self._checker = None
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path) -> Policy:
        return cls(json.loads(Path(path).read_text("utf-8")))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.artifact, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec.from_dict(self.artifact["model_spec"])

    @property
    def cache_lifetime(self) -> float | None:
        """How long the provider is asked to keep what it caches, in seconds. None for its default."""
        standard = self.artifact.get("standard_spec")
        if standard is None or standard["cache_ttl_seconds"] == (
            self.spec.cache_ttl_seconds
        ):
            return None
        return self.spec.cache_ttl_seconds

    @property
    def cache_control(self) -> dict:
        """The cache mark for a Claude request you assemble yourself."""
        return cache_mark(self.cache_lifetime)

    @property
    def header_lifetime(self) -> float | None:
        """How long the provider is asked to keep instructions and tools, where that is longer than the rest."""
        spec = self.artifact.get("header_spec")
        return spec["cache_ttl_seconds"] if spec else None

    def at_default_lifetime(self) -> Policy:
        """The same cut and pings, planned under the provider's default cache lifetime."""
        if self.cache_lifetime is None and self.header_lifetime is None:
            return self
        dropped = ("standard_spec", "header_spec")
        artifact = {k: v for k, v in self.artifact.items() if k not in dropped}
        standard = self.artifact.get("standard_spec", self.artifact["model_spec"])
        return Policy({**artifact, "model_spec": standard})

    @property
    def recipe(self) -> RetentionRecipe | agent.AgentRecipe | None:
        """The cut the policy applies, or None when it keeps the whole history."""
        if self.agentic:
            return agent.AgentRecipe(**self.artifact["recipe"])
        for row in self.artifact["recipes"]:
            if row["eligible"]:
                return RetentionRecipe(**row["recipe"])
        return None

    @property
    def max_pings(self) -> int:
        if self.agentic:
            return max(self.recipe.tool_pings, self.recipe.user_pings)
        return (self.artifact.get("planner_options") or {}).get("max_pings", 1)

    def _per_1000(self, costs, count):
        """Costs per 1,000 turns or calls: in dollars for a known model, else in input tokens."""
        price = (self.artifact.get("pricing") or {}).get("input_price_per_million")
        scale = 1000 / count if count else None
        if price is None:
            return {
                "unit": "input-token equivalents",
                **{
                    k: v * scale if scale is not None else None
                    for k, v in costs.items()
                },
            }
        return {
            "unit": "USD",
            **{
                k: v * scale * price / 1e6 if scale is not None else None
                for k, v in costs.items()
            },
        }

    @property
    def summary(self) -> dict:
        """The decision and what it costs on the fitted threads, as plain numbers."""
        if self.agentic:
            return self._agent_summary
        solved = self.artifact.get("solve") or {}
        chosen = solved.get("chosen") or {}
        recipe = self.recipe
        keep = solved.get("keep_everything_cost")
        costs = {
            "policy": chosen.get("expected_cost"),
            "keep_everything": keep,
        }
        alone = (solved.get("caching_alone") or {}).get("expected_cost")
        if alone is not None:
            # Everything kept, at the cheapest pings and lifetime: what caching can do by itself.
            costs["caching_alone"] = alone
        fitted = self.artifact.get("fitted") or {}
        return {
            "model": self.spec.model_id,
            "threads": len(self.artifact["training_threads"]),
            "turns": solved.get("turns"),
            "labelled_turns": fitted.get("labelled_turns", 0),
            "unknown_requirements": fitted.get("unknown_requirements"),
            "assessed_requirements": fitted.get("assessed_requirements"),
            "unknown_turns": fitted.get("unknown_turns"),
            "historical_requirements": fitted.get("historical_requirements"),
            "keep": None
            if recipe is None
            else {
                "questions": recipe.input_runs,
                "answers": recipe.output_runs,
                "retrieved": recipe.retrieval_runs,
                "matching_retrieved": recipe.protect_query_matches,
                "always_under_tokens": recipe.keep_under_tokens,
                **(
                    {"cited_answer_runs": recipe.cited_answer_runs}
                    if recipe.cited_answer_runs != 0
                    else {}
                ),
            },
            "max_pings": self.max_pings,
            "cache_lifetime_seconds": self.cache_lifetime,
            "coverage": chosen.get("coverage"),
            "saving": solved.get("saving_fraction"),
            "cost_per_1000_turns": None
            if None in costs.values()
            else self._per_1000(costs, solved.get("turns")),
        }

    @property
    def _agent_summary(self) -> dict:
        solved, recipe = self.artifact.get("solve") or {}, self.recipe
        chosen = solved.get("chosen") or {}
        costs = {
            "policy": chosen.get("cost"),
            "keep_everything": solved.get("keep_everything_cost"),
            "caching_alone": (solved.get("caching_alone") or {}).get("cost"),
        }
        return {
            "model": self.spec.model_id,
            "threads": len(self.artifact.get("training_threads") or ()),
            "turns": solved.get("turns"),
            "calls": solved.get("calls"),
            "relying_calls": solved.get("relying_calls"),
            "keep": None
            if recipe.rounds is None
            else {"rounds": recipe.rounds, "ratio": recipe.ratio},
            "pings": {"tool": recipe.tool_pings, "user": recipe.user_pings},
            "max_pings": self.max_pings,
            "cache_lifetime_seconds": self.cache_lifetime,
            "header_lifetime_seconds": self.header_lifetime,
            "coverage": chosen.get("coverage"),
            "missed_calls": chosen.get("missed_calls"),
            "saving": solved.get("saving_fraction"),
            "cost_per_1000_calls": None
            if None in costs.values()
            else self._per_1000(costs, solved.get("calls")),
        }

    def _describe_agent(self) -> str:
        summary, recipe = self.summary, self.recipe
        lines = agent.sentences(recipe)
        if summary["cache_lifetime_seconds"]:
            lines.insert(
                1,
                f"Ask for {_duration(summary['cache_lifetime_seconds'])} cache entries.",
            )
        if summary["header_lifetime_seconds"]:
            lines.insert(
                1,
                f"Ask for {_duration(summary['header_lifetime_seconds'])} cache entries for the "
                "instructions and tools that open every request.",
            )
        if recipe.rounds is not None and summary["coverage"] is not None:
            kept = f"{summary['coverage']:.0%} of the calls that rely on an earlier result find it in view."
            if summary["missed_calls"]:
                share = summary["missed_calls"] / summary["calls"]
                kept += f" The others, {share:.1%} of all calls, fetch it back, and that is in the cost."
            lines.append(kept)
        costs, saving = summary["cost_per_1000_calls"], summary["saving"]
        if costs and saving is not None:
            if costs["unit"] == "USD":
                lines.append(
                    f"${costs['policy']:,.2f} per 1,000 calls against "
                    f"${costs['keep_everything']:,.2f} keeping everything: {saving:.0%} less."
                )
                if costs["caching_alone"] < costs["keep_everything"]:
                    lines.append(
                        f"Caching alone, at its best, would cost ${costs['caching_alone']:,.2f}."
                    )
            else:
                lines.append(f"{saving:.0%} cheaper than keeping everything.")
        return "\n".join(lines)

    def describe(self) -> str:
        """The decision in sentences."""
        if self.agentic:
            return self._describe_agent()
        summary, recipe = self.summary, self.recipe
        lines = []
        if recipe is None:
            lines.append(
                "Keep the whole history in view."
                if summary["labelled_turns"]
                else "Keep the whole history in view: without labels only the caching is fitted."
            )
        else:
            lines.append(
                f"Keep {_window(recipe.input_runs, 'questions')}, "
                f"{_window(recipe.output_runs, 'answers')} and "
                f"{_window(recipe.retrieval_runs, 'retrieved text')}."
            )
            if recipe.protect_query_matches and recipe.retrieval_runs is not None:
                lines.append(
                    "Earlier retrieved text that matches the new question stays too."
                )
            if recipe.cited_answer_runs != 0:
                lines.append(
                    f"Keep visible sources cited by {_window(recipe.cited_answer_runs, 'answers')}."
                )
            if recipe.keep_under_tokens:
                lines.append(
                    f"Parts of at most {recipe.keep_under_tokens} tokens always stay."
                )
        if summary["cache_lifetime_seconds"]:
            lines.append(
                f"Ask for {_duration(summary['cache_lifetime_seconds'])} cache entries."
            )
        pings = summary["max_pings"]
        lines.append(
            f"Send up to {pings} keep-warm ping{'' if pings == 1 else 's'} while a user is away."
            if pings
            else "Send no keep-warm pings."
        )
        if recipe is not None and summary["coverage"] is not None:
            kept = f"{summary['coverage']:.0%} of what the labelled turns need from earlier turns stays in view."
            if summary["coverage"] < 0.995:
                kept += " The rest is fetched back, and that call is in the cost."
            lines.append(kept)
        costs, saving = summary["cost_per_1000_turns"], summary["saving"]
        if costs and saving is not None:
            if costs["unit"] == "USD":
                lines.append(
                    f"${costs['policy']:,.2f} per 1,000 turns against "
                    f"${costs['keep_everything']:,.2f} keeping everything: {saving:.0%} less."
                )
                alone = costs.get("caching_alone")
                if recipe is not None and alone and alone < costs["keep_everything"]:
                    lines.append(
                        f"Caching alone, at its best, would cost ${alone:,.2f}."
                    )
            else:
                lines.append(f"{saving:.0%} cheaper than keeping everything.")
        return "\n".join(lines)

    def session(self, thread_id: str, *, system_prompt=None, tools=None) -> Session:
        """The session of one conversation. Keep it for as long as the conversation lives."""
        with self._lock:
            # Conversations open at the same moment on a server: set up once, and in full.
            if self._compiled is None and not self.agentic:
                checker = CacheBreakChecker()
                weakref.finalize(self, checker.close)
                self._checker = checker
                self._compiled = WorkloadPolicy(self.artifact)
        header = self.artifact.get("header") or {}
        return Session(
            self,
            thread_id,
            header.get("system_prompt", "") if system_prompt is None else system_prompt,
            header.get("default_tools", []) if tools is None else tools,
        )


@dataclass(frozen=True)
class Turn:
    """One request to send: its parts in order, with archived ones reduced to a notice."""

    parts: tuple[Source, ...]
    archived_ids: frozenset
    tokens: int
    action: str
    plan: object = None

    @property
    def messages(self) -> list[dict]:
        """One chat message per part: answers from the assistant, everything else from the user."""
        return [
            {"role": "assistant" if p.kind == "output" else "user", "content": p.text}
            for p in self.parts
        ]


def _counters(usage) -> dict:
    """Provider usage as the counters the session takes. Unknown shapes count as no usage."""
    counted = canonical_usage(usage)
    if counted is None:
        return {}
    return {
        "input_tokens": counted["input"],
        "cached_input_tokens": counted["cache_read"]
        if cache_read_reported(usage)
        else None,
        "cache_write_input_tokens": counted["cache_write"],
    }


def _thought(usage) -> int:
    """Tokens of reasoning in a reply, as far as the provider counts them apart."""
    return reasoning_tokens(usage) or 0


class Session:
    """One conversation under a policy.

    `prepare` returns the request for a new user message. `complete` records the
    answer. `next_ping_at` says when a keep-warm ping is due. The session keeps
    the full history itself, so an archived part can always be reopened.
    """

    def __init__(self, policy: Policy, thread_id: str, system_prompt, tools):
        self.policy, self.thread_id = policy, thread_id
        self.system_prompt, self.tools = system_prompt, tools
        self._model = policy.spec.model_id
        if policy.agentic:
            self._engine = agent.AgentEngine(policy.artifact, thread_id)
        else:
            self._engine = WorkloadSession(
                policy._compiled,
                CacheScope("compache", "declared", self._model, thread_id),
                policy._checker,
            )
        self._history: list[Source] = []
        self._run = 0
        self._pending = None

    def copy(self) -> Session | None:
        """The session as it stands, for a conversation that branches. None where it cannot be copied."""
        if not self.policy.agentic:
            return None
        made = Session.__new__(Session)
        made.__dict__.update(self.__dict__)
        made._engine, made._history = self._engine.copy(), list(self._history)
        return made

    def follows(self, other: Session) -> None:
        """Take over how warm the cache is from a conversation that went on from the same start."""
        self._engine.follows(other._engine)

    def _request(self, run, parts):
        return Request(
            self.thread_id,
            f"{self.thread_id}:{run}",
            run,
            self._model,
            tuple(self._history) + tuple(parts),
        )

    def _retrieved(self, run, items, first) -> list[Source]:
        """Retrieved items of one run as parts, numbered from `first`."""
        prefix, parts = f"{self.thread_id}:{run}", []
        for k, item in enumerate(items, first):
            fields = {"text": item} if isinstance(item, str) else dict(item)
            part_id = fields.get("part_id", f"{prefix}:retrieval:{k}")
            if (
                not isinstance(part_id, str)
                or not part_id.strip()
                or part_id.strip() != part_id
            ):
                raise ContractError(
                    "a retrieval part_id must be a nonempty trimmed string"
                )
            if part_id == f"{prefix}:output":
                raise ContractError(
                    "a retrieval part_id collides with the current answer"
                )
            if not isinstance(fields.get("text"), str):
                raise ContractError("a question and every retrieved item need text")
            parts.append(
                Source(
                    part_id,
                    "retrieval",
                    run,
                    fields["text"],
                    fields.get("document_id", part_id),
                    fields.get("version", "1"),
                    fields.get("chunk_id", part_id),
                )
            )
        return parts

    def prepare(
        self, question: str, retrieved=(), *, at=None, system_prompt=None, tools=None
    ) -> Turn:
        """The request to send for a new user message and the text retrieved for it.

        Each retrieved item is text, or a dict with `text` and optionally
        `part_id`, `document_id`, `version` and `chunk_id`. An explicit part ID
        identifies this occurrence and must be unique in the thread. Call again with the same
        arguments to retry a request that was never answered.
        """
        run = self._run + 1
        parts = [Source(f"{self.thread_id}:{run}:input", "input", run, question)]
        parts += self._retrieved(run, retrieved, 1)
        if not isinstance(question, str):
            raise ContractError("a question and every retrieved item need text")
        request = self._request(run, parts)
        plan = self._engine.prepare(
            request,
            started_at=time.time() if at is None else at,
            system_prompt=self.system_prompt
            if system_prompt is None
            else system_prompt,
            tools=self.tools if tools is None else tools,
        )
        self._pending = (request, tuple(parts))
        view = plan.view
        return Turn(
            view.parts,
            view.archived_ids,
            view.tokens,
            "keep" if plan.decision["action"] == "keep" else "cut",
            plan,
        )

    def extend(self, turn: Turn, retrieved, *, at=None) -> Turn:
        """Add text retrieved while `turn` is being answered, such as the results of a tool loop.

        The history stays as `prepare` sent it and the new items follow it, so
        the cached prefix is untouched. An agent's policy decides again here,
        on the results of one round of tool calls at a time, and `at` is when
        the request that carries them goes out. Returns the turn to pass to
        `complete`.
        """
        if self._pending is None or self._pending[0] is not turn.plan.request:
            raise ContractError("no prepared turn to extend")
        request, parts = self._pending
        found = sum(p.kind == "retrieval" for p in parts)
        added = self._retrieved(request.run_index, retrieved, found + 1)
        if at is None:
            at = time.time()
        plan = self._engine.extend(turn.plan, added, at=at)
        self._pending = (plan.request, parts + tuple(added))
        view, action = plan.view, turn.action
        if self.policy.agentic:
            action = plan.decision["action"]
        return Turn(view.parts, view.archived_ids, view.tokens, action, plan)

    def replied(
        self,
        turn: Turn,
        text: str = "",
        calls=(),
        *,
        at=None,
        usage=None,
        place=None,
    ):
        """Record a reply that calls tools, and so does not end `turn`. Returns when the next ping is due.

        `text` is what it said and `calls` the arguments of its tool calls. An
        agent's policy learns from it what the provider holds in its cache and
        when a ping is due while the tools run. Without it the policy goes by
        its own count. `place` says which message will carry the reply, where
        that keeps the reply's reasoning in later requests.
        """
        if self._pending is None or self._pending[0] is not turn.plan.request:
            raise ContractError("no prepared turn to reply to")
        if not self.policy.agentic:
            return None
        counted = canonical_usage(usage) or {}
        self._engine.replied(
            turn.plan,
            at=time.time() if at is None else at,
            text=text,
            calls=calls,
            output_tokens=counted.get("output"),
            thought=_thought(usage),
            place=place,
            **_counters(usage),
        )
        return self.next_ping_at

    def thought(self, place, tokens: int) -> None:
        """Say that a message of the conversation holds reasoning of about that size."""
        self._engine.thought(place, tokens)

    def carries(self, places) -> None:
        """Say which messages of the request still hold reasoning."""
        self._engine.carries(places)

    def without(self, places) -> list:
        """Those of the messages with reasoning that go out without it."""
        return self._engine.without(places)

    def reopened(self, name: str, text: str) -> None:
        """Say that fetched-back text now stands at the end of the conversation, under `name`."""
        if self.policy.agentic:
            self._engine.reopened(name, self.policy.spec.part_tokens(text))

    def lost(self) -> None:
        """Say that the provider has nothing of the conversation in its cache any more."""
        if self.policy.agentic:
            self._engine.flow.at = None

    def warm(self, at: float) -> bool:
        """Whether the provider is taken to have the conversation in its cache."""
        return self.policy.agentic and self._engine.flow.warm(at)

    def shows_again(self, name: str) -> bool:
        """Whether text that was fetched back under `name` still stands where it was put."""
        return self.policy.agentic and self._engine.shows_again(name)

    def complete(
        self,
        turn: Turn,
        answer: str,
        *,
        at=None,
        usage=None,
        last_usage=None,
        cited_part_ids=(),
        place=None,
    ):
        """Record the answer to `turn`. Returns when the next ping is due, if one is.

        `usage` is the provider's usage for the answer call, as a dict in
        Anthropic's or OpenAI's shape. With it the session tracks real input
        sizes and notices a cache that stopped hitting. Without it the request
        is taken to be cached the way the model's rules say. When the turn made
        several calls, `usage` is that of the first and `last_usage` that of
        the last: the first shows what the turn found cached, the last what it
        leaves there. If the host records which visible retrievals the answer
        cited, pass their part IDs.
        """
        if self._pending is None or self._pending[0] is not turn.plan.request:
            raise ContractError("no prepared turn to complete")
        request, parts = self._pending
        source = Source(
            f"{request.run_id}:output",
            "output",
            request.run_index,
            answer,
            cited_part_ids=cited_part_ids,
        )
        if source.cited_part_ids:
            visible = {
                p.part_id
                for p in turn.parts
                if p.kind == "retrieval" and p.part_id not in turn.archived_ids
            }
            if not set(source.cited_part_ids) <= visible:
                raise ContractError(
                    "cited_part_ids must name visible retrievals in the answered request"
                )
        counters = _counters(usage)
        final = _counters(last_usage)
        spec = self.policy.spec
        if self.policy.agentic:
            # Its calls were told of one by one: what counts is what the last one left.
            counters = final or counters
            counters["place"] = place
            if place is not None:
                counters["thought"] = _thought(last_usage or usage)
        elif counters and final:
            # The first call says what the turn found cached, the last what it leaves.
            left = final["input_tokens"]
            if final["cache_write_input_tokens"]:
                left = min(
                    left,
                    (final["cached_input_tokens"] or 0)
                    + final["cache_write_input_tokens"],
                )
            left = left // spec.cache_block_tokens * spec.cache_block_tokens
            cacheable = spec.cache_ttl_seconds > 0 and left >= spec.cache_min_tokens
            counters = {
                "input_tokens": final["input_tokens"],
                "cached_input_tokens": counters["cached_input_tokens"],
                "reusable_input_tokens": left if cacheable else 0,
            }
        if not counters:
            cached = turn.tokens // spec.cache_block_tokens * spec.cache_block_tokens
            cacheable = spec.cache_ttl_seconds > 0 and cached >= spec.cache_min_tokens
            counters = {"reusable_input_tokens": cached if cacheable else 0}
        self._engine.complete(
            turn.plan,
            source,
            ended_at=time.time() if at is None else at,
            **counters,
        )
        self._history.extend((*parts, source))
        self._run, self._pending = request.run_index, None
        return self.next_ping_at

    @property
    def next_ping_at(self):
        """Epoch seconds at which the next keep-warm ping is due, or None."""
        return self._engine.next_ping_at

    def ping(self, *, at=None) -> Turn | None:
        """The request to send as a keep-warm ping, or None when it is no longer worth sending.

        Send it with the smallest output the provider allows, then call `ping_sent`.
        """
        if not self._run and not self.policy.agentic:
            return None
        view = self._engine.ping_view(
            run_id=f"{self.thread_id}:{self._run}",
            at=time.time() if at is None else at,
        )
        if view is None:
            return None
        return Turn(view.parts, view.archived_ids, view.tokens, "ping")

    def ping_sent(self, *, at, usage=None, completed_at=None) -> bool:
        """Record a ping sent at `at`. True when it refreshed the cache.

        Pass the ping's own usage. Without it the refresh cannot be confirmed
        and no further ping is planned for this wait.
        """
        counted = canonical_usage(usage) or {}
        return self._engine.acknowledge_ping(
            run_id=f"{self.thread_id}:{self._run}",
            at=at,
            completed_at=completed_at,
            output_tokens=counted.get("output"),
            **_counters(usage),
        )

    def reopen(self, reference: str, offset: int = 0, *, search=("retrieval",)) -> dict:
        """Resolve the model's `reopen` tool call against this conversation's archive.

        Words that are not a reference are looked for in retrieved text. `search`
        adds `"input"` and `"output"`, for a conversation whose questions hold its sources.
        """
        if self._pending is not None:
            request = self._pending[0]
        elif self._run:
            request = self._request(self._run + 1, ())
        else:
            raise ContractError("nothing to reopen before the first turn")
        arguments = {"reference": reference}
        if offset:
            arguments["offset"] = offset
        if not self.policy.agentic:
            return _reopen(request, arguments, search=tuple(search))
        return _reopen(
            request,
            arguments,
            search=tuple(search),
            within=self._engine.archived_ids,
            page=_AGENT_PAGE,
        )
