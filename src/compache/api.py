"""Fit a policy on recorded threads, price it on others, and apply it to live ones."""

from __future__ import annotations

import json
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    cache_read_reported,
    canonical_usage,
    load_threads,
    read_jsonl,
    recorded,
)
from .wire import cache_mark

__all__ = ["Policy", "Session", "Turn", "fit", "replay"]

_HORIZON_SECONDS = 86400


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
    return load_threads(rows, model=spec.model_id), spec, found, model


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
    tools and when the record was read. Without labels only the pings are
    fitted and the whole history stays in view. Where the threads record
    several model calls in a turn, those are priced.

    Every candidate is replayed against the provider's cache rules at the
    threads' own timestamps, each thread priced by a model of returns fitted on
    the others, and the cheapest wins after paying for what it would fetch back.
    Where a known model can be asked for another cache lifetime, as Claude can
    for an hour, every candidate is priced under that too.
    `observed_until` is when the record was cut, for a record that ends while
    conversations were still going. `options` go to the solver.
    """
    labels = _labels(labels, threads)
    threads, spec, found, model = _threads(threads, model)
    if system_prompt is None:
        system_prompt = found["system_prompt"] or ""
    tools = list(found["tools"] or ()) if tools is None else list(tools)
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
        **{
            "calls_per_run": _calls(threads),
            "lifetimes": cache_lifetimes(model),
            **options,
        },
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
        "system_prompt": system_prompt,
        "default_tools": tools,
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
    threads, _, found, _ = _threads(threads, policy.spec)
    header = policy.artifact.get("header") or {}
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
            "cost_per_1000_turns": policy._per_1000_turns(paired, len(common_runs)),
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
            **policy._per_1000_turns({"policy": cost}, turns),
            **policy._per_1000_turns({"keep_everything": keep}, baseline_turns),
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
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != "compache-workload-planner-v1"
        ):
            raise ContractError("expected a fitted policy")
        self.artifact = artifact
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
        if "standard_spec" not in self.artifact:
            return None
        return self.spec.cache_ttl_seconds

    @property
    def cache_control(self) -> dict:
        """The cache mark for a Claude request you assemble yourself."""
        return cache_mark(self.cache_lifetime)

    @property
    def recipe(self) -> RetentionRecipe | None:
        """The cut the policy applies, or None when it keeps the whole history."""
        for row in self.artifact["recipes"]:
            if row["eligible"]:
                return RetentionRecipe(**row["recipe"])
        return None

    @property
    def max_pings(self) -> int:
        return (self.artifact.get("planner_options") or {}).get("max_pings", 1)

    def _per_1000_turns(self, costs, turns):
        """Costs per 1,000 turns: in dollars for a known model, else in input tokens."""
        price = (self.artifact.get("pricing") or {}).get("input_price_per_million")
        scale = 1000 / turns if turns else None
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
            else self._per_1000_turns(costs, solved.get("turns")),
        }

    def describe(self) -> str:
        """The decision in sentences."""
        summary, recipe = self.summary, self.recipe
        lines = []
        if recipe is None:
            lines.append(
                "Keep the whole history in view."
                if summary["labelled_turns"]
                else "Keep the whole history in view: without labels only the pings are fitted."
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
            if self._compiled is None:
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
        self._engine = WorkloadSession(
            policy._compiled,
            CacheScope("compache", "declared", self._model, thread_id),
            policy._checker,
        )
        self._history: list[Source] = []
        self._run = 0
        self._pending = None

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

    def extend(self, turn: Turn, retrieved) -> Turn:
        """Add text retrieved while `turn` is being answered, such as the results of a tool loop.

        The history stays as `prepare` sent it and the new items follow it, so
        the cached prefix is untouched. Returns the turn to pass to `complete`.
        """
        if self._pending is None or self._pending[0] is not turn.plan.request:
            raise ContractError("no prepared turn to extend")
        request, parts = self._pending
        found = sum(p.kind == "retrieval" for p in parts)
        added = self._retrieved(request.run_index, retrieved, found + 1)
        plan = self._engine.extend(turn.plan, added)
        self._pending = (plan.request, parts + tuple(added))
        view = plan.view
        return Turn(view.parts, view.archived_ids, view.tokens, turn.action, plan)

    def complete(
        self,
        turn: Turn,
        answer: str,
        *,
        at=None,
        usage=None,
        last_usage=None,
        cited_part_ids=(),
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
        if counters and final:
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
        elif not counters:
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
        if not self._run:
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
        return _reopen(request, arguments, search=tuple(search))
