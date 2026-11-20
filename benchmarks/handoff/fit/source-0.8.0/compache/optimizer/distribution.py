"""Finite-horizon expected thread cost with a trace-fitted demand distribution.

The demand model exposes only the arrival gap and payload at request time. A
response, its duration and future returns are revealed after the view decision.
All candidate actions use the same finite quadrature. The continuation can
postpone one approved rewrite of the current history; new content is appended.
The host constructs new eligible views and replans at each actual decision.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from math import expm1, inf, isfinite, log1p
from operator import itemgetter
from random import Random
from statistics import NormalDist, fmean, pstdev
from types import MappingProxyType

from compache.traces import ContractError

from .contracts import ModelSpec, number, seconds

SCHEMA = "compache-thread-demand-v2"
SINGLE_GENERATION_SCHEMA = "compache-thread-demand-v1"


def validate_calls_per_run(calls_per_run, run_ids):
    """Validate a run-ID sidecar and return only explicitly observed counts.

    Count the first model call and each subsequent foreground call on the
    same serial, append-only prompt. Exclude timers, rewrites, recovery and
    independent model requests. Missing and null counts mean unknown.
    """
    if calls_per_run is None:
        return {}
    if not isinstance(calls_per_run, Mapping):
        raise ContractError("calls_per_run must map run IDs to counts or null")
    allowed = set(run_ids)
    observed = {}
    for rid, count in calls_per_run.items():
        if rid not in allowed:
            raise ContractError("calls_per_run contains a foreign run ID")
        if count is not None:
            number(count, "calls_per_run", 1, integer=True)
            observed[rid] = count
    return observed


def _arriving(run, spec):
    texts = [run.input.text, *(p.text for p in run.retrievals if p.exposed_to_model)]
    if any(t is None for t in texts) or run.output.text is None:
        raise ContractError(
            "distribution fitting needs visible user, retrieval and output text"
        )
    return sum(spec.part_tokens(t) for t in texts)


def fit_distribution(
    threads,
    spec: ModelSpec,
    *,
    observation_end,
    horizon_seconds=86400,
    calls_per_run=None,
):
    """Fit observable return/size pairs and response/duration pairs by position.

    A landmark contributes to return probabilities only when its entire idle
    horizon was observable, regardless of whether an early return occurred.
    Terminal probability means no return inside that horizon. Changes in
    retained history from previous compaction are never fitted as demand.
    The optional calls_per_run sidecar supplies observed foreground counts.
    The canonical trace's single answer record alone does not measure them.
    """
    number(horizon_seconds, "horizon_seconds", 0)
    if horizon_seconds <= 0:
        raise ContractError("a positive idle observation horizon is required")
    cutoff = seconds(observation_end)
    threads = sorted(threads, key=lambda t: t.thread_id)
    counts = validate_calls_per_run(
        calls_per_run, (r.run_id for thread in threads for r in thread.runs)
    )
    rows = defaultdict(
        lambda: {
            "response": [],
            "returns": [],
            "no_return": 0,
            "censored": 0,
            "calls_per_run": [],
        }
    )
    thread_ids, run_ids = set(), set()
    reported = 0
    for thread in threads:
        if thread.thread_id in thread_ids:
            raise ContractError("duplicate fitting thread")
        thread_ids.add(thread.thread_id)
        sizes = [_arriving(r, spec) for r in thread.runs]
        for i, run in enumerate(thread.runs):
            if run.run_id in run_ids or run.generation.model != spec.model_id:
                raise ContractError(
                    "distribution fitting requires unique runs of the specified model"
                )
            run_ids.add(run.run_id)
            start, end = seconds(run.generation.started_at), seconds(run.ended_at)
            if end > cutoff:
                raise ContractError("observation_end precedes a run end")
            row = rows[run.run_index]
            visible = spec.part_tokens(run.output.text)
            if run.generation.usage is not None:
                output = dict(run.generation.usage)["output"]
                reported += 1
            else:
                output = spec.text_tokens(run.output.text)
            row["response"].append([output, visible, end - start])
            row["calls_per_run"].append(counts.get(run.run_id))
            if cutoff - end < horizon_seconds:
                row["censored"] += 1
                continue
            if i + 1 < len(thread.runs):
                gap = seconds(thread.runs[i + 1].generation.started_at) - end
                if gap <= horizon_seconds:
                    row["returns"].append([gap, sizes[i], sizes[i + 1]])
                    continue
            row["no_return"] += 1
    if not rows:
        raise ContractError("distribution fitting needs at least one thread")
    if not any(r["returns"] or r["no_return"] for r in rows.values()):
        raise ContractError(
            "no fully observed idle horizons; supply a supported shorter horizon or later observation cutoff"
        )
    return {
        "schema_version": SCHEMA,
        "model_id": spec.model_id,
        "token_ruler": [spec.bytes_per_token, spec.part_overhead_tokens],
        "observation_end": observation_end,
        "horizon_seconds": horizon_seconds,
        "training_threads": len(thread_ids),
        "training_runs": len(run_ids),
        "reported_output_runs": reported,
        "by_position": {str(k): rows[k] for k in sorted(rows)},
    }


@dataclass(frozen=True)
class _Row:
    responses: tuple
    returns: tuple
    no_return: int
    censored: int
    observed_runs: int
    observed_calls: int

    @property
    def unknown_runs(self):
        return len(self.responses) - self.observed_runs


@dataclass(frozen=True, eq=False)
class _Demand:
    positions: MappingProxyType
    pooled: _Row
    horizon: float
    model: str
    ruler: tuple
    threads: int

    def call_forecast(self, pos):
        row = self.positions.get(pos)
        if row and row.observed_runs:
            return row.observed_calls / row.observed_runs, "position"
        if self.pooled.observed_runs:
            return self.pooled.observed_calls / self.pooled.observed_runs, "pooled"
        return 1.0, "one_call_assumption"

    @cached_property
    def return_support(self):
        # Keep original gap/growth pairs available for late timer receipts,
        # including rare returns absent from an earlier set of quadrature draws.
        return lru_cache(maxsize=512)(
            lambda pos, sampler: _return_support(self, pos, sampler)
        )

    @cached_property
    def kernel(self):
        # A compiled profile owns its small quadrature cache. Dropping the
        # profile releases both data and cache across installations.
        return lru_cache(maxsize=512)(
            lambda pos, strata, seed, sampler, start=0: _outcomes(
                self, pos, strata, seed, sampler, start
            )
        )


def freeze_distribution(profile, spec):
    """Validate an artifact once and reuse its immutable representation during replay."""
    if isinstance(profile, _Demand):
        if profile.model != spec.model_id or profile.ruler != (
            spec.bytes_per_token,
            spec.part_overhead_tokens,
        ):
            raise ContractError("distribution model or token ruler mismatch")
        return profile
    fields = {
        "schema_version",
        "model_id",
        "token_ruler",
        "observation_end",
        "horizon_seconds",
        "training_threads",
        "training_runs",
        "reported_output_runs",
        "by_position",
    }
    if (
        not isinstance(profile, dict)
        or set(profile) != fields
        or profile["schema_version"] not in (SCHEMA, SINGLE_GENERATION_SCHEMA)
    ):
        raise ContractError("invalid distribution profile")
    if profile["model_id"] != spec.model_id or profile["token_ruler"] != [
        spec.bytes_per_token,
        spec.part_overhead_tokens,
    ]:
        raise ContractError("distribution model or token ruler mismatch")
    horizon = profile["horizon_seconds"]
    number(horizon, "horizon_seconds")
    if horizon <= 0:
        raise ContractError("a positive horizon is required")
    seconds(profile["observation_end"])
    number(profile["training_threads"], "training_threads", 1, integer=True)
    number(
        profile["training_runs"],
        "training_runs",
        profile["training_threads"],
        integer=True,
    )
    number(
        profile["reported_output_runs"],
        "reported_output_runs",
        0,
        profile["training_runs"],
        integer=True,
    )
    raw = profile["by_position"]
    if (
        not isinstance(raw, dict)
        or not raw
        or set(raw) != {str(k) for k in range(1, len(raw) + 1)}
    ):
        raise ContractError("demand positions must be consecutive and start at one")
    positions = {}
    row_fields = {"response", "returns", "no_return", "censored"}
    if profile["schema_version"] == SCHEMA:
        row_fields.add("calls_per_run")
    for k, row in raw.items():
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ContractError("invalid demand row")
        for name in ("no_return", "censored"):
            number(row[name], name, integer=True)
        if (
            not isinstance(row["returns"], list)
            or not isinstance(row["response"], list)
            or not row["response"]
        ):
            raise ContractError(
                "demand observations must be arrays with a response at each position"
            )
        for obs in row["returns"]:
            if not isinstance(obs, list) or len(obs) != 3:
                raise ContractError("invalid return observation")
            number(obs[0], "return gap", 0, horizon)
            number(obs[1], "prior arrival", integer=True)
            number(obs[2], "arrival", integer=True)
        for obs in row["response"]:
            if not isinstance(obs, list) or len(obs) != 3:
                raise ContractError("invalid response observation")
            number(obs[0], "output", integer=True)
            number(obs[1], "rendered output", integer=True)
            number(obs[2], "duration")
        observed = []
        if "calls_per_run" in row:
            counts = row["calls_per_run"]
            if not isinstance(counts, list) or len(counts) != len(row["response"]):
                raise ContractError("calls_per_run must include every response")
            for count in counts:
                if count is not None:
                    number(count, "calls_per_run", 1, integer=True)
                    observed.append(count)
        if (
            len(row["response"])
            != len(row["returns"]) + row["no_return"] + row["censored"]
        ):
            raise ContractError("inconsistent demand observation counts")
        positions[int(k)] = _Row(
            tuple(map(tuple, row["response"])),
            tuple(map(tuple, row["returns"])),
            row["no_return"],
            row["censored"],
            len(observed),
            sum(observed),
        )
    if (
        len(positions[1].responses) != profile["training_threads"]
        or sum(len(row.responses) for row in positions.values())
        != profile["training_runs"]
    ):
        raise ContractError("inconsistent demand training counts")
    pooled = _Row(
        tuple(s for row in positions.values() for s in row.responses),
        tuple(s for row in positions.values() for s in row.returns),
        sum(row.no_return for row in positions.values()),
        sum(row.censored for row in positions.values()),
        sum(row.observed_runs for row in positions.values()),
        sum(row.observed_calls for row in positions.values()),
    )
    if not pooled.returns and not pooled.no_return:
        raise ContractError("fitted return distribution is entirely censored")
    return _Demand(
        MappingProxyType(positions),
        pooled,
        horizon,
        spec.model_id,
        (spec.bytes_per_token, spec.part_overhead_tokens),
        profile["training_threads"],
    )


def request_charge(spec, input_tokens, shared_prefix_tokens, output_tokens):
    """Price one request in fresh-input-token equivalents, matching replay.Cache."""
    number(input_tokens, "input_tokens", integer=True)
    number(shared_prefix_tokens, "shared_prefix_tokens", 0, input_tokens, integer=True)
    number(output_tokens, "output_tokens", integer=True)
    return _charge(spec, input_tokens, shared_prefix_tokens, output_tokens)


def _input_buckets(spec, size, shared):
    if spec.cache_ttl_seconds <= 0 or size < spec.cache_min_tokens:
        return size, 0, 0
    read = shared // spec.cache_block_tokens * spec.cache_block_tokens
    if read < spec.cache_min_tokens:
        read = 0
    return 0, size - read, read


def _charge(spec, size, shared, output):
    fresh, writes, reads = _input_buckets(spec, size, shared)
    return (
        fresh
        + reads * spec.read_multiplier
        + writes * spec.write_multiplier
        + output * spec.output_multiplier
    )


def repeat_ping_charge(
    spec, input_tokens, reusable_tokens, output_tokens, *, uncached_input_rate=None
):
    """Forecast a repeat's rounded reads and price its residual at the receipt rate."""
    if uncached_input_rate is None:
        return request_charge(spec, input_tokens, reusable_tokens, output_tokens)
    number(uncached_input_rate, "uncached_input_rate")
    number(input_tokens, "input_tokens", integer=True)
    number(reusable_tokens, "reusable_tokens", 0, input_tokens, integer=True)
    number(output_tokens, "output_tokens", integer=True)
    _, _, reads = _input_buckets(spec, input_tokens, reusable_tokens)
    return (
        reads * spec.read_multiplier
        + (input_tokens - reads) * uncached_input_rate
        + output_tokens * spec.output_multiplier
    )


def repeat_input_usage(spec, input_tokens, additional_calls):
    """Price the initial input view on subsequent physical calls in one run.

    Each call is modeled as reusing that view inside the TTL. Block rounding
    and cache eligibility still apply, including fresh billing with no cache.
    Tool-loop suffixes, extra output and refresh timestamps are unobserved.
    """
    number(input_tokens, "input_tokens", integer=True)
    number(additional_calls, "additional_calls", integer=True)
    fresh, writes, reads = _input_buckets(spec, input_tokens, input_tokens)
    return {
        "input_tokens": additional_calls * input_tokens,
        "fresh_input_tokens": additional_calls * fresh,
        "cache_write_tokens": additional_calls * writes,
        "cache_read_tokens": additional_calls * reads,
        "cost_input_equivalents": additional_calls
        * _charge(spec, input_tokens, input_tokens, 0),
    }


@dataclass(frozen=True)
class _View:
    name: str
    size: int
    shared: int
    rewrite_shared: int
    extra: float
    admitted: bool
    admission_margin_tokens: int


def _views(candidates):
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 32:
        raise ContractError(
            "supply 1–32 feasible views, with the current unmodified view first"
        )
    result = []
    names = set()
    for c in candidates:
        required = {
            "name",
            "input_tokens",
            "shared_prefix_tokens",
            "rewrite_prefix_tokens",
        }
        if not isinstance(c, dict) or not required <= set(c) <= required | {
            "extra_cost",
            "admitted",
            "admission_margin_tokens",
        }:
            raise ContractError("invalid distribution view fields")
        name, size, shared, rewrite = (
            c[k]
            for k in (
                "name",
                "input_tokens",
                "shared_prefix_tokens",
                "rewrite_prefix_tokens",
            )
        )
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ContractError("candidate names must be unique and nonempty")
        number(size, "input_tokens", integer=True)
        number(shared, "shared_prefix_tokens", 0, size, integer=True)
        number(
            rewrite,
            "rewrite_prefix_tokens",
            0,
            min(size, result[0].size if result else size),
            integer=True,
        )
        extra = c.get("extra_cost", 0)
        number(extra, "extra_cost")
        if not result and (rewrite != size or extra):
            raise ContractError(
                "first (keep) view must share its full current view and have zero rewrite cost"
            )
        admitted = c.get("admitted", True)
        if type(admitted) is not bool:
            raise ContractError("admitted must be boolean")
        margin = c.get("admission_margin_tokens", 0)
        number(margin, "admission_margin_tokens", integer=True)
        result.append(_View(name, size, shared, rewrite, extra, admitted, margin))
        names.add(name)
    return tuple(result)


def _quadrature(ordered, branches, seed, start=0):
    """Sample the sorted eligible cohort, exactly for small cohorts."""
    count = len(ordered) - start
    if count <= branches:
        return tuple((k, v / count) for k, v in Counter(ordered[start:]).items())
    rng = Random(seed)
    samples = Counter()
    for i in range(branches):
        lo = start + i * count // branches
        hi = start + (i + 1) * count // branches
        samples[ordered[rng.randrange(lo, hi)]] += hi - lo
    return tuple((s, n / count) for s, n in samples.items())


def _return_support(demand, pos, sampler):
    row = demand.positions.get(pos, demand.pooled)
    if not row.returns and not row.no_return:
        row = demand.pooled
    if not row.returns:
        return (), row.no_return
    returns = []
    if sampler == "brownian":
        deltas = [log1p(new) - log1p(old) for _, old, new in row.returns]
        drift, sigma = fmean(deltas), pstdev(deltas)
        # Keep pairing between observed return gaps and growth ranks. Transform
        # the empirical growth rank into a normal innovation on log arrivals.
        ranks = {}
        seen = 0
        for value, count in sorted(Counter(deltas).items()):
            ranks[value] = (seen + count / 2) / len(deltas)
            seen += count
        for (gap, old, new), delta in zip(row.returns, deltas):
            shift = drift + sigma * NormalDist().inv_cdf(ranks[delta])
            returns.append((gap, old, new, shift))
    else:
        returns = [(*v, 0.0) for v in row.returns]
    return tuple(sorted(returns)), row.no_return


def _outcomes(demand, pos, branches, seed, sampler, start):
    row = demand.positions.get(pos, demand.pooled)
    responses = _quadrature(sorted(row.responses), branches, seed + 2 * pos)
    returns, stops = demand.return_support(pos, sampler)
    count = len(returns) - start
    total = count + stops
    if not count:
        return responses, float(bool(stops)), ()
    scale = count / total
    return (
        responses,
        stops / total,
        tuple(
            (obs, p * scale)
            for obs, p in _quadrature(returns, branches, seed + 2 * pos + 1, start)
        ),
    )


# Runtime exploration is bounded independently of the fitted company size.
# Exhausting it hands control back to the incumbent; partial scores cannot win.
MAX_SEARCH_STATES = 25000


class _BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class _PingContinuation:
    dispatch: float
    observed: float
    deadlines: tuple
    input_tokens: int
    reusable_tokens: int
    output_tokens: int
    uncached_input_rate: float | None
    return_start: int
    returns: int
    stops: int

    @classmethod
    def from_receipt(cls, receipt, search, duration, max_pings):
        required = {
            "elapsed_seconds",
            "remaining_due_after_start_seconds",
            "input_tokens",
            "reusable_tokens",
            "output_tokens",
        }
        if not isinstance(receipt, Mapping) or not (
            required
            <= receipt.keys()
            <= required | {"observed_elapsed_seconds", "uncached_input_rate"}
        ):
            raise ContractError("invalid after_ping fields")
        dispatch = receipt["elapsed_seconds"]
        observed = receipt.get("observed_elapsed_seconds", dispatch)
        number(dispatch, "elapsed_seconds", duration)
        number(observed, "observed_elapsed_seconds", dispatch)
        held = receipt["input_tokens"]
        reusable = receipt["reusable_tokens"]
        output = receipt["output_tokens"]
        number(held, "input_tokens", integer=True)
        number(reusable, "reusable_tokens", 0, held, integer=True)
        number(output, "output_tokens", integer=True)
        rate = receipt.get("uncached_input_rate")
        if rate is not None:
            number(rate, "uncached_input_rate")
        slots = receipt["remaining_due_after_start_seconds"]
        if not isinstance(slots, (list, tuple)) or len(slots) > max(0, max_pings - 1):
            raise ContractError("remaining deadlines exceed the original ping budget")
        deadlines = []
        previous = None
        for slot in slots:
            number(slot, "remaining ping deadline")
            if previous is not None and slot <= previous:
                raise ContractError("remaining ping deadlines must strictly increase")
            previous = slot
            if slot > dispatch and slot >= observed:
                deadlines.append(slot)
        # The newly confirmed cache expires from dispatch, even if its receipt
        # arrives late. Only a consecutive, still-warm chain is actionable.
        last_refresh = dispatch
        for i, due in enumerate(deadlines):
            if (
                due - last_refresh >= search.spec.cache_ttl_seconds
                or due > duration + search.demand.horizon
            ):
                deadlines = deadlines[:i]
                break
            last_refresh = due
        returns, stops = search.demand.return_support(search.index, search.sampler)
        start = bisect_right(returns, observed - duration, key=itemgetter(0))
        return cls(
            dispatch,
            observed,
            tuple(deadlines),
            held,
            reusable,
            output,
            rate,
            start,
            len(returns) - start,
            stops,
        )

    def anchor_views(self, views, answer):
        """Move all candidate baselines to the measured keep input including its answer."""
        anchored = []
        original = views[0].size
        for i, view in enumerate(views):
            held = self.input_tokens + view.size - original
            if i and held <= 0:
                # The forecast cut would remove the entire measured prompt.
                # Retain its existing size estimate, as in edit accounting.
                held = view.size + answer
            shared = min(view.rewrite_shared, held, self.input_tokens)
            anchored.append(
                replace(view, size=held, rewrite_shared=shared if i else held)
            )
        return tuple(anchored)


class _Search:
    def __init__(
        self,
        profile,
        spec,
        run_index,
        candidates,
        *,
        ping_allowed,
        max_pings,
        ping_lead_seconds,
        lookahead,
        reserve_tokens,
        branches,
        seed,
        sampler,
        current_arrival_tokens,
        shared_header_tokens,
        ambient_header_tokens,
        ambient_hit_probability,
    ):
        number(run_index, "run_index", 1, integer=True)
        number(lookahead, "lookahead", 1, 4, integer=True)
        number(branches, "branches", 1, 12, integer=True)
        number(seed, "seed", 0, integer=True)
        number(max_pings, "max_pings", 0, 8, integer=True)
        number(ping_lead_seconds, "ping_lead_seconds")
        number(reserve_tokens, "reserve_tokens", 0, spec.context_tokens, integer=True)
        if type(ping_allowed) is not bool or sampler not in ("empirical", "brownian"):
            raise ContractError("invalid ping availability or demand sampler")
        if current_arrival_tokens is not None:
            number(current_arrival_tokens, "current_arrival_tokens", integer=True)
        number(shared_header_tokens, "shared_header_tokens", integer=True)
        number(ambient_header_tokens, "ambient_header_tokens", integer=True)
        number(ambient_hit_probability, "ambient_hit_probability", 0, 1)
        self.shared_header_tokens = shared_header_tokens
        self.ambient_header_tokens = ambient_header_tokens
        self.ambient_hit_probability = ambient_hit_probability
        self.demand = freeze_distribution(profile, spec)
        self.views = _views(candidates)
        self.spec = spec
        self.expanded_states = 0
        self.index = run_index
        self.depth = lookahead
        self.cap = spec.context_tokens - reserve_tokens
        self.interval = spec.cache_ttl_seconds - ping_lead_seconds
        self.max_pings = (
            max_pings
            if ping_allowed and 0 < self.interval < spec.cache_ttl_seconds
            else 0
        )
        self.branches = branches
        self.seed = seed
        self.sampler = sampler
        # Scope the memo tables to this decision, with numeric state rather than
        # thread IDs, model answers or hypothetical future demand paths as keys.
        self.arrived = lru_cache(maxsize=8192)(self._arrived)
        self.after = lru_cache(maxsize=8192)(self._after)

    def count_state(self):
        self.expanded_states += 1
        if self.expanded_states > MAX_SEARCH_STATES:
            raise _BudgetExceeded

    def clear(self):
        self.arrived.cache_clear()
        self.after.cache_clear()

    def metadata(self, *, idle=False):
        row = self.demand.positions.get(self.index)
        start = self.index + int(idle)
        forecasts = {}
        for pos in range(start, start + self.depth):
            calls, source = self.demand.call_forecast(pos)
            cohort = self.demand.positions.get(pos)
            forecasts[str(pos)] = {
                "expected_calls": calls,
                "source": source,
                "observed_runs": cohort.observed_runs if cohort else 0,
                "unknown_runs": cohort.unknown_runs if cohort else 0,
            }
        return {
            "objective": "expected cumulative modeled cost",
            "unit": "fresh_input_token_equivalents",
            "lookahead": self.depth,
            "expanded_states": self.expanded_states,
            "max_search_states": MAX_SEARCH_STATES,
            "terminal_value": 0,
            "idle_horizon_seconds": self.demand.horizon,
            "sampler": self.sampler,
            "quadrature_strata": self.branches,
            "seed": self.seed,
            "training_threads": self.demand.threads,
            "calls_per_run": {
                "training_observed_runs": self.demand.pooled.observed_runs,
                "training_unknown_runs": self.demand.pooled.unknown_runs,
                "by_position": forecasts,
            },
            "shared_header_tokens_now": self.shared_header_tokens,
            "ambient_header_tokens": self.ambient_header_tokens,
            "ambient_hit_probability": self.ambient_hit_probability,
            "position_threads": len(row.responses) if row else 0,
            "position_return_landmarks": len(row.returns) + row.no_return if row else 0,
            "pooled_return_fallback_at_position": not row
            or not (row.returns or row.no_return),
            "ping_actions_enabled": self.max_pings > 0,
            "continuation": "Eligible current-history rewrites can be deferred. Newly arriving content appends until the next real decision; the host revalidates actual views.",
        }

    def kernel(self, pos):
        return self.demand.kernel(pos, self.branches, self.seed, self.sampler, 0)

    def arrival_size(self, observation, previous):
        _, old, new, shift = observation
        if self.sampler == "empirical":
            return new
        log_size = log1p(old if previous is None else previous) + shift
        if log_size > log1p(self.spec.context_tokens + 1):
            # All larger current arrivals exceed the hard cap by themselves.
            return self.spec.context_tokens + 1
        return max(0, round(expm1(log_size)))

    def _arrived(self, pos, depth, view, growth, shared, arrival):
        """View already chosen on visible arrival; integrate hidden response and future."""
        self.count_state()
        choice = self.views[view]
        size = choice.size + growth
        # This error allowance is measured for the actual edit at arrival.
        # Future simulated edits have their own growth distribution and reserve.
        margin = choice.admission_margin_tokens if pos == self.index else 0
        if not choice.admitted or size + margin > self.cap:
            return inf
        input_cost = _charge(self.spec, size, shared, 0)
        if (
            pos > self.index
            and self.ambient_hit_probability
            and self.ambient_header_tokens > shared
        ):
            # External activity may preserve a common system/tool prefix after
            # the thread's cache expires. It overlaps the thread prefix. The
            # realized provider hit is unknown until after the view decision.
            with_header = _charge(
                self.spec, size, min(size, self.ambient_header_tokens), 0
            )
            input_cost += self.ambient_hit_probability * (with_header - input_cost)
        responses, _, _ = self.kernel(pos)
        total = input_cost
        # The view is chosen before knowing the run's physical call count.
        # Its initial prompt recurs on every extra call, even after a cold
        # first send. This cost is linear in the count, so its empirical mean
        # is exact for this term regardless of the response quadrature.
        expected_calls, _ = self.demand.call_forecast(pos)
        if expected_calls > 1:
            total += (expected_calls - 1) * _charge(self.spec, size, size, 0)
        for (out, history_out, duration), p in responses:
            if size + margin + out > self.spec.context_tokens:
                return inf
            continuation = (
                min(
                    self.after(
                        pos, depth - 1, view, growth, history_out, duration, arrival
                    )
                )
                if depth > 1
                else 0.0
            )
            total += p * (self.spec.output_multiplier * out + continuation)
        return total

    def _after(
        self,
        pos,
        depth,
        view,
        growth,
        history_out,
        duration,
        arrival,
        held_cache_tokens=None,
        after_ping=None,
    ):
        """Choose a schedule before observing the next return, including idle exits."""
        self.count_state()
        size = self.views[view].size + growth
        held = size + history_out
        if after_ping is None:
            initial_prefix = (
                size if held_cache_tokens is None else min(size, held_cache_tokens)
            )
            repeat_prefix = held
            output = self.spec.ping_output_tokens
            refreshed = 0
            start = 0
            deadlines = tuple(
                i * self.interval
                for i in range(1, self.max_pings + 1)
                if duration < self.interval
                and i * self.interval <= duration + self.demand.horizon
            )
        else:
            # The ping already paid for the appended answer. Even a lapse can
            # use its cached blocks on a return before the new TTL expires.
            initial_prefix = repeat_prefix = after_ping.reusable_tokens
            output = after_ping.output_tokens
            refreshed = after_ping.dispatch
            start = after_ping.return_start
            deadlines = after_ping.deadlines if self.max_pings else ()
        if (
            size < self.spec.cache_min_tokens
            or held > self.cap
            or held + output > self.spec.context_tokens
        ):
            deadlines = ()
        _, p_stop, returns = self.demand.kernel(
            pos, self.branches, self.seed, self.sampler, start
        )
        ping_costs = [0.0]
        if deadlines:
            if after_ping is None:
                first = _charge(self.spec, held, initial_prefix, output)
                repeat = _charge(self.spec, held, repeat_prefix, output)
            else:
                first = repeat = repeat_ping_charge(
                    self.spec,
                    held,
                    repeat_prefix,
                    output,
                    uncached_input_rate=after_ping.uncached_input_rate,
                )
            ping_costs.extend(first + n * repeat for n in range(len(deadlines)))
        scores = []
        next_views = range(len(self.views)) if view == 0 else (view,)
        for n, stop_cost in enumerate(ping_costs):
            total = p_stop * stop_cost
            for obs, p in returns:
                gap = obs[0]
                incoming = self.arrival_size(obs, arrival)
                future_growth = growth + history_out + incoming
                # An arriving user cancels a timer even at the same timestamp.
                end = duration + gap
                sent = bisect_left(deadlines, end, hi=n)
                last_refresh = deadlines[sent - 1] if sent else refreshed
                warm = end - last_refresh < self.spec.cache_ttl_seconds
                cached = (repeat_prefix if sent else initial_prefix) if warm else 0
                choices = []
                for other in next_views:
                    rewrite = other != view
                    # All newly appended demand comes after the historical
                    # checkpoint. A later rewrite retains only its original
                    # matching prefix, irrespective of appended history size.
                    shared = (
                        min(cached, self.views[other].rewrite_shared)
                        if rewrite
                        else cached
                    )
                    cost = self.arrived(
                        pos + 1, depth, other, future_growth, shared, incoming
                    )
                    choices.append(cost + (self.views[other].extra if rewrite else 0))
                total += p * (ping_costs[sent] + min(choices))
            scores.append(total)
        return tuple(scores)


def _cost_dict(costs):
    return {k: (v if isfinite(v) else None) for k, v in costs.items()}


def plan_distribution(
    profile,
    spec,
    *,
    run_index,
    candidates,
    warm,
    ping_allowed=True,
    max_pings=1,
    ping_lead_seconds=45,
    lookahead=3,
    reserve_tokens=0,
    branches=3,
    seed=0,
    sampler="empirical",
    current_arrival_tokens=None,
    shared_header_tokens=0,
    ambient_header_tokens=0,
    ambient_hit_probability=0.0,
):
    """Choose the feasible current view with lowest expected cost over `lookahead` runs.

    Supply the unmodified view first, followed by dependency-approved checkpoints.
    `shared_prefix_tokens` compares each view to the previous actual request.
    `rewrite_prefix_tokens` compares each view to the unmodified current request.
    Both must include only truly matching prompt bytes under the host token ruler.
    `shared_header_tokens` is the available scoped common system/default-tools
    prefix now. Later requests use the separate ambient forecast. Prefixes
    rooted at the same first token overlap, so only the larger match is priced.
    """
    if type(warm) is not bool:
        raise ContractError("warm must be boolean")
    search = _Search(
        profile,
        spec,
        run_index,
        candidates,
        ping_allowed=ping_allowed,
        max_pings=max_pings,
        ping_lead_seconds=ping_lead_seconds,
        lookahead=lookahead,
        reserve_tokens=reserve_tokens,
        branches=branches,
        seed=seed,
        sampler=sampler,
        current_arrival_tokens=current_arrival_tokens,
        shared_header_tokens=shared_header_tokens,
        ambient_header_tokens=ambient_header_tokens,
        ambient_hit_probability=ambient_hit_probability,
    )
    try:
        costs = {
            v.name: search.arrived(
                run_index,
                lookahead,
                i,
                0,
                max(v.shared if warm else 0, min(v.size, shared_header_tokens)),
                current_arrival_tokens,
            )
            + v.extra
            for i, v in enumerate(search.views)
        }
        best = min(costs, key=costs.get)
        viable = isfinite(costs[best])
        return {
            "action": best if viable else None,
            "expected_cost": costs[best] if viable else None,
            "costs": _cost_dict(costs),
            "status": "ok" if viable else "no_feasible_view_in_modeled_horizon",
            **search.metadata(),
        }
    except _BudgetExceeded:
        return {
            "action": None,
            "expected_cost": None,
            "costs": {},
            "status": "search_budget_exhausted",
            **search.metadata(),
        }
    finally:
        search.clear()


def plan_distribution_idle(
    profile,
    spec,
    *,
    run_index,
    candidates,
    answer_tokens,
    duration_seconds,
    held_cache_tokens=None,
    after_ping=None,
    ping_allowed=True,
    max_pings=1,
    ping_lead_seconds=45,
    lookahead=3,
    reserve_tokens=0,
    branches=3,
    seed=0,
    sampler="empirical",
    current_arrival_tokens=None,
    shared_header_tokens=0,
    ambient_header_tokens=0,
    ambient_hit_probability=0.0,
):
    """Optimize a cancel-on-return ping schedule and costs over `lookahead` future runs.

    The first view is the actual submitted input. `answer_tokens` is the rendered
    history footprint of the known response. Any additional views are eligible
    deferred cuts of that same prompt. Supply `held_cache_tokens` when the host
    knows the completed request's reusable prefix capacity. It caps reads on
    the next request or first ping, until a new request refreshes the cache.
    Due times are relative to request start.

    After a confirmed ping, pass its actual dispatch offset in elapsed_seconds,
    optional receipt offset in observed_elapsed_seconds, remaining offered
    deadlines in remaining_due_after_start_seconds, and its input_tokens,
    reusable_tokens and output_tokens together in after_ping. Scores then cover
    future spend only, conditional on no return through receipt time. The ping
    input includes the foreground answer. A repeat sends the same input and
    uses the measured output count as its output forecast. Its optional
    uncached_input_rate is the measured fresh/write cost per unread input token.
    This rate applies to the repeat's remaining unread tokens. Each new foreground
    run keeps its original modeled ping budget and output assumption.
    """
    number(answer_tokens, "answer_tokens", integer=True)
    number(duration_seconds, "duration_seconds")
    if held_cache_tokens is not None:
        number(held_cache_tokens, "held_cache_tokens", integer=True)
    search = _Search(
        profile,
        spec,
        run_index,
        candidates,
        ping_allowed=ping_allowed,
        max_pings=max_pings,
        ping_lead_seconds=ping_lead_seconds,
        lookahead=lookahead,
        reserve_tokens=reserve_tokens,
        branches=branches,
        seed=seed,
        sampler=sampler,
        current_arrival_tokens=current_arrival_tokens,
        shared_header_tokens=shared_header_tokens,
        ambient_header_tokens=ambient_header_tokens,
        ambient_hit_probability=ambient_hit_probability,
    )
    extra = {}
    continuation = None
    try:
        if after_ping is not None:
            continuation = _PingContinuation.from_receipt(
                after_ping, search, duration_seconds, max_pings
            )
            total = continuation.returns + continuation.stops
            extra["support"] = {
                "returns": continuation.returns,
                "no_return": continuation.stops,
                "return_probability": continuation.returns / total if total else None,
            }
            if (
                continuation.observed >= duration_seconds + search.demand.horizon
                or not total
            ):
                return {
                    "pings": 0,
                    "due_after_start_seconds": [],
                    "costs": {},
                    "expected_cost": None,
                    "status": "idle_horizon_elapsed"
                    if total
                    else "no_observed_idle_survivors",
                    **extra,
                    **search.metadata(idle=True),
                }
            search.views = continuation.anchor_views(search.views, answer_tokens)
            answer_tokens = 0  # Already included in every rebased candidate.
        values = search.after(
            run_index,
            lookahead,
            0,
            0,
            answer_tokens,
            duration_seconds,
            current_arrival_tokens,
            held_cache_tokens,
            continuation,
        )
        best = min(range(len(values)), key=values.__getitem__)
        viable = isfinite(values[best])
        n = best if viable else 0
        return {
            "pings": n,
            "due_after_start_seconds": list(continuation.deadlines[:n])
            if continuation is not None
            else [i * search.interval for i in range(1, n + 1)],
            "costs": _cost_dict(dict(enumerate(values))),
            "expected_cost": values[best] if viable else None,
            "status": "ok" if viable else "no_feasible_view_in_modeled_horizon",
            **extra,
            **search.metadata(idle=True),
        }
    except _BudgetExceeded:
        return {
            "pings": 0,
            "due_after_start_seconds": [],
            "expected_cost": None,
            "costs": {},
            "status": "search_budget_exhausted",
            **extra,
            **search.metadata(idle=True),
        }
    finally:
        search.clear()
