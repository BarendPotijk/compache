"""Measure a workload from recorded threads, in the terms of the cost calculator.

The calculator prices a workload described by a set of numbers: tokens per
turn, how long threads run and how that depends on how far they have come, how
long users stay away, and how often and how far back a turn needs something
earlier. `profile` reads those numbers off recorded threads and labels and
returns a link that opens the calculator with them.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from statistics import NormalDist, fmean
from urllib.parse import urlencode

from .errors import ContractError
from .labels import normalize_label
from .models import model_spec, pricing
from .optimizer.contracts import seconds
from .optimizer.distribution import measure_cache_reads
from .optimizer.parts import Source, placeholder
from .record import is_record
from .record import load as load_record
from .threads import load_threads, read_jsonl, recorded

__all__ = ["CALCULATOR_URL", "history_resent", "profile", "workload"]

CALCULATOR_URL = "https://barendpotijk.github.io/compache/"
_KINDS = ("input", "retrieval", "output")
_LINK_VERSION = "7"
# Turns after the first by which an effect of position is halfway to its long-thread value.
_HALF_TURNS = (0.5, 1, 2, 3, 5, 8, 13, 21, 34)
_Z95 = 1.6448536269514722
_NORMAL = NormalDist()
# A request that carries less than this share of the conversation so far has left text out.
_RESENT = 0.8
# The calculator's limits on thread length.
_MAX_TURNS, _MAX_TURNS_P95 = 160, 320


def _quantile(values, q):
    """Nearest-rank quantile of a nonempty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _spread(values):
    return {"median": _quantile(values, 0.5), "p95": _quantile(values, 0.95)}


def _rows(source):
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return list(read_jsonl(source))
    return list(source)


def _nearest(alternatives, origin, target):
    """How far back each kind of part lies in the alternative whose evidence is nearest."""
    best = None
    for alternative in alternatives:
        far = {}
        for source in alternative["sources"]:
            kind, turn = origin.get(source["part_id"], (None, target))
            if turn < target:
                far[kind] = max(far.get(kind, 0), target - turn)
        if not far:
            continue
        key = (max(far.values()), len(alternative["sources"]))
        if best is None or key < best[0]:
            best = (key, far)
    return best[1] if best else {}


# -- thread length -----------------------------------------------------------


def _fit_length(lengths) -> dict:
    """Thread length two ways, and which describes these threads better.

    A lognormal has a median and a 95th percentile. The other law gives the
    chance of another turn after turn n: it starts at `first`, tends to `long`
    and is halfway `half_turn` turns after the first. Real traffic usually has many one-turn
    threads and long threads that keep going, which only the second can show.
    """
    median, p95 = _quantile(lengths, 0.5), _quantile(lengths, 0.95)
    top = max(lengths)
    reach = [sum(n >= turn for n in lengths) for turn in range(1, top + 2)]
    # Lognormal, rounded to a whole turn as the calculator draws it.
    sigma = math.log(p95 / median) / _Z95 if p95 > median else 0.0

    def share(turn):
        if sigma == 0:
            return 1.0 if turn == median else 0.0
        upper = _NORMAL.cdf((math.log(turn + 0.5) - math.log(median)) / sigma)
        lower = (
            _NORMAL.cdf((math.log(turn - 0.5) - math.log(median)) / sigma)
            if turn > 1
            else 0.0
        )
        return upper - lower

    lognormal = sum(math.log(max(share(n), 1e-12)) for n in lengths)

    def likelihood(first, long, decay):
        total = 0.0
        for i, weight in enumerate(decay):
            chance = min(1 - 1e-9, max(1e-9, long - (long - first) * weight))
            goes_on = reach[i + 1]
            total += goes_on * math.log(chance) + (reach[i] - goes_on) * math.log(
                1 - chance
            )
        return total

    best = None
    for half in _HALF_TURNS:
        decay = [0.5 ** (turn / half) for turn in range(top)]
        coarse = max(
            (likelihood(f / 20, g / 20, decay), f / 20, g / 20)
            for f in range(21)
            for g in range(21)
        )
        fine = max(
            (likelihood(f, g, decay), f, g)
            for f in (min(1.0, max(0.0, coarse[1] + k / 100)) for k in range(-5, 6))
            for g in (min(1.0, max(0.0, coarse[2] + k / 100)) for k in range(-5, 6))
        )
        if best is None or fine[0] > best[0]:
            best = (fine[0], fine[1], fine[2], half)
    return {
        "median": median,
        "p95": p95,
        "one_turn_share": lengths.count(1) / len(lengths),
        "another_turn": {
            "after_first": round(best[1], 2),
            "in_long_threads": round(best[2], 2),
            "half_turn": best[3],
        },
        # A handful of threads cannot tell the two apart.
        "law": "continuation"
        if len(lengths) >= 20 and best[0] > lognormal + 2
        else "lognormal",
    }


# -- return delay ------------------------------------------------------------


def _log_density(value, mean, sigma):
    return -0.5 * ((value - mean) / sigma) ** 2 - math.log(sigma)


def _long_breaks(logs):
    """Two kinds of return, if the delays clearly hold two: the usual one and a long break.

    Returns the chance each delay is a break, or None when one lognormal
    describes them about as well.
    """
    count = len(logs)
    if count < 40:
        return None
    ordered = sorted(logs)
    cut = int(count * 0.8)
    parts = [ordered[:cut], ordered[cut:]]
    weight = 0.2
    means = [fmean(p) for p in parts]
    sigmas = [
        max(0.05, math.sqrt(fmean((x - m) ** 2 for x in p)))
        for p, m in zip(parts, means)
    ]
    breaks = [0.0] * count
    for _ in range(200):
        for i, x in enumerate(logs):
            usual = (1 - weight) * math.exp(_log_density(x, means[0], sigmas[0]))
            late = weight * math.exp(_log_density(x, means[1], sigmas[1]))
            breaks[i] = late / (usual + late) if usual + late > 0 else 0.0
        total = sum(breaks)
        if not 1e-6 < total < count - 1e-6:
            return None
        before = (weight, *means)
        weight = total / count
        means = [
            sum((1 - b) * x for b, x in zip(breaks, logs)) / (count - total),
            sum(b * x for b, x in zip(breaks, logs)) / total,
        ]
        sigmas = [
            max(
                0.05,
                math.sqrt(
                    sum((1 - b) * (x - means[0]) ** 2 for b, x in zip(breaks, logs))
                    / (count - total)
                ),
            ),
            max(
                0.05,
                math.sqrt(
                    sum(b * (x - means[1]) ** 2 for b, x in zip(breaks, logs)) / total
                ),
            ),
        ]
        if max(abs(a - b) for a, b in zip(before, (weight, *means))) < 1e-7:
            break
    mean = fmean(logs)
    sigma = max(0.05, math.sqrt(fmean((x - mean) ** 2 for x in logs)))
    one = sum(_log_density(x, mean, sigma) for x in logs)
    two = sum(
        math.log(
            max(
                1e-300,
                (1 - weight) * math.exp(_log_density(x, means[0], sigmas[0]))
                + weight * math.exp(_log_density(x, means[1], sigmas[1])),
            )
        )
        for x in logs
    )
    # Two kinds must be far apart, both present, and clearly better than one.
    clear = (
        0.02 <= weight <= 0.6
        and means[1] - means[0] >= math.log(8)
        and math.exp(means[1]) >= 30
        and -2 * two + 5 * math.log(count) < -2 * one + 2 * math.log(count) - 10
    )
    return (breaks, weight, math.exp(means[1])) if clear else None


def _fit_delay(delays) -> dict:
    """The return delay in minutes: its median and spread, long breaks, and what position does.

    `delays` pairs the turn a delay follows with its length. With an effect of
    position the median and spread are those of long threads, and
    `first_factor` says how much longer the wait after turn one is.
    """
    logs = [math.log(max(minutes, 1e-3)) for _, minutes in delays]
    found = {
        "median": None,
        "p95": None,
        "first_factor": 1.0,
        "half_turn": None,
        "break_share": 0.0,
        "break_median": None,
    }
    usual = list(zip((turn for turn, _ in delays), logs))
    minutes = [value for _, value in delays]
    split = _long_breaks(logs)
    if split is not None:
        chances, found["break_share"], found["break_median"] = split
        minutes = [m for m, chance in zip(minutes, chances) if chance < 0.5]
        usual = [pair for pair, chance in zip(usual, chances) if chance < 0.5]
    found.update(_spread(minutes))
    after_first = sum(turn == 1 for turn, _ in usual)
    later = sum(turn >= 3 for turn, _ in usual)
    if len(usual) < 30 or after_first < 8 or later < 8:
        return found
    best = None
    for half in _HALF_TURNS:
        xs = [0.5 ** ((turn - 1) / half) for turn, _ in usual]
        mean_x, mean_y = fmean(xs), fmean(y for _, y in usual)
        spread_x = sum((x - mean_x) ** 2 for x in xs)
        if spread_x <= 0:
            continue
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, (_, y) in zip(xs, usual)) / spread_x
        )
        base = mean_y - slope * mean_x
        errors = [y - base - slope * x for x, (_, y) in zip(xs, usual)]
        miss = sum(e * e for e in errors)
        if best is None or miss < best[0]:
            best = (miss, half, base, slope, errors, spread_x)
    if best is None:
        return found
    miss, half, base, slope, errors, spread_x = best
    noise = math.sqrt(miss / max(1, len(usual) - 2) / spread_x)
    # An effect counts when it moves the median by a quarter and stands clear of the noise.
    if abs(slope) >= math.log(1.25) and abs(slope) >= 3 * noise:
        found.update(
            median=math.exp(base),
            p95=math.exp(base + _quantile(errors, 0.95)),
            first_factor=math.exp(slope),
            half_turn=half,
        )
    return found


# -- everything else ---------------------------------------------------------


def history_resent(threads, spec) -> float | None:
    """The share of the conversation so far that a request carried, where usage says.

    Near one for an app that resends its whole history. Lower when the app
    already leaves earlier text out, and then keeping everything is more than
    it pays today.
    """
    shares = []
    for thread in threads:
        held = 0
        for run in thread.runs:
            request = held + spec.part_tokens(run.input.text)
            request += sum(
                spec.part_tokens(p.text) for p in run.retrievals if p.exposed_to_model
            )
            if run.generation.usage and run.run_index > 1 and request:
                sent = dict(run.generation.usage)["input"]
                shares.append(min(1.0, sent / request))
            held = request + spec.part_tokens(run.output.text)
    return _quantile(shares, 0.5) if len(shares) >= 5 else None


def _notice_tokens(spec) -> int:
    """Tokens of the notice the runtime leaves in place of one archived part."""
    return spec.part_tokens(placeholder("thread", Source("part", "retrieval", 1, "")))


def _shared_prompt_share(threads, lifetime_seconds):
    """Of the requests that find their own thread's cache cold, the share another thread warmed.

    A request is cold at a thread's start and after a gap longer than the cache
    lifetime. Another thread has warmed the shared prompt when one of its turns
    began within that lifetime before the request and had finished by then.
    """
    turns = sorted(
        (seconds(run.started_at), seconds(run.ended_at), number)
        for number, thread in enumerate(threads)
        for run in thread.runs
    )
    starts = [turn[0] for turn in turns]
    cold = warmed = 0
    for number, thread in enumerate(threads):
        previous = None
        for run in thread.runs:
            at = seconds(run.started_at)
            if previous is None or at - previous >= lifetime_seconds:
                cold += 1
                k = bisect_left(starts, at) - 1
                while k >= 0 and starts[k] > at - lifetime_seconds:
                    if turns[k][2] != number and turns[k][1] <= at:
                        warmed += 1
                        break
                    k -= 1
            previous = at
    return warmed / cold if cold else None


def measure(threads, spec, labels=()) -> dict:
    """Sizes, lengths, return delays, calls and labelled dependencies of a set of threads."""
    threads = list(threads)
    by_thread = {}
    for row in labels:
        by_thread.setdefault(row.get("thread_id"), []).append(row)
    tokens = {kind: [] for kind in _KINDS}
    lengths, delays, durations, calls, reasoning = [], [], [], [], []
    pieces = []
    retrieving = runs = 0
    lags = {kind: [] for kind in _KINDS}
    labelled = known = needing = 0
    for thread in threads:
        lengths.append(len(thread.runs))
        previous = None
        for run in thread.runs:
            runs += 1
            tokens["input"].append(spec.part_tokens(run.input.text))
            retrieved = [p.text for p in run.retrievals if p.exposed_to_model]
            if retrieved:
                retrieving += 1
                pieces.append(len(retrieved))
                tokens["retrieval"].append(sum(map(spec.part_tokens, retrieved)))
            tokens["output"].append(spec.part_tokens(run.output.text))
            usage = dict(run.generation.usage) if run.generation.usage else None
            if usage:
                reasoning.append(
                    max(0, usage["output"] - spec.text_tokens(run.output.text))
                )
            if run.calls:
                calls.append(run.calls)
            start, end = seconds(run.started_at), seconds(run.ended_at)
            durations.append(end - start)
            if previous is not None:
                delays.append((run.run_index - 1, max(0.0, start - previous) / 60))
            previous = end
        origin = thread.part_origin
        index = {run.run_id: run.run_index for run in thread.runs}
        for row in by_thread.get(thread.thread_id, ()):
            label = normalize_label(row, thread)
            target = index[label["target_run_id"]]
            labelled += 1
            if target == 1:
                # A first turn has nothing earlier to need.
                continue
            requirements = label["task_requirements"]
            if label["status"] != "complete" or any(
                r["historical_context_needed"] == "unknown" or not r["alternatives"]
                for r in requirements
            ):
                continue
            known += 1
            far = {}
            for requirement in requirements:
                if requirement["historical_context_needed"] == "yes":
                    for kind, lag in _nearest(
                        requirement["alternatives"], origin, target
                    ).items():
                        far[kind] = max(far.get(kind, 0), lag)
            needing += bool(far)
            for kind, lag in far.items():
                lags[kind].append(lag)
    resent = history_resent(threads, spec)
    return {
        "threads": len(lengths),
        "turns": runs,
        "tokens_per_turn": {
            "input": _spread(tokens["input"]),
            # Of the turns that retrieve anything.
            "retrieval": _spread(tokens["retrieval"] or [0]),
            "output": _spread(tokens["output"]),
        },
        "turns_that_retrieve": retrieving / runs,
        "reasoning_tokens": _quantile(reasoning, 0.5) if reasoning else None,
        "calls_per_turn": fmean(calls) if calls else None,
        "answer_seconds": _quantile(durations, 0.5),
        "cache_read_when_live": measure_cache_reads(threads, spec)["share"],
        "history_resent": resent,
        # What archiving one turn's retrieved text leaves behind: a notice per part.
        "archive_notice_tokens": round(_notice_tokens(spec) * fmean(pieces))
        if pieces
        else None,
        "turns_per_thread": _fit_length(lengths),
        "idle_minutes": _fit_delay(delays) if delays else None,
        "prompt_warmed_by_others": _shared_prompt_share(threads, spec.cache_ttl_seconds)
        if len(lengths) > 1
        else None,
        "labelled_turns": labelled,
        "assessed_turns": known,
        "turns_needing_history": needing,
        "needs": {
            kind: {
                "share_of_turns": len(values) / known,
                "turns_back": _spread(values) if values else None,
            }
            for kind, values in lags.items()
        }
        if known
        else None,
    }


def workload(measured, spec, model, header_tokens, notes=None) -> tuple[str, dict]:
    """The calculator's settings for a measured workload: a model preset and a scenario."""
    tokens, lengths = measured["tokens_per_turn"], measured["turns_per_thread"]
    header = header_tokens
    notes = [] if notes is None else notes
    scenario = {
        "inputTokens": max(1, tokens["input"]["median"]),
        "outputTokens": tokens["output"]["median"],
        "tokenMedian": tokens["retrieval"]["median"],
        "tokenP95": tokens["retrieval"]["p95"],
        "systemTokens": header,
        "lengthDistribution": lengths["law"],
        "plannedTurns": min(_MAX_TURNS, lengths["median"]),
        "plannedTurnsP95": min(_MAX_TURNS_P95, lengths["p95"]),
        "continuationProbability": 1,
        "retentionRequirement": 0.8,
        "pingOutputTokens": spec.ping_output_tokens,
        "recoveryOutputTokens": spec.expansion_output_tokens,
    }
    if lengths["p95"] > _MAX_TURNS:
        notes.append(
            f"The calculator caps a thread at {_MAX_TURNS} turns. These run longer."
        )
    if lengths["law"] == "continuation":
        another = lengths["another_turn"]
        scenario.update(
            continueFirst=another["after_first"],
            continueLong=another["in_long_threads"],
            continueHalfTurn=another["half_turn"],
        )
    if 0 < measured["turns_that_retrieve"] < 1:
        scenario["retrievalShare"] = round(measured["turns_that_retrieve"], 3)
    if measured["reasoning_tokens"]:
        scenario["reasoningTokens"] = measured["reasoning_tokens"]
    if measured["calls_per_turn"] and measured["calls_per_turn"] > 1:
        scenario["callsPerTurn"] = round(min(100.0, measured["calls_per_turn"]), 2)
    if measured["answer_seconds"] > 0:
        scenario["answerSeconds"] = round(min(86400.0, measured["answer_seconds"]), 1)
    if measured["cache_read_when_live"] is not None:
        scenario["cacheHitShare"] = round(measured["cache_read_when_live"], 3)
    if measured["archive_notice_tokens"]:
        scenario["archivedTokens"] = min(10000, measured["archive_notice_tokens"])
    delay = measured["idle_minutes"]
    if delay:
        scenario["gapMedianMinutes"] = round(max(0.01, delay["median"]), 2)
        scenario["gapP95Minutes"] = round(
            max(scenario["gapMedianMinutes"], delay["p95"]), 2
        )
        if delay["half_turn"]:
            scenario["gapFirstFactor"] = round(
                min(1000.0, max(0.01, delay["first_factor"])), 2
            )
            scenario["gapHalfTurn"] = delay["half_turn"]
        if delay["break_share"]:
            scenario["breakShare"] = round(delay["break_share"], 3)
            scenario["breakMedianMinutes"] = round(
                min(1e6, max(1.0, delay["break_median"])), 1
            )
    else:
        notes.append(
            "No thread has a second turn, so the return delay keeps the calculator's default."
        )
    warmed = measured["prompt_warmed_by_others"]
    if warmed and header >= spec.cache_min_tokens:
        scenario["sharedPrefixWarm"] = round(warmed, 3)
    if measured["needs"]:
        for kind, found in measured["needs"].items():
            scenario[f"{kind}DependencyProbability"] = round(found["share_of_turns"], 3)
            back = found["turns_back"] or {"median": 1, "p95": 1}
            scenario[f"{kind}DependencyTurnsBack"] = min(_MAX_TURNS, back["median"])
            scenario[f"{kind}DependencyTurnsBackP95"] = back["p95"]
    else:
        notes.append(
            "No labels, so what later turns need keeps the calculator's defaults, measured on DocChat."
        )
    priced = pricing(model)
    if priced and priced.get("preset"):
        preset = priced["preset"]
    else:
        # A model without a price sheet: prices relative to one dollar per million input tokens.
        preset = "custom"
        scenario["model"] = {
            "context": spec.context_tokens,
            "maxInput": spec.context_tokens,
            "inputRate": 1.0,
            "cachedRate": spec.read_multiplier,
            "outputRate": spec.output_multiplier,
            "writeMultiplier": spec.write_multiplier,
            "longThreshold": spec.long_input_threshold or 0,
            "longInputMultiplier": spec.long_input_factor,
            "longOutputMultiplier": spec.long_output_factor,
        }
        scenario["ttlMinutes"] = spec.cache_ttl_seconds / 60
        scenario["cacheMinTokens"] = spec.cache_min_tokens
        scenario["cacheBlockTokens"] = spec.cache_block_tokens
        notes.append(
            "Custom model: prices are relative to $1 per million input tokens."
        )
    return preset, scenario


def header_size(spec, system_prompt, tools) -> int:
    """Tokens of what opens every request."""
    return spec.text_tokens(system_prompt) + (
        spec.text_tokens(json.dumps(list(tools), ensure_ascii=False)) if tools else 0
    )


def _decision(policy) -> dict:
    """A fitted policy as the calculator takes it."""
    recipe = policy.recipe
    return {
        "questions": recipe.input_runs if recipe else None,
        "answers": recipe.output_runs if recipe else None,
        "retrieved": recipe.retrieval_runs if recipe else None,
        "keepUnder": recipe.keep_under_tokens if recipe else 0,
        "maxPings": policy.max_pings,
    }


def profile(
    threads, labels=None, *, model=None, system_prompt=None, tools=None, policy=None
) -> dict:
    """A workload profile of `threads`, with a link that opens the calculator filled in.

    Everything the fit reads by turn position has a number here: the chance of
    another turn, the return delay and long breaks, how long answers take and
    how many model calls a turn makes. The dependency numbers come from the
    labels: for each labelled turn the alternative whose evidence lies nearest.
    Without labels the calculator keeps its own defaults for what later turns
    need, and the profile says so.
    """
    if labels is None and is_record(threads):
        labels = load_record(threads)[1]
    rows = _rows(threads)
    found = recorded(rows)
    if model is None:
        model = found["model"]
        if model is None:
            raise ContractError(
                "the threads do not record which model answered: name it with --model or model="
            )
    spec = model_spec(model)
    if system_prompt is None:
        system_prompt = found["system_prompt"] or ""
    if tools is None:
        tools = found["tools"] or ()
    loaded = load_threads(rows, model=spec.model_id)
    measured = measure(loaded, spec, _rows(labels))
    notes = []
    preset, scenario = workload(
        measured, spec, model, header_size(spec, system_prompt, tools), notes
    )
    if policy is not None:
        from .api import Policy

        loaded = policy if isinstance(policy, Policy) else Policy.load(policy)
        if loaded.spec.model_id != spec.model_id:
            raise ContractError(
                f"the policy is for {loaded.spec.model_id}, the threads are priced as {spec.model_id}"
            )
        scenario["policy"] = _decision(loaded)
    source = f"{measured['threads']:,} threads, {measured['turns']:,} turns"
    query = {
        "v": _LINK_VERSION,
        "model": preset,
        "scenario": json.dumps(scenario, separators=(",", ":")),
        "source": source,
    }
    return {
        "schema_version": "compache-profile-v1",
        "source": {
            "threads": measured["threads"],
            "turns": measured["turns"],
            "labelled_turns": measured["labelled_turns"],
            "model": spec.model_id,
        },
        "measured": measured,
        "calculator": {"model": preset, "scenario": scenario, "source": source},
        "link": CALCULATOR_URL + "#" + urlencode(query),
        "notes": notes,
    }
