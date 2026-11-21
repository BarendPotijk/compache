"""Measure a workload from recorded threads, in the terms of the cost calculator.

The calculator prices a workload described by a handful of numbers: tokens per
turn, thread length, how long users stay away, and how often and how far back a
turn needs something earlier. `profile` reads those numbers off recorded
threads and labels and returns a link that opens the calculator with them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode

from .labels import normalize_label
from .models import model_spec, pricing
from .optimizer.contracts import seconds
from .threads import load_threads, read_jsonl, system_prompt_of

__all__ = ["CALCULATOR_URL", "profile"]

CALCULATOR_URL = "https://barendpotijk.github.io/compache/"
_KINDS = ("input", "retrieval", "output")
_LINK_VERSION = "6"


def _quantile(values, q):
    """Nearest-rank quantile of a nonempty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


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


def measure(threads, spec, labels=()) -> dict:
    """Sizes, lengths, idle gaps and labelled dependencies of a set of threads."""
    by_thread = {}
    for row in labels:
        by_thread.setdefault(row.get("thread_id"), []).append(row)
    tokens = {kind: [] for kind in _KINDS}
    lengths, gaps = [], []
    lags = {kind: [] for kind in _KINDS}
    labelled = known = needing = 0
    for thread in threads:
        lengths.append(len(thread.runs))
        previous = None
        for run in thread.runs:
            tokens["input"].append(spec.part_tokens(run.input.text))
            tokens["retrieval"].append(
                sum(
                    spec.part_tokens(p.text)
                    for p in run.retrievals
                    if p.exposed_to_model
                )
            )
            usage = dict(run.generation.usage) if run.generation.usage else None
            tokens["output"].append(
                usage["output"] if usage else spec.part_tokens(run.output.text)
            )
            if previous is not None:
                gaps.append(max(0.0, seconds(run.started_at) - previous) / 60)
            previous = seconds(run.ended_at)
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

    def spread(values):
        return {"median": _quantile(values, 0.5), "p95": _quantile(values, 0.95)}

    return {
        "threads": len(lengths),
        "turns": sum(lengths),
        "tokens_per_turn": {kind: spread(values) for kind, values in tokens.items()},
        "turns_per_thread": spread(lengths),
        "idle_minutes": spread(gaps) if gaps else None,
        "labelled_turns": labelled,
        "assessed_turns": known,
        "turns_needing_history": needing,
        "needs": {
            kind: {
                "share_of_turns": len(values) / known,
                "turns_back": spread(values) if values else None,
            }
            for kind, values in lags.items()
        }
        if known
        else None,
    }


def profile(threads, labels=None, *, model, system_prompt=None, tools=()) -> dict:
    """A workload profile of `threads`, with a link that opens the calculator filled in.

    The dependency numbers come from the labels: for each labelled turn the
    alternative whose evidence lies nearest. Without labels the calculator keeps
    its own defaults for what later turns need, and the profile says so.
    """
    spec = model_spec(model)
    rows = _rows(threads)
    prompts = [p for p in map(system_prompt_of, rows) if p]
    if system_prompt is None:
        system_prompt = max(set(prompts), key=prompts.count) if prompts else ""
    loaded = load_threads(rows, model=spec.model_id)
    measured = measure(loaded, spec, _rows(labels))
    tokens, lengths = measured["tokens_per_turn"], measured["turns_per_thread"]
    header = spec.text_tokens(system_prompt) + (
        spec.text_tokens(json.dumps(list(tools), ensure_ascii=False)) if tools else 0
    )
    scenario = {
        "inputTokens": max(1, tokens["input"]["median"]),
        "outputTokens": tokens["output"]["median"],
        "tokenMedian": tokens["retrieval"]["median"],
        "tokenP95": tokens["retrieval"]["p95"],
        "systemTokens": header,
        "lengthDistribution": "lognormal",
        "plannedTurns": lengths["median"],
        "plannedTurnsP95": lengths["p95"],
        "continuationProbability": 1,
        "retentionRequirement": 0.8,
        "pingOutputTokens": spec.ping_output_tokens,
        "recoveryOutputTokens": spec.expansion_output_tokens,
    }
    notes = []
    if measured["idle_minutes"]:
        scenario["gapMedianMinutes"] = round(
            max(0.01, measured["idle_minutes"]["median"]), 2
        )
        scenario["gapP95Minutes"] = round(
            max(scenario["gapMedianMinutes"], measured["idle_minutes"]["p95"]), 2
        )
    else:
        notes.append(
            "No thread has a second turn, so the return delay keeps the calculator's default."
        )
    if measured["needs"]:
        for kind, found in measured["needs"].items():
            scenario[f"{kind}DependencyProbability"] = round(found["share_of_turns"], 3)
            back = found["turns_back"] or {"median": 1, "p95": 1}
            scenario[f"{kind}DependencyTurnsBack"] = back["median"]
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
