"""Expected static-header reuse from the calendar of completed training requests."""

from collections import defaultdict
from hashlib import sha256
from heapq import heappop, heappush
from itertools import groupby

from compache.errors import ContractError

from .cache_breaks import CacheScope
from .contracts import digest, seconds
from .shared_prefix import COMPONENTS, SharedPrefixCatalog, _tool_parts, _wire


def header_traffic_signature(
    system_prompt, default_tools, component_order=COMPONENTS, *, tool_fragments=None
):
    """Identify ordered, exactly framed shared components without persisting text."""
    order = tuple(component_order)
    if order not in (COMPONENTS, COMPONENTS[::-1]):
        raise ContractError(
            "expected one system_prompt and one default_tools component"
        )
    tools, _ = _tool_parts(default_tools, tool_fragments)
    fragments = {
        "system_prompt": _wire(system_prompt),
        "default_tools": b"".join(tools),
    }
    return digest([(name, sha256(fragments[name]).hexdigest()) for name in order])


def _bucket():
    return dict(
        runs=0,
        eligible_runs=0,
        full_header_hits=0,
        partial_header_hits=0,
        eligible_header_tokens=0,
        matched_header_tokens=0,
    )


def _report(counts):
    result = {}
    for name, bucket in counts.items():
        capacity, runs = bucket["eligible_header_tokens"], bucket["eligible_runs"]
        hits = bucket["full_header_hits"] + bucket["partial_header_hits"]
        result[name] = {
            **bucket,
            "probability": bucket["matched_header_tokens"] / capacity
            if capacity
            else 0.0,
            "hit_rate": hits / runs if runs else 0.0,
            "support": (
                "no_runs"
                if not bucket["runs"]
                else "no_eligible_runs"
                if not runs
                else "no_cross_thread_matches"
                if not hits
                else "calendar_estimate"
            ),
        }
        if name == "warm_returns":
            result[name].update(
                probability=None, hit_rate=None, support="own_thread_warm_excluded"
            )
    result["ambient_hit_probability"] = result["cold_returns"]["probability"]
    result["first_run_hit_probability"] = result["first_runs"]["probability"]
    return result


def _scope_counts(calendar, header, spec):
    counts = {k: _bucket() for k in ("first_runs", "cold_returns", "warm_returns")}
    # A completion is visible only after its request began and finished. Own
    # timestamps track eligible full requests, even with a sub-minimum header.
    completions, offers, previous = [], [], {}
    block = spec.cache_block_tokens
    minimum = max(1, spec.cache_min_tokens)
    ttl = spec.cache_ttl_seconds
    for start, batch in groupby(sorted(calendar), key=lambda entry: entry[0]):
        while completions and completions[0][0] <= start:
            _, opened, thread, request_cacheable, prefix_capacity = heappop(completions)
            if request_cacheable and opened + ttl > start:
                previous[thread] = max(opened, previous.get(thread, opened))
                if prefix_capacity:
                    heappush(offers, (-prefix_capacity, -opened, thread))
        # Only the highest-capacity active offer is needed for a cold arrival.
        while offers and -offers[0][1] + ttl <= start:
            heappop(offers)
        for _, end, thread, index, total in batch:
            can_cache = total // block * block >= minimum and ttl > 0
            capacity = min(total, header.static_tokens) // block * block
            capacity = capacity if can_cache and capacity >= minimum else 0
            own_warm = thread in previous and previous[thread] + ttl > start
            category = (
                "first_runs"
                if index == 1
                else "warm_returns"
                if own_warm
                else "cold_returns"
            )
            bucket = counts[category]
            bucket["runs"] += 1
            if capacity:
                bucket["eligible_runs"] += 1
                bucket["eligible_header_tokens"] += capacity
                if category != "warm_returns" and offers:
                    # By construction this thread has no eligible active entry.
                    matched = min(capacity, -offers[0][0])
                    bucket["matched_header_tokens"] += matched
                    bucket[
                        "full_header_hits"
                        if matched == capacity
                        else "partial_header_hits"
                    ] += 1
            heappush(completions, (end, start, thread, can_cache, capacity))
    return counts


def fit_shared_header_traffic(
    threads,
    spec,
    *,
    system_prompt,
    default_tools,
    tokenize=None,
    count_tokens=None,
    component_order=COMPONENTS,
    provider="declared",
    cache_key="",
    tool_fragments=None,
):
    """Fit a calendar-based opportunity rate for a declared fixed common header.

    One fit describes one provider, model, routing key and fixed header; company
    scopes stay independent. Canonical traces have no provider or routing key
    field, so mixed routes must be fitted in separate batches. Supply completed
    successful requests on their original, globally comparable wall clocks.

    Read counts in these traces do not identify the system/tool contribution.
    A successful neighbor offers its eligible static blocks only once its run
    has completed, and strictly before expiry measured from its request start.
    Simultaneous starts cannot warm one another. The forecast for cold returns
    is matched static tokens divided by eligible static tokens; first requests
    have a separate denominator, and warm returns never train either rate.
    """
    if not isinstance(provider, str) or not provider or not isinstance(cache_key, str):
        raise ContractError("provider must be nonempty and cache_key must be a string")
    catalog = SharedPrefixCatalog()
    header = catalog.measure(
        CacheScope("calendar", provider, spec.model_id, "calendar", cache_key),
        system_prompt=system_prompt,
        default_tools=default_tools,
        spec=spec,
        tokenize=tokenize,
        count_tokens=count_tokens,
        component_order=component_order,
        tool_fragments=tool_fragments,
    )
    signature = header_traffic_signature(
        system_prompt, default_tools, component_order, tool_fragments=tool_fragments
    )
    calendars = defaultdict(list)
    seen = set()
    reported = estimated = 0
    for thread in threads:
        scope = digest([thread.company_id, provider, spec.model_id, cache_key])
        identity = (scope, thread.thread_id)
        if identity in seen:
            raise ContractError(
                "duplicate training thread in shared-header traffic fit"
            )
        seen.add(identity)
        history = header.total_tokens
        for run in thread.runs:
            if run.generation.model != spec.model_id:
                raise ContractError(
                    "shared-header traffic fit requires the declared model"
                )
            start, end = seconds(run.generation.started_at), seconds(run.ended_at)
            # Full-canonical traces append only visible parts and the last output.
            # Missing text contributes zero body tokens, a conservative lower bound.
            added = [run.input, *(p for p in run.retrievals if p.exposed_to_model)]
            history += sum(spec.part_tokens(part.text) for part in added)
            if run.generation.usage is not None:
                total = dict(run.generation.usage)["input"]
                reported += 1
            else:
                total = history
                estimated += 1
            calendars[scope].append(
                (start, end, thread.thread_id, run.run_index, total)
            )
            history += spec.part_tokens(run.output.text)
    pooled = {k: _bucket() for k in ("first_runs", "cold_returns", "warm_returns")}
    by_scope = {}
    for scope, calendar in sorted(calendars.items()):
        counts = _scope_counts(calendar, header, spec)
        by_scope[scope] = _report(counts)
        for kind, bucket in counts.items():
            for key, value in bucket.items():
                pooled[kind][key] += value
    return {
        "schema_version": "compache-shared-header-traffic-v1",
        "source": "modeled_training_calendar",
        "calendar_start_assumption": "empty_until_first_observed_completion",
        "model_id": spec.model_id,
        "provider": provider,
        "cache_key_digest": digest(cache_key),
        "cache_ttl_seconds": spec.cache_ttl_seconds,
        "cache_min_tokens": spec.cache_min_tokens,
        "cache_block_tokens": spec.cache_block_tokens,
        "header_signature": signature,
        "component_order": list(header.component_order),
        "system_prompt_tokens": header.system_tokens,
        "default_tool_tokens": header.default_tool_tokens,
        "static_tokens": header.static_tokens,
        "count_source": header.source,
        "reported_input_runs": reported,
        "estimated_input_runs": estimated,
        "training_threads": len(seen),
        "by_scope": by_scope,
        **_report(pooled),
    }
