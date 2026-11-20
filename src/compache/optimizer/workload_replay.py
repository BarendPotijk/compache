"""Chronological held-out accounting for the workload planner and a raw no-ping reference."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from compache.traces import ContractError

from .cache_breaks import CacheBreakChecker, CacheScope
from .contracts import digest, number, seconds
from .distribution import repeat_input_usage, validate_calls_per_run
from compache.dependencies import label_index

from .parts import Request, Source
from .host_ledger import compare_host_ledger, host_contract
from .retention import RetentionRecipe, assess, matched_prefix_tokens, render_recipe
from .shared_prefix import SharedPrefixCatalog
from .workload import WorkloadPolicy, WorkloadSession

__all__ = ["compare_host_ledger", "replay_workload"]


def _block_capacity(tokens, spec):
    rounded = tokens // spec.cache_block_tokens * spec.cache_block_tokens
    return (
        rounded
        if spec.cache_ttl_seconds > 0 and rounded >= spec.cache_min_tokens
        else 0
    )


def _charge(view, prior, at, output_tokens, spec, header, reuse, order):
    """Price one logical provider invocation. Static headers may hit without a thread history."""
    own = 0
    if prior is not None:
        previous, refreshed = prior
        if 0 <= at - refreshed < spec.cache_ttl_seconds:
            # This replay fixes system_prompt and default_tools for the entire
            # workload. Its remaining framing is stable within each thread.
            own = matched_prefix_tokens(
                previous, view, stable_overhead_tokens=view.overhead_tokens
            )
    shared = max(own, reuse.eligible_tokens)
    reads = min(view.tokens, _block_capacity(shared, spec))
    eligible = spec.cache_ttl_seconds > 0 and view.tokens >= spec.cache_min_tokens
    writes = view.tokens - reads if eligible else 0
    fresh = view.tokens - reads - writes
    cold_rate = spec.write_multiplier if eligible else 1.0
    cold_input_cost = view.tokens * cold_rate * spec.input_price_factor(view.tokens)
    cost = spec.token_charge(fresh, writes, reads, output_tokens)
    return {
        "input_tokens": view.tokens,
        "fresh_input_tokens": fresh,
        "cache_write_tokens": writes,
        "cache_read_tokens": reads,
        "output_tokens": output_tokens,
        "cost_input_equivalents": cost,
        "cold_input_cost": cold_input_cost,
        "input_cost": cost - output_tokens * spec.output_rate(view.tokens),
        "catalog_eligible_tokens": reuse.eligible_tokens,
        **_header_usage(reads, header, order, 1),
    }


def _header_usage(reads_per_call, header, order, calls):
    left = min(reads_per_call, header.static_tokens)
    attribution = {}
    for name in order:
        size = (
            header.system_tokens
            if name == "system_prompt"
            else header.default_tool_tokens
        )
        attribution[name] = min(left, size)
        left -= attribution[name]
    left = attribution["default_tools"]
    tool_reads = []
    for size in header.tool_breakdown:
        tool_reads.append(min(left, size))
        left -= tool_reads[-1]
    return {
        "system_read_tokens": calls * attribution["system_prompt"],
        "default_tool_read_tokens": calls * attribution["default_tools"],
        "system_prompt_input_tokens": calls * header.system_tokens,
        "default_tools_input_tokens": calls * header.default_tool_tokens,
        "each_default_tool_input_tokens": tuple(
            calls * n for n in header.tool_breakdown
        ),
        "each_default_tool_read_tokens": [calls * n for n in tool_reads],
    }


@dataclass
class _ThreadReplay:
    scope: CacheScope
    session: WorkloadSession | None
    prior: tuple | None = None
    pending: tuple | None = None
    failed: bool = False


def _run_arm(
    threads,
    labels,
    policy,
    *,
    candidate,
    observation_end,
    system,
    tools,
    tokenize,
    count_tokens,
    component_order,
    share_headers,
    provider,
    cache_key,
    calls_per_run,
    modeled_ping_output_tokens,
):
    spec = policy.spec
    catalog = SharedPrefixCatalog()
    check = CacheBreakChecker()
    contexts = {}
    events = []
    serial = count()
    ledger = []
    failures = []

    # At an exact tie an acknowledged completion updates the cache first, an
    # actual arrival cancels its timer second, and an idle ping is considered last.
    def enqueue(at, priority, tid, data):
        heappush(events, (at, priority, tid, next(serial), data))

    for thread in threads:
        tid = thread.thread_id
        scope = CacheScope(thread.company_id, provider, spec.model_id, tid, cache_key)
        session = (
            WorkloadSession(
                policy,
                scope,
                check,
                catalog=catalog,
                tokenize=tokenize,
                count_tokens=count_tokens,
                component_order=component_order,
                share_headers=share_headers,
            )
            if candidate
            else None
        )
        contexts[tid] = _ThreadReplay(scope, session)
        canonical = Request.from_thread(thread, thread.runs[-1].run_id).parts
        for run in thread.runs:
            start = seconds(run.started_at)
            send = seconds(run.generation.started_at)
            if start < send:
                enqueue(start, 1, tid, ("arrive", run))
            enqueue(send, 1, tid, ("send", run, canonical))

    try:
        while events:
            at, priority, tid, _, data = heappop(events)
            if at > observation_end:
                break
            ctx = contexts[tid]
            if ctx.failed:
                continue
            if priority == 1:
                if data[0] == "arrive":
                    if candidate:
                        check.begin_return(
                            ctx.scope, run_id=data[1].run_id, started_at=at
                        )
                    continue
                _, run, canonical = data
                req = Request(
                    tid,
                    run.run_id,
                    run.run_index,
                    run.generation.model,
                    canonical[: len(run.generation.request_part_ids)],
                )
                if run.output.text is None or any(p.text is None for p in req.parts):
                    failures.append(
                        {
                            "thread_id": tid,
                            "run_id": run.run_id,
                            "reason": "missing_visible_text",
                        }
                    )
                    ctx.failed = True
                    continue
                if candidate:
                    try:
                        plan = ctx.session.prepare(
                            req, started_at=at, system_prompt=system, tools=tools
                        )
                    except ContractError as exc:
                        reason = {
                            "no admissible workload view for this request": "input_not_admissible",
                            "request superseded by newer cache observation": "ambiguous_request_clock",
                        }.get(str(exc))
                        if reason is None:
                            raise
                        failures.append(
                            {
                                "thread_id": tid,
                                "run_id": run.run_id,
                                "reason": reason,
                            }
                        )
                        ctx.failed = True
                        continue
                    view, header, reuse = plan.view, plan.header, plan.reuse
                    action = plan.decision["action"]
                else:
                    # The actual baseline sends the available raw request whenever
                    # admitted. It does not consult a forecast of future feasibility.
                    plan = None
                    header = catalog.measure(
                        ctx.scope,
                        system_prompt=system,
                        default_tools=tools,
                        spec=spec,
                        tokenize=tokenize,
                        count_tokens=count_tokens,
                        component_order=component_order,
                        share=share_headers,
                    )
                    view = render_recipe(
                        req,
                        spec,
                        RetentionRecipe(),
                        cut=False,
                        overhead_tokens=policy.overhead_tokens + header.total_tokens,
                    )
                    reuse = catalog.lookup(header, at=at, request_tokens=view.tokens)
                    action = "keep"
                if (
                    view.tokens > spec.context_tokens - policy.options["reserve_tokens"]
                    or view.wire_bytes + header.wire_bytes > spec.byte_limit
                ):
                    failures.append(
                        {
                            "thread_id": tid,
                            "run_id": run.run_id,
                            "reason": "input_not_admissible",
                        }
                    )
                    ctx.failed = True
                    continue
                # The answering decision is frozen before consuming reference labels or the observed answer.
                usage = (
                    dict(run.generation.usage)
                    if run.generation.usage is not None
                    else None
                )
                out = (
                    usage["output"]
                    if usage is not None
                    else spec.text_tokens(run.output.text)
                )
                charge = _charge(
                    view, ctx.prior, at, out, spec, header, reuse, component_order
                )
                grade = assess(labels.get(run.run_id), view.full_part_ids)
                row = dict(
                    thread_id=tid,
                    run_id=run.run_id,
                    run_index=run.run_index,
                    role="answer",
                    at=at,
                    action=action,
                    historical_required=grade["historical_requirements_assessed"],
                    historical_retained=grade["historical_requirements_retained"],
                    unassessable=not grade["assessable"],
                    changed=bool(view.archived_ids),
                    assessed_miss=bool(grade["missed_requirements"]),
                    **charge,
                )
                ctx.pending = plan, header, view, row, run
                enqueue(seconds(run.ended_at), 0, tid, None)
            elif priority == 0:
                plan, header, view, row, run = ctx.pending
                observed_calls = calls_per_run.get(run.run_id)
                row["foreground_calls"] = observed_calls or 1
                row["call_count_source"] = (
                    "supplied" if observed_calls else "assumed_one"
                )
                ledger.append(row)
                if observed_calls is not None and observed_calls > 1:
                    additional = observed_calls - 1
                    repeat = repeat_input_usage(spec, view.tokens, additional)
                    ledger.append(
                        dict(
                            thread_id=tid,
                            run_id=run.run_id,
                            run_index=run.run_index,
                            role="foreground_repeat",
                            at=at,
                            action="repeat_initial_view",
                            calls=additional,
                            output_tokens=0,
                            **repeat,
                            **_header_usage(
                                repeat["cache_read_tokens"] // additional,
                                header,
                                component_order,
                                additional,
                            ),
                        )
                    )
                ctx.pending = None
                if candidate:
                    answer = Source(
                        run.output.part_id, "output", run.run_index, run.output.text
                    )
                    # The trace associates its input meter with the historical
                    # generation. Its normalized part IDs omit the submitted
                    # system/tools, framing and rewrites. Even keep-all may
                    # send different bytes. Never promote this meter (or our
                    # modeled charge) to an observed input for a replay arm.
                    record = ctx.session.complete(
                        plan,
                        answer,
                        ended_at=at,
                        answer_history_tokens=spec.part_tokens(answer.text),
                        reusable_input_tokens=_block_capacity(view.tokens, spec),
                        cached_input_tokens=None,
                    )
                    if record["due_times"]:
                        enqueue(
                            record["due_times"][0],
                            2,
                            tid,
                            (run.run_id, run.run_index, header),
                        )
                else:
                    catalog.record(
                        header,
                        at=row["at"],
                        request_tokens=view.tokens,
                        cached_input_tokens=None,
                    )
                ctx.prior = view, row["at"]
            else:
                run_id, pos, header = data
                view = ctx.session.ping_view(run_id=run_id, at=at)
                if view is None:
                    continue
                reuse = catalog.lookup(header, at=at, request_tokens=view.tokens)
                charge = _charge(
                    view,
                    ctx.prior,
                    at,
                    modeled_ping_output_tokens,
                    spec,
                    header,
                    reuse,
                    component_order,
                )
                confirmed = ctx.session.acknowledge_ping(
                    run_id=run_id,
                    at=at,
                    modeled_cached_input_tokens=charge["cache_read_tokens"],
                    modeled_cache_write_input_tokens=charge["cache_write_tokens"],
                    modeled_output_tokens=charge["output_tokens"],
                )
                # A dispatched ping is chargeable even when its receipt cancels later timers.
                ledger.append(
                    dict(
                        thread_id=tid,
                        run_id=run_id,
                        run_index=pos,
                        role="ping",
                        at=at,
                        action="ping",
                        receipt=ctx.session.last_ping_receipt,
                        **charge,
                    )
                )
                if confirmed:
                    ctx.prior = view, at
                    if ctx.session.next_ping_at is not None:
                        enqueue(ctx.session.next_ping_at, 2, tid, data)
    finally:
        check.close()
    return _summarize(ledger, failures, threads)


def _summarize(ledger, failures, threads):
    planned = sum(len(t.runs) for t in threads)
    answers = [r for r in ledger if r["role"] == "answer"]
    repeats = [r for r in ledger if r["role"] == "foreground_repeat"]
    pings = [r for r in ledger if r["role"] == "ping"]
    first = [r for r in answers if r["run_index"] == 1]
    graded = sum(r["historical_required"] for r in answers)
    kept = sum(r["historical_retained"] for r in answers)
    meters = (
        "input_tokens",
        "cache_write_tokens",
        "fresh_input_tokens",
        "cache_read_tokens",
        "output_tokens",
        "system_prompt_input_tokens",
        "default_tools_input_tokens",
        "system_read_tokens",
        "default_tool_read_tokens",
        "cost_input_equivalents",
    )
    per_thread = defaultdict(float)
    for row in ledger:
        per_thread[row["thread_id"]] += row["cost_input_equivalents"]
    return {
        "totals": {name: sum(r[name] for r in ledger) for name in meters},
        "planned_answers": planned,
        "answers": len(answers),
        "pings": len(pings),
        "ping_receipt_outcomes": dict(Counter(r["receipt"]["outcome"] for r in pings)),
        "foreground_calls": sum(r["foreground_calls"] for r in answers),
        "runs_with_supplied_call_counts": sum(
            r["call_count_source"] == "supplied" for r in answers
        ),
        "repeat_input_tokens": sum(r["input_tokens"] for r in repeats),
        "repeat_read_tokens": sum(r["cache_read_tokens"] for r in repeats),
        "repeat_input_cost_equivalents": sum(
            r["cost_input_equivalents"] for r in repeats
        ),
        "complete": planned == len(answers),
        "failures": failures,
        "first_answers": len(first),
        "first_answers_with_shared_reads": sum(
            r["cache_read_tokens"] > 0 for r in first
        ),
        "first_input_cost_if_cold": sum(r["cold_input_cost"] for r in first),
        "first_actual_input_cost": sum(r["input_cost"] for r in first),
        "first_each_default_tool_input_tokens": [
            sum(v) for v in zip(*(r["each_default_tool_input_tokens"] for r in first))
        ],
        "first_each_default_tool_read_tokens": [
            sum(v) for v in zip(*(r["each_default_tool_read_tokens"] for r in first))
        ],
        "first_system_read_tokens": sum(r["system_read_tokens"] for r in first),
        "first_default_tool_read_tokens": sum(
            r["default_tool_read_tokens"] for r in first
        ),
        "historical_required": graded,
        "historical_retained": kept,
        "coverage": kept / graded if graded else None,
        "unassessable_runs": sum(r["unassessable"] for r in answers),
        "changed_unassessable_runs": sum(
            r["unassessable"] and r["changed"] for r in answers
        ),
        "runs_with_a_known_miss": sum(r["assessed_miss"] for r in answers),
        "actions": dict(Counter(r["action"] for r in answers)),
        "per_thread_cost": dict(per_thread),
        "ledger": ledger,
    }


def replay_workload(
    threads,
    labels,
    artifact,
    *,
    observation_end,
    system_prompt,
    default_tools,
    tokenize=None,
    count_tokens=None,
    component_order=("system_prompt", "default_tools"),
    share_headers=True,
    provider="declared",
    cache_key="",
    calls_per_run=None,
    modeled_ping_output_tokens=None,
    include_host_contract=False,
    **policy_options,
):
    """Replay a frozen planner and a preserve-all/no-ping control on identical external demand.

    One cache domain per company, provider and model. All clocks use trace times
    and the explicit observation cutoff. Captured text is distinct from priced
    token estimates. Input header parameters apply uniformly to the cohort.
    Reported output meters can price the frozen response. Normalized traces
    lack a matched submitted view for the replay's input; both arms model it.
    """
    threads, labels = list(threads), list(labels)
    if not threads:
        raise ContractError("held-out evaluation requires at least one thread")
    policy = WorkloadPolicy(artifact, **policy_options)
    # This is the synthetic receipt at dispatch. Planning sees only the
    # configured forecast until the ping has actually returned.
    if modeled_ping_output_tokens is None:
        modeled_ping_output_tokens = policy.spec.ping_output_tokens
    number(modeled_ping_output_tokens, "modeled_ping_output_tokens", integer=True)
    if any(digest(t.thread_id) in policy.training_threads for t in threads):
        raise ContractError("held-out evaluation includes a training thread")
    cutoff = (
        seconds(observation_end)
        if isinstance(observation_end, str)
        else observation_end
    )
    number(cutoff, "observation_end", -1e20)
    if any(seconds(t.runs[-1].ended_at) > cutoff for t in threads):
        raise ContractError("observation cutoff precedes a held-out answer")
    counts = validate_calls_per_run(
        calls_per_run, {r.run_id for t in threads for r in t.runs}
    )
    indexed = label_index(threads, labels)
    options = {
        "observation_end": cutoff,
        "system": system_prompt,
        "tools": default_tools,
        "tokenize": tokenize,
        "count_tokens": count_tokens,
        "component_order": component_order,
        "share_headers": share_headers,
        "provider": provider,
        "cache_key": cache_key,
    }
    candidate = _run_arm(
        threads,
        indexed,
        policy,
        candidate=True,
        calls_per_run=counts,
        modeled_ping_output_tokens=modeled_ping_output_tokens,
        **options,
    )
    baseline = _run_arm(
        threads,
        indexed,
        policy,
        candidate=False,
        calls_per_run=counts,
        modeled_ping_output_tokens=modeled_ping_output_tokens,
        **options,
    )
    total = baseline["totals"]["cost_input_equivalents"]
    delta = (
        total - candidate["totals"]["cost_input_equivalents"]
        if candidate["complete"] and baseline["complete"]
        else None
    )
    result = {
        "schema_version": "compache-workload-replay-v1",
        "reference": "preserve-all / no-pings",
        "modeled_cost_unit": "fresh input token equivalents",
        "ping_output_scenario": {
            "initial_forecast_tokens": policy.spec.ping_output_tokens,
            "modeled_receipt_tokens": modeled_ping_output_tokens,
        },
        "heldout_threads": len(threads),
        "threads_by_company": dict(Counter(t.company_id for t in threads)),
        "shared_header_reuse": share_headers,
        "control_scope": "Declared raw-history comparison on a fixed observed workload. Host deployment and invoice savings are unmeasured.",
        "pricing_scope": "Includes every dispatched ping through observation_end. Held-out foreground counts price repeated sends of each arm's initial view within the TTL. Missing counts assume one call. Aggregate foreground output is charged once. Tool-loop suffix growth, intra-run refresh timing and recovery are outside this projection.",
        "timing": "Per-company cache. User arrival cancels timers; generation start sets price and TTL, entries become visible at completed run end. Ties order by completion, arrival, ping, then thread ID.",
        "candidate": candidate,
        "baseline": baseline,
        "savings_input_equivalents": delta,
        "savings_fraction": delta / total if delta is not None and total else None,
    }
    reported = sum(r.generation.usage is not None for t in threads for r in t.runs)
    if reported:
        result["input_accounting"] = {
            "historical_input_counters": reported,
            "provider_anchors_used": 0,
            "reason": "The normalized trace omits the complete submitted envelope for each modeled view.",
        }
    if include_host_contract:
        result["host_contract"] = host_contract(threads, policy, **options)
        if counts:
            result["host_contract"]["foreground_calls_per_run"] = counts
    return result
