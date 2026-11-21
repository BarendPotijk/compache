"""Match an installed host policy's full model-call export to one frozen replay."""

from collections import Counter

from compache.errors import ContractError

from .cache_breaks import CacheScope
from .contracts import ModelSpec, digest, number, seconds
from .header_traffic import header_traffic_signature
from .shared_prefix import SharedPrefixCatalog


def count_foreground_calls(
    events, *, complete_run_ids, model=None, provider=None, cache_key=None
):
    """Count successful serial calls with a shared initial view in fully exported runs.

    The exporter assigns roles. A foreground_step is a subsequent or intermediate
    model invocation in the same run. A tool execution by itself is not a call.
    A physical call ID may be delivered more than once by a tracing transport.
    """
    if isinstance(complete_run_ids, (str, bytes)) or complete_run_ids is None:
        raise ContractError("complete_run_ids must identify the fully exported runs")
    ids = list(complete_run_ids)
    if any(not isinstance(r, str) or not r.strip() for r in ids) or len(ids) != len(
        set(ids)
    ):
        raise ContractError("complete_run_ids must contain unique nonempty run IDs")
    counts = dict.fromkeys(ids, 0)
    answers = Counter()
    seen = {}
    route = {"model": model, "provider": provider, "cache_key": cache_key}
    declared = {}
    for row in events:
        if not isinstance(row, dict):
            raise ContractError("physical call events must be objects")
        run = row.get("run_id")
        call = row.get("call_id")
        role, status = row.get("role"), row.get("status")
        if any(not isinstance(v, str) or not v.strip() for v in (run, call)):
            raise ContractError("physical calls need nonempty run_id and call_id")
        if role not in (
            "answer",
            "foreground_step",
            "ping",
            "compaction",
            "recovery",
            "support",
        ) or status not in ("succeeded", "failed"):
            raise ContractError(
                "physical calls need an explicit supported role and status"
            )
        foreground = (
            run in counts
            and status == "succeeded"
            and role in ("answer", "foreground_step")
        )
        if foreground:
            for field, expected in route.items():
                if field not in row:
                    continue
                value = row[field]
                if not isinstance(value, str) or (
                    field != "cache_key" and not value.strip()
                ):
                    raise ContractError(f"invalid foreground {field}")
                expected = declared.setdefault(
                    (run, field), value if expected is None else expected
                )
                if value != expected:
                    raise ContractError(
                        f"foreground call uses a different {field} for its run"
                    )
        identity = (run, role, status)
        if call in seen:
            if seen[call] != identity:
                raise ContractError("conflicting records for a physical call ID")
            continue
        seen[call] = identity
        if foreground:
            counts[run] += 1
            answers[run] += role == "answer"
    if any(count == 0 for count in counts.values()):
        raise ContractError("a fully exported run has no successful foreground call")
    if any(answers[run] != 1 for run in counts):
        raise ContractError("each complete run must have one successful final answer")
    return counts


def host_contract(
    threads,
    policy,
    *,
    observation_end,
    system,
    tools,
    tokenize,
    count_tokens,
    component_order,
    share_headers,
    provider,
    cache_key,
):
    spec = policy.spec
    header = SharedPrefixCatalog().measure(
        CacheScope(
            threads[0].company_id,
            provider,
            spec.model_id,
            threads[0].thread_id,
            cache_key,
        ),
        system_prompt=system,
        default_tools=tools,
        spec=spec,
        tokenize=tokenize,
        count_tokens=count_tokens,
        component_order=component_order,
        share=share_headers,
    )
    # Hash each complete source conversation once, including its clock,
    # visible evidence, requested order, captured answers and reported usage.
    ordered = sorted(threads, key=lambda t: t.thread_id)
    answers = []
    for thread in ordered:
        for r in thread.runs:
            out = (
                dict(r.generation.usage)["output"]
                if r.generation.usage is not None
                else spec.text_tokens(r.output.text)
                if r.output.text is not None
                else None
            )
            answers.append(
                [
                    thread.thread_id,
                    r.run_id,
                    seconds(r.started_at),
                    seconds(r.generation.started_at),
                    out,
                ]
            )
    return {
        "schema_version": "compache-host-comparison-v1",
        "workload_digest": digest([digest(t.to_dict()) for t in ordered]),
        "model_spec": spec.to_dict(),
        "provider": provider,
        "cache_key_digest": digest(cache_key),
        "cache_namespace": "per company, provider, model and cache_key",
        "scope_digest": digest([(t.thread_id, t.company_id) for t in ordered]),
        "share_headers": share_headers,
        "component_order": list(component_order),
        "header_digest": header_traffic_signature(system, tools, component_order),
        "header_ruler": header.source,
        "header_tokens": [header.system_tokens, header.default_tool_tokens],
        "default_tool_tokens": list(header.tool_breakdown),
        "stable_overhead_tokens": policy.overhead_tokens,
        "reserve_tokens": policy.options["reserve_tokens"],
        "accounting": "all model calls in window; same frozen questions, retrievals, foreground answers and timestamps",
        "observation_end": observation_end,
        "foreground_end_times": {
            r.run_id: seconds(r.ended_at) for thread in ordered for r in thread.runs
        },
        "answers": answers,
    }


def compare_host_ledger(replay, host):
    """Reprice a complete installed-policy call export on this replay's exact contract.

    The host supplies total exported call count and an explicit completeness
    attestation, covering failed attempts, timers, summaries and recovery. Their
    accuracy requires the host's independent metering export. Unobserved usage
    must never be filled with zero.
    """
    contract = replay.get("host_contract")
    if (
        not isinstance(contract, dict)
        or not isinstance(host, dict)
        or host.get("host_contract") != contract
    ):
        raise ContractError(
            "host ledger and replay must include the same workload, provider, pricing and timing contract"
        )
    name = host.get("policy_name")
    if not isinstance(name, str) or not name.strip():
        raise ContractError("host ledger requires the installed policy_name")
    if (
        not isinstance(host.get("events"), list)
        or type(host.get("includes_all_calls")) is not bool
    ):
        raise ContractError(
            "host ledger requires events and an explicit includes_all_calls attestation"
        )
    expected_calls = host.get("deployment_call_count")
    number(expected_calls, "deployment_call_count", integer=True)
    spec = ModelSpec.from_dict(contract["model_spec"])
    planned = {
        (tid, rid): (start, send, out)
        for tid, rid, start, send, out in contract["answers"]
    }
    seen, answered, roles = set(), set(), Counter()
    foreground_counts, foreground_outputs, first_send = Counter(), Counter(), {}
    last_send, answer_send = {}, {}
    observed_counts = contract.get("foreground_calls_per_run", {})
    cost = 0.0
    failures = 0
    for row in host["events"]:
        if not isinstance(row, dict):
            raise ContractError("host event must be an object")
        call = row.get("call_id")
        if not isinstance(call, str) or not call.strip() or call in seen:
            raise ContractError("host call IDs must be nonempty and unique")
        seen.add(call)
        if (
            row.get("provider", contract["provider"]) != contract["provider"]
            or row.get("model", spec.model_id) != spec.model_id
        ):
            raise ContractError("host event is outside the contracted provider/model")
        pair = (row.get("thread_id"), row.get("run_id"))
        if not all(isinstance(v, str) for v in pair) or pair not in planned:
            raise ContractError("host event refers to a foreign thread or run")
        start, send, out = planned[pair]
        at = row.get("at")
        number(at, "host event time", start, contract["observation_end"])
        role, status = row.get("role"), row.get("status")
        if role not in (
            "answer",
            "foreground_step",
            "ping",
            "compaction",
            "recovery",
            "support",
        ) or status not in ("succeeded", "failed"):
            raise ContractError("host event must declare a supported role and status")
        for key in (
            "input_tokens",
            "fresh_input_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
            "output_tokens",
        ):
            number(row.get(key), key, integer=True)
        fresh, writes, reads, output = (
            row[key]
            for key in (
                "fresh_input_tokens",
                "cache_write_tokens",
                "cache_read_tokens",
                "output_tokens",
            )
        )
        if fresh + writes + reads != row["input_tokens"]:
            raise ContractError(
                "host input token buckets must partition all input tokens"
            )
        if (reads or writes) and (
            spec.cache_ttl_seconds <= 0 or row["input_tokens"] < spec.cache_min_tokens
        ):
            raise ContractError(
                "host event used cache outside the declared eligibility rules"
            )
        if reads and (reads < spec.cache_min_tokens or reads % spec.cache_block_tokens):
            raise ContractError(
                "host cache read violates the declared minimum or block size"
            )
        if role == "answer" and status == "succeeded":
            if pair in answered:
                raise ContractError("duplicate foreground answer")
            if output != out or (pair[1] not in observed_counts and send != at):
                raise ContractError("incompatible foreground clock/output")
            answered.add(pair)
            answer_send[pair] = at
        if role in ("answer", "foreground_step") and status == "succeeded":
            end = contract.get("foreground_end_times", {}).get(
                pair[1], contract["observation_end"]
            )
            if not send <= at <= end:
                raise ContractError(
                    "foreground call outside its run's generation window"
                )
            foreground_counts[pair] += 1
            foreground_outputs[pair] += output
            first_send[pair] = min(first_send.get(pair, at), at)
            last_send[pair] = max(last_send.get(pair, at), at)
        failures += status == "failed"
        roles[role] += 1
        cost += spec.token_charge(fresh, writes, reads, output)
    complete = (
        replay["candidate"]["complete"]
        and len(answered) == len(planned)
        and len(seen) == expected_calls
        and host["includes_all_calls"]
        and all(
            foreground_counts[pair] == observed_counts.get(pair[1], 1)
            and first_send.get(pair) == send
            and answer_send.get(pair) == last_send.get(pair)
            and foreground_outputs[pair] == out
            for pair, (_, send, out) in planned.items()
        )
    )
    candidate_cost = replay["candidate"]["totals"]["cost_input_equivalents"]
    delta = cost - candidate_cost if complete else None
    return {
        "reference": name,
        "complete": complete,
        "expected_answers": len(planned),
        "successful_answers": len(answered),
        "missing_answers": sorted(planned.keys() - answered),
        "deployment_call_count": expected_calls,
        "exported_calls": len(seen),
        "calls_by_role": dict(roles),
        "failed_calls_charged": failures,
        "host_exported_cost_input_equivalents": cost,
        "candidate_cost_input_equivalents": candidate_cost,
        "savings_input_equivalents": delta,
        "savings_fraction": delta / cost if delta is not None and cost > 0 else None,
    }
