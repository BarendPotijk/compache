"""Fixed-trajectory TTL repricing; this is not a provider cache simulator."""

from __future__ import annotations

from collections import defaultdict
from compache.optimizer.contracts import number, seconds
from compache.traces import ContractError

# Standard Codex credit rates per million tokens, as supplied by the caller's
# pricing source: https://learn.chatgpt.com/docs/pricing#token-rates
INPUT_RATE = 2.5
CACHED_INPUT_RATE = 0.25
OUTPUT_RATE = 12.5
ROLES = frozenset(("answer", "summary", "ping", "recovery", "codex_overhead"))


def _price(input_tokens, cached_tokens, output_tokens):
    return (
        (input_tokens - cached_tokens) * INPUT_RATE
        + cached_tokens * CACHED_INPUT_RATE
        + output_tokens * OUTPUT_RATE
    ) / 1_000_000


def _api_price(i, c, w, o):
    factor = 2 if i > 272_000 else 1
    return (
        (i - c - w) * 0.10 * factor
        + c * 0.01 * factor
        + w * 0.125 * factor
        + o * 0.50 * (1.5 if factor == 2 else 1)
    ) / 1_000_000


def reprice(
    rows,
    ttls=(300, 1800),
    shared_cached_floor=0,
    exclude_recovery=False,
    pricing="codex_credits",
):
    """Reprice the *same* observed call trajectory at each TTL.

    Each row requires arm, replicate, sequence, logical_at, role, ok, usage.
    Time is a caller-provided cadence assumption, not observed cache evidence.
    Cache warmth is approximated by the previous successful nonempty input in
    the same arm/replicate, without prefix identity or branch tracking. A warm
    row reuses its own observed cached-input count; a cold row retains at most
    the explicit shared floor, bounded by observed cached input. All calls can refresh the cache.
    Recovery exclusion removes only its direct cost; it does not alter cache
    state, future calls, or their token counts.
    """
    if pricing not in ("codex_credits", "openai_usd"):
        raise ContractError("unknown pricing")
    unit = "USD" if pricing == "openai_usd" else "credits"
    number(shared_cached_floor, "shared_cached_floor", integer=True)
    if type(exclude_recovery) is not bool:
        raise ContractError("exclude_recovery must be boolean")
    ttls = tuple(ttls)
    if not ttls or len(set(ttls)) != len(ttls):
        raise ContractError("TTLs must be nonempty and distinct")
    for ttl in ttls:
        number(ttl, "ttl", 1)
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("call must be an object")
        for key in (
            "arm",
            "replicate",
            "sequence",
            "logical_at",
            "role",
            "ok",
            "usage",
        ):
            if key not in row:
                raise ContractError(f"missing {key}")
        if (
            not isinstance(row["arm"], str)
            or not row["arm"]
            or row["role"] not in ROLES
            or type(row["ok"]) is not bool
        ):
            raise ContractError("invalid call identity, role, or ok")
        if not isinstance(row["replicate"], (str, int)) or isinstance(
            row["replicate"], bool
        ):
            raise ContractError("invalid replicate")
        number(row["sequence"], "sequence", integer=True)
        timestamp = (
            seconds(row["logical_at"])
            if isinstance(row["logical_at"], str)
            else row["logical_at"]
        )
        number(timestamp, "logical_at", -1e20)
        usage = row["usage"]
        if usage is not None:
            if not isinstance(usage, dict):
                raise ContractError("usage must be object or null")
            names = ("input_tokens", "cached_input_tokens", "output_tokens")
            if pricing == "openai_usd":
                names += ("cache_write_input_tokens",)
            if any(usage.get(k) is None for k in names):
                usage = None
            else:
                for k in names:
                    number(usage[k], k, integer=True)
                if (
                    usage["cached_input_tokens"]
                    + (
                        usage["cache_write_input_tokens"]
                        if pricing == "openai_usd"
                        else 0
                    )
                    > usage["input_tokens"]
                ):
                    raise ContractError("cached and written input exceed input")
        groups[(row["arm"], row["replicate"])].append((row, timestamp, usage))

    results = []
    for (arm, replicate), calls in sorted(
        groups.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        calls.sort(key=lambda x: x[0]["sequence"])
        if len({x[0]["sequence"] for x in calls}) != len(calls):
            raise ContractError("duplicate sequence")
        if any(calls[i][1] < calls[i - 1][1] for i in range(1, len(calls))):
            raise ContractError("logical_at must follow sequence")
        for ttl in ttls:
            last = None
            total = 0.0
            actual = 0.0
            complete = True
            answer_warm = answer_cold = 0
            details = []
            role_costs = {
                role: {"scenario_cost": 0.0, "observed_cost": 0.0, "complete": True}
                for role in ROLES
            }
            for row, timestamp, usage in calls:
                if not row["ok"]:
                    complete = False
                warm = last is not None and timestamp - last < ttl
                if row["role"] == "answer":
                    if warm:
                        answer_warm += 1
                    else:
                        answer_cold += 1
                included = not (exclude_recovery and row["role"] == "recovery")
                scenario_cost = observed_cost = None
                cached = None
                if usage is None:
                    if included:
                        complete = False
                        role_costs[row["role"]]["complete"] = False
                else:
                    i, c, o = (
                        usage[k]
                        for k in (
                            "input_tokens",
                            "cached_input_tokens",
                            "output_tokens",
                        )
                    )
                    cached = c if warm else min(i, c, shared_cached_floor)
                    if pricing == "openai_usd":
                        w = usage["cache_write_input_tokens"]
                        modeled_write = w + c - cached
                        scenario_cost = _api_price(i, cached, modeled_write, o)
                        observed_cost = _api_price(i, c, w, o)
                    else:
                        modeled_write = None
                        scenario_cost, observed_cost = (
                            _price(i, cached, o),
                            _price(i, c, o),
                        )
                    if included:
                        total += scenario_cost
                        actual += observed_cost
                        role_costs[row["role"]]["scenario_cost"] += scenario_cost
                        role_costs[row["role"]]["observed_cost"] += observed_cost
                    if row["ok"] and i > 0:
                        last = timestamp
                if not row["ok"] and included:
                    role_costs[row["role"]]["complete"] = False
                detail = {
                    "sequence": row["sequence"],
                    "role": row["role"],
                    "warm": warm,
                    "modeled_cached_input_tokens": cached,
                    "observed_cached_input_tokens": None
                    if usage is None
                    else usage["cached_input_tokens"],
                    "scenario_cost": scenario_cost,
                    "observed_cost": observed_cost,
                    "cost_unit": unit,
                    "modeled_cache_write_input_tokens": modeled_write
                    if usage is not None and pricing == "openai_usd"
                    else None,
                    "included": included,
                }
                if pricing == "codex_credits":
                    detail.update(
                        scenario_credits=scenario_cost, observed_credits=observed_cost
                    )
                details.append(detail)
            count = answer_warm + answer_cold
            for costs in role_costs.values():
                if not costs["complete"]:
                    costs["scenario_cost"] = costs["observed_cost"] = None
            result = {
                "arm": arm,
                "replicate": replicate,
                "ttl_seconds": ttl,
                "complete": complete,
                "scenario_cost": total if complete else None,
                "observed_cost": actual if complete else None,
                "cost_unit": unit,
                "by_role": role_costs,
                "answer_warm_count": answer_warm,
                "answer_cold_count": answer_cold,
                "answer_warm_fraction": answer_warm / count if count else None,
                "calls": details,
            }
            if pricing == "codex_credits":
                result.update(
                    scenario_credits=result["scenario_cost"],
                    observed_credits=result["observed_cost"],
                )
            results.append(result)
    aggregates = []
    for arm in sorted({r["arm"] for r in results}):
        for ttl in ttls:
            subset = [r for r in results if r["arm"] == arm and r["ttl_seconds"] == ttl]
            complete = all(r["complete"] for r in subset)
            warm = sum(r["answer_warm_count"] for r in subset)
            cold = sum(r["answer_cold_count"] for r in subset)
            role_totals = {}
            for role in ROLES:
                ok = all(r["by_role"][role]["complete"] for r in subset)
                role_totals[role] = {
                    "complete": ok,
                    "scenario_cost": sum(
                        r["by_role"][role]["scenario_cost"] for r in subset
                    )
                    if ok
                    else None,
                    "observed_cost": sum(
                        r["by_role"][role]["observed_cost"] for r in subset
                    )
                    if ok
                    else None,
                }
            aggregate = {
                "arm": arm,
                "ttl_seconds": ttl,
                "replicates": len(subset),
                "complete": complete,
                "scenario_cost": sum(r["scenario_cost"] for r in subset)
                if complete
                else None,
                "observed_cost": sum(r["observed_cost"] for r in subset)
                if complete
                else None,
                "cost_unit": unit,
                "by_role": role_totals,
                "answer_warm_count": warm,
                "answer_cold_count": cold,
                "answer_warm_fraction": warm / (warm + cold) if warm + cold else None,
            }
            if pricing == "codex_credits":
                aggregate.update(
                    scenario_credits=aggregate["scenario_cost"],
                    observed_credits=aggregate["observed_cost"],
                )
            aggregates.append(aggregate)
    return {
        "replicates": results,
        "aggregates": aggregates,
        "assumptions": {
            "cadence": "logical_at is an assumed return-time trajectory, not proof of observed cache state",
            "cache": "single last-refresh approximation per arm/replicate; prefix branches are not modeled",
            "capacity": "warm reuse equals observed cached input; cold reuse is capped by observed cached input and the caller-supplied shared floor",
            "trajectory": "same calls and token counts for each TTL; policy is not reoptimized",
            "recovery": "direct-cost exclusion only; no downstream counterfactual"
            if exclude_recovery
            else "included",
            "pricing": (
                "credits per million: input 2.5, cached input 0.25, output 12.5; no cache-write charge"
                if pricing == "codex_credits"
                else "USD per million: input 0.10, cache read 0.01, cache write 0.125, output 0.50; input over 272000 doubles input rates and multiplies output by 1.5; 300s TTL is a hypothetical sensitivity, not a supported live setting"
            ),
        },
    }


def reprice_prefix(
    rows,
    ttls=(300, 1800),
    guard_tokens=128,
    block_tokens=128,
    cache_min_tokens=1024,
    exclude_recovery=False,
):
    """Model prefix reuse on a fixed trajectory, independent of physical cache reads."""
    for name, value, minimum in (
        ("guard_tokens", guard_tokens, 0),
        ("block_tokens", block_tokens, 1),
        ("cache_min_tokens", cache_min_tokens, 0),
    ):
        number(value, name, minimum, integer=True)
    if type(exclude_recovery) is not bool:
        raise ContractError("exclude_recovery must be boolean")
    ttls = tuple(ttls)
    if not ttls or len(set(ttls)) != len(ttls):
        raise ContractError("TTLs must be nonempty and distinct")
    for ttl in ttls:
        number(ttl, "ttl", 1)
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("call must be object")
        for key in (
            "arm",
            "replicate",
            "sequence",
            "logical_at",
            "role",
            "ok",
            "usage",
            "cache_blocks",
        ):
            if key not in row:
                raise ContractError(f"missing {key}")
        if (
            not isinstance(row["arm"], str)
            or not row["arm"]
            or row["role"] not in ROLES
            or type(row["ok"]) is not bool
        ):
            raise ContractError("invalid call identity, role, or ok")
        if not isinstance(row["replicate"], (str, int)) or isinstance(
            row["replicate"], bool
        ):
            raise ContractError("invalid replicate")
        number(row["sequence"], "sequence", integer=True)
        timestamp = (
            seconds(row["logical_at"])
            if isinstance(row["logical_at"], str)
            else row["logical_at"]
        )
        number(timestamp, "logical_at", -1e20)
        blocks = row["cache_blocks"]
        if (
            not isinstance(blocks, (list, tuple))
            or not blocks
            or any(
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
                for x in blocks
            )
        ):
            raise ContractError(
                "cache_blocks must be nonempty ordered lowercase SHA256 hashes"
            )
        usage = row["usage"]
        if usage is not None:
            if not isinstance(usage, dict):
                raise ContractError("usage must be object or null")
            fields = ("input_tokens", "cached_input_tokens", "output_tokens")
            if any(usage.get(k) is None for k in fields):
                usage = None
            else:
                for k in fields:
                    number(usage[k], k, integer=True)
                if usage["cached_input_tokens"] > usage["input_tokens"]:
                    raise ContractError("cached input exceeds input")
        groups[(row["arm"], row["replicate"])].append(
            (row, timestamp, tuple(blocks), usage)
        )
    results = []
    for (arm, replicate), calls in sorted(
        groups.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        calls.sort(key=lambda item: item[0]["sequence"])
        if len({x[0]["sequence"] for x in calls}) != len(calls):
            raise ContractError("duplicate sequence")
        if any(calls[i][1] < calls[i - 1][1] for i in range(1, len(calls))):
            raise ContractError("logical_at must follow sequence")
        for ttl in ttls:
            entries = []
            complete = True
            total = physical = 0.0
            counts = {"warm": 0, "first": 0, "expiry": 0, "rewrite": 0}
            details = []
            role_costs = {
                role: {"scenario_cost": 0.0, "observed_cost": 0.0, "complete": True}
                for role in ROLES
            }
            for row, timestamp, blocks, usage in calls:
                included = not (exclude_recovery and row["role"] == "recovery")
                matching = [
                    e
                    for e in entries
                    if len(e[0]) <= len(blocks) and blocks[: len(e[0])] == e[0]
                ]
                live = [e for e in matching if timestamp - e[2] < ttl]
                capacity = max((e[1] for e in live), default=0)
                cached = None if usage is None else min(usage["input_tokens"], capacity)
                reason = (
                    "warm"
                    if cached
                    else "first"
                    if not entries
                    else "expiry"
                    if matching and not live
                    else "rewrite"
                )
                if row["role"] == "answer":
                    counts[reason] += 1
                modeled_cost = physical_cost = None
                if not row["ok"] or (usage is None and included):
                    complete = False
                if included and (not row["ok"] or usage is None):
                    role_costs[row["role"]]["complete"] = False
                if usage is not None:
                    i, c, o = (
                        usage[k]
                        for k in (
                            "input_tokens",
                            "cached_input_tokens",
                            "output_tokens",
                        )
                    )
                    modeled_cost, physical_cost = _price(i, cached, o), _price(i, c, o)
                    if included:
                        total += modeled_cost
                        physical += physical_cost
                        role_costs[row["role"]]["scenario_cost"] += modeled_cost
                        role_costs[row["role"]]["observed_cost"] += physical_cost
                    if row["ok"] and i > 0:
                        # Matching live prefixes are refreshed; unrelated branches remain.
                        entries = [
                            (
                                b,
                                cap,
                                timestamp
                                if (
                                    len(b) <= len(blocks)
                                    and blocks[: len(b)] == b
                                    and timestamp - at < ttl
                                )
                                else at,
                            )
                            for b, cap, at in entries
                        ]
                        new_capacity = max(
                            0, (i - guard_tokens) // block_tokens * block_tokens
                        )
                        if new_capacity >= cache_min_tokens:
                            entries = [e for e in entries if e[0] != blocks]
                            entries.append((blocks, new_capacity, timestamp))
                        entries = entries[-80:]
                details.append(
                    {
                        "sequence": row["sequence"],
                        "role": row["role"],
                        "cache_status": reason,
                        "modeled_cached_input_tokens": cached,
                        "observed_cached_input_tokens": None
                        if usage is None
                        else usage["cached_input_tokens"],
                        "scenario_cost": modeled_cost,
                        "observed_cost": physical_cost,
                        "cost_unit": "credits",
                        "included": included,
                    }
                )
            for costs in role_costs.values():
                if not costs["complete"]:
                    costs["scenario_cost"] = costs["observed_cost"] = None
            count = sum(counts.values())
            results.append(
                {
                    "arm": arm,
                    "replicate": replicate,
                    "ttl_seconds": ttl,
                    "complete": complete,
                    "scenario_cost": total if complete else None,
                    "observed_cost": physical if complete else None,
                    "cost_unit": "credits",
                    "answer_counts": counts,
                    "answer_warm_fraction": counts["warm"] / count if count else None,
                    "by_role": role_costs,
                    "calls": details,
                }
            )
    aggregates = []
    for arm in sorted({r["arm"] for r in results}):
        for ttl in ttls:
            subset = [r for r in results if r["arm"] == arm and r["ttl_seconds"] == ttl]
            complete = all(r["complete"] for r in subset)
            counts = {
                key: sum(r["answer_counts"][key] for r in subset)
                for key in ("warm", "first", "expiry", "rewrite")
            }
            n = sum(counts.values())
            aggregates.append(
                {
                    "arm": arm,
                    "ttl_seconds": ttl,
                    "replicates": len(subset),
                    "complete": complete,
                    "scenario_cost": sum(r["scenario_cost"] for r in subset)
                    if complete
                    else None,
                    "observed_cost": sum(r["observed_cost"] for r in subset)
                    if complete
                    else None,
                    "cost_unit": "credits",
                    "answer_counts": counts,
                    "answer_warm_fraction": counts["warm"] / n if n else None,
                }
            )
    return {
        "replicates": results,
        "aggregates": aggregates,
        "assumptions": {
            "model": "intended ordered message-boundary prefix projection; not native snapshot cache or exact billing",
            "capacity": f"whole prior observed input minus {guard_tokens} guard tokens, rounded down to {block_tokens}; guard may not cover harness differences",
            "cadence": "logical_at is assumed return timing, not proof of physical cache state",
            "trajectory": "identical observed calls and token counts for all TTLs; no policy reoptimization or quality claim",
            "recovery": "direct cost exclusion only; downstream trajectory unchanged"
            if exclude_recovery
            else "included",
            "pricing": "Codex credits per million: input 2.5, cached input 0.25, output 12.5; no cache-write charge",
        },
    }
