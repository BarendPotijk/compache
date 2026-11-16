"""Aggregate observed live usage; never estimate missing counters or cache state."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Any, Mapping, Sequence

PRICING_LABEL = "standard-rate credit equivalent of reported usage, not subscription billing statement"
# Explicit scenario rates per token, not an assertion about subscription billing.
RATES = {
    "gpt-6-luna": {
        "input": Decimal("0.0000025"),
        "cached": Decimal("0.00000025"),
        "output": Decimal("0.0000125"),
    }
}
ROLES = ("answer", "summary", "ping", "recovery")
ARMS = ("opencode", "compache")
FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")


def _usage(row: Mapping[str, Any]) -> dict[str, int] | None:
    u = row.get("usage")
    if row.get("ok") is not True or not isinstance(u, Mapping):
        return None
    if any(type(u.get(k)) is not int or u[k] < 0 for k in FIELDS):
        return None
    if u["cached_input_tokens"] > u["input_tokens"]:
        return None
    return {k: u[k] for k in FIELDS}


def cache_bucket(usage: Mapping[str, int]) -> str:
    if usage["cached_input_tokens"] == 0:
        return "miss"
    if usage["cached_input_tokens"] == usage["input_tokens"]:
        return "hit"
    return "partial"


def _total(rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    valid = [_usage(row) for row in rows]
    complete = all(u is not None for u in valid)
    known = [u for u in valid if u is not None]
    sums = {k: sum(u[k] for u in known) for k in FIELDS}
    rates = RATES[model]
    cost = (
        (Decimal(sums["input_tokens"] - sums["cached_input_tokens"]) * rates["input"])
        + Decimal(sums["cached_input_tokens"]) * rates["cached"]
        + Decimal(sums["output_tokens"]) * rates["output"]
    )
    return {
        "rows": len(rows),
        "valid_usage_rows": len(known),
        "complete": complete,
        "known_usage": sums,
        "usage": sums if complete else None,
        "standard_rate_credit_equivalent": float(cost) if complete else None,
        "cache_buckets": dict(Counter(cache_bucket(u) for u in known)),
        "exact_runtime_overhead_tokens": None,
    }


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    expected_counts: Mapping[tuple[str, int, str], int],
    other_overhead: Any = None,
) -> dict[str, Any]:
    """Summarize 3 paired replicates. expected_counts includes every arm/replicate/role.

    Rows are answer, summary, ping or recovery model calls. Each supplies ok and
    usage; failed/missing records invalidate the relevant total. Counts must be
    supplied externally from the experiment plan, including explicit zeroes.
    """
    if model not in RATES:
        raise ValueError("unknown model; explicit scenario rates required")
    keys = {(a, r, role) for a in ARMS for r in range(1, 4) for role in ROLES}
    if set(expected_counts) != keys or any(
        type(v) is not int or v < 0 for v in expected_counts.values()
    ):
        raise ValueError(
            "expected_counts must specify nonnegative counts for every arm, replicate and role"
        )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {
        key: [] for key in keys
    }
    for row in rows:
        key = (row.get("arm"), row.get("replicate"), row.get("role"))
        if key not in grouped:
            raise ValueError(f"invalid row group: {key}")
        grouped[key].append(row)
    coverage = {key: len(grouped[key]) == expected_counts[key] for key in keys}
    by_rep = {}
    for arm in ARMS:
        by_rep[arm] = {}
        for rep in range(1, 4):
            role_totals = {
                role: _total(grouped[(arm, rep, role)], model) for role in ROLES
            }
            all_rows = [row for role in ROLES for row in grouped[(arm, rep, role)]]
            actual = _total(all_rows, model)
            no_recovery = _total(
                [row for row in all_rows if row["role"] != "recovery"], model
            )
            for total, roles in ((actual, ROLES), (no_recovery, ROLES[:-1])):
                if not all(coverage[(arm, rep, role)] for role in roles):
                    total.update(
                        complete=False, usage=None, standard_rate_credit_equivalent=None
                    )
            by_rep[arm][rep] = {
                "actual": actual,
                "zero_recovery_direct_exclusion": no_recovery,
                "by_role": role_totals,
                "expected_counts_match": {
                    role: coverage[(arm, rep, role)] for role in ROLES
                },
            }
    pairs = []
    for rep in range(1, 4):
        baseline = by_rep["opencode"][rep]["actual"]["standard_rate_credit_equivalent"]
        treatment = by_rep["compache"][rep]["actual"]["standard_rate_credit_equivalent"]
        pairs.append(
            {
                "replicate": rep,
                "savings": baseline - treatment
                if baseline is not None and treatment is not None
                else None,
            }
        )
    totals = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        actual = _total(arm_rows, model)
        no_recovery = _total(
            [row for row in arm_rows if row["role"] != "recovery"], model
        )
        for total, roles in ((actual, ROLES), (no_recovery, ROLES[:-1])):
            if not all(
                coverage[(arm, rep, role)] for rep in range(1, 4) for role in roles
            ):
                total.update(
                    complete=False, usage=None, standard_rate_credit_equivalent=None
                )
        totals[arm] = {
            "actual": actual,
            "zero_recovery_direct_exclusion": no_recovery,
            "by_role": {
                role: _total([row for row in arm_rows if row["role"] == role], model)
                for role in ROLES
            },
        }
    # Suppress role totals too when planned calls are missing or extra.
    for arm in ARMS:
        for rep in range(1, 4):
            for role in ROLES:
                if not coverage[(arm, rep, role)]:
                    by_rep[arm][rep]["by_role"][role].update(
                        complete=False, usage=None, standard_rate_credit_equivalent=None
                    )
        for role in ROLES:
            if not all(coverage[(arm, rep, role)] for rep in range(1, 4)):
                totals[arm]["by_role"][role].update(
                    complete=False, usage=None, standard_rate_credit_equivalent=None
                )
    a, b = (totals[arm]["actual"]["standard_rate_credit_equivalent"] for arm in ARMS)
    return {
        "pricing_label": PRICING_LABEL,
        "model": model,
        "rates_per_token": {k: str(v) for k, v in RATES[model].items()},
        "by_replicate": by_rep,
        "paired_savings": pairs,
        "aggregate": totals,
        "aggregate_savings": a - b if a is not None and b is not None else None,
        "other_overhead_separate": other_overhead,
        "counterfactual_note": "Direct exclusion of recovery calls only; later carryover is not modeled.",
        "cache_note": "Buckets use reported cached-input counters; they do not identify TTL or cache cause.",
    }
