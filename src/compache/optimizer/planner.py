"""Pure expected-value keep-warm decisions; no scheduling or provider calls."""

from __future__ import annotations

from dataclasses import dataclass
from compache.traces import ContractError
from .contracts import Policy, digest, number
from .fit import cdf, validate_profile


@dataclass(frozen=True)
class RetentionDecision:
    action: str
    pings: int
    due_times: tuple[float, ...]
    expected_saving: float
    values: tuple[float, ...]
    reason: str


def end_run(
    policy: Policy,
    profile: dict | None,
    *,
    held_input: int,
    answer_tokens: int,
    cold_view: int,
    run_index: int,
    request_started_at: float,
    ended_at: float,
    shared_tokens=0,
    later_calls=None,
    reduces=False,
) -> RetentionDecision:
    for key, value in [
        ("held_input", held_input),
        ("answer_tokens", answer_tokens),
        ("cold_view", cold_view),
        ("shared_tokens", shared_tokens),
    ]:
        number(value, key, integer=True)
    for key, value in [
        ("request_started_at", request_started_at),
        ("ended_at", ended_at),
    ]:
        number(value, key, -1e20)
    number(run_index, "run_index", 1, integer=True)
    if ended_at < request_started_at:
        raise ContractError("run ends before request starts")
    if shared_tokens > min(held_input, cold_view):
        raise ContractError("shared prefix exceeds held or cold view")
    if profile is not None:
        validate_profile(profile)
        if policy.profile_hash != digest(profile):
            raise ContractError("policy/profile hash mismatch")
    spec = policy.spec
    interval = spec.cache_ttl_seconds - policy.ping_lead_seconds

    def lapse(reason):
        return RetentionDecision("lapse", 0, (), 0, (), reason)

    if profile is None or not policy.max_pings:
        return lapse("no fitted curve or pings disabled")
    if interval <= 0:
        return lapse("TTL does not exceed lead")
    if ended_at >= request_started_at + interval:
        return lapse("first ping deadline already passed")
    k = held_input + answer_tokens
    if k > (policy.warm_trigger - policy.reserve) * spec.context_tokens:
        return lapse("prefix above keep-warm ceiling")
    if held_input < spec.cache_min_tokens:
        return lapse("prefix was not cacheable")
    row = profile["cadence"]["by_run_index"].get(str(run_index))
    curve = row["curve"] if row else profile["cadence"]["pooled"]
    if not curve["observations"] or not curve["returns"]:
        return lapse("no observed return support")
    calls = (
        (row["later_calls"] if row else profile["cadence"]["later_calls"])
        if later_calls is None
        else later_calls
    )
    number(calls, "later_calls", integer=True)
    # F is measured from this request start, while the fitted curve uses idle time.
    duration = ended_at - request_started_at

    def f(t):
        return cdf(curve, t - duration)

    saving = (
        spec.write_multiplier * max(0, cold_view - shared_tokens)
        - spec.read_multiplier * max(0, k - shared_tokens)
        - spec.read_multiplier * max(0, k - cold_view) * calls
    )
    values = []
    ping_cost = 0.0
    for n in range(1, policy.max_pings + 1):
        sent = 1 - f(n * interval)
        first = (
            spec.read_multiplier * held_input + spec.write_multiplier * answer_tokens
        )
        recurring = spec.read_multiplier * k
        ping_cost += sent * (
            (first if n == 1 else recurring)
            + spec.ping_output_tokens * spec.output_multiplier
        )
        bridged = max(
            0, f(n * interval + spec.cache_ttl_seconds) - f(spec.cache_ttl_seconds)
        )
        values.append(bridged * saving - ping_cost)
    best = max(range(len(values)), key=lambda i: (values[i], -i))
    if values[best] > 0:
        n = best + 1
        return RetentionDecision(
            "keep_warm",
            n,
            tuple(request_started_at + i * interval for i in range(1, n + 1)),
            values[best],
            tuple(values),
            "positive expected value under supplied curve and economics",
        )
    # Explicit compact scheduling is not enabled: the gate will plan on a cold start.
    return RetentionDecision(
        "lapse", 0, (), 0, tuple(values), "no positive-value ping plan"
    )
