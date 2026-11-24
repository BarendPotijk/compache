"""What compache saved, read off the record of the calls it carried.

Every call is on record with what the provider billed and how much text
compache left out of it. That gives its price, and what it would have cost
with everything in view: the same request, longer by what was left out, read
from a cache that held all of the one before it wherever that was still alive
under the provider's default lifetime, and whatever other conversations had
left there. So the comparison is with caching at its best, not with none.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from . import record
from .errors import ContractError
from .models import cache_lifetimes, model_spec, pricing
from .optimizer.contracts import ModelSpec
from .optimizer.distribution import _charge

__all__ = ["lines", "savings"]

_COUNTED = ("answer", "reopen", "retry", "ping", "note")
# The lifetime of the entries that a provider counts apart.
_HOUR = 3600
# What an unknown model is taken to charge for reading its cache, in input tokens.
_UNKNOWN = ModelSpec("unknown", read_multiplier=0.1, write_multiplier=1.0)


@dataclass(frozen=True)
class _Held:
    """How a conversation would stand with everything in view, as of its latest call."""

    # Tokens of its request, and when that went out.
    whole: int
    at: float
    # Tokens of its first request.
    first: int
    # What the latest would have read from the cache, its output, and its prices.
    shared: int
    output: int
    plain: ModelSpec


def _sheets(*names) -> tuple[ModelSpec, dict, dict | None]:
    """A model's prices under its default lifetime and under each other one, and what a token costs."""
    for name in filter(None, names):
        try:
            spec = model_spec(name)
        except ContractError:
            continue
        others = {s.cache_ttl_seconds: s for s in cache_lifetimes(name)}
        return spec, others, pricing(name)
    return _UNKNOWN, {}, None


def _new(requested) -> dict:
    return {
        "requested": requested,
        "calls": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "output_tokens": 0,
        "left_out_tokens": 0,
        "reopened": 0,
        "pings": 0,
        "retries": 0,
        "spent": 0.0,
        "without": 0.0,
    }


def savings(source) -> dict:
    """What the calls of a record cost, and what they would have with everything in view.

    `source` is a record folder or the rows of one's `calls.jsonl`. Amounts are
    in dollars where the model has a price sheet. Calls of other models are
    counted under `unpriced`, in input tokens.
    """
    rows = (
        record.load_calls(source) if isinstance(source, (str, Path)) else list(source)
    )
    rows = [r for r in rows if r.get("kind") in _COUNTED and r.get("usage")]
    models: dict[str, dict] = {}
    priced: set[str] = set()
    sheets: dict = {}
    held: dict = {}
    alone: set = set()
    threads: set = set()
    # When a model was last called, and until when its cache lives only for what compache did.
    called: dict = {}
    kept: dict = {}
    for row in sorted(rows, key=lambda r: r["at"]):
        model = row.get("model") or row.get("requested") or "unknown"
        requested = row.get("requested") or model
        if model not in sheets:
            sheets[model] = _sheets(row.get("model"), row.get("requested"))
        spec, others, price = sheets[model]
        dollars = price["input_price_per_million"] / 1e6 if price else 1.0
        if price:
            priced.add(model)
        entry, usage = models.setdefault(model, _new(requested)), row["usage"]
        size, output = usage["input"], usage["output"]
        read, written = usage["cache_read"], usage["cache_write"]
        paid = others.get(row.get("ttl"), spec)
        spent = paid.token_charge(max(0, size - read - written), written, read, output)
        long, plain = None, spec
        if row.get("long_write") and paid is spec:
            # What opens a request is written at the price of its own lifetime.
            long = others.get(row.get("header_ttl") or _HOUR)
            more = (long.write_multiplier if long else 0) - spec.write_multiplier
            spent += (
                min(row["long_write"], written)
                * more
                * bool(long)
                * (spec.input_price_factor(size))
            )
            if not row.get("header_ttl"):
                # The application asks for that itself, so it would have paid it anyway.
                plain = long or spec
        if row.get("cost") is not None and price:
            # What the provider says it billed, where it says so.
            spent = row["cost"] / dollars
        entry["spent"] += spent * dollars
        entry["input_tokens"] += size
        entry["cache_read_tokens"] += read
        entry["output_tokens"] += output
        thread, kind, at = row["thread"], row["kind"], row["at"]
        threads.add(thread)
        if row.get("left_alone"):
            alone.add(thread)
        before = held.get(thread)
        if kind == "answer":
            entry["calls"] += 1
            left = 0
            if row.get("sent_bytes"):
                # Its tokens by what the provider counted for the text that did go out.
                left = round(row["left_bytes"] * size / row["sent_bytes"])
            whole = max(0, size + left)
            first = before.first if before is not None else whole
            # What another conversation had cached, such as the same instructions, it would have read
            # too, unless that was there only for an entry that compache made to last, or a ping.
            alive = at - called.get(model, float("-inf")) < spec.cache_ttl_seconds
            shared = min(read, first) if alive or at >= kept.get(model, 0) else 0
            if before is not None and at - before.at < plain.cache_ttl_seconds:
                shared = max(shared, before.whole)
            shared = min(shared, whole)
            if row.get("sent_bytes"):
                without = _charge(plain, whole, shared, output)
            else:
                # Left as the application wrote it: it cost what it cost.
                without = spent
            entry["without"] += without * dollars
            entry["left_out_tokens"] += max(0, left)
            held[thread] = _Held(whole, at, first, shared, output, plain)
        elif kind in ("reopen", "retry"):
            entry["reopened" if kind == "reopen" else "retries"] += 1
            if before is not None:
                # The answer came with this call. The one before it would not have been made.
                plainly = (before.plain, before.whole, before.shared)
                entry["without"] += dollars * (
                    _charge(*plainly, output) - _charge(*plainly, before.output)
                )
                held[thread] = replace(before, at=at, output=output)
        elif kind == "ping":
            entry["pings"] += 1
        if kind in ("answer", "reopen", "retry"):
            called[model] = at
        lasts = row.get("ttl") or row.get("header_ttl")
        if kind == "ping" or lasts:
            until = at + (lasts or spec.cache_ttl_seconds)
            kept[model] = max(kept.get(model, 0), until)
    known = [entry for name, entry in models.items() if name in priced]
    spent = sum(e["spent"] for e in known)
    without = sum(e["without"] for e in known)
    return {
        "calls": sum(e["calls"] for e in models.values()),
        "conversations": len(threads),
        "left_alone": len(alone),
        "since": min((r["at"] for r in rows), default=None),
        "until": max((r["at"] for r in rows), default=None),
        "spent": spent,
        "without": without,
        "saving": 1 - spent / without if without else None,
        "models": {m: e for m, e in models.items() if m in priced},
        "unpriced": {m: e for m, e in models.items() if m not in priced},
    }


def _dollars(amount: float) -> str:
    return f"${amount:,.2f}" if amount >= 1 else f"${amount:,.4f}"


def _count(n, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def lines(found: dict) -> list[str]:
    """What was saved, in a few lines."""
    if not found["calls"]:
        return ["No calls on record yet."]
    since = datetime.fromtimestamp(found["since"], timezone.utc).date()
    until = datetime.fromtimestamp(found["until"], timezone.utc).date()
    when = f"on {since}" if since == until else f"from {since} to {until}"
    said = [
        f"{_count(found['calls'], 'call')} in {_count(found['conversations'], 'conversation')} {when}."
    ]
    if found["without"]:
        saving = found["saving"]
        more = (
            f"{saving:.0%} less" if saving >= 0.005 else f"{max(0, -saving):.0%} more"
        )
        said.append(
            f"{_dollars(found['spent'])} instead of {_dollars(found['without'])} "
            f"with everything in view and cached: {more}."
        )
    for model, entry in sorted(
        found["models"].items(), key=lambda pair: -pair[1]["spent"]
    ):
        whole = entry["input_tokens"] + entry["left_out_tokens"]
        parts = [
            _count(entry["calls"], "call"),
            f"{_dollars(entry['spent'])} of {_dollars(entry['without'])}",
        ]
        if entry["input_tokens"]:
            parts.append(
                f"{entry['cache_read_tokens'] / entry['input_tokens']:.0%} read from the cache"
            )
        if whole:
            parts.append(f"{entry['left_out_tokens'] / whole:.0%} left out")
        if entry["reopened"]:
            parts.append(f"{entry['reopened']:,} fetched back")
        if entry["pings"]:
            parts.append(_count(entry["pings"], "ping"))
        said.append(f"  {model}: {', '.join(parts)}")
    for model, entry in found["unpriced"].items():
        whole = entry["input_tokens"] + entry["left_out_tokens"]
        share = entry["left_out_tokens"] / whole if whole else 0
        said.append(
            f"  {model}: {_count(entry['calls'], 'call')}, {share:.0%} of the input left out, no price sheet"
        )
    if found["left_alone"]:
        said.append(
            f"{_count(found['left_alone'], 'conversation')} went out as written, "
            "to learn from what replies rely on when everything is in view."
        )
    return said
