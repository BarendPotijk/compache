"""What compache saved, read off the record of the calls it carried.

Every call is on record with what the provider billed and how much text
compache left out of it. That gives its price, and what it would have cost
with everything in view: the same request, longer by what was left out, read
from a cache that held all of the one before it wherever that was still alive
under the provider's default lifetime, and whatever other conversations had
left there. So the comparison is with caching at its best, not with none.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import record
from .errors import ContractError
from .models import cache_lifetimes, model_spec, pricing
from .optimizer.contracts import ModelSpec
from .optimizer.distribution import _charge

__all__ = ["lines", "savings"]

_COUNTED = ("answer", "reopen", "retry", "ping", "note")
# What an unknown model is taken to charge for reading its cache, in input tokens.
_UNKNOWN = ModelSpec("unknown", read_multiplier=0.1, write_multiplier=1.0)


def _sheets(model: str | None) -> tuple[ModelSpec, dict, dict | None]:
    """A model's prices under its default lifetime and under each other one, and what a token costs."""
    try:
        spec = model_spec(str(model or ""))
    except ContractError:
        return _UNKNOWN, {}, None
    others = {s.cache_ttl_seconds: s for s in cache_lifetimes(str(model))}
    return spec, others, pricing(str(model))


def _new() -> dict:
    return {
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
    for row in sorted(rows, key=lambda r: r["at"]):
        model = row.get("model") or "unknown"
        if model not in sheets:
            sheets[model] = _sheets(row.get("model"))
        spec, others, price = sheets[model]
        dollars = price["input_price_per_million"] / 1e6 if price else 1.0
        if price:
            priced.add(model)
        entry, usage = models.setdefault(model, _new()), row["usage"]
        size, output = usage["input"], usage["output"]
        read, written = usage["cache_read"], usage["cache_write"]
        paid = others.get(row.get("ttl"), spec)
        spent = paid.token_charge(max(0, size - read - written), written, read, output)
        entry["spent"] += spent * dollars
        entry["input_tokens"] += size
        entry["cache_read_tokens"] += read
        entry["output_tokens"] += output
        threads.add(row["thread"])
        if row.get("left_alone"):
            alone.add(row["thread"])
        kind = row["kind"]
        if kind == "answer":
            entry["calls"] += 1
            left = 0
            if row.get("sent_bytes"):
                # Its tokens by what the provider counted for the text that did go out.
                left = round(row["left_bytes"] * size / row["sent_bytes"])
            whole = max(0, size + left)
            before = held.get(row["thread"])
            first = before[2] if before is not None else whole
            # What another conversation had cached, such as the same instructions, it would have read too.
            shared = min(read, first)
            if before is not None and row["at"] - before[1] < spec.cache_ttl_seconds:
                shared = max(shared, before[0])
            shared = min(shared, whole)
            if row.get("sent_bytes"):
                entry["without"] += _charge(spec, whole, shared, output) * dollars
            else:
                # Left as the application wrote it: it cost what it cost.
                entry["without"] += spent * dollars
            entry["left_out_tokens"] += max(0, left)
            held[row["thread"]] = (whole, row["at"], first)
        elif kind == "ping":
            entry["pings"] += 1
        elif kind == "reopen":
            entry["reopened"] += 1
        elif kind == "retry":
            entry["retries"] += 1
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
        said.append(
            f"  {model}: {_count(entry['calls'], 'call')}, no price sheet, so only recorded"
        )
    if found["left_alone"]:
        said.append(
            f"{_count(found['left_alone'], 'conversation')} went out as written, "
            "to learn from what replies rely on when everything is in view."
        )
    return said
