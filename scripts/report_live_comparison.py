"""Report completed six-arm native snapshot comparison; never infer missing usage."""

from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from compache.optimizer.measurement import aggregate, ARMS, ROLES
from compache.optimizer.cache_scenarios import reprice_prefix


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def audit_events(path, row):
    if not path.exists():
        return [row.get("usage")]
    events = read_lines(path)
    updates = [
        e["params"]["tokenUsage"]
        for e in events
        if e.get("method") == "thread/tokenUsage/updated"
        and e.get("params", {}).get("threadId") == row.get("session_id")
    ]
    if not updates:
        raise ValueError("missing raw usage notification")
    fields = (
        ("inputTokens", "input_tokens"),
        ("cachedInputTokens", "cached_input_tokens"),
        ("outputTokens", "output_tokens"),
    )
    prev = {k: 0 for k, _ in fields}
    deltas = []
    for update in updates:
        total = update.get("total", {})
        last = update.get("last", {})
        if any(type(total.get(k)) is not int or total[k] < prev[k] for k, _ in fields):
            raise ValueError("nonmonotonic usage total")
        delta = {name: total[k] - prev[k] for k, name in fields}
        if all(v == 0 for v in delta.values()):
            continue
        if any(last.get(k) != delta[name] for k, name in fields):
            raise ValueError("total/last usage mismatch")
        if delta["cached_input_tokens"] > delta["input_tokens"]:
            raise ValueError("invalid cached usage")
        deltas.append(delta)
        prev = {k: total[k] for k, _ in fields}
    if (
        not deltas
        or deltas[-1] != row.get("usage", {})
        and any(
            deltas[-1][name] != row.get("usage", {}).get(name) for _, name in fields
        )
    ):
        raise ValueError("last/ledger usage mismatch")
    return deltas


def retry_overhead(root):
    totals = {
        arm: {
            "model_calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
        }
        for arm in ARMS
    }
    for p in sorted((root / "excluded-transport-attempts").glob("*/overhead.json")):
        arm = p.parent.name.split("-")[0]
        if arm not in totals:
            raise ValueError("unknown retry arm")
        d = read_json(p)
        calls = d["model_calls"]
        sums = {
            k: sum(x[k] for x in calls)
            for k in ("inputTokens", "cachedInputTokens", "outputTokens")
        }
        if sums != d["totals"]:
            raise ValueError("retry usage total mismatch")
        z = totals[arm]
        z["model_calls"] += len(calls)
        for k, name in (
            ("inputTokens", "input_tokens"),
            ("cachedInputTokens", "cached_input_tokens"),
            ("outputTokens", "output_tokens"),
        ):
            z[name] += sums[k]
    for z in totals.values():
        z["credit_equivalent"] = (
            (z["input_tokens"] - z["cached_input_tokens"]) * 2.5e-6
            + z["cached_input_tokens"] * 0.25e-6
            + z["output_tokens"] * 12.5e-6
        )
    return totals


def load_experiment(root):
    plan = read_json(root / "plan.json")
    n = plan["planned_answers_per_arm"]
    rows = []
    counts = {}
    errors = []
    answer_records = {}
    for arm in ARMS:
        for rep in range(1, 4):
            p = root / "runs" / f"{arm}-{rep}"
            try:
                ledger = read_lines(p / "ledger.jsonl")
                answers = read_lines(p / "answers.jsonl")
                expected = read_json(p / "expected-counts.json")
                complete = read_json(p / "complete.json")
                actual = Counter(x["role"] for x in ledger)
                if (
                    set(expected) != set(ROLES)
                    or complete.get("counts") != expected
                    or complete.get("calls") != len(ledger)
                ):
                    raise ValueError("completion/count mismatch")
                if any(actual[role] != expected[role] for role in ROLES):
                    raise ValueError("ledger/count mismatch")
                if (
                    len(answers) != n
                    or complete.get("answers") != n
                    or actual["answer"] != n
                ):
                    raise ValueError("incomplete answers")
                if [a.get("run_index") for a in answers] != list(range(1, n + 1)):
                    raise ValueError("answer sequence mismatch")
                if [x.get("sequence") for x in ledger] != list(
                    range(1, len(ledger) + 1)
                ):
                    raise ValueError("call sequence mismatch")
                if any(
                    x.get("arm") != arm
                    or x.get("replicate") != rep
                    or x.get("ok") is not True
                    for x in ledger
                ):
                    raise ValueError("failed call or wrong identity")
                for row in ledger:
                    row["_provider_usage"] = audit_events(
                        p / f"call-{row['sequence']:04d}" / "events.jsonl", row
                    )
                for role in ROLES:
                    counts[arm, rep, role] = expected[role]
                rows.extend(ledger)
                answer_records[f"{arm}-{rep}"] = answers
            except (
                FileNotFoundError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"{arm}-{rep}: {exc}")
    if errors:
        raise ValueError("Incomplete experiment: " + "; ".join(errors))
    return plan, rows, counts, answer_records


def build(root):
    plan, rows, counts, answer_records = load_experiment(root)
    expanded = []
    logical = []
    extra = {
        "model_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }
    sequences = {}
    for row in rows:
        pieces = row["_provider_usage"]
        if any(
            not isinstance(part, dict)
            or any(
                type(part.get(k)) is not int
                for k in ("input_tokens", "cached_input_tokens", "output_tokens")
            )
            for part in pieces
        ):
            raise ValueError("missing usage")
        sums = {
            k: sum(part[k] for part in pieces)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens")
        }
        logical.append({**row, "usage": sums})
        key = (row["arm"], row["replicate"])
        for i, part in enumerate(pieces):
            sequences[key] = sequences.get(key, 0) + 1
            is_extra = i < len(pieces) - 1
            if is_extra:
                extra["model_calls"] += 1
                for k in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    extra[k] += part[k]
            expanded.append(
                {
                    **row,
                    "logical_sequence": row["sequence"],
                    "sequence": sequences[key],
                    "role": "codex_overhead" if is_extra else row["role"],
                    "usage": part,
                    "wall_seconds": 0 if is_extra else row.get("wall_seconds", 0),
                }
            )
    measured = aggregate(logical, model=plan["model"], expected_counts=counts)
    if not all(measured["aggregate"][a]["actual"]["complete"] for a in ARMS):
        raise ValueError("incomplete measured usage coverage")
    actual = reprice_prefix(expanded)
    no_recovery = reprice_prefix(expanded, exclude_recovery=True)
    for group in (actual, no_recovery):
        if (
            not all(r["complete"] for r in group["replicates"])
            or len(group["replicates"]) != 12
        ):
            raise ValueError("incomplete modeled cost coverage")
    scenarios = {}
    for ttl in (300, 1800):
        subsets = [r for r in actual["replicates"] if r["ttl_seconds"] == ttl]
        pairs = []
        for rep in range(1, 4):
            b = next(
                x for x in subsets if x["arm"] == "opencode" and x["replicate"] == rep
            )
            c = next(
                x for x in subsets if x["arm"] == "compache" and x["replicate"] == rep
            )
            pairs.append(
                {
                    "replicate": rep,
                    "opencode": b["scenario_cost"],
                    "compache": c["scenario_cost"],
                    "savings": b["scenario_cost"] - c["scenario_cost"],
                    "savings_percent": (b["scenario_cost"] - c["scenario_cost"])
                    / b["scenario_cost"]
                    * 100
                    if b["scenario_cost"]
                    else None,
                }
            )
        baseline = sum(p["opencode"] for p in pairs)
        treatment = sum(p["compache"] for p in pairs)
        scenarios[str(ttl)] = {
            "paired_savings": pairs,
            "opencode": baseline,
            "compache": treatment,
            "aggregate_savings": baseline - treatment,
            "savings_percent": (baseline - treatment) / baseline * 100
            if baseline
            else None,
        }
    timeline = []
    for ttl in (300, 1800):
        for r in [x for x in actual["replicates"] if x["ttl_seconds"] == ttl]:
            cumulative = 0.0
            cumulative_input = 0
            cumulative_output = 0
            cumulative_wall = 0.0
            index = {(x["arm"], x["replicate"], x["sequence"]): x for x in expanded}
            for call in r["calls"]:
                cumulative += call["scenario_cost"]
                original = index[(r["arm"], r["replicate"], call["sequence"])]
                cumulative_input += original["usage"]["input_tokens"]
                cumulative_output += original["usage"]["output_tokens"]
                cumulative_wall += original.get("wall_seconds", 0)
                timeline.append(
                    {
                        "ttl_seconds": ttl,
                        "arm": r["arm"],
                        "replicate": r["replicate"],
                        "sequence": call["sequence"],
                        "logical_sequence": original["logical_sequence"],
                        "run_index": original["run_index"],
                        "role": call["role"],
                        "logical_at": original["logical_at"],
                        "input_tokens": original["usage"]["input_tokens"],
                        "observed_cached_input_tokens": call[
                            "observed_cached_input_tokens"
                        ],
                        "modeled_cached_input_tokens": call[
                            "modeled_cached_input_tokens"
                        ],
                        "output_tokens": original["usage"]["output_tokens"],
                        "modeled_credit_cost": call["scenario_cost"],
                        "cumulative_modeled_credit_cost": cumulative,
                        "cache_status": call["cache_status"],
                        "started_at_utc": original.get("started_at_utc"),
                        "ended_at_utc": original.get("ended_at_utc"),
                        "wall_seconds": original.get("wall_seconds"),
                        "cumulative_wall_seconds": cumulative_wall,
                        "cumulative_input_tokens": cumulative_input,
                        "cumulative_output_tokens": cumulative_output,
                    }
                )
    diagnostics = {}
    for arm in ARMS:
        for rep in range(1, 4):
            key = f"{arm}-{rep}"
            calls = [r for r in rows if r["arm"] == arm and r["replicate"] == rep]
            answers = answer_records[key]
            formats = Counter(
                a.get("response_format", "unknown_legacy") for a in answers
            )
            diagnostics[key] = {
                "calls_by_role": dict(Counter(c["role"] for c in calls)),
                "max_input_tokens": max(c["usage"]["input_tokens"] for c in calls),
                "total_wall_seconds": sum(c.get("wall_seconds", 0) for c in calls),
                "missing_authored_requirements": sum(
                    len(
                        (a.get("diagnostic") or {}).get(
                            "missing_authored_requirements", []
                        )
                    )
                    for a in answers
                )
                if arm == "compache"
                else None,
                "rewrite_events": sum(
                    c.get("prefix_transition") == "rewritten"
                    or c.get("prefix_transition") == "rewrite"
                    for c in calls
                ),
                "summary_calls": sum(c["role"] == "summary" for c in calls),
                "response_formats": dict(formats),
            }
    retries = retry_overhead(root)
    for arm in ARMS:
        retry = retries[arm]
        total = measured["aggregate"][arm]["actual"]
        for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
            retry["combined_" + field] = total["usage"][field] + retry[field]
        retry["combined_credit_equivalent"] = (
            total["standard_rate_credit_equivalent"] + retry["credit_equivalent"]
        )
    setup = (
        read_json(root / "setup-overhead.json")
        if (root / "setup-overhead.json").exists()
        else None
    )
    note = (
        read_json(root / "continuation-note.json")
        if (root / "continuation-note.json").exists()
        else None
    )
    return {
        "complete": True,
        "retry_overhead": retries,
        "codex_additional_invocations": extra,
        "partial_setup_overhead": setup,
        "diagnostics": diagnostics,
        "continuation_note": note,
        "model": plan["model"],
        "measured": measured,
        "modeled_with_recovery": actual,
        "modeled_zero_recovery_direct_exclusion": no_recovery,
        "modeled_paired_savings": scenarios,
        "timeline": timeline,
        "limitations": [
            "Physical cached counters are observed for fresh ephemeral snapshots; modeled prefix reuse is not observed cache behavior.",
            "Runtime input overhead is included in usage but cannot be split exactly.",
            "Additional provider invocations use the same modeled snapshot blocks and logical time; their native correction tails are not observable.",
            "The direct recovery exclusion does not remove downstream carryover.",
            "OpenCode is a source-derived adapter, not the native executable.",
        ],
    }


def write_report(root):
    result = build(root)
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "timeline.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["timeline"][0]), delimiter="\t")
        w.writeheader()
        w.writerows(result["timeline"])
    lines = [
        "# Native snapshot comparison",
        "",
        "Completed six runs. Credit equivalents use standard rates for reported usage; they are not a subscription billing statement.",
        "",
        "| Arm | Input | Physical cached input | Output | Reported usage credit equivalent |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        total = result["measured"]["aggregate"][arm]["actual"]
        u = total["usage"]
        lines.append(
            f"| {arm} | {u['input_tokens']:,} | {u['cached_input_tokens']:,} | {u['output_tokens']:,} | {total['standard_rate_credit_equivalent']:.4f} |"
        )
    lines += [
        "",
        f"Additional Codex invocations within successful logical calls: {result['codex_additional_invocations']['model_calls']} (included in measured totals and modeled scenarios). Their correction tails are not visible; modeled cache reuse assumes the same snapshot blocks.",
    ]
    lines += [
        "",
        "Discarded transport attempts are retry overhead, separate from package recovery. Combined totals include successful calls and retries; setup and controller activity are incomplete.",
        "",
        "| Arm | Retry model calls | Retry input | Retry cached | Retry output | Retry credits | Combined credits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, z in result["retry_overhead"].items():
        lines.append(
            f"| {arm} | {z['model_calls']} | {z['input_tokens']:,} | {z['cached_input_tokens']:,} | {z['output_tokens']:,} | {z['credit_equivalent']:.4f} | {z['combined_credit_equivalent']:.4f} |"
        )
    values = result["retry_overhead"].values()
    lines.append(
        f"| Both | {sum(z['model_calls'] for z in values)} | {sum(z['input_tokens'] for z in values):,} | {sum(z['cached_input_tokens'] for z in values):,} | {sum(z['output_tokens'] for z in values):,} | {sum(z['credit_equivalent'] for z in values):.4f} | {sum(z['combined_credit_equivalent'] for z in values):.4f} |"
    )
    if result["partial_setup_overhead"] is not None:
        lines += [
            "",
            "Setup probe counts are partial; see setup-overhead.json. Controller usage is unavailable.",
        ]
    lines += [
        "",
        "Modeled prefix-cache savings are counterfactual; physical cache behavior differs because each snapshot uses a fresh ephemeral thread.",
        "",
        "| TTL | OpenCode credits | Compache credits | Savings | Savings % |",
        "|---|---:|---:|---:|---:|",
    ]
    for ttl, v in result["modeled_paired_savings"].items():
        pct = (
            f"{v['savings_percent']:.1f}%"
            if v["savings_percent"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {ttl} seconds | {v['opencode']:.4f} | {v['compache']:.4f} | {v['aggregate_savings']:.4f} | {pct} |"
        )
        for pair in v["paired_savings"]:
            pct = (
                f"{pair['savings_percent']:.1f}%"
                if pair["savings_percent"] is not None
                else "unavailable"
            )
            lines.append(
                f"| {ttl}s, replicate {pair['replicate']} | {pair['opencode']:.4f} | {pair['compache']:.4f} | {pair['savings']:.4f} | {pct} |"
            )
    lines += [
        "",
        "| Run | Answer / summary / ping / recovery calls | Max input | Wall seconds | Missing requirements | Rewrites | Plain text / unknown format |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, d in result["diagnostics"].items():
        counts = "/".join(str(d["calls_by_role"].get(x, 0)) for x in ROLES)
        formats = d["response_formats"]
        lines.append(
            f"| {key} | {counts} | {d['max_input_tokens']:,} | {d['total_wall_seconds']:.1f} | {d['missing_authored_requirements'] if d['missing_authored_requirements'] is not None else 'not assessed'} | {d['rewrite_events']} | {formats.get('plain_text', 0)} / {formats.get('unknown_legacy', 0)} |"
        )
    if result["continuation_note"] is not None:
        lines += [
            "",
            "The run continued after an initial protocol pause. External-tool isolation changed for affected replicas after discarded attempts; see continuation-note.json for details.",
        ]
    lines += [
        "",
        "Recovery is included in the primary totals. The separate recovery exclusion removes direct calls only and does not model later carryover.",
        "",
        *["- " + x for x in result["limitations"]],
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    try:
        write_report(args.root)
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(str(e))
