#!/usr/bin/env python3
"""Build a seeded, fictional sharded RAG stress workload (revision 2).

Planning-token counts use ceil(UTF-8 bytes / 4) + 6 per retrieval part,
matching default ModelSpec.part_tokens. They are not measured provider tokens.
"""

from __future__ import annotations
import argparse, gzip, hashlib, json, random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_long_rag_workload import handbook_expansion

SEED = 20261116
LENGTHS = (24, 36, 48, 60, 72, 84, 96, 108, 120, 144)
DOMAINS = ("returns", "delivery", "pricing", "hosting", "people", "warranty")
NAMES = (
    "Alder Harbor",
    "Bramble Forge",
    "Cedar Compass",
    "Dune Foundry",
    "Elm Lantern",
    "Fable Orchard",
    "Granite Loom",
    "Hearth Atlas",
    "Indigo Quay",
    "Juniper Mill",
    "Kestrel Vale",
    "Linden Bridge",
    "Morrow Field",
    "Northbank Craft",
    "Opal Ferry",
    "Pine Standard",
    "Quartz Grove",
    "River Anchor",
    "Sable Ridge",
    "Thistle Workshop",
    "Umber Crest",
    "Violet Wharf",
    "Willow Mason",
    "Yarrow Basin",
    "Zephyr Works",
    "Ashbrook Supply",
    "Birch Row",
    "Copper Meridian",
    "Driftwood House",
    "Elder Port",
)
BASE = datetime(2026, 4, 6, tzinfo=timezone.utc)
DEFAULT = Path("work/scaled-rag-dataset")


def compact(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as z,
    ):
        for row in rows:
            z.write((compact(row) + "\n").encode())


def iso(d):
    return d.isoformat().replace("+00:00", "Z")


def src(part, quote):
    return {"part_id": part, "quote": quote}


def token_estimate(text):
    return (len(text.encode("utf-8")) + 3) // 4 + 6


def profile(n):
    r = random.Random(SEED + 10007 * n)
    p = {
        "company_id": f"company-{n:02}",
        "name": NAMES[n - 1],
        "business": (
            "homewares",
            "specialty tools",
            "outdoor equipment",
            "business software",
            "logistics",
            "workplace services",
        )[(n - 1) % 6],
        "location": (
            "Rotterdam",
            "Utrecht",
            "Eindhoven",
            "Groningen",
            "Maastricht",
            "Zwolle",
        )[(n * 7) % 6],
        "warm_seconds": r.randint(2, 24),
        "medium_seconds": r.randint(360, 750),
        "long_seconds": r.randint(1900, 3900),
        "medium_probability": round(r.uniform(0.07, 0.24), 3),
        "long_probability": round(r.uniform(0.03, 0.14), 3),
        "dependency_probability": round(r.uniform(0.58, 0.82), 3),
        "long_lag_probability": round(r.uniform(0.35, 0.7), 3),
        "output_style": r.choice(("concise", "brief", "detailed")),
        "domain_count": r.randint(4, 6),
        "bundle_target": r.randint(16800, 47000),
    }
    # Preserve three pre-outcome pilot IDs. Their revised content is independently seeded.
    if n in (24, 26, 29):
        p["bundle_target"] = {26: 39000, 24: 41500, 29: 46500}[n]
    return p


def opaque(tid, number, prefix="C"):
    return (
        prefix
        + hashlib.sha256(f"{SEED}:{tid}:{number}:{prefix}".encode())
        .hexdigest()[:10]
        .upper()
    )


def domain_fact(domain, case, r, p):
    """A single authoritative case occurrence and its natural current question."""
    if domain == "returns":
        days = r.randrange(18, 64)
        cond = r.choice(
            (
                "proof of purchase",
                "the original order confirmation",
                "a transaction receipt",
            )
        )
        fact = f"Case {case}: the approved return window is {days} days from delivery; {cond} is required."
        q = f"For returns case {case}, what is the approved window and what proof is required?"
    elif domain == "delivery":
        days = r.randrange(2, 12)
        region = r.choice(
            (
                "mainland Netherlands",
                "Belgium",
                "the Amsterdam service area",
                "the Utrecht service area",
            )
        )
        fact = f"Case {case}: the in-stock delivery estimate is {days} business days after dispatch, for {region}."
        q = f"For delivery case {case}, what is the estimate and which destination does it cover?"
    elif domain == "pricing":
        fee = r.randrange(190, 990, 10)
        seats = r.randrange(8, 65)
        fact = f"Case {case}: the quoted standard plan is €{fee} per month and includes {seats} seats."
        q = f"For pricing case {case}, what is the monthly standard-plan quote and seat allowance?"
    elif domain == "hosting":
        city = r.choice(
            ("Rotterdam", "Utrecht", "Eindhoven", "Zwolle", "Groningen", "Maastricht")
        )
        days = r.randrange(14, 91)
        fact = f"Case {case}: production customer records are stored in {city}; encrypted backups are retained for {days} days."
        q = f"For hosting case {case}, where are production customer records stored and for how long are backups retained?"
    elif domain == "people":
        days = r.randrange(2, 13)
        approval = r.choice(
            ("the line manager", "People Operations", "the department lead")
        )
        fact = f"Case {case}: the annual paid training allowance is {days} days, subject to approval by {approval}."
        q = f"For people case {case}, what is the annual paid training allowance and whose approval is required?"
    else:
        months = r.randrange(12, 49)
        days = r.randrange(3, 16)
        fact = f"Case {case}: the limited warranty runs for {months} months from delivery and assessment usually takes {days} business days."
        q = f"For warranty case {case}, what is the coverage duration and usual assessment time?"
    return fact, q


def corpus(p, domain):
    """Procedural handbook plus a fictional tabular register of distinct matters.

    Tabular entries have varied scope, status, owner and action, with independent
    identifiers and values; they are separate cases, not repeated padding text.
    """
    n = int(p["company_id"][-2:])
    r = random.Random(SEED + n * 199 + DOMAINS.index(domain))
    handbook = (
        handbook_expansion(domain, "Alder & Finch", n)
        .split("| Case |")[0]
        .replace("Alder & Finch", p["name"])
    )
    rows = [
        f"{p['name']} / {domain} procedure and case register. The register entries below are case-specific; approved case terms do not silently amend the general handbook.",
        handbook,
        "Register columns: case reference | intake | category | stage | owner | evidence | next action | review date. Each row concerns a distinct matter.",
    ]
    categories = {
        "returns": (
            "unused item",
            "packaging dispute",
            "clearance status",
            "order evidence",
            "damaged item",
        ),
        "delivery": (
            "dispatch tracking",
            "destination check",
            "carrier exception",
            "stock confirmation",
            "holiday calendar",
        ),
        "pricing": (
            "seat adjustment",
            "invoice query",
            "renewal estimate",
            "plan comparison",
            "billing contact",
        ),
        "hosting": (
            "backup retention",
            "residency scope",
            "data classification",
            "processor query",
            "recovery exercise",
        ),
        "people": (
            "training request",
            "leave coordination",
            "manager approval",
            "part-time schedule",
            "handover coverage",
        ),
        "warranty": (
            "defect assessment",
            "delivery evidence",
            "repair route",
            "wear exclusion",
            "claim photographs",
        ),
    }
    stages = (
        "received",
        "awaiting evidence",
        "owner review",
        "resolved",
        "follow-up due",
        "referred",
    )
    evidence = (
        "order record",
        "customer email",
        "signed case note",
        "service log",
        "manager note",
        "invoice extract",
        "system record",
    )
    actions = (
        "confirm scope",
        "request missing proof",
        "record decision",
        "route to owner",
        "check applicable date",
        "close on confirmation",
        "resolve discrepancy",
        "schedule review",
    )
    owners = ("care", "finance", "operations", "people", "fulfilment", "security")
    # Target size is filled with complete records; a fixed reserve leaves room for
    # inserted authoritative occurrences and keeps every full bundle under 50k.
    target_bytes = max(60000, 4 * (p["bundle_target"] - 120))
    total = sum(len(x.encode()) + 1 for x in rows)
    j = 0
    while total < target_bytes:
        j += 1
        case = opaque(f"{p['company_id']}-{domain}-register", j, "R")
        row = (
            f"{case} | {2025 + r.randrange(2)}-{1 + r.randrange(12):02}-{1 + r.randrange(27):02} | {r.choice(categories[domain])}"
            f" | {r.choice(stages)} | {r.choice(owners)} | {r.choice(evidence)} | {r.choice(actions)}"
            f" | review {2026 + r.randrange(2)}-{1 + r.randrange(12):02}-{1 + r.randrange(27):02} | "
            f"quantity {r.randrange(1, 240)}; channel {r.choice(('portal', 'email', 'phone', 'internal'))}; "
            f"priority {r.choice(('routine', 'time-sensitive', 'owner review'))}; "
            f"outcome {r.choice(('pending', 'documented', 'customer response due', 'escalated'))}"
        )
        rows.append(row)
        total += len(row.encode()) + 1
    return rows


def schedule(length, p, r):
    """Preselect target/source pairs so only needed historical notes are introduced."""
    used = set()
    result = {}
    kinds = (
        "input",
        "history",
        "prior_output",
        "mixed",
        "fresh",
        "version",
        "correction",
        "arithmetic",
    )
    for i in range(1, length + 1):
        forced = i in (52, 82, 85, 89, 93)
        if i <= 2 or not (r.random() < p["dependency_probability"] or forced):
            continue
        lag = (
            r.randint(51, i - 1)
            if i > 52 and (r.random() < p["long_lag_probability"] or forced)
            else r.choice(
                (1, min(i - 1, r.randint(6, 20)), min(i - 1, r.randint(21, 50)))
            )
        )
        available = [j for j in range(1, i) if j not in used and (j > 8 or forced)]
        if forced:
            distant = [j for j in available if i - j >= (51 if i in (52, 82) else 21)]
            if distant:
                available = distant
        if not available:
            continue
        source = min(available, key=lambda j: abs((i - j) - lag))
        kind = kinds[((i + int(p["company_id"][-2:])) // 3) % len(kinds)]
        if kind == "correction" and i - source < 2:
            eligible = [j for j in available if i - j >= 2]
            if not eligible:
                continue
            source = min(eligible, key=lambda j: abs((i - j) - lag))
        used.add(source)
        result[i] = (kind, source)
    return result


def make_thread(n, k, p):
    length = LENGTHS[k]
    tid = f"scaled-{n:02}-{k + 1:02}"
    cid = p["company_id"]
    r = random.Random(SEED + n * 1000 + k)
    domains = DOMAINS[: p["domain_count"]]
    id_rng = random.Random(SEED ^ (n * 0x9E3779B1) ^ (k * 0x85EBCA77))
    cases = [f"C{x:010X}" for x in id_rng.sample(range(1 << 40), length)]
    facts = []
    questions = []
    dom = []
    for i, case in enumerate(cases, 1):
        d = domains[(i + n + k) % len(domains)]
        f, q = domain_fact(d, case, r, p)
        dom.append(d)
        facts.append(f)
        questions.append(q)
    plans = schedule(length, p, r)
    notes = defaultdict(list)
    note_values = {}
    for target, (kind, source) in plans.items():
        if kind not in ("input", "mixed", "correction", "arithmetic"):
            continue
        case = cases[source - 1]
        units = r.randint(20, 160)
        note = f"For procurement case {case}, my initial allocation is {units} units."
        notes[source].append(note)
        note_values[target] = (note, units, None)
        if kind == "correction":
            corrected = units + r.randint(4, 35)
            correction = f"Correction for procurement case {case}: change my allocation to {corrected} units."
            notes[source + 1].append(correction)
            note_values[target] = (note, units, correction)
    corpora = {d: corpus(p, d) for d in domains}
    thread = {
        "company_id": cid,
        "thread_id": tid,
        "scenario_id": "fictional-scaled-rag-revised",
        "provenance": "fabricated authored fixture; usage unavailable",
        "runs": [],
    }
    labels = []
    history = []
    previous = []
    design = []
    families = Counter()
    lags = Counter()
    cursor = BASE + timedelta(days=n * 10 + k)
    split = "heldout" if n >= 21 else ("validation" if k >= 8 else "train")
    for i in range(1, length + 1):
        rid = f"{tid}-run-{i:03}"
        iid = f"{tid}-r{i:03}-input"
        oid = f"{tid}-r{i:03}-output"
        pid = f"{tid}-r{i:03}-retrieval"
        domain = dom[i - 1]
        kind, source = plans.get(i, ("direct", None))
        lag = i - source if source else 0
        if kind == "direct" and i % 11 == 0:
            kind = "abstention"
        elif kind == "direct" and i % 7 == 0:
            kind = "scoped_negative"
        q = " ".join(notes.get(i, []) + [questions[i - 1]])
        historical = ""
        alts = []
        extra = []
        need = "yes"
        source_kind = "mixed"
        old = previous[source - 1] if source else None
        if kind == "input":
            note, units, _ = note_values[i]
            q += f" For procurement case {old['case']}, what allocation did I previously propose?"
            historical = note
            alts = [[src(old["iid"], note)]]
            source_kind = "input"
        elif kind == "history":
            q += f" Also return to the earlier {old['domain']} case {old['case']} and report its recorded terms."
            historical = old["fact"]
            alts = [[src(old["pid"], old["fact"])], [src(old["oid"], old["fact"])]]
        elif kind == "prior_output":
            q += f" Also quote the first sentence of your answer in turn {source}, preserving punctuation."
            historical = old["fact"]
            alts = [[src(old["oid"], old["fact"])]]
            source_kind = "output"
        elif kind == "mixed":
            note, units, _ = note_values[i]
            q += f" For earlier case {old['case']}, report both my proposed procurement allocation and the recorded {old['domain']} terms."
            historical = note + " " + old["fact"]
            alts = [
                [src(old["iid"], note), src(old["pid"], old["fact"])],
                [src(old["iid"], note), src(old["oid"], old["fact"])],
            ]
        elif kind == "fresh":
            q += f" Also report the recorded terms for case {old['case']}, using the current case extract."
            historical = old["fact"]
            extra.append(old["fact"])
            alts = [
                [src(pid, old["fact"])],
                [src(old["pid"], old["fact"])],
                [src(old["oid"], old["fact"])],
            ]
            need = "no"
        elif kind == "version":
            revised_fact, _ = domain_fact(old["domain"], old["case"], r, p)
            while revised_fact == old["fact"]:
                revised_fact, _ = domain_fact(old["domain"], old["case"], r, p)
            amendment = f"Revised case record superseding the original: {revised_fact}"
            q += f" For earlier case {old['case']}, compare the original recorded terms with the revised case record."
            historical = old["fact"] + " " + amendment
            extra.append(amendment)
            alts = [
                [src(old["pid"], old["fact"]), src(pid, amendment)],
                [src(old["oid"], old["fact"]), src(pid, amendment)],
            ]
        elif kind == "correction":
            note, units, correction = note_values[i]
            corrected = int(correction.split("to ")[-1].split()[0])
            correction_run = previous[source]
            q += f" For procurement case {old['case']}, compare my original allocation with the later correction."
            historical = f"The original allocation was {units} units; the corrected allocation is {corrected} units."
            alts = [[src(old["iid"], note), src(correction_run["iid"], correction)]]
            source_kind = "input"
        elif kind == "arithmetic":
            note, units, _ = note_values[i]
            rate = 3 + (n + i) % 7
            rate_text = f"Planning schedule for case {old['case']}: the rate is {rate} credits per unit."
            q += f" For procurement case {old['case']}, calculate the total from my allocation and the current planning schedule."
            historical = f"The planning total is {units * rate} credits."
            extra.append(rate_text)
            alts = [[src(old["iid"], note), src(pid, rate_text)]]
        elif kind == "abstention":
            q += " What was the exact loss in the unlogged pilot? State whether it is documented."
        elif kind == "scoped_negative":
            q += " For in-stock deliveries, does the published estimate guarantee arrival?"
            extra.append(
                "For in-stock deliveries, the published estimate is not a guaranteed arrival date."
            )
        # The relevant record is interleaved at a deterministic varying position.
        rows = corpora[domain]
        insert_at = 2 + r.randrange(len(rows) - 2)
        snippets = [facts[i - 1], *extra]
        text = "\n".join(rows[:insert_at] + snippets + rows[insert_at:])
        answer = facts[i - 1] + (" " + historical if historical else "")
        if kind == "abstention":
            answer += " The exact loss in the unlogged pilot is not documented in the available material."
        if kind == "scoped_negative":
            answer += " " + extra[-1]
        if p["output_style"] == "detailed":
            answer += " These case-specific terms should be checked against the applicable documented scope."
        ret = {
            "part_id": pid,
            "document_id": f"{cid}-{domain}-register",
            "version": "v2",
            "chunk_id": f"{domain}-{i:03}",
            "text": text,
            "exposed_to_model": True,
        }
        end = cursor + timedelta(seconds=1)
        run = {
            "run_id": rid,
            "run_index": i,
            "started_at": iso(cursor),
            "ended_at": iso(end),
            "history_policy": "full_canonical",
            "input": {"part_id": iid, "text": q},
            "output": {"part_id": oid, "text": answer},
            "retrievals": [ret],
            "generation": {
                "observation_id": hashlib.sha256(rid.encode()).hexdigest()[:16],
                "started_at": iso(cursor),
                "ended_at": iso(end),
                "model": "gpt-6-luna",
                "usage": None,
                "request_part_ids": history + [iid, pid],
            },
        }
        thread["runs"].append(run)

        def req(num, description, sk, alternatives, needed):
            return {
                "requirement_id": f"req-{num}",
                "description": description,
                "source_kind": sk,
                "necessity": "inferred",
                "alternatives": [{"sources": a} for a in alternatives],
                "evidence_note": "Authored sufficient evidence; not measured causal source use.",
                "question_quote": q,
                "fulfillment": "fulfilled",
                "historical_context_needed": needed,
            }

        reqs = [
            req(
                1,
                "Report the current case terms.",
                "retrieval",
                [[src(pid, facts[i - 1])]],
                "no",
            )
        ]
        if alts:
            reqs.append(
                req(2, "Answer the requested case follow-up.", source_kind, alts, need)
            )
        if kind == "scoped_negative":
            reqs.append(
                req(
                    2,
                    "State the in-stock guarantee scope.",
                    "retrieval",
                    [[src(pid, extra[-1])]],
                    "no",
                )
            )
        if kind == "abstention":
            reqs.append(
                req(
                    2,
                    "State that the unlogged pilot loss is unavailable.",
                    "unknown",
                    [],
                    "unknown",
                )
            )

        def claim_req(x):
            return {
                key: v
                for key, v in x.items()
                if key
                not in ("question_quote", "fulfillment", "historical_context_needed")
            }

        claims = [
            {
                "claim_id": "claim-1",
                "target_quote": facts[i - 1],
                "relation": "supported",
                "requirements": [claim_req(reqs[0])],
            }
        ]
        if historical:
            claims.append(
                {
                    "claim_id": "claim-2",
                    "target_quote": historical,
                    "relation": "supported",
                    "requirements": [claim_req(reqs[1])],
                }
            )
        if kind == "scoped_negative":
            claims.append(
                {
                    "claim_id": "claim-2",
                    "target_quote": extra[-1],
                    "relation": "supported",
                    "requirements": [claim_req(reqs[1])],
                }
            )
        if kind == "abstention":
            claims.append(
                {
                    "claim_id": "claim-2",
                    "target_quote": "The exact loss in the unlogged pilot is not documented in the available material.",
                    "relation": "uncertain",
                    "requirements": [],
                }
            )
        edges = []
        seen = set()
        for alt in alts:
            for s in alt:
                sid = s["part_id"]
                if sid == pid or sid in seen:
                    continue
                seen.add(sid)
                origin = next(
                    x for x in previous if sid in (x["iid"], x["oid"], x["pid"])
                )
                sk = (
                    "input"
                    if sid == origin["iid"]
                    else "output"
                    if sid == origin["oid"]
                    else "retrieval"
                )
                edges.append(
                    {
                        "source_part_id": sid,
                        "source_run_id": origin["rid"],
                        "source_kind": sk,
                        "relation": "possibly_used",
                        "target_claim_ids": ["claim-2"],
                        "target_task_requirement_ids": ["req-2"],
                        "evidence_note": "Earlier visible occurrence is an authored sufficient route.",
                    }
                )
        labels.append(
            {
                "schema_version": "compache-dependency-label-v1",
                "thread_id": tid,
                "target_run_id": rid,
                "status": "complete",
                "coverage_notes": [
                    "Fabricated output and authored evidence. Live prior-output requirements must refer to actual answers."
                ],
                "task_requirements": reqs,
                "claims": claims,
                "historical_dependencies": edges,
                "unresolved": [],
            }
        )
        design.append(
            {
                "run_id": rid,
                "family": kind,
                "lag": lag,
                "source_run_id": old["rid"] if old else None,
                "domain": domain,
                "case_id": cases[i - 1],
                "current_expected": facts[i - 1],
                "historical_expected": historical or None,
                "actual_prior_output_required": kind == "prior_output",
                "answer_position": insert_at,
            }
        )
        previous.append(
            {
                "iid": iid,
                "oid": oid,
                "pid": pid,
                "rid": rid,
                "fact": facts[i - 1],
                "case": cases[i - 1],
                "domain": domain,
            }
        )
        history += [iid, pid, oid]
        families[kind] += 1
        if lag:
            lags[
                "1-5"
                if lag <= 5
                else "6-20"
                if lag <= 20
                else "21-50"
                if lag <= 50
                else "51+"
            ] += 1
        u = r.random()
        gap = (
            p["long_seconds"]
            if u < p["long_probability"]
            else p["medium_seconds"]
            if u < p["long_probability"] + p["medium_probability"]
            else p["warm_seconds"]
        )
        cursor = end + timedelta(seconds=gap)
    sizes = [token_estimate(run["retrievals"][0]["text"]) for run in thread["runs"]]
    words = [len(run["retrievals"][0]["text"].split()) for run in thread["runs"]]
    meta = {
        "thread_id": tid,
        "company_id": cid,
        "split": split,
        "turn_count": length,
        "scenario": "fictional-scaled-rag-revised",
        "company_profile": p,
        "case_ids": cases,
        "families": dict(families),
        "historical_lags": dict(lags),
        "run_design": design,
        "retrieval_planning_tokens": {"min": min(sizes), "max": max(sizes)},
        "retrieval_word_estimate": {
            "min": round(min(words) * 1.3),
            "max": round(max(words) * 1.3),
        },
    }
    return thread, labels, meta


def build(output, companies=30, selected=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    families = Counter()
    lags = Counter()
    profiles = []
    for n in (
        sorted(set(n for n, _ in selected)) if selected else range(1, companies + 1)
    ):
        p = profile(n)
        profiles.append(p)
        for k in [k for nn, k in selected if nn == n] if selected else range(10):
            thread, labels, entry = make_thread(n, k, p)
            tid = entry["thread_id"]
            trace = f"traces/{tid}.jsonl.gz"
            label = f"labels/{tid}.jsonl.gz"
            write_gz(output / trace, [thread])
            write_gz(output / label, labels)
            entry.update(
                trace_path=trace,
                labels_path=label,
                trace_sha256=sha(output / trace),
                labels_sha256=sha(output / label),
            )
            rows.append(entry)
            families.update(entry["families"])
            lags.update(entry["historical_lags"])
    manifest = {
        "schema_version": "compache-scaled-rag-revised-v1",
        "seed": SEED,
        "thread_count": len(rows),
        "turn_count": sum(x["turn_count"] for x in rows),
        "companies": profiles,
        "threads": rows,
        "split_counts": dict(Counter(x["split"] for x in rows)),
        "family_counts": dict(families),
        "historical_lag_counts": dict(lags),
        "planning_token_method": "ModelSpec.part_tokens defaults: ceil UTF-8 bytes/4 plus 6 per part; estimate only",
        "word_estimate_method": "1.3 times whitespace words; separate rough estimate only",
        "interpretation": "Authored fictional stress workload; no natural-traffic prevalence or public benchmark claim.",
        "usage": "null for fabricated outputs; live provider usage must be measured",
        "answer_status": "fabricated scaffolding; live prior-output dependencies require actual run answers",
        "train_thread_ids": [x["thread_id"] for x in rows if x["split"] == "train"],
        "validation_thread_ids": [
            x["thread_id"] for x in rows if x["split"] == "validation"
        ],
        "heldout_thread_ids": [x["thread_id"] for x in rows if x["split"] == "heldout"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--companies", type=int, default=30)
    ap.add_argument(
        "--select", action="append", help="company:thread ordinal (1-based)"
    )
    args = ap.parse_args()
    selected = (
        [(int(x.split(":")[0]), int(x.split(":")[1]) - 1) for x in args.select]
        if args.select
        else None
    )
    m = build(args.output, args.companies, selected)
    print(
        f"Generated {m['thread_count']} threads / {m['turn_count']} turns in {args.output}"
    )
