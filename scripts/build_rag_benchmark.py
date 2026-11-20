#!/usr/bin/env python3
"""Build the deterministic, fictional Compache standard-RAG v1 benchmark."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "rag_v1"
SEED = 681_204
BASE = datetime(2026, 1, 5, 9, tzinfo=timezone.utc)

COMPANIES = [
    {"id": "alder", "name": "Alder & Finch", "business": "outdoor equipment", "cadence": 9, "return": 28, "standard": 5, "express": 2, "region": "Rotterdam", "leave": 26, "budget": 420},
    {"id": "briar", "name": "Briar Works", "business": "home furnishings", "cadence": 23, "return": 35, "standard": 7, "express": 3, "region": "Utrecht", "leave": 24, "budget": 560},
    {"id": "cinder", "name": "Cinder & Co.", "business": "specialty food", "cadence": 47, "return": 21, "standard": 4, "express": 1, "region": "Eindhoven", "leave": 29, "budget": 310},
]
FAMILIES = ["direct_lookup", "constraint_carryover", "output_reference", "old_retrieval", "multi_source_comparison", "exact_quote", "fresh_duplicate", "unselected_decoy", "version_supersession", "topic_return", "unsupported_answer", "unanswerable", "masked_content", "irrelevant_citation", "ellipsis_anaphora"]

def digest(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]

def iso(d: datetime) -> str:
    return d.isoformat().replace("+00:00", "Z")

def part(tid: str, i: int, kind: str, j: int = 0) -> str:
    return f"{tid}-r{i:02d}-{kind}" + (str(j) if j else "")

def business_days(n: int) -> str:
    return f"{n} business day" + ("" if n == 1 else "s")

def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values), encoding="utf-8")

def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    corpus: list[dict] = []
    threads: list[dict] = []
    truth: list[dict] = []
    for ci, co in enumerate(COMPANIES):
        cid, name = co["id"], co["name"]
        docs = {
            "returns1": (f"{name} accepts unused {co['business']} within {co['return']} days after delivery. The customer must include proof of purchase, and clearance goods are excluded. Items must be returned in their original packaging.", "returns", "v1"),
            "returns2": (f"Effective 1 March 2026, {name} accepts unused {co['business']} within {co['return'] + 7} days after delivery. Proof of purchase remains required; clearance goods are still excluded. This revision replaces the prior return window.", "returns", "v2"),
            "standard": (f"For in-stock orders, {name} standard delivery usually takes {co['standard']} business days. The estimate begins after dispatch and excludes public holidays. A tracking email is sent when the parcel leaves the warehouse.", "delivery", "v1"),
            "express": (f"{name} express delivery usually takes {business_days(co['express'])} for in-stock orders. Express service is available only in mainland Netherlands. The estimate begins after dispatch and excludes public holidays.", "delivery", "v1"),
            "brand": (f"The approved {name} customer motto is ‘Made for everyday journeys.’ Use it exactly in campaign material. Do not imply a lifetime guarantee; product cover is limited to the written warranty.", "brand", "v3"),
            "staff": (f"At {name}, eligible full-time employees receive {co['leave']} paid leave days per calendar year. Part-time allowance is prorated by contracted hours. Leave requests require manager approval before booking.", "people", "v2"),
            "finance": (f"{name} reported serving 640 active customers in its 2025 annual review. The document does not state a 2022 revenue figure. Customer count is not a revenue measure.", "annual-review", "2025"),
            "masked": (None, "operations", "v1"),
            "region": (f"{name} stores EU customer records in {co['region']}, Netherlands. Backups are encrypted and retained for 30 days. The residency statement applies to production customer records, not employee email.", "hosting", "v2"),
            "budget": (f"The {name} team plan costs €{co['budget']} per month for up to 25 seats. Additional seats cost €18 each per month. Annual billing is invoiced at the start of the term.", "pricing", "v2"),
        }
        for key, (text, family, version) in docs.items():
            corpus.append({"company_id": cid, "document_id": f"{cid}-{family}", "version": version, "chunk_id": key, "title": f"{name} {family.replace('-', ' ').title()}", "effective_at": "2026-03-01" if key == "returns2" else None, "text": text, **({"content_status": "masked"} if text is None else {})})

        for fi, family in enumerate(FAMILIES):
            serial = ci * len(FAMILIES) + fi + 1
            tid = f"rag-{serial:03d}"
            # Every thread has 3-6 actual turns; selected intervening topics vary deterministically.
            length = 3 + ((serial + ci) % 4)
            runs = []
            for ri in range(1, length + 1):
                is_target = ri == length
                retrieval_keys: list[tuple[str, bool]] = []
                if ri == 1:
                    q = f"What return conditions should a customer of {name} know?"
                    answer = f"Unused {co['business']} may be returned within {co['return']} days after delivery when kept in original packaging and accompanied by proof of purchase; clearance goods are excluded."
                    retrieval_keys = [("returns1", True)]
                    if family == "old_retrieval":
                        q = f"What was {name}'s return window?"
                        answer = f"The return window is {co['return']} days after delivery."
                elif not is_target:
                    # These are distinct, coherent requests, never repeated shipping filler.
                    topic = (ri + fi + ci) % 4
                    if topic == 0:
                        q, answer, retrieval_keys = f"Where are {name}'s EU customer records stored?", f"In {co['region']}, Netherlands; the stated residency policy applies to production customer records.", [("region", True)]
                    elif topic == 1:
                        q, answer, retrieval_keys = f"How much annual leave does a full-time {name} employee receive?", f"{co['leave']} paid days per calendar year, subject to manager approval for each request.", [("staff", True)]
                    elif topic == 2:
                        q, answer, retrieval_keys = f"What does the {name} team plan cost each month, and how many seats are included?", f"€{co['budget']} per month includes up to 25 seats; extra seats cost €18 each monthly.", [("budget", True)]
                    else:
                        q, answer, retrieval_keys = f"What delivery estimate applies to in-stock orders at {name}?", f"For in-stock orders, standard delivery usually takes {business_days(co['standard'])} after dispatch; estimates exclude public holidays.", [("standard", True)]
                else:
                    q, answer = f"Remind me of the return window at {name}.", f"The return window is {co['return']} days after delivery."
                    if family == "direct_lookup": retrieval_keys = [("returns1", True)]
                    elif family == "constraint_carryover":
                        q, answer = "Which return conditions apply?", f"Unused {co['business']} may be returned within {co['return']} days after delivery when kept in original packaging and accompanied by proof of purchase; clearance goods are excluded."
                    elif family == "output_reference":
                        q, answer = "Turn the second bullet from your earlier comparison into a concise heading.", "Proof of purchase required"
                    elif family == "old_retrieval": q, answer = "What packaging did the first policy require?", "Items must be returned in their original packaging."
                    elif family == "multi_source_comparison":
                        q, answer = f"For in-stock orders, how many business days faster is express than standard delivery at {name}?", f"The stated estimates make express {co['standard'] - co['express']} business days faster for in-stock orders in mainland Netherlands. Both are usual durations after dispatch, excluding public holidays."
                        retrieval_keys = [("standard", True), ("express", True)]
                    elif family == "exact_quote": q, answer = f"Quote the approved {name} motto exactly, including punctuation.", "‘Made for everyday journeys.’"
                    elif family == "fresh_duplicate":
                        q = f"Does {name}'s stated return window exceed 28 days?"
                        exceeds = co["return"] > 28
                        answer = f"{'Yes' if exceeds else 'No'}. The stated window is {co['return']} days, which {'exceeds' if exceeds else 'does not exceed'} 28 days."
                    elif family == "unselected_decoy": q, answer = f"What is the {name} return window?", f"The return window is {co['return']} days after delivery."; retrieval_keys = [("returns1", False)]
                    elif family == "version_supersession": q, answer = f"What is the currently effective {name} return window?", f"The current window is {co['return'] + 7} days after delivery, effective 1 March 2026."; retrieval_keys = [("returns2", True)]
                    elif family == "topic_return": q, answer = "Returning to the first question, how long is the return window?", f"The first policy allowed returns within {co['return']} days after delivery."
                    elif family == "unsupported_answer": q, answer = "Does the policy also cover opened goods?", "Yes, opened goods can be returned within 90 days."
                    elif family == "unanswerable": q, answer = f"What was {name}'s revenue in 2022?", "I cannot determine 2022 revenue from the available information."; retrieval_keys = [("finance", True)]
                    elif family == "masked_content": q, answer = f"What is the confidential {name} warehouse alarm code?", "The available record is masked, so I cannot recover the alarm code."; retrieval_keys = [("masked", True)]
                    elif family == "irrelevant_citation": q, answer = f"What was {name}'s revenue in 2022?", f"{name} had €4 million in revenue. [annual-review]"; retrieval_keys = [("finance", True)]
                    elif family == "ellipsis_anaphora": q, answer = "Would the option you selected arrive before standard delivery?", f"On the stated estimates, yes: for in-stock orders in mainland Netherlands, express usually takes {business_days(co['express'])} versus {business_days(co['standard'])} for standard delivery; both estimates begin after dispatch and exclude public holidays."; retrieval_keys = [("express", True), ("standard", True)]

                # Additional turns provide independently useful context or a varied intervening topic.
                if ri == 1 and family == "constraint_carryover":
                    q = f"For this conversation, keep answers to one sentence. What return conditions apply at {name}?"
                    answer = f"Unused {co['business']} may be returned within {co['return']} days after delivery when kept in original packaging and accompanied by proof of purchase; clearance goods are excluded."
                if ri == 1 and family == "output_reference":
                    q = f"Compare return window and proof-of-purchase rule for {name}; use two bullets."
                    answer = f"• Return window: {co['return']} days\n• Proof of purchase is required"
                if ri == 1 and family in ("exact_quote", "ellipsis_anaphora", "multi_source_comparison"):
                    retrieval_keys = [("brand", True)] if family == "exact_quote" else [("standard", True)] if family == "multi_source_comparison" else [("standard", True), ("express", True)]
                    if family == "exact_quote": q, answer = f"What is {name}'s approved motto?", "The approved motto is ‘Made for everyday journeys.’"
                    elif family == "multi_source_comparison": q, answer = f"What is {name}'s standard delivery estimate?", f"For in-stock orders, standard delivery usually takes {business_days(co['standard'])} after dispatch; estimates exclude public holidays."
                    else: q, answer = f"Which {name} delivery option would you choose when arrival speed matters?", f"For in-stock mainland Netherlands orders, express usually takes {business_days(co['express'])} and standard usually takes {business_days(co['standard'])}; I would choose express when speed matters."
                if family == "fresh_duplicate" and is_target: retrieval_keys = [("returns1", True)]
                if family == "unsupported_answer" and is_target: retrieval_keys = [("returns1", True)]

                day_offset = ci * 31 + fi
                if family == "version_supersession":
                    day_offset += 90
                start = BASE + timedelta(days=day_offset, minutes=(ri - 1) * co["cadence"], seconds=rng.randrange(55))
                end = start + timedelta(seconds=5 + rng.randrange(10))
                inp, out, runid = part(tid, ri, "input"), part(tid, ri, "output"), f"{tid}-run-{ri:02d}"
                retrieved = []
                for j, (key, visible) in enumerate(retrieval_keys, 1):
                    doc = next(d for d in corpus if d["company_id"] == cid and d["chunk_id"] == key)
                    retrieved.append({"part_id": part(tid, ri, "retrieval", j), "document_id": doc["document_id"], "version": doc["version"], "chunk_id": key, "text": doc["text"], "exposed_to_model": visible, **({"content_status": "masked"} if doc["text"] is None else {})})
                history_parts = [p for old in runs for p in [old["input"], *[x for x in old["retrievals"] if x["exposed_to_model"]], old["output"]]]
                req = [p["part_id"] for p in history_parts] + [inp] + [x["part_id"] for x in retrieved if x["exposed_to_model"]]
                run = {"run_id": runid, "run_index": ri, "started_at": iso(start), "ended_at": iso(end), "input": {"part_id": inp, "text": q}, "retrievals": retrieved, "output": {"part_id": out, "text": answer}, "generation": {"observation_id": digest("gen:" + runid), "started_at": iso(start + timedelta(seconds=2)), "ended_at": iso(end), "request_part_ids": req, "model": "synthetic-fixture", "usage": None}, "history_policy": "full_canonical"}
                runs.append(run)
            thread = {"thread_id": tid, "company_id": cid, "scenario_id": family, "runs": runs}
            threads.append(thread)
            target = runs[-1]
            # Truth refers only to visible source occurrences. Each listed quote is an exact substring.
            visible = {x["part_id"]: x for r in runs for x in [r["input"], *r["retrievals"], r["output"]]}
            def src(pid: str, quote: str | None = None) -> dict:
                text = visible[pid]["text"]
                assert text is not None and quote and quote in text, (pid, quote)
                return {"part_id": pid, "quote": quote}
            def latest(kind: str, chunk: str | None = None) -> str:
                for run in reversed(runs[:-1]):
                    for item in reversed([run["input"], *run["retrievals"], run["output"]]):
                        if kind in item["part_id"] and (chunk is None or item.get("chunk_id") == chunk): return item["part_id"]
                raise ValueError((family, kind, chunk))
            reqs: list[dict] = []
            status = "supported"
            notes: list[str] = []
            previous = runs[-2]
            if family in ("direct_lookup", "fresh_duplicate", "unselected_decoy"):
                current = target["retrievals"][0]["part_id"] if target["retrievals"] and target["retrievals"][0]["exposed_to_model"] else None
                earlier = latest("retrieval", "returns1")
                fact_quote = f"within {co['return']} days after delivery" if family != "fresh_duplicate" else f"within {co['return']} days"
                alternatives = [[src(earlier, fact_quote)]]
                prior_summary = next(r["output"]["part_id"] for r in runs[:-1] if f"within {co['return']} days after delivery" in r["output"]["text"])
                prior_quote = f"within {co['return']} days" if family == "fresh_duplicate" else f"within {co['return']} days after delivery"
                alternatives.append([src(prior_summary, prior_quote)])
                if current:
                    alternatives.append([src(current, fact_quote)])
                description = "Compare the stated return window with the 28-day threshold" if family == "fresh_duplicate" else "State the return window and when it starts"
                reqs = [{"description": description, "alternatives": alternatives}]
            elif family == "constraint_carryover":
                pid = next(r["input"]["part_id"] for r in runs[:-1] if "one sentence" in r["input"]["text"])
                policy = latest("retrieval", "returns1")
                earlier_answer = next(r["output"]["part_id"] for r in runs[:-1] if "clearance goods are excluded" in r["output"]["text"])
                reqs = [
                    {"description": "Honor the prior user instruction to use one sentence; target input does not repeat it", "alternatives": [[src(pid, "keep answers to one sentence")]]},
                    {"description": "State the return window and proof requirement", "alternatives": [[src(policy, visible[policy]["text"])], [src(earlier_answer, f"within {co['return']} days after delivery when kept in original packaging and accompanied by proof of purchase")]]},
                    {"description": "Specify that the window starts after delivery", "alternatives": [[src(policy, f"within {co['return']} days after delivery")], [src(earlier_answer, f"within {co['return']} days after delivery")]]},
                    {"description": "Require original packaging", "alternatives": [[src(policy, "Items must be returned in their original packaging")], [src(earlier_answer, "kept in original packaging")]]},
                    {"description": "Preserve the clearance exclusion", "alternatives": [[src(policy, "clearance goods are excluded")], [src(earlier_answer, "clearance goods are excluded")]]},
                ]
            elif family == "output_reference":
                pid = next(r["output"]["part_id"] for r in runs[:-1] if "Proof of purchase is required" in r["output"]["text"])
                reqs = [{"description": "Transform the second bullet from the prior answer", "alternatives": [[src(pid, "• Proof of purchase is required")]]}]
            elif family in ("old_retrieval", "topic_return"):
                pid = latest("retrieval", "returns1")
                quote = "Items must be returned in their original packaging." if family == "old_retrieval" else f"{name} accepts unused {co['business']} within {co['return']} days after delivery."
                reference = "Resolve 'the first question' to its return-policy topic" if family == "topic_return" else "Resolve 'the first policy' to its earlier return-policy turn"
                if family == "old_retrieval":
                    reqs = [{"description": reference, "alternatives": [[src(runs[0]["input"]["part_id"], runs[0]["input"]["text"])]]}, {"description": "Recover the packaging requirement that the earlier assistant answer omitted", "alternatives": [[src(pid, quote)]]}]
                else:
                    earlier_output = next(r["output"]["part_id"] for r in runs[:-1] if f"within {co['return']} days after delivery" in r["output"]["text"])
                    reqs = [{"description": reference, "alternatives": [[src(runs[0]["input"]["part_id"], runs[0]["input"]["text"])]]}, {"description": "Recover the earlier return window", "alternatives": [[src(pid, quote)], [src(earlier_output, f"within {co['return']} days after delivery")]]}]
            elif family == "multi_source_comparison":
                standards: list[str] = []
                expresses: list[str] = []
                for run in runs:
                    for item in [*run["retrievals"], run["output"]]:
                        pid = item["part_id"]
                        if pid == target["output"]["part_id"]:
                            continue
                        if "retrieval" in pid and item.get("exposed_to_model"):
                            if item.get("chunk_id") == "standard": standards.append(pid)
                            if item.get("chunk_id") == "express": expresses.append(pid)
                        elif pid == run["output"]["part_id"] and f"{co['standard']} business days" in item["text"] and "standard" in item["text"].lower():
                            standards.append(pid)
                        if pid == run["output"]["part_id"] and f"{co['express']} business day" in item["text"] and "express" in item["text"].lower():
                            expresses.append(pid)
                pairs: dict[tuple[str, ...], list[dict]] = {}
                for std in standards:
                    for exp in expresses:
                        source_ids = tuple(dict.fromkeys((std, exp)))
                        pairs[source_ids] = [src(pid, visible[pid]["text"]) for pid in source_ids]
                reqs = [{"description": "Compare express and standard business-day estimates", "alternatives": list(pairs.values())}]
            elif family == "exact_quote":
                pid = latest("retrieval", "brand")
                prior_output = next(r["output"]["part_id"] for r in runs[:-1] if "‘Made for everyday journeys.’" in r["output"]["text"])
                reqs = [{"description": "Reproduce the approved motto exactly", "alternatives": [[src(pid, "The approved "+name+" customer motto is ‘Made for everyday journeys.’")], [src(prior_output, "‘Made for everyday journeys.’")]]}]
            elif family == "version_supersession":
                pid = target["retrievals"][0]["part_id"]
                reqs = [{"description": "Use current policy and its effective date", "alternatives": [[src(pid, f"Effective 1 March 2026, {name} accepts unused {co['business']} within {co['return'] + 7} days after delivery.")]]}]
            elif family == "unsupported_answer":
                status = "unsupported"; notes = ["The source covers unused goods and does not support opened goods or a 90-day window."]
            elif family == "unanswerable":
                status = "unanswerable"; notes = ["The retrieved annual review explicitly says the requested 2022 revenue figure is not stated."]
            elif family == "masked_content":
                status = "unassessable_masked"; notes = ["The only matching record is masked; unknown content is not evidence of no dependency."]
            elif family == "irrelevant_citation":
                status = "unsupported"; notes = ["The review states customer count, not 2022 revenue; the citation is irrelevant."]
            elif family == "ellipsis_anaphora":
                antecedent = next(r["output"]["part_id"] for r in runs[:-1] if "I would choose express when speed matters" in r["output"]["text"])
                source_ids = []
                for run in runs:
                    for item in [*run["retrievals"], run["output"]]:
                        pid = item["part_id"]
                        text = item["text"] or ""
                        if pid == target["output"]["part_id"]:
                            continue
                        if "retrieval" in pid and item.get("exposed_to_model"):
                            if item.get("chunk_id") in ("standard", "express"): source_ids.append(pid)
                        elif "business day" in text and "standard" in text.lower() and "express" in text.lower():
                            source_ids.append(pid)
                standards = [pid for pid in source_ids if visible[pid].get("chunk_id") == "standard" or "standard" in (visible[pid]["text"] or "").lower()]
                expresses = [pid for pid in source_ids if visible[pid].get("chunk_id") == "express" or "express" in (visible[pid]["text"] or "").lower()]
                pairs: dict[tuple[str, ...], list[dict]] = {}
                for std in standards:
                    for exp in expresses:
                        ids = tuple(dict.fromkeys((std, exp)))
                        pairs[ids] = [src(pid, visible[pid]["text"]) for pid in ids]
                reqs = [{"description": "Resolve the selected option to express delivery from the explicit earlier choice", "alternatives": [[src(antecedent, "I would choose express when speed matters.")]]}, {"description": "Compare both delivery estimates; alternatives include document retrievals and a prior assistant summary where they state the values", "alternatives": list(pairs.values())}]
            truth_scope = "Alternatives are authored sufficient visible evidence routes for the target task, not a claim of unique necessity, causal use, or exhaustive coverage of equivalent sets; unseen equivalent routes require adjudication. Prior assistant outputs establish conversational availability and are labeled as output occurrences, not independent documentary verification."
            notes.append(truth_scope)
            truth.append({"thread_id": tid, "target_run_id": target["run_id"], "scenario_id": family, "expected_requirements": reqs, "expected_answer_status": status, "notes": notes})

    for name, rows in (("corpus.jsonl", corpus), ("threads.jsonl", threads), ("ground_truth.jsonl", truth)):
        write_jsonl(OUT / name, rows)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*OUT.glob("*.jsonl"), OUT / "question-design.json"])}
    texts = [p["text"] for t in threads for r in t["runs"] for p in [r["input"], *r["retrievals"], r["output"]] if p["text"] is not None]
    manifest = {"dataset_id": "compache-standard-rag-v1", "schema_version": "compache-rag-fixture-v1", "seed": SEED, "generator": "scripts/build_rag_benchmark.py", "provenance": "Fictional authored policies and deterministic simulation; benchmark design, not a measurement of real traffic or causal use.", "history_policy": "Full canonical input, exposed retrieval, and output history. Per-generation request_part_ids is authoritative.", "truth_semantics": "Requirements are AND; alternatives are OR and list sufficient visible evidence routes, not unique necessity or causal use. Earlier assistant outputs establish conversational availability and are labeled output occurrences, not documentary verification.", "counts": {"companies": len(COMPANIES), "threads": len(threads), "question_families": len(FAMILIES), "runs": sum(len(t["runs"]) for t in threads), "corpus_chunks": len(corpus), "ground_truth_rows": len(truth), "part_text_utf8_bytes": sum(len(s.encode()) for s in texts), "part_text_whitespace_words": sum(len(s.split()) for s in texts), "tokens": None}, "company_cadence_minutes": {x["id"]: x["cadence"] for x in COMPANIES}, "simulation": {"base_time_utc": iso(BASE), "duration_seconds": "5-14 inclusive", "start_jitter_seconds": "0-54 inclusive", "thread_length_rule": "3 + ((1-based thread serial + company index) modulo 4)", "usage": None, "usage_note": "Usage was not simulated or measured; null means unavailable, not zero."}, "scope_and_exclusions": ["Three fictional companies crossed with all 15 starter families; balanced by design, not prevalence.", "No production traffic, live retrieval system, causal intervention, tokenizer, latency, or billed-usage claim.", "No credentials, network calls, external APIs, or remote writes.", "Masked source content remains null and unavailable.", "Generated observation rows and provider ingestion adapters are outside this benchmark artifact."], "sha256": hashes}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

if __name__ == "__main__":
    print(json.dumps(build()["counts"], sort_keys=True))
