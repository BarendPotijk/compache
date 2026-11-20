"""Conformance checks for the authored RAG benchmark and its frozen truth."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "benchmarks" / "rag_v1"
BUILDER = REPO / "scripts" / "build_rag_benchmark.py"

def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line]

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = read_jsonl("corpus.jsonl")
        cls.threads = read_jsonl("threads.jsonl")
        cls.truth = read_jsonl("ground_truth.jsonl")
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    def test_balanced_45_threads_and_at_least_150_runs(self) -> None:
        self.assertEqual(len(self.threads), 45)
        self.assertGreaterEqual(sum(len(t["runs"]) for t in self.threads), 150)
        self.assertTrue(all(3 <= len(t["runs"]) <= 6 for t in self.threads))
        self.assertEqual(len({t["company_id"] for t in self.threads}), 3)
        per_company = Counter(t["company_id"] for t in self.threads)
        self.assertEqual(set(per_company.values()), {15})
        per_pair = Counter((t["company_id"], t["scenario_id"]) for t in self.threads)
        self.assertEqual(len(per_pair), 45)
        self.assertTrue(all(n == 1 for n in per_pair.values()))
        self.assertEqual({r["scenario_id"] for r in self.truth}, {t["scenario_id"] for t in self.threads})

    def test_full_history_and_exact_generation_visibility(self) -> None:
        for thread in self.threads:
            prior: list[str] = []
            for i, run in enumerate(thread["runs"], 1):
                self.assertEqual(run["run_index"], i)
                self.assertEqual(run["history_policy"], "full_canonical")
                expected = prior + [run["input"]["part_id"]] + [p["part_id"] for p in run["retrievals"] if p["exposed_to_model"]]
                self.assertEqual(run["generation"]["request_part_ids"], expected)
                self.assertNotIn(run["output"]["part_id"], expected)
                prior = expected + [run["output"]["part_id"]]
                self.assertIsNone(run["generation"]["usage"])

    def test_frozen_truth_quotes_are_exact_and_sources_visible(self) -> None:
        by_thread = {t["thread_id"]: t for t in self.threads}
        for label in self.truth:
            thread = by_thread[label["thread_id"]]
            target = next(r for r in thread["runs"] if r["run_id"] == label["target_run_id"])
            visible = {p["part_id"]: p for r in thread["runs"] for p in [r["input"], *r["retrievals"], r["output"]]}
            available = set(target["generation"]["request_part_ids"])
            for requirement in label["expected_requirements"]:
                for alternative in requirement["alternatives"]:
                    self.assertTrue(alternative, "an alternative represents a nonempty AND set")
                    for source in alternative:
                        self.assertIn(source["part_id"], available)
                        self.assertNotEqual(source["part_id"], target["output"]["part_id"])
                        text = visible[source["part_id"]]["text"]
                        self.assertIsInstance(text, str)
                        self.assertIn(source["quote"], text)

    def test_masked_unknowns_and_unsupported_answers_are_preserved(self) -> None:
        masked = [c for c in self.corpus if c.get("content_status") == "masked"]
        self.assertTrue(masked)
        self.assertTrue(all(c["text"] is None for c in masked))
        masked_truth = [x for x in self.truth if x["expected_answer_status"] == "unassessable_masked"]
        self.assertEqual(len(masked_truth), 3)
        self.assertTrue(all("unknown content" in x["notes"][0] for x in masked_truth))
        self.assertTrue(any(x["expected_answer_status"] == "unsupported" for x in self.truth))
        self.assertTrue(any(x["expected_answer_status"] == "unanswerable" for x in self.truth))
        for thread in self.threads:
            for run in thread["runs"]:
                for item in run["retrievals"]:
                    if item.get("content_status") == "masked":
                        self.assertIsNone(item["text"])

    def test_arithmetic_and_effective_date_cases_match_their_answers(self) -> None:
        for thread in self.threads:
            target = thread["runs"][-1]
            if thread["scenario_id"] == "fresh_duplicate":
                co = next(c for c in ("alder", "briar", "cinder") if c == thread["company_id"])
                days = {"alder": 28, "briar": 35, "cinder": 21}[co]
                affirmative = "Yes." in target["output"]["text"]
                self.assertEqual(affirmative, days > 28)
                self.assertIn(thread["company_id"], target["input"]["text"].lower())
                self.assertIn("28 days", target["input"]["text"])
                self.assertNotIn("that ", target["input"]["text"].lower())
            if thread["scenario_id"] == "version_supersession":
                self.assertGreaterEqual(target["started_at"], "2026-03-01T00:00:00Z")
                self.assertIn("effective 1 March 2026", target["output"]["text"])
                self.assertEqual(target["retrievals"][0]["version"], "v2")
        cinder = next(x for x in self.corpus if x["company_id"] == "cinder" and x["chunk_id"] == "express")
        self.assertIn("1 business day for in-stock orders", cinder["text"])
        self.assertNotIn("1 business days", cinder["text"])

    def test_constraint_carryover_and_reference_roles_are_separate(self) -> None:
        for thread in self.threads:
            scenario = thread["scenario_id"]
            target = thread["runs"][-1]
            truth = next(x for x in self.truth if x["thread_id"] == thread["thread_id"])
            if scenario == "constraint_carryover":
                constraint = next(req for req in truth["expected_requirements"] if "one sentence" in req["description"])
                old_part = constraint["alternatives"][0][0]["part_id"]
                old_run = next(r for r in thread["runs"] if old_part == r["input"]["part_id"])
                self.assertLess(old_run["run_index"], target["run_index"])
                self.assertIn("one sentence", old_run["input"]["text"])
                self.assertNotIn("one sentence", target["input"]["text"])
                # A separate content requirement must use policy evidence; the format instruction cannot stand in for it.
                content = next(req for req in truth["expected_requirements"] if "return window" in req["description"])
                self.assertTrue(any("retrieval" in source["part_id"] for alt in content["alternatives"] for source in alt))
                required = {"starts after delivery", "original packaging", "clearance exclusion"}
                descriptions = " ".join(req["description"] for req in truth["expected_requirements"]).lower()
                self.assertTrue(all(text in descriptions for text in required))
                clearance = next(req for req in truth["expected_requirements"] if "clearance" in req["description"])
                self.assertTrue(any("output" in src["part_id"] for alt in clearance["alternatives"] for src in alt))
                for phrase in ("after delivery", "original packaging", "proof of purchase", "clearance goods are excluded"):
                    self.assertIn(phrase, target["output"]["text"])
            if scenario == "topic_return":
                resolution, fact = truth["expected_requirements"]
                self.assertIn("first question", resolution["description"])
                self.assertEqual(resolution["alternatives"][0][0]["part_id"], thread["runs"][0]["input"]["part_id"])
                self.assertIn("return window", fact["description"])
                self.assertIn("retrieval", fact["alternatives"][0][0]["part_id"])
            if scenario == "old_retrieval":
                self.assertIn("return window", thread["runs"][0]["input"]["text"].lower())
                self.assertNotIn("original packaging", thread["runs"][0]["output"]["text"].lower())
                self.assertIn("packaging", target["input"]["text"].lower())
                self.assertIn("original packaging", target["output"]["text"].lower())
                package_fact = next(req for req in truth["expected_requirements"] if "packaging" in req["description"])
                self.assertEqual(len(package_fact["alternatives"]), 1)
                self.assertIn("retrieval", package_fact["alternatives"][0][0]["part_id"])

    def test_alternatives_cover_valid_visible_evidence_and_unambiguous_anaphora(self) -> None:
        truth_by_thread = {x["thread_id"]: x for x in self.truth}
        for thread in self.threads:
            family = thread["scenario_id"]
            truth = truth_by_thread[thread["thread_id"]]
            if family == "direct_lookup":
                fact = truth["expected_requirements"][0]
                sources = {alt[0]["part_id"] for alt in fact["alternatives"]}
                self.assertGreaterEqual(len(sources), 2)  # repeated historical and fresh identical evidence
            elif family == "exact_quote":
                fact = truth["expected_requirements"][0]
                self.assertGreaterEqual(len(fact["alternatives"]), 2)
                self.assertTrue(any("output" in alt[0]["part_id"] for alt in fact["alternatives"]))
            elif family == "multi_source_comparison":
                fact = truth["expected_requirements"][0]
                self.assertGreaterEqual(len(fact["alternatives"]), 3)
                for alt in fact["alternatives"]:
                    self.assertEqual(len(alt), 2)  # both values are jointly required
                    self.assertTrue(any("standard" in source["quote"].lower() for source in alt))
                    self.assertTrue(any("express" in source["quote"].lower() for source in alt))
                # The first turn's retrieval and assistant summary are both visible and sufficient routes for standard's value.
                alternatives = fact["alternatives"]
                self.assertTrue(any(any("r01-retrieval" in s["part_id"] for s in alt) for alt in alternatives))
                self.assertTrue(any(any("r01-output" in s["part_id"] for s in alt) for alt in alternatives))
                for run in thread["runs"][:-1]:
                    if "standard delivery usually takes" in run["output"]["text"].lower():
                        self.assertTrue(any(run["output"]["part_id"] in [s["part_id"] for s in alt] for alt in alternatives))
            elif family == "ellipsis_anaphora":
                antecedent = next(r for r in thread["runs"][:-1] if "I would choose express when speed matters" in r["output"]["text"])
                self.assertIn("selected", truth["expected_requirements"][0]["description"])
                self.assertIn("the option you selected", thread["runs"][-1]["input"]["text"])
                self.assertIn(antecedent["output"]["part_id"], thread["runs"][-1]["generation"]["request_part_ids"])
                fact = truth["expected_requirements"][1]
                self.assertTrue(any("r01-output" in s["part_id"] for alt in fact["alternatives"] for s in alt))
                self.assertTrue(any(any(s["part_id"].endswith("r01-retrieval1") for s in alt) and any(s["part_id"].endswith("r01-retrieval2") for s in alt) for alt in fact["alternatives"]))

    def test_intermediate_delivery_summaries_preserve_scope_and_qualifiers(self) -> None:
        for thread in self.threads:
            if thread["scenario_id"] == "multi_source_comparison":
                first = thread["runs"][0]["output"]["text"].lower()
                self.assertIn("usually", first)
                self.assertIn("in-stock", first)
            if thread["scenario_id"] == "ellipsis_anaphora":
                first = thread["runs"][0]["output"]["text"].lower()
                self.assertIn("mainland netherlands", first)
                self.assertIn("usually", first)
                self.assertIn("in-stock", first)

    def test_manifest_hashes_and_simulation_disclosures(self) -> None:
        self.assertEqual(self.manifest["schema_version"], "compache-rag-fixture-v1")
        self.assertIsNone(self.manifest["simulation"]["usage"])
        self.assertIsNone(self.manifest["counts"]["tokens"])
        self.assertEqual(set(self.manifest["sha256"]), {"corpus.jsonl", "threads.jsonl", "ground_truth.jsonl", "question-design.json"})
        for filename, expected in self.manifest["sha256"].items():
            actual = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)
        self.assertEqual(len(set(self.manifest["company_cadence_minutes"].values())), 3)
        for row in self.truth:
            self.assertTrue(any("sufficient visible evidence routes" in note for note in row["notes"]))
            self.assertTrue(any("documentary verification" in note for note in row["notes"]))

    def test_builder_recreates_byte_identical_artifacts(self) -> None:
        before = {p.name: p.read_bytes() for p in DATA.iterdir() if p.is_file()}
        subprocess.run([sys.executable, str(BUILDER)], check=True, cwd=REPO, capture_output=True, text=True)
        after = {p.name: p.read_bytes() for p in DATA.iterdir() if p.is_file()}
        self.assertEqual(after, before)

if __name__ == "__main__":
    unittest.main()
