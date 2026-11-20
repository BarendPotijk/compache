import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from compache.optimizer.contracts import digest
from compache.setup_dependencies import (
    dependency_preview, fit_content_recipes, prepare_dependency_sample,
)
from compache.traces import ContractError


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "rag_workload"


class DependencySetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "sample.jsonl"
        with (FIXTURE / "threads.jsonl").open() as original:
            self.threads = [json.loads(next(original)) for _ in range(5)]
        self.source.write_text("".join(json.dumps(t) + "\n" for t in self.threads))
        self.spec = json.loads((FIXTURE / "roomy-model.json").read_text())
        self.labels = self.root / "labels.jsonl"

    def prepare(self, name="review", **kwargs):
        directory = self.root / name
        report = prepare_dependency_sample(
            self.source, directory, allow_content=True, **kwargs
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        packets = []
        for filename, sha in manifest["sha256"].items():
            data = (directory / filename).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha)
            self.assertEqual(stat.S_IMODE((directory / filename).stat().st_mode), 0o600)
            packets.append(json.loads(data))
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(report["packets"], len(packets))
        return packets

    def write_labels(self, packets):
        targets = {p["target_run_id"] for p in packets}
        with (FIXTURE / "labels.jsonl").open() as original:
            matches = [r for r in map(json.loads, original) if r["target_run_id"] in targets]
        self.assertEqual(len(matches), len(targets))
        self.labels.write_text("".join(json.dumps(r) + "\n" for r in matches))
        return matches

    def fit(self, **kwargs):
        return fit_content_recipes(
            self.source, self.labels, self.spec, allow_content=True, **kwargs
        )

    def test_no_content_approval_performs_no_io(self):
        with patch.object(Path, "open", side_effect=AssertionError("file access")):
            self.assertEqual(dependency_preview()["max_threads"], 3)
            for approved in (False, None, "yes", 1):
                with self.assertRaises(ContractError):
                    prepare_dependency_sample(self.source, self.root / "out", allow_content=approved)
                with self.assertRaises(ContractError):
                    fit_content_recipes(self.source, self.labels, self.spec, allow_content=approved)
        self.assertFalse((self.root / "out").exists())

    def test_private_deterministic_whole_thread_packets_only(self):
        packets = self.prepare(max_threads=2, seed=8)
        self.assertEqual(packets, self.prepare("again", max_threads=2, seed=8))
        tids = {p["thread_id"] for p in packets}
        self.assertEqual(len(tids), 2)
        for thread in self.threads:
            if thread["thread_id"] not in tids:
                continue
            targets = [p for p in packets if p["thread_id"] == thread["thread_id"]]
            self.assertEqual([p["target_run_id"] for p in targets], [r["run_id"] for r in thread["runs"]])
            for run, p in zip(thread["runs"], targets):
                self.assertEqual([v["part_id"] for v in p["request_parts"]], run["generation"]["request_part_ids"])
                self.assertNotIn(run["output"]["part_id"], [v["part_id"] for v in p["request_parts"]])
                for name in ("generation", "company_id", "provenance", "started_at", "usage"):
                    self.assertNotIn(name, p)
        with self.assertRaises(FileExistsError):
            self.prepare(max_threads=2)
        self.assertEqual(len(list((self.root / "review/packets").glob("*.json"))), len(packets))

    def test_assess_with_validated_labels_only_and_no_cadence_fit(self):
        packets = self.prepare()
        gold = self.write_labels(packets)
        with patch("compache.optimizer.workload.fit_distribution", side_effect=AssertionError("metadata fit")):
            result = self.fit()
        self.assertEqual(result["training_threads"], sorted({digest(p["thread_id"]) for p in packets}))
        self.assertEqual(len(result["recipes"]), 2)
        self.assertEqual(result["report"]["selected_runs"], len(packets))
        for card, score in zip(result["report"]["recipe_results"], result["recipes"]):
            self.assertEqual(set(card), {
                "recipe", "historical_retained", "historical_required", "coverage",
                "changed_runs", "unknown_runs", "unknown_changed_runs", "unknown_change_rate",
                "eligible", "reasons",
            })
            for field in set(card) - {"reasons"}:
                self.assertEqual(card[field], score[field])
            self.assertEqual(card["eligible"], not bool(card["reasons"]))
        self.assertEqual(sum(result["report"]["label_status"].values()), len(packets))
        self.assertNotIn(packets[0]["thread_id"], json.dumps(result))
        self.assertNotIn(packets[0]["target_question"]["text"], json.dumps(result))
        self.assertNotIn("demand", result)
        self.assertEqual(
            sum(result["report"]["requirements_needing_history"].values()),
            sum(len(r["task_requirements"]) for r in gold),
        )

    def test_scope_and_evidence_fail_closed_without_echoing_content(self):
        packets = self.prepare()
        rows = self.write_labels(packets)
        bad = {**rows[0], "target_run_id": "PRIVATE-CUSTOMER-ID"}
        for invalid in (rows[1:], rows + [rows[0]], [bad] + rows[1:]):
            self.labels.write_text("".join(json.dumps(r) + "\n" for r in invalid))
            with self.assertRaises(ContractError) as error:
                self.fit()
            self.assertNotIn("PRIVATE-CUSTOMER-ID", str(error.exception))
        rows[0]["task_requirements"][0]["question_quote"] = "PRIVATE-FABRICATED-QUOTE"
        self.labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ContractError, "Invalid dependency evidence"):
            self.fit()
        self.spec["model_id"] = "mismatch"
        with self.assertRaisesRegex(ContractError, "different model"):
            self.fit()

    def test_short_trace_reports_lookback_and_withholds_untested_cuts(self):
        packets = self.prepare(max_threads=1)
        self.write_labels(packets)
        fitted = self.fit(max_threads=1)
        self.assertEqual(
            fitted["report"]["required_lookback_runs_by_kind"],
            {"input": {}, "retrieval": {"2": 1}, "output": {}},
        )
        for score in fitted["recipes"]:
            self.assertEqual(score["coverage"], 1)
            self.assertFalse(score["eligible"])
            self.assertEqual(score["setup_admission_reason"], "no_cut_exercised_in_sample")
        for card in fitted["report"]["recipe_results"]:
            self.assertEqual(card["reasons"], ["no_cut_exercised_in_sample"])

    def test_failed_preparation_leaves_no_partial_bundle(self):
        with patch("compache.setup_dependencies.build_packet", side_effect=OSError):
            with self.assertRaises(OSError):
                self.prepare()
        self.assertFalse((self.root / "review").exists())

    def test_masked_turns_keep_full_denominator_and_withhold_cuts(self):
        result = fit_content_recipes(
            FIXTURE / "threads.jsonl", FIXTURE / "labels.jsonl", self.spec,
            allow_content=True, max_threads=45,
        )
        report = result["report"]
        self.assertEqual(report["selected_runs"], 207)
        self.assertEqual(report["projection"]["scored_runs"], 204)
        self.assertEqual(report["projection"]["affected_threads"], 3)
        self.assertEqual(report["projection"]["unscorable_runs"], 3)
        self.assertEqual(sum(report["label_status"].values()), 207)
        for card in report["recipe_results"]:
            self.assertFalse(card["eligible"])
            self.assertIn("incomplete_visible_history", card["reasons"])

    def test_all_masked_inputs_accept_only_valid_labels_and_make_no_estimate(self):
        thread = self.threads[0]
        thread["runs"] = thread["runs"][:1]
        run = thread["runs"][0]
        run["input"].update(text=None, content_status="masked")
        self.source.write_text(json.dumps(thread) + "\n")
        label = {"schema_version": "compache-dependency-label-v1", "thread_id": thread["thread_id"],
                 "target_run_id": run["run_id"], "status": "unassessable",
                 "coverage_notes": ["Question masked"], "claims": [], "task_requirements": [],
                 "historical_dependencies": [], "unresolved": ["Question masked"]}
        self.labels.write_text(json.dumps(label) + "\n")
        result = self.fit()["report"]
        self.assertEqual(result["projection"]["scored_runs"], 0)
        self.assertEqual(result["projection"]["unscorable_runs"], 1)
        self.assertTrue(all(card["coverage"] is None for card in result["recipe_results"]))
        label["status"] = "complete"
        self.labels.write_text(json.dumps(label) + "\n")
        with self.assertRaisesRegex(ContractError, "Invalid dependency evidence"):
            self.fit()
