from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "rag_long"
sys.path.insert(0, str(ROOT / "src"))
from compache.dependencies import build_packet, validate_label
from compache.traces import load_threads


class LongWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global DATA
        cls.temp = tempfile.TemporaryDirectory()
        DATA = Path(cls.temp.name)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_long_rag_workload.py"),
                "--output",
                str(DATA),
            ],
            check=True,
            capture_output=True,
        )
        cls.addClassCleanup(cls.temp.cleanup)
        cls.threads = load_threads(DATA / "threads.jsonl")
        cls.by_thread = {x.thread_id: x for x in cls.threads}
        cls.labels = [
            json.loads(line)
            for line in (DATA / "labels.jsonl").read_text().splitlines()
        ]
        cls.design = [
            json.loads(line)
            for line in (DATA / "question_design.jsonl").read_text().splitlines()
        ]
        cls.manifest = json.loads((DATA / "manifest.json").read_text())
        cls.label_by_run = {x["target_run_id"]: x for x in cls.labels}
        cls.design_by_run = {x["run_id"]: x for x in cls.design}

    def test_fixed_long_split_and_complete_label_coverage(self) -> None:
        self.assertEqual(len(self.threads), 7)
        self.assertEqual(len(self.labels), 7 * 72)
        self.assertEqual(
            self.manifest["train_thread_ids"], [t.thread_id for t in self.threads[:6]]
        )
        self.assertEqual(self.manifest["heldout_thread_ids"], ["long-alder-heldout"])
        for thread in self.threads:
            self.assertEqual(len(thread.runs), 72)
            for run in thread.runs:
                label = self.label_by_run[run.run_id]
                errors = validate_label(label, build_packet(thread, run.run_id))
                self.assertEqual(errors, [], f"{run.run_id}: {errors}")

    def test_selected_context_is_topic_matched_and_meets_long_scale(self) -> None:
        counts = [len(run.retrievals) for thread in self.threads for run in thread.runs]
        self.assertGreaterEqual(min(counts), 3)
        self.assertLessEqual(max(counts), 5)
        self.assertGreater(
            min(self.manifest["final_full_history_prompt_words"].values()), 850_000
        )
        self.assertGreater(
            self.manifest["final_prompt_token_estimate_range"]["low"], 1_000_000
        )
        self.assertIsNone(self.threads[0].runs[-1].generation.usage)
        self.assertIn(
            "MUST be replaced by actual native model answers",
            self.manifest["answer_status"],
        )
        self.assertTrue((DATA / "handbook.md").is_file())

    def test_history_families_have_correct_visible_alternatives(self) -> None:
        for thread in self.threads:
            prefix = thread.thread_id
            version = self.label_by_run[f"{prefix}-run-009"]["task_requirements"][0]
            self.assertEqual(version["historical_context_needed"], "yes")
            self.assertEqual(len(version["alternatives"][0]["sources"]), 2)
            duplicate = self.label_by_run[f"{prefix}-run-015"]["task_requirements"][0]
            self.assertEqual(duplicate["historical_context_needed"], "no")
            self.assertEqual(len(duplicate["alternatives"]), 2)
            old = self.label_by_run[f"{prefix}-run-030"]["task_requirements"][0]
            self.assertEqual(old["historical_context_needed"], "yes")
            self.assertEqual(
                old["alternatives"][0]["sources"][0]["part_id"],
                f"{prefix}-r001-retrieval1",
            )
            output_ref = self.label_by_run[f"{prefix}-run-018"]["task_requirements"][0]
            self.assertEqual(
                output_ref["alternatives"][0]["sources"][0]["part_id"],
                f"{prefix}-r006-output",
            )
            run36 = next(
                x
                for x in self.by_thread[prefix].to_dict()["runs"]
                if x["run_index"] == 36
            )
            decoy = next(
                x for x in run36["retrievals"] if x["chunk_id"] == "returns_old"
            )
            self.assertFalse(decoy["exposed_to_model"])
            self.assertNotIn(decoy["part_id"], run36["generation"]["request_part_ids"])
            unselected = self.label_by_run[f"{prefix}-run-036"]["task_requirements"][0]
            self.assertEqual(unselected["historical_context_needed"], "yes")
            self.assertEqual(
                unselected["alternatives"][0]["sources"][0]["part_id"],
                f"{prefix}-r009-retrieval1",
            )
            topic_return = self.label_by_run[f"{prefix}-run-037"]["task_requirements"][
                0
            ]
            self.assertEqual(topic_return["historical_context_needed"], "no")

    def test_case_registers_and_offline_timing_are_frozen(self) -> None:
        manual = (DATA / "handbook.md").read_text()
        self.assertIn("BIL-01-001", manual)
        self.assertIn("BIL-07-001", manual)
        self.assertEqual(
            self.manifest["cadence"]["offline_medium_return_gap_seconds"], 420
        )
        self.assertEqual(
            self.manifest["cadence"]["offline_long_return_gap_seconds"], 2400
        )
        self.assertEqual(
            set(self.manifest["cadence"]["long_gap_after_turn_by_thread"].values()),
            set(range(60, 67)),
        )
        self.assertEqual(
            set(self.manifest["question_family_counts"].values()) & {0}, set()
        )

    def test_generator_recreates_identical_artifacts(self) -> None:
        names = (
            "threads.jsonl",
            "labels.jsonl",
            "corpus.jsonl",
            "question_design.jsonl",
            "handbook.md",
            "manifest.json",
        )
        before = {
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in names
        }
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_long_rag_workload.py"),
                "--output",
                str(DATA),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        after = {
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in names
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
