"""Cross-component checks on the complete benchmark evidence, independent of truth."""
import copy
import json
import random
import unittest
from pathlib import Path

from compache.dependencies import build_packet, load_label_schema
from compache.traces import export_langfuse, load_threads, normalize_langfuse

REPO = Path(__file__).resolve().parents[1]


class EvidencePipelineTests(unittest.TestCase):
    def test_every_run_roundtrips_without_truth_or_visibility_changes(self):
        threads = load_threads(REPO / "benchmarks/rag_v1/threads.jsonl")
        rows = export_langfuse(threads)
        self.assertEqual(len(rows), 3 * sum(len(t.runs) for t in threads))
        self.assertNotIn("scenario_id", json.dumps(rows))
        rows.extend(copy.deepcopy(rows[::17]))
        random.Random(71).shuffle(rows)
        for row in rows:
            for key in ("input", "output"):
                if key in row:
                    row[key] = json.dumps(row[key])
        restored = {t.thread_id: t for t in normalize_langfuse(rows)}
        self.assertEqual(set(restored), {t.thread_id for t in threads})
        for original in threads:
            expected = original.to_dict()
            expected.pop("scenario_id", None)
            actual = restored[original.thread_id]
            self.assertEqual(expected, actual.to_dict())
            for run in original.runs:
                before = build_packet(original, run.run_id)
                after = build_packet(actual, run.run_id)
                self.assertEqual(before, after)
                self.assertEqual([p["part_id"] for p in after["request_parts"]], list(run.generation.request_part_ids))
                self.assertNotIn("scenario_id", json.dumps(after))
                self.assertNotIn(run.output.part_id, run.generation.request_part_ids)

    def test_skill_and_package_use_identical_label_schema(self):
        skill_schema = REPO / "skills/compache-label-dependencies/references/label.schema.json"
        self.assertEqual(json.loads(skill_schema.read_text()), load_label_schema())


if __name__ == "__main__":
    unittest.main()
