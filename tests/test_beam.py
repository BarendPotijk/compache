"""Contract checks for the BEAM adapter, on authored rows in BEAM's layout."""

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "beam"))
import beam

from compache.labels import build_packet, label_index
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.session import WorkloadPolicy, fit_workload
from compache.threads import Thread


def message(i, role, text, anchor=None):
    return {
        "id": i,
        "index": None,
        "role": role,
        "content": text,
        "question_type": "main_question" if role == "user" else None,
        "time_anchor": anchor,
    }


def row(conversation_id="7", *, restart=False, second_date="March-02-2025"):
    first = [
        message(0, "user", "I keep bees on a flat roof in Leiden.", "March-01-2025"),
        message(1, "assistant", "Noted: a rooftop hive in Leiden."),
        message(2, "user", "The hive is named Clover."),
        message(3, "assistant", "Clover it is."),
    ]
    offset = 0 if restart else 4
    second = [
        message(offset, "user", "I moved the hive to Delft.", second_date),
        message(offset + 1, "assistant", "Updated: the hive is now in Delft."),
        message(offset + 2, "user", "The harvest came to nine jars."),
        message(offset + 3, "assistant", "Nine jars recorded. " * 40),
    ]
    probes = {
        "abstention": [
            {
                "question": "What colour is the hive?",
                "ideal_response": "The chat does not say.",
                "rubric": ["States that the colour is not given"],
            }
        ],
        "information_extraction": [
            {
                "question": "What is the hive called?",
                "answer": "Clover.",
                "source_chat_ids": [2],
                "rubric": ["Clover"],
            }
        ],
        "knowledge_update": [
            {
                "question": "Where is the hive now?",
                "answer": "Delft.",
                "source_chat_ids": {"original_info": [0], "updated_info": [4, 5]},
                "rubric": ["Delft"],
            }
        ],
        "event_ordering": [
            {
                "question": "In what order did things happen?",
                "answer": "Named, moved, harvested.",
                "source_chat_ids": [2, [4], [6, 7]],
                "rubric": ["Naming before the move"],
            }
        ],
        "summarization": [
            {
                "question": "Summarise the chat.",
                "ideal_summary": "A hive was named, moved and harvested.",
                "source_chat_ids": [],
                "rubric": ["Mentions the move"],
            }
        ],
    }
    return {
        "conversation_id": conversation_id,
        "conversation_seed": {"category": "Hobbies & Crafts", "id": 7, "title": "Bees"},
        "chat": [first, second],
        "probing_questions": repr(probes),
    }


def moment(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class BeamWorkloadTests(unittest.TestCase):
    def test_conversation_becomes_a_valid_thread_with_probe_runs(self):
        thread, labels, entry = beam.convert(row(), "100K")
        parsed = Thread.from_dict(thread)
        self.assertEqual(parsed.thread_id, "beam-100k-07")
        self.assertEqual(parsed.company_id, "hobbies-crafts")
        self.assertEqual(
            {
                k: entry[k]
                for k in ("chat_runs", "probe_runs", "turn_count", "sessions")
            },
            {"chat_runs": 4, "probe_runs": 5, "turn_count": 9, "sessions": 2},
        )
        self.assertEqual(
            (
                entry["assessed_probes"],
                entry["unknown_probes"],
                entry["ambiguous_probes"],
            ),
            (3, 2, 0),
        )
        third = thread["runs"][2]
        self.assertEqual(third["input"]["text"], "I moved the hive to Delft.")
        self.assertEqual(
            third["generation"]["request_part_ids"],
            [f"beam-100k-07-m{i:04d}" for i in range(5)],
        )
        probe = thread["runs"][4]
        self.assertEqual(probe["input"]["text"], "What colour is the hive?")
        self.assertEqual(probe["output"]["text"], "The chat does not say.")
        self.assertEqual(len(probe["generation"]["request_part_ids"]), 9)
        self.assertEqual(
            [x["run_id"] for x in labels],
            [r["run_id"] for r in thread["runs"][4:]],
        )

    def test_labels_pass_the_package_validator(self):
        thread, labels, _ = beam.convert(row(), "100K")
        beam.verify(thread, labels)
        visible = {}
        for run in thread["runs"]:
            official = build_packet(thread, run["run_id"])
            self.assertEqual(beam.packet(thread["thread_id"], run, visible), official)
            for kind in ("input", "output"):
                visible[run[kind]["part_id"]] = {
                    "part_id": run[kind]["part_id"],
                    "origin_run_id": run["run_id"],
                    "origin_run_index": run["run_index"],
                    "kind": kind,
                    "text": run[kind]["text"],
                    "content_status": "available",
                }
        self.assertEqual(len(label_index([Thread(thread)], labels)), len(labels))
        self.assertEqual(set(labels[1]), {"thread_id", "run_id", "category", "needs"})
        labels[1]["needs"][0][0]["quote"] = "absent"
        with self.assertRaisesRegex(ValueError, "quote not found"):
            beam.verify(thread, labels)

    def test_evidence_ids_map_to_parts_of_the_right_kind(self):
        thread, labels, _ = beam.convert(row(), "100K")
        asked = {run["run_id"]: run["input"]["text"] for run in thread["runs"]}
        by_question = {asked[x["run_id"]]: x["needs"] for x in labels}

        def cited(question):
            (alternative,) = by_question[question]
            return [s["part_id"][-5:] for s in alternative]

        self.assertEqual(cited("What is the hive called?"), ["m0002"])
        self.assertEqual(
            by_question["What is the hive called?"][0][0]["quote"],
            "The hive is named Clover.",
        )
        self.assertEqual(cited("Where is the hive now?"), ["m0000", "m0004", "m0005"])
        self.assertEqual(
            cited("In what order did things happen?"),
            ["m0002", "m0004", "m0006", "m0007"],
        )
        long_quote = by_question["In what order did things happen?"][0][-1]["quote"]
        self.assertEqual(len(long_quote), beam.QUOTE_CHARS)
        for question in ("What colour is the hive?", "Summarise the chat."):
            self.assertIsNone(by_question[question])
        with self.assertRaises(ValueError):
            beam.evidence_ids(["m2"])

    def test_restarted_message_numbering_makes_evidence_ambiguous(self):
        thread, labels, entry = beam.convert(row(restart=True), "1M")
        beam.verify(thread, labels)
        self.assertEqual(
            (
                entry["assessed_probes"],
                entry["unknown_probes"],
                entry["ambiguous_probes"],
            ),
            (0, 2, 3),
        )
        self.assertTrue(all(x["needs"] is None for x in labels))

    def test_malformed_conversations_are_rejected(self):
        broken = row()
        broken["chat"][1].pop()
        with self.assertRaisesRegex(ValueError, "alternating"):
            beam.convert(broken, "100K")
        undated = row()
        undated["chat"][1][0]["time_anchor"] = "sometime"
        with self.assertRaisesRegex(ValueError, "session date"):
            beam.convert(undated, "100K")
        with self.assertRaises(ValueError):
            beam.convert(row(), "100K", clock="instant")

    def test_clock_is_declared_reproducible_and_keeps_sessions_apart(self):
        thread, _, _ = beam.convert(row(), "100K", clock="slow")
        self.assertEqual(thread, beam.convert(row(), "100K", clock="slow")[0])
        self.assertNotEqual(
            thread, beam.convert(row(), "100K", clock="slow", seed=1)[0]
        )
        runs = thread["runs"]
        self.assertEqual(runs[0]["started_at"], "2025-03-01T09:00:00Z")
        self.assertEqual(runs[2]["started_at"], "2025-03-02T09:00:00Z")
        # A long answer takes longer to arrive.
        self.assertGreater(
            moment(runs[3]["ended_at"]) - moment(runs[3]["started_at"]),
            moment(runs[1]["ended_at"]) - moment(runs[1]["started_at"]),
        )
        self.assertEqual(
            (
                moment(runs[4]["started_at"]) - moment(runs[3]["ended_at"])
            ).total_seconds(),
            beam.PROBE_DELAY_SECONDS,
        )
        # A session dated before the previous one ended waits out a break instead.
        same_day, _, _ = beam.convert(row(second_date="March-01-2025"), "100K")
        gap = moment(same_day["runs"][2]["started_at"]) - moment(
            same_day["runs"][1]["ended_at"]
        )
        self.assertEqual(gap.total_seconds(), beam.MIN_SESSION_BREAK_SECONDS)
        shares = {
            name: sum(beam.idle_seconds(name, 1, "t", i) <= 300 for i in range(4000))
            / 4000
            for name in beam.CLOCKS
        }
        self.assertGreater(shares["fast"], 0.95)
        self.assertTrue(0.5 < shares["mixed"] < 0.65)
        self.assertLess(shares["slow"], 0.1)

    def test_split_depends_only_on_thread_ids_inside_each_size(self):
        entries = [
            {"thread_id": beam.thread_name(size, i), "size": size}
            for size, n in (("100K", 20), ("500K", 35), ("1M", 35))
            for i in range(1, n + 1)
        ]
        beam.assign_splits(entries)
        counts = {
            size: sum(e["split"] == "test" for e in entries if e["size"] == size)
            for size in ("100K", "500K", "1M")
        }
        self.assertEqual(counts, {"100K": 7, "500K": 12, "1M": 12})
        chosen = {e["thread_id"]: e["split"] for e in entries}
        again = [dict(e, split=None) for e in reversed(entries) if e["size"] != "500K"]
        beam.assign_splits(again)
        self.assertTrue(all(chosen[e["thread_id"]] == e["split"] for e in again))
        with self.assertRaises(ValueError):
            beam.assign_splits(entries, test_fraction=1)

    def test_build_writes_readable_shards_and_refuses_to_overwrite(self):
        rows = [
            ("100K", row("1")),
            ("100K", row("2")),
            ("500K", row("3", restart=True)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "beam"
            manifest = beam.build(rows, out, clock="fast")
            self.assertEqual(manifest, json.loads((out / "manifest.json").read_text()))
            self.assertEqual(
                (
                    manifest["thread_count"],
                    manifest["chat_runs"],
                    manifest["turn_count"],
                ),
                (3, 12, 27),
            )
            self.assertEqual(
                (
                    manifest["assessed_probes"],
                    manifest["unknown_probes"],
                    manifest["ambiguous_probes"],
                ),
                (6, 6, 3),
            )
            self.assertEqual(manifest["clock"]["scenario"], "fast")
            self.assertEqual(manifest["source"]["revision"], beam.SOURCE["revision"])
            self.assertEqual(
                manifest["split"]["counts"],
                {"100K/test": 1, "100K/train": 1, "500K/train": 1},
            )
            loaded = list(beam.iter_threads(out))
            self.assertEqual(
                [t["thread_id"] for t, _, _ in loaded],
                ["beam-100k-01", "beam-100k-02", "beam-500k-03"],
            )
            for thread, labels, entry in loaded:
                beam.verify(thread, labels)
                digest = hashlib.sha256(
                    (out / entry["trace_path"]).read_bytes()
                ).hexdigest()
                self.assertEqual(digest, entry["trace_sha256"])
            with self.assertRaises(SystemExit):
                beam.build(rows, out)
            other = Path(tmp) / "again"
            beam.build(rows, other, clock="fast")
            self.assertEqual(
                (out / "traces/beam-100k-01.jsonl.gz").read_bytes(),
                (other / "traces/beam-100k-01.jsonl.gz").read_bytes(),
            )
            (Path(tmp) / beam.SOURCE["files"]["100K"][0]).write_bytes(b"not parquet")
            with self.assertRaisesRegex(SystemExit, "pinned"):
                next(beam.read_source(tmp, ["100K"]))

    def test_converted_threads_feed_the_workload_fit(self):
        converted = [beam.convert(row(str(i)), "100K") for i in (1, 2)]
        threads = [Thread.from_dict(thread) for thread, _, _ in converted]
        labels = [label for _, rows, _ in converted for label in rows]
        artifact = fit_workload(
            threads, labels, ModelSpec("replay"), observation_end="2026-01-01T00:00:00Z"
        )
        scored = artifact["recipes"][0]
        # Chat runs carry no label and count as unsampled. Two questions per thread are unknown.
        self.assertEqual(scored["unlabeled_runs"], 8)
        self.assertEqual(
            (scored["historical_required"], scored["unknown_requirements"]), (6, 4)
        )
        self.assertIsInstance(WorkloadPolicy(artifact).recipes, dict)


if __name__ == "__main__":
    unittest.main()
