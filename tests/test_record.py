"""A record of conversations without their text, and a fit from it."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import compache
from compache.cli import main
from compache.record import Recorder, is_record, load, load_calls, thread_name
from compache.threads import load_threads

START = 1_790_000_000
QUESTION = "What does the next section of the contract say about it?"
ANSWER = "It says the supplier carries the cost."


def passage(n):
    return f"Clause {n:03d} of the agreement. " + "wording " * 700


def conversation(tid, turns=9, gap=3600):
    """A thread with its text, and labels: every turn after the second needs what was retrieved two turns back."""
    runs, labels = [], []
    for n in range(1, turns + 1):
        at = START + n * gap
        runs.append(
            {
                "started_at": at,
                "ended_at": at + 6,
                "input": QUESTION,
                "retrievals": [passage(n), "Unrelated. " + "filler " * 700],
                "output": ANSWER,
            }
        )
        needs = [[f"{tid}:{n - 2}:retrieval:1"]] if n > 2 else []
        labels.append({"thread_id": tid, "run": n, "needs": needs})
    return {"thread_id": tid, "runs": runs}, labels


def record_of(folder, pairs, **options):
    """The same conversations as a record would hold them: sizes and labels."""
    recorder = Recorder(folder, **options)
    for thread, labels in pairs:
        for n, (run, label) in enumerate(zip(thread["runs"], labels), 1):
            refs = [f"R{n - 2}.1"] if label["needs"] else []
            recorder.turn(
                thread["thread_id"],
                n,
                started_at=run["started_at"],
                ended_at=run["ended_at"],
                model="claude-sonnet-5-5",
                sizes={
                    "input": len(run["input"].encode()),
                    "retrievals": [len(t.encode()) for t in run["retrievals"]],
                    "output": len(run["output"].encode()),
                    "system": 31,
                },
                asked=n > 1,
                refs=refs if n > 1 else None,
            )
    return recorder


class RecordTests(unittest.TestCase):
    def test_a_record_holds_sizes_times_counters_and_labels_and_no_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [conversation("user-4711@example.com")]
            recorder = record_of(tmp, pairs)
            recorder.call(
                "user-4711@example.com",
                2,
                "ping",
                at=START + 2000,
                model="claude-sonnet-5-5",
                usage={
                    "input": 9000,
                    "output": 1,
                    "cache_read": 8990,
                    "cache_write": 0,
                },
                cost=0.0004,
            )
            self.assertTrue(is_record(tmp))
            self.assertFalse(is_record(Path(tmp) / "turns.jsonl"))
            written = (Path(tmp) / "turns.jsonl").read_text() + (
                Path(tmp) / "calls.jsonl"
            ).read_text()
            for word in ("contract", "supplier", "Clause", "example.com", "4711"):
                self.assertNotIn(word, written)
            name = thread_name("user-4711@example.com")
            threads, labels = load(tmp)
            self.assertEqual([t["thread_id"] for t in threads], [name])
            first = threads[0]["runs"][0]
            self.assertEqual(first["input"], {"bytes": len(QUESTION.encode())})
            self.assertEqual(threads[0]["system_bytes"], 31)
            self.assertEqual(labels[0], {"thread_id": name, "run": 1, "needs": []})
            self.assertEqual(labels[2]["needs"], [[f"{name}:1:retrieval:1"]])
            self.assertEqual(
                threads[0]["runs"][2]["cited_part_ids"], [f"{name}:1:retrieval:1"]
            )
            self.assertEqual(load_calls(tmp)[0]["kind"], "ping")
            # The sizes come back as stand-in text of the same length.
            loaded = load_threads(tmp, model="claude-sonnet-5-5")[0]
            self.assertEqual(
                [len(p.text) for p in loaded.runs[0].retrievals],
                [len(t.encode()) for t in pairs[0][0]["runs"][0]["retrievals"]],
            )
            self.assertEqual(set(loaded.runs[0].input.text), {"."})

    def test_what_was_not_recorded_whole_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(tmp, hashed=False)
            sizes = {"input": 20, "retrievals": [4000], "output": 30}

            def turn(tid, n, **more):
                recorder.turn(
                    tid,
                    n,
                    started_at=START + 60 * n,
                    ended_at=START + 60 * n + 5,
                    model="m",
                    sizes=sizes,
                    **more,
                )

            for n in (1, 2, 3, 5):
                turn("gap", n)
            for n in (4, 5):
                turn("late", n)
            turn("noted", 1)
            turn("noted", 2, asked=True, refs=["R1.1", "A1", "R2.1", "R9.9", "Q2"])
            turn("noted", 3, asked=True)
            turn("noted", 4, asked=True, refs=["Q1"], older=True)
            turn(
                "noted",
                4,
                asked=True,
                refs=[],
                usage={"input": 9, "output": 1, "cache_read": 0, "cache_write": 0},
                read_reported=False,
            )
            threads, labels = load(tmp)
            by_id = {t["thread_id"]: t for t in threads}
            # Up to the first gap, and not at all without a first turn.
            self.assertEqual(len(by_id["gap"]["runs"]), 3)
            self.assertNotIn("late", by_id)
            noted = {
                row["run"]: row["needs"]
                for row in labels
                if row["thread_id"] == "noted"
            }
            # The last line written for a turn counts. A turn asked without an answer has no label.
            self.assertEqual(
                noted, {1: [], 2: [["noted:1:retrieval:1", "noted:1:output"]], 4: []}
            )
            runs = by_id["noted"]["runs"]
            self.assertEqual(
                runs[1]["cited_part_ids"],
                ["noted:1:retrieval:1", "noted:2:retrieval:1"],
            )
            self.assertEqual(runs[3]["generation"], {"cache_read_reported": False})
            load_threads(tmp)

    def test_a_reply_that_pointed_past_its_list_is_marked_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(tmp, hashed=False)
            sizes = {"input": 20, "retrievals": [], "output": 30}
            for n in (1, 2):
                recorder.turn(
                    "t",
                    n,
                    started_at=START + n,
                    ended_at=START + n,
                    model="m",
                    sizes=sizes,
                    asked=True,
                    refs=["A1"],
                    older=True,
                )
            self.assertEqual([row["needs"] for row in load(tmp)[1]], [[], None])


class FitTests(unittest.TestCase):
    def test_a_fit_from_sizes_decides_as_a_fit_from_the_text(self):
        pairs = [conversation(f"t{i}") for i in range(5)]
        threads = [t for t, _ in pairs]
        labels = [row for _, rows in pairs for row in rows]
        options = dict(model="claude-sonnet-5-5", share_headers=False, lifetimes=())
        from_text = compache.fit(
            threads, labels, system_prompt="." * 31, **options
        ).summary
        with tempfile.TemporaryDirectory() as tmp:
            record_of(tmp, pairs, hashed=False)
            policy = compache.fit(tmp, **options)
            from_sizes = policy.summary
            for key in ("turns", "labelled_turns", "max_pings", "coverage"):
                self.assertEqual(from_sizes[key], from_text[key], key)
            # Keeping what matches the new question cannot be judged without words, so it is left out.
            self.assertFalse(from_sizes["keep"].pop("matching_retrieved"))
            from_text["keep"].pop("matching_retrieved")
            self.assertEqual(from_sizes["keep"], from_text["keep"])
            self.assertAlmostEqual(
                from_sizes["cost_per_1000_turns"]["policy"],
                from_text["cost_per_1000_turns"]["policy"],
                places=6,
            )
            self.assertIsNotNone(from_sizes["keep"])
            # Another record is priced with its own labels.
            with tempfile.TemporaryDirectory() as other:
                record_of(other, [conversation("h0"), conversation("h1")])
                held = compache.replay(policy, other)
            self.assertEqual(held["turns"], 18)
            self.assertEqual(held["needs_history"], 14)
            self.assertGreater(held["saving"], 0.2)
            # The command line takes the folder for threads and labels alike.
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main(["check", tmp])
            self.assertEqual(status, 0)
            self.assertIn("5 threads, 45 turns.", out.getvalue())
            self.assertIn("Labels: 45 turns.", out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["profile", tmp, "--json"])
            measured = json.loads(out.getvalue())["measured"]
            self.assertEqual(
                (measured["labelled_turns"], measured["turns_needing_history"]),
                (45, 35),
            )


if __name__ == "__main__":
    unittest.main()
