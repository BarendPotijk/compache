"""The solver picks the cut and the ping count by replayed cost on its own threads."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from helpers import END, labels_for, thread

from compache.cli import main
from compache.optimizer import solve_workload
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.workload import WorkloadPolicy
from compache.optimizer.workload_replay import replay_workload
from compache.traces import ContractError, Thread, write_threads

HEADER = {"system_prompt": "Answer from the supplied notes.", "default_tools": []}


def threads(gap, *, count=4, n=6, long_answers=False):
    out = []
    for i in range(count):
        data = thread(tid=f"t{i}", n=n, gap=gap).to_dict()
        if long_answers:
            for run in data["runs"]:
                run["output"]["text"] = "An observed final answer. " * 60
        out.append(Thread.from_dict(data))
    return out


class WorkloadSolveTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            "test",
            context_tokens=100_000,
            cache_ttl_seconds=300,
            cache_min_tokens=16,
            cache_block_tokens=8,
        )

    def solve(self, sample, labels=(), **options):
        return solve_workload(
            sample,
            labels,
            self.spec,
            observation_end=END,
            folds=2,
            share_headers=False,
            **HEADER,
            **options,
        )

    def test_ping_count_follows_how_users_return(self):
        # A run takes 30 s. Returns 10 s later are already warm, returns 370 s later
        # land just past the five-minute lifetime, and returns an hour later are out of reach.
        chosen = {
            gap: self.solve(threads(gap))["solve"]["chosen"] for gap in (40, 400, 3630)
        }
        self.assertEqual(chosen[40]["max_pings"], 0)
        self.assertGreater(chosen[400]["max_pings"], 0)
        self.assertEqual(chosen[3630]["max_pings"], 0)
        self.assertTrue(all(c["recipe"] is None for c in chosen.values()))

    def test_solved_policy_carries_its_cut_and_ping_count(self):
        sample = threads(400, long_answers=True)
        labels = [row for t in sample for row in labels_for(t, need_old=True)]
        policy = self.solve(sample, labels)
        solved = policy["solve"]
        chosen = solved["chosen"]
        # Every later turn needs the first document, so the cut keeps retrievals
        # and drops the long answers.
        self.assertEqual(
            (chosen["recipe"]["retrieval_runs"], chosen["recipe"]["output_runs"]),
            (None, 0),
        )
        self.assertEqual(chosen["coverage"], 1)
        complete = [c for c in solved["candidates"] if c["complete"]]
        self.assertEqual(
            chosen["expected_cost"], min(c["expected_cost"] for c in complete)
        )
        keep = next(
            c
            for c in solved["candidates"]
            if c["recipe"] is None and c["max_pings"] == 0
        )
        self.assertEqual(solved["keep_everything_cost"], keep["replayed_cost"])
        self.assertAlmostEqual(
            solved["saving_fraction"],
            1 - chosen["expected_cost"] / keep["replayed_cost"],
        )
        self.assertGreater(solved["saving_fraction"], 0)
        self.assertEqual(policy["planner_options"], {"max_pings": chosen["max_pings"]})
        admitted = [row["recipe"] for row in policy["recipes"] if row["eligible"]]
        self.assertEqual(admitted, [chosen["recipe"]])
        # Only the cheapest cut at each retention level is priced by replay.
        self.assertIn(
            "off_the_retention_frontier",
            {row["admission"] for row in policy["recipes"]},
        )
        loaded = WorkloadPolicy(policy)
        self.assertEqual(loaded.options["max_pings"], chosen["max_pings"])
        self.assertEqual(len(loaded.recipes), 1)
        self.assertEqual(WorkloadPolicy(policy, max_pings=0).options["max_pings"], 0)
        self.assertEqual(policy, self.solve(sample, labels))
        self.assertEqual(len(policy["training_threads"]), 4)

    def test_a_cut_must_beat_what_recovery_costs(self):
        sample = threads(3630, n=7)
        labels = []
        for t in sample:
            rows = labels_for(t)
            # Each turn needs the previous document, and the last needs the first again.
            for row, needed in zip(rows[1:], (1, 2, 3, 4, 5, 1)):
                source = t.runs[needed - 1].retrievals[0]
                requirement = row["task_requirements"][0]
                requirement["historical_context_needed"] = "yes"
                requirement["alternatives"] = [
                    {"sources": [{"part_id": source.part_id, "quote": source.text}]}
                ]
            labels.extend(rows)
        policy = self.solve(sample, labels)
        chosen = policy["solve"]["chosen"]
        self.assertEqual(chosen["recipe"]["retrieval_runs"], 1)
        self.assertAlmostEqual(chosen["coverage"], 5 / 6)
        self.assertGreater(chosen["recovery_cost"], 0)
        self.assertEqual(
            chosen["expected_cost"], chosen["replayed_cost"] + chosen["recovery_cost"]
        )
        # Without the band no cut loses evidence, and the solver falls back to one that keeps it all.
        strict = self.solve(sample, labels, tradeoff_floor=0.95)["solve"]["chosen"]
        self.assertTrue(strict["recipe"] is None or strict["coverage"] >= 0.95)

    def test_invalid_requests(self):
        sample = threads(400)
        with self.assertRaisesRegex(ContractError, "at least two"):
            self.solve(sample[:1])
        with self.assertRaisesRegex(ContractError, "ping_counts"):
            self.solve(sample, max_pings=2)
        with self.assertRaisesRegex(ContractError, "ping_counts"):
            self.solve(sample, ping_counts=(1, -1))

    def test_command_writes_a_policy_that_replay_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train, heldout = (
                threads(400),
                [
                    Thread.from_dict(
                        json.loads(
                            json.dumps(thread(tid="held", n=6, gap=400).to_dict())
                        )
                    )
                ],
            )
            write_threads(train, base / "train.jsonl")
            write_threads(heldout, base / "held.jsonl")
            (base / "model.json").write_text(json.dumps(self.spec.to_dict()))
            (base / "header.json").write_text(json.dumps(HEADER))

            def call(*args):
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    status = main([str(a) for a in args])
                self.assertEqual(status, 0)
                return json.loads(out.getvalue())

            common = ("--header", base / "header.json", "--observation-end", END)
            solved = call(
                "planner",
                "solve",
                "--threads",
                base / "train.jsonl",
                "--model-spec",
                base / "model.json",
                "--folds",
                "2",
                "--output",
                base / "policy.json",
                *common,
            )
            self.assertEqual(solved["training_threads"], 4)
            self.assertGreater(solved["chosen"]["max_pings"], 0)
            policy = json.loads((base / "policy.json").read_text())
            self.assertEqual(
                policy["planner_options"]["max_pings"], solved["chosen"]["max_pings"]
            )
            replayed = call(
                "planner",
                "replay",
                "--threads",
                base / "held.jsonl",
                "--policy",
                base / "policy.json",
                "--no-share-headers",
                "--output",
                base / "replay.json",
                *common,
            )
            self.assertTrue(replayed["candidate_complete"])
            report = json.loads((base / "replay.json").read_text())
            self.assertGreater(report["candidate"]["pings"], 0)
            direct = replay_workload(
                heldout, [], policy, observation_end=END, share_headers=False, **HEADER
            )
            self.assertEqual(direct["candidate"]["pings"], report["candidate"]["pings"])


if __name__ == "__main__":
    unittest.main()
