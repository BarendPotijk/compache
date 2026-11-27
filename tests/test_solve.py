"""The solver picks the cut and the ping count by replayed cost on its own threads."""

import unittest
from dataclasses import replace

from helpers import END, labels_for, thread

from compache import Policy, replay
from compache.optimizer import solve_workload
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.session import WorkloadPolicy
from compache.threads import ContractError, Thread

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

    def test_the_cache_lifetime_follows_how_users_return(self):
        hour = replace(self.spec, cache_ttl_seconds=3600, write_multiplier=2.0)
        solved = {
            gap: self.solve(threads(gap), lifetimes=(hour,))
            for gap in (40, 1500, 7000, 20_000)
        }
        chosen = {
            gap: artifact["solve"]["chosen"]["cache_ttl_seconds"]
            for gap, artifact in solved.items()
        }
        # Back in seconds: the default is warm anyway. After 25 minutes only the longer one is,
        # after two hours with pings. After many hours neither is, so the cheaper write wins.
        self.assertEqual(chosen, {40: 300, 1500: 3600, 7000: 3600, 20_000: 300})
        pings = {g: a["solve"]["chosen"]["max_pings"] for g, a in solved.items()}
        self.assertEqual(pings, {40: 0, 1500: 0, 7000: 1, 20_000: 0})
        # Every candidate is priced under both.
        for artifact in solved.values():
            priced = [c["cache_ttl_seconds"] for c in artifact["solve"]["candidates"]]
            self.assertEqual(priced.count(300), priced.count(3600))
        # The policy carries what it won under, and what the provider does by default.
        slow = solved[1500]
        self.assertEqual(slow["model_spec"], hour.to_dict())
        self.assertEqual(slow["standard_spec"], self.spec.to_dict())
        self.assertEqual(WorkloadPolicy(slow).spec, hour)
        self.assertNotIn("standard_spec", solved[40])
        # Its saving is against what an application does on its own: the default lifetime, no pings.
        alone = self.solve(threads(1500))["solve"]["keep_everything_cost"]
        self.assertEqual(slow["solve"]["keep_everything_cost"], alone)
        self.assertGreater(slow["solve"]["saving_fraction"], 0.2)
        # Nothing was cut here, so caching alone is all there is to it.
        self.assertEqual(slow["solve"]["caching_alone"], slow["solve"]["chosen"])

    def test_a_lifetime_changes_nothing_but_the_cache(self):
        hour = replace(self.spec, cache_ttl_seconds=3600, write_multiplier=2.0)
        for lifetimes, refused in (
            ((replace(hour, read_multiplier=0.2),), "nothing else"),
            ((replace(self.spec, write_multiplier=1.0),), "another one"),
            ((hour, replace(hour, write_multiplier=1.5)), "another one"),
        ):
            with self.assertRaisesRegex(ContractError, refused):
                self.solve(threads(1500), lifetimes=lifetimes)

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

    def test_replay_rejects_threads_used_only_to_choose_the_cut(self):
        demand = threads(3630)
        evidence = thread(tid="separate-evidence", n=6, gap=3630)
        artifact = self.solve(
            demand, labels_for(evidence), retention_threads=lambda: iter([evidence])
        )
        with self.assertRaisesRegex(ContractError, "training thread"):
            replay(Policy(artifact), [evidence])

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
        # Raising the minimum excludes this cheaper cut.
        strict = self.solve(sample, labels, retention_floor=0.95)["solve"]["chosen"]
        self.assertTrue(strict["recipe"] is None or strict["coverage"] >= 0.95)

    def test_independent_labels_allow_a_cut_but_missing_labels_do_not(self):
        sample = threads(3630)
        labels = [r for t in sample for r in labels_for(t)]
        solved = self.solve(sample, labels)["solve"]["chosen"]
        self.assertIsNotNone(solved["recipe"])
        self.assertIsNone(solved["coverage"])
        # The last run's demand could include any part archived earlier.
        sparse = [r for r in labels if r["target_run_id"] != sample[0].runs[-1].run_id]
        self.assertIsNone(self.solve(sample, sparse)["solve"]["chosen"]["recipe"])
        self.assertIsNone(self.solve(sample)["solve"]["chosen"]["recipe"])

    def test_live_cache_meters_change_the_fitted_schedule(self):
        from datetime import datetime, timedelta

        def fit_share(share, caching=True, **options):
            sample = []
            for t in threads(400, n=12):
                data = t.to_dict()
                prev = size = 0
                for i, run in enumerate(data["runs"]):
                    if not i:
                        start = datetime.fromisoformat(run["started_at"])
                    else:
                        start += timedelta(seconds=120 if i % 4 == 0 else 400)
                    run["started_at"] = run["generation"]["started_at"] = (
                        start.isoformat()
                    )
                    run["ended_at"] = run["generation"]["ended_at"] = (
                        start + timedelta(seconds=30)
                    ).isoformat()
                    size += sum(
                        self.spec.part_tokens(p["text"])
                        for p in (run["input"], *run["retrievals"])
                    )
                    reads = int(prev * share) if i % 4 == 0 else 0
                    run["generation"]["usage"] = dict(
                        input=size,
                        output=5,
                        cache_read=reads,
                        cache_write=size - reads if caching else 0,
                    )
                    prev = size
                    size += self.spec.part_tokens(run["output"]["text"])
                sample.append(Thread(data))
            return self.solve(sample, **options)

        low, high = fit_share(0.1), fit_share(1)
        self.assertEqual(low["cache_reads"]["observations"], 8)
        self.assertAlmostEqual(low["cache_reads"]["share"], 0.1, places=3)
        self.assertEqual(low["planner_options"]["max_pings"], 0)
        self.assertGreater(high["planner_options"]["max_pings"], 0)
        # Both timely returns follow a declared prefix edit. They give no
        # evidence of the provider hit share on an otherwise matching prefix.
        overlap = {f"t{i}-r{n}": 0.5 for i in range(4) for n in (4, 8)}
        edited = fit_share(0.1, next_endpoint_reuse_fraction=overlap)
        self.assertEqual(edited["cache_reads"], {"share": None, "observations": 0})
        # An application that never switched caching on reports zero reads and
        # zero writes. Those are not misses, so the provider's rules are assumed.
        off = fit_share(0, caching=False)
        self.assertEqual(off["cache_reads"], {"share": None, "observations": 0})
        self.assertEqual(
            off["planner_options"]["max_pings"], high["planner_options"]["max_pings"]
        )

    def test_invalid_requests(self):
        sample = threads(400)
        with self.assertRaisesRegex(ContractError, "at least two"):
            self.solve(sample[:1])
        with self.assertRaisesRegex(ContractError, "ping_counts"):
            self.solve(sample, max_pings=2)
        with self.assertRaisesRegex(ContractError, "ping_counts"):
            self.solve(sample, ping_counts=(1, -1))


if __name__ == "__main__":
    unittest.main()
