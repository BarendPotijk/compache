import unittest

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.gate import Request, Source, handle
from compache.optimizer.integration import recover
from compache.optimizer.replay import assess
from compache.optimizer.retention import (
    RetentionRecipe,
    fit_recipe,
    fit_recipes,
    matched_prefix_tokens,
    recipe_grid,
    render_recipe,
    reopen_part,
)
from compache.optimizer.workload import WorkloadPolicy, fit_workload
from compache.traces import ContractError, Thread
from test_optimizer import END, labels_for, thread


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("test")
        self.thread = thread(n=5)
        self.req = Request.from_thread(self.thread, self.thread.runs[-1].run_id)

    def test_typed_cut_keeps_current_and_complete_retrieval_groups(self):
        # No retrieval at run four. The last completed retrieval group is run three.
        req = Request(
            "t",
            "r5",
            5,
            "test",
            tuple(p for p in self.req.parts if p.part_id != "t-r4-doc"),
        )
        view = render_recipe(
            req, self.spec, RetentionRecipe(2, 1, 1), protected_ids={"t-r1-in"}
        )
        self.assertEqual(
            view.full_part_ids,
            {
                "t-r1-in",
                "t-r3-in",
                "t-r3-doc",
                "t-r4-in",
                "t-r4-out",
                "t-r5-in",
                "t-r5-doc",
            },
        )
        self.assertEqual(
            [p.part_id for p in req.parts], [p.part_id for p in view.parts]
        )
        for p, q in zip(req.parts, view.parts):
            if p.part_id in view.full_part_ids:
                self.assertIs(q, p)
            else:
                self.assertIn(handle(req.thread_id, p), q.text)
                # Both input and output archives flow through the normal recovery protocol.
                restored = recover(
                    req,
                    {"tool": "reopen", "handle": handle(req.thread_id, p)},
                    allow_reopen=True,
                )
                self.assertEqual(
                    (restored["part_id"], restored["text"]), (p.part_id, p.text)
                )
        with self.assertRaises(ContractError):
            reopen_part(req, handle("other-thread", req.parts[0]))

    def test_checkpoint_does_not_reopen_or_roll_on_ordinary_new_turn(self):
        req = Request.from_thread(self.thread, self.thread.runs[2].run_id)
        cut = render_recipe(
            req, self.spec, RetentionRecipe(1, 1, 1), overhead_tokens=100
        )
        appended = render_recipe(
            self.req,
            self.spec,
            RetentionRecipe(0, 0, 0),
            archived_ids=cut.archived_ids,
            cut=False,
            overhead_tokens=100,
        )
        self.assertEqual(appended.archived_ids, cut.archived_ids)
        self.assertEqual(appended.wire[: len(cut.wire)], cut.wire)
        self.assertEqual(
            matched_prefix_tokens(cut, appended, stable_overhead_tokens=100), cut.tokens
        )
        recut = render_recipe(
            self.req,
            self.spec,
            RetentionRecipe(0, 0, 0),
            archived_ids=cut.archived_ids,
            protected_ids={"t-r1-out"},
            overhead_tokens=100,
        )
        self.assertIn("t-r1-out", recut.full_part_ids)
        self.assertIn("t-r1-doc", recut.archived_ids)
        self.assertEqual(
            matched_prefix_tokens(cut, recut, stable_overhead_tokens=100),
            100 + sum(cut.part_tokens[:2]),
        )

    def test_source_version_change_invalidates_prefix_even_with_identical_text(self):
        from dataclasses import replace

        first = render_recipe(
            self.req, self.spec, RetentionRecipe(), overhead_tokens=100
        )
        changed = list(self.req.parts)
        changed[1] = replace(changed[1], version="renumbered-citation")
        second = render_recipe(
            replace(self.req, parts=tuple(changed)),
            self.spec,
            RetentionRecipe(),
            overhead_tokens=100,
        )
        self.assertEqual(
            matched_prefix_tokens(first, second, stable_overhead_tokens=100),
            100 + first.part_tokens[0],
        )

    def test_fit_scores_prior_evidence_and_missing_labels(self):
        recipe = RetentionRecipe(None, None, 1)
        rows = labels_for(self.thread, need_old=True)
        scored = fit_recipe([self.thread], rows, self.spec, recipe)
        self.assertEqual(
            (scored["historical_retained"], scored["historical_required"]), (1, 4)
        )
        self.assertEqual(scored["direct_retained"], 1)
        self.assertFalse(scored["eligible"])
        # A turn without a label is unsampled. It is reported and does not gate.
        permissive = fit_recipe(
            [self.thread], rows[:-1], self.spec, recipe, retention_floor=0
        )
        self.assertEqual(permissive["unknown_changed_runs"], 1)
        self.assertEqual(permissive["unlabeled_changed_runs"], 1)
        self.assertEqual(permissive["unknown_changed_requirements"], 0)
        self.assertTrue(permissive["eligible"])
        # The original rule still rejects a recipe that changes any such turn.
        strict = fit_recipe(
            [self.thread],
            rows[:-1],
            self.spec,
            recipe,
            retention_floor=0,
            max_unknown_change_rate=0,
        )
        self.assertFalse(strict["eligible"])
        self.assertEqual(strict["admission"], "above_unknown_change_cap")
        unassessed = fit_recipe(
            [self.thread],
            [],
            self.spec,
            recipe,
            retention_floor=0,
            max_unknown_change_rate=1,
        )
        self.assertIsNone(unassessed["coverage"])
        self.assertFalse(unassessed["eligible"])
        full = fit_recipe([self.thread], [], self.spec, RetentionRecipe())
        self.assertTrue(full["eligible"])

    def test_combining_individually_passing_recipes_checks_sufficient_alternatives(
        self,
    ):
        labels = labels_for(self.thread, need_old=True)
        prior = self.thread.runs[0].input
        for row in labels[1:]:
            req = row["task_requirements"][0]
            req["source_kind"] = "mixed"
            req["alternatives"].append(
                {"sources": [{"part_id": prior.part_id, "quote": prior.text}]}
            )
        remove_inputs = RetentionRecipe(0, None, None)
        remove_docs = RetentionRecipe(None, None, 0)
        first, second = fit_recipes(
            [self.thread], labels, self.spec, [remove_inputs, remove_docs]
        )
        self.assertTrue(first["eligible"])
        self.assertTrue(second["individual_eligible"])
        self.assertFalse(second["eligible"])
        self.assertEqual(second["joint_envelope"]["coverage"], 0)

    def test_unknown_requirement_on_a_changed_turn_counts_as_lost(self):
        rows = labels_for(self.thread, need_old=True)
        rows[-1]["task_requirements"].append(
            dict(
                rows[-1]["task_requirements"][0],
                requirement_id="t2",
                source_kind="unknown",
                historical_context_needed="unknown",
                alternatives=[],
            )
        )
        # Dropping old inputs and outputs keeps every document a later turn needs.
        recipe = RetentionRecipe(0, 0, None)
        # By default the floor covers assessed requirements. Both figures are reported.
        assessed = fit_recipe(
            [self.thread], rows, self.spec, recipe, retention_floor=0.9
        )
        self.assertTrue(assessed["eligible"])
        self.assertEqual(assessed["unknowns"], "excluded")
        self.assertEqual(assessed["coverage_lower_bound"], 0.8)
        worst = fit_recipe(
            [self.thread],
            rows,
            self.spec,
            recipe,
            retention_floor=0.9,
            unknowns="worst_case",
        )
        self.assertEqual(
            (worst["historical_retained"], worst["historical_required"]), (4, 4)
        )
        self.assertEqual(worst["coverage"], 1)
        self.assertEqual(worst["unknown_changed_requirements"], 1)
        self.assertEqual(worst["coverage_lower_bound"], 0.8)
        self.assertFalse(worst["eligible"])
        self.assertEqual(worst["admission"], "unknown_requirements_could_break_floor")
        roomy = fit_recipe(
            [self.thread],
            rows,
            self.spec,
            recipe,
            retention_floor=0.8,
            unknowns="worst_case",
        )
        self.assertTrue(roomy["eligible"])
        # A recipe that changes nothing on the unknown turn is not charged for it.
        wide = fit_recipe(
            [self.thread],
            rows,
            self.spec,
            RetentionRecipe(5, 5, 5),
            retention_floor=1,
            unknowns="worst_case",
        )
        self.assertEqual(wide["unknown_changed_requirements"], 0)
        self.assertEqual(wide["coverage_lower_bound"], 1)
        with self.assertRaises(ContractError):
            fit_recipe([self.thread], rows, self.spec, recipe, unknowns="ignored")

    def test_grid_pairs_dialogue_and_retrieval_windows_below_the_longest_thread(self):
        grid = recipe_grid(144)
        self.assertEqual(len(grid), len(set(grid)))
        self.assertEqual(
            {r.retrieval_runs for r in grid}, {None, 0, 1, 2, 5, 10, 20, 50, 100}
        )
        self.assertEqual({r.input_runs for r in grid}, {None, 1, 2, 5, 10, 20, 50, 100})
        self.assertTrue(all(r.input_runs == r.output_runs for r in grid))
        self.assertNotIn(RetentionRecipe(None, None, None, True), grid)
        self.assertEqual(len(grid), 2 * (8 * 9 - 1))
        # The narrowest combination of two candidates is itself a candidate, so a
        # combined cut never needs a recipe the fit did not score.
        members = set(grid)

        def narrower(a, b):
            return b if a is None else a if b is None else min(a, b)

        for left in grid[::7]:
            for right in grid[::5]:
                meet = RetentionRecipe(
                    narrower(left.input_runs, right.input_runs),
                    narrower(left.output_runs, right.output_runs),
                    narrower(left.retrieval_runs, right.retrieval_runs),
                    left.protect_query_matches and right.protect_query_matches,
                )
                self.assertIn(meet, members)
        self.assertEqual(
            recipe_grid(1, protect_query_matches=(True,)),
            (RetentionRecipe(None, None, 0, True),),
        )
        self.assertEqual(max(r.retrieval_runs or 0 for r in recipe_grid(100)), 50)
        for invalid in (0, 2.5, True):
            with self.assertRaises(ContractError):
                recipe_grid(invalid)
        with self.assertRaises(ContractError):
            recipe_grid(10, protect_query_matches=())

    def long_answers(self):
        data = thread(n=5).to_dict()
        for run in data["runs"]:
            run["output"]["text"] = "An observed final answer. " * 40
        return Thread.from_dict(data)

    def test_cheapest_selection_admits_one_exercised_recipe(self):
        subject = self.long_answers()
        rows = labels_for(subject, need_old=True)
        grid = recipe_grid(len(subject.runs))
        scores = fit_recipes([subject], rows, self.spec, grid, select="cheapest")
        admitted = [s for s in scores if s["eligible"]]
        self.assertEqual(len(admitted), 1)
        # Every later turn needs the first document, so no retrieval window passes.
        # The narrowest dialogue window is the cheapest of the rest.
        self.assertEqual(
            admitted[0]["recipe"],
            {
                "input_runs": 1,
                "output_runs": 1,
                "retrieval_runs": None,
                "protect_query_matches": True,
            },
        )
        self.assertEqual(admitted[0]["admission"], "admitted")
        self.assertEqual(admitted[0]["joint_envelope"]["recipe"], admitted[0]["recipe"])
        passing = [s for s in scores if s["individual_eligible"]]
        self.assertGreater(len(passing), 1)
        self.assertTrue(all(s["recipe"]["retrieval_runs"] is None for s in passing))
        self.assertEqual(
            admitted[0]["projected_tokens"], min(s["projected_tokens"] for s in passing)
        )
        self.assertLess(admitted[0]["compact_fraction"], 1)
        self.assertEqual(
            {s["admission"] for s in scores},
            {"admitted", "cheaper_recipe_selected", "below_retention_floor"},
        )
        # With short messages every passing cut costs more than it removes,
        # because a placeholder outweighs the message it replaces.
        short = fit_recipes(
            [self.thread],
            labels_for(self.thread, need_old=True),
            self.spec,
            grid,
            select="cheapest",
        )
        self.assertFalse(any(s["eligible"] for s in short))
        self.assertIn("saves_no_tokens", {s["admission"] for s in short})
        # A window that first cuts on an unlabelled turn was never tested.
        untested = fit_recipes(
            [subject],
            rows[:-1],
            self.spec,
            [RetentionRecipe(3, 3, None)],
            select="cheapest",
        )[0]
        self.assertTrue(untested["individual_eligible"])
        self.assertEqual(untested["changed_historical_required"], 0)
        self.assertFalse(untested["eligible"])
        self.assertEqual(untested["admission"], "cut_met_no_assessed_requirement")
        with self.assertRaises(ContractError):
            fit_recipes([subject], rows, self.spec, grid, select="best")

    def test_fit_scores_match_rendering_every_turn(self):
        data = thread(n=7, body_size=40).to_dict()
        # One question names an earlier document, so query protection takes effect.
        data["runs"][4]["input"]["text"] = "Compare this item with doc-2 please."
        subject = Thread.from_dict(data)
        rows = labels_for(subject, need_old=True)
        rows[3]["task_requirements"][0]["historical_context_needed"] = "unknown"
        del rows[5]
        by_run = {row["target_run_id"]: row for row in rows}
        grid = recipe_grid(len(subject.runs))
        scores = fit_recipes([subject], rows, self.spec, grid)
        parts = Request.from_thread(subject, subject.runs[-1].run_id).parts
        for recipe, score in zip(grid, scores):
            archived, expected = (
                frozenset(),
                dict.fromkeys(
                    ("projected", "required", "retained", "changed", "unknown_changed"),
                    0,
                ),
            )
            for run in subject.runs:
                request = Request(
                    subject.thread_id,
                    run.run_id,
                    run.run_index,
                    "test",
                    parts[: len(run.generation.request_part_ids)],
                )
                view = render_recipe(request, self.spec, recipe, archived_ids=archived)
                archived = view.archived_ids
                grade = assess(by_run.get(run.run_id), view.full_part_ids)
                expected["projected"] += view.tokens
                expected["required"] += grade["historical_requirements_assessed"]
                expected["retained"] += grade["historical_requirements_retained"]
                expected["changed"] += bool(archived)
                expected["unknown_changed"] += (
                    bool(archived) and not grade["assessable"]
                )
            self.assertEqual(
                expected,
                {
                    "projected": score["projected_tokens"],
                    "required": score["historical_required"],
                    "retained": score["historical_retained"],
                    "changed": score["changed_runs"],
                    "unknown_changed": score["unknown_changed_runs"],
                },
                recipe,
            )
        tokens = {r: s["projected_tokens"] for r, s in zip(grid, scores)}
        self.assertNotEqual(
            tokens[RetentionRecipe(None, None, 2, True)],
            tokens[RetentionRecipe(None, None, 2, False)],
        )

    def test_workload_fit_searches_the_grid_and_loads_one_cut(self):
        subject = self.long_answers()
        rows = labels_for(subject, need_old=True)
        artifact = fit_workload([subject], rows, self.spec, observation_end=END)
        self.assertEqual(len(artifact["recipes"]), len(recipe_grid(5)))
        self.assertEqual(
            artifact["recipe_selection"],
            {
                "select": "cheapest",
                "unknowns": "excluded",
                "max_unknown_change_rate": None,
            },
        )
        policy = WorkloadPolicy(artifact)
        admitted = [
            (f"cut-{i}", RetentionRecipe(**row["recipe"]))
            for i, row in enumerate(artifact["recipes"])
            if row["eligible"]
        ]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(policy.recipes, dict(admitted))
        # A supplied list keeps its order and may admit several compatible cuts.
        supplied = fit_workload(
            [subject],
            rows,
            self.spec,
            observation_end=END,
            recipes=[RetentionRecipe(2, 2, None), RetentionRecipe(1, 1, None)],
        )
        self.assertEqual(supplied["recipe_selection"]["select"], "supplied")
        self.assertEqual(len(WorkloadPolicy(supplied).recipes), 2)

    def test_foreign_pins_hidden_sources_and_invalid_configuration(self):
        with self.assertRaises(ContractError):
            RetentionRecipe(retrieval_runs=True)
        with self.assertRaises(ContractError):
            render_recipe(
                self.req, self.spec, RetentionRecipe(), protected_ids={"impossible"}
            )
        with self.assertRaises(ContractError):
            render_recipe(
                self.req, self.spec, RetentionRecipe(), archived_ids={"t-r5-doc"}
            )
        with self.assertRaises(ContractError):
            render_recipe(
                Request("t", "r1", 1, "test", (Source("x", "input", 1, None),)),
                self.spec,
                RetentionRecipe(),
            )
        with self.assertRaises(ContractError):
            fit_recipes([self.thread, self.thread], [], self.spec, [RetentionRecipe()])


if __name__ == "__main__":
    unittest.main()
