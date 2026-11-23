import unittest

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.parts import Request, Source, handle
from compache.optimizer.retention import assess
from compache.optimizer.retention import (
    RetentionRecipe,
    evidence_windows,
    fit_recipe,
    fit_recipes,
    matched_prefix_tokens,
    recipe_grid,
    render_recipe,
    reopen_part,
    size_ladder,
)
from compache.optimizer.session import WorkloadPolicy, fit_workload
from compache.threads import ContractError, Thread
from helpers import END, labels_for, thread


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("test")
        self.thread = thread(n=5)
        self.req = Request.from_thread(self.thread, self.thread.runs[-1].run_id)

    def test_cited_sources_expire_causally_in_fit_and_render(self):
        data = self.thread.to_dict()
        first = data["runs"][0]["retrievals"][0]["part_id"]
        data["runs"][0]["cited_part_ids"] = [first]
        # The later mention cannot restore an already archived occurrence.
        data["runs"][3]["cited_part_ids"] = [first]
        annotated = Thread(data)
        for age, expected in (
            (0, [False] * 4),
            (1, [True, False, False, False]),
            (2, [True, True, False, False]),
            (None, [True] * 4),
        ):
            with self.subTest(age=age):
                recipe = RetentionRecipe(retrieval_runs=0, cited_answer_runs=age)
                archived = set()
                visible = []
                for r in annotated.runs:
                    req = Request.from_thread(annotated, r.run_id)
                    view = render_recipe(req, self.spec, recipe, archived_ids=archived)
                    archived = view.archived_ids
                    if r.run_index > 1:
                        visible.append(first in view.full_part_ids)
                self.assertEqual(visible, expected)
                scored = fit_recipe(
                    [annotated], labels_for(annotated, need_old=True), self.spec, recipe
                )
                self.assertEqual(scored["historical_retained"], sum(visible))
        self.assertEqual(
            self.req.parts, Request.from_thread(annotated, self.req.run_id).parts
        )
        observed = Request.from_thread(annotated, self.req.run_id)
        self.assertEqual(
            [handle("t", p) for p in self.req.parts],
            [handle("t", p) for p in observed.parts],
        )

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
                # Both input and output archives reopen by their handle.
                restored = reopen_part(req, handle(req.thread_id, p))
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

    def test_grid_gives_each_kind_its_own_window_below_the_longest_thread(self):
        grid = recipe_grid(144)
        self.assertEqual(len(grid), len(set(grid)))
        windows = {None, 0, 1, 2, 5, 10, 20, 50, 100}
        for name in ("input_runs", "output_runs", "retrieval_runs"):
            self.assertEqual({getattr(r, name) for r in grid}, windows)
        # User and assistant messages are cut independently.
        self.assertIn(RetentionRecipe(None, 0, None, False), grid)
        self.assertIn(RetentionRecipe(5, 50, 2, True), grid)
        self.assertNotIn(RetentionRecipe(None, None, None, False), grid)
        # Protection only varies where a retrieval window can cut.
        self.assertFalse(
            any(r.protect_query_matches for r in grid if r.retrieval_runs is None)
        )
        self.assertEqual(len(grid), 9 * 9 * (1 + 2 * 8) - 1)
        # The narrowest combination of two candidates is itself a candidate, so a
        # combined cut never needs a recipe the fit did not score.
        members = set(grid)

        def narrower(a, b):
            return b if a is None else a if b is None else min(a, b)

        for flags in ((True, False), (True,), (False,)):
            sample = recipe_grid(20, protect_query_matches=flags)
            members = set(sample)
            for left in sample[::7]:
                for right in sample[::5]:
                    meet = RetentionRecipe(
                        narrower(left.input_runs, right.input_runs),
                        narrower(left.output_runs, right.output_runs),
                        narrower(left.retrieval_runs, right.retrieval_runs),
                        left.protect_query_matches and right.protect_query_matches,
                    )
                    self.assertIn(meet, members)
        # A thread without retrievals needs no retrieval windows.
        chat = recipe_grid(144, kinds={"input", "output"})
        self.assertEqual(len(chat), 9 * 9 - 1)
        self.assertTrue(
            all(r.retrieval_runs is None and not r.protect_query_matches for r in chat)
        )
        self.assertEqual(
            recipe_grid(1, kinds={"retrieval"}, protect_query_matches=(True,)),
            (RetentionRecipe(None, None, 0, True),),
        )
        self.assertEqual(max(r.retrieval_runs or 0 for r in recipe_grid(100)), 50)
        for invalid in (0, 2.5, True):
            with self.assertRaises(ContractError):
                recipe_grid(invalid)
        with self.assertRaises(ContractError):
            recipe_grid(10, protect_query_matches=())
        with self.assertRaises(ContractError):
            recipe_grid(10, kinds={"summary"})

    def long_answers(self):
        data = thread(n=5).to_dict()
        for run in data["runs"]:
            run["output"]["text"] = "An observed final answer. " * 40
        return Thread.from_dict(data)

    def test_cheapest_selection_admits_one_exercised_recipe(self):
        subject = self.long_answers()
        rows = labels_for(subject, need_old=True)
        grid = recipe_grid(len(subject.runs))
        # A generator is enough: threads are scored one at a time.
        scores = fit_recipes(iter([subject]), rows, self.spec, grid, select="cheapest")
        admitted = [s for s in scores if s["eligible"]]
        self.assertEqual(len(admitted), 1)
        # Every later turn needs the first document, so no retrieval window passes.
        # Dropping the long answers saves tokens. Dropping the short questions
        # would cost some, because a placeholder outweighs each of them.
        self.assertEqual(
            admitted[0]["recipe"],
            {
                "input_runs": None,
                "output_runs": 0,
                "retrieval_runs": None,
                "protect_query_matches": False,
                "keep_under_tokens": 0,
            },
        )
        self.assertEqual(admitted[0]["min_thread_coverage"], 1)
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
            {
                "admitted",
                "cheaper_recipe_selected",
                "saves_no_tokens",
                "below_retention_floor",
            },
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
        data = thread(n=8, body_size=40).to_dict()
        # One question names an earlier document, so query protection takes effect.
        data["runs"][5]["input"]["text"] = "Compare this item with doc-2 please."
        # Retrieval windows count groups: one run has none and one has two chunks.
        data["runs"][3]["retrievals"] = []
        extra = dict(
            data["runs"][5]["retrievals"][0], part_id="t-r6-doc-b", chunk_id="two"
        )
        data["runs"][5]["retrievals"].append(extra)
        history = []
        for run in data["runs"]:
            current = [run["input"]["part_id"]] + [
                p["part_id"] for p in run["retrievals"]
            ]
            run["generation"]["request_part_ids"] = history + current
            history = history + current + [run["output"]["part_id"]]
        subject = Thread.from_dict(data)
        rows = labels_for(subject, need_old=True)
        rows[3]["task_requirements"][0]["historical_context_needed"] = "unknown"
        del rows[5]
        by_run = {row["target_run_id"]: row for row in rows}
        # Questions are 15 tokens, answers 13 and documents 31, so each size
        # threshold spares a different kind.
        grid = recipe_grid(len(subject.runs), keep_under_tokens=(0, 14, 20))
        self.assertEqual(size_ladder([subject], self.spec), (0, 13, 15))
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
        # Sparing the short answers changes what a cut of both removes.
        self.assertNotEqual(
            tokens[RetentionRecipe(0, 0, None, False)],
            tokens[RetentionRecipe(0, 0, None, False, 14)],
        )
        self.assertEqual(
            tokens[RetentionRecipe(None, 0, None, False, 14)],
            scores[0]["canonical_tokens"],
        )

    def test_workload_fit_searches_the_grid_and_loads_one_cut(self):
        subject = self.long_answers()
        rows = labels_for(subject, need_old=True)
        artifact = fit_workload([subject], rows, self.spec, observation_end=END)
        candidates = recipe_grid(
            5,
            keep_under_tokens=size_ladder([subject], self.spec),
            extra_windows=evidence_windows([subject], rows),
        )
        self.assertEqual(len(artifact["recipes"]), len(candidates))
        self.assertEqual(
            artifact["recipe_selection"],
            {
                "select": "cheapest",
                "tradeoff_floor": None,
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

    def test_a_recipe_under_the_floor_must_pay_for_what_it_loses(self):
        subject = thread(n=7)
        rows = labels_for(subject)
        # Each turn needs the previous document, and the last needs the first again.
        for row, needed in zip(rows[1:], (1, 2, 3, 4, 5, 1)):
            source = subject.runs[needed - 1].retrievals[0]
            requirement = row["task_requirements"][0]
            requirement["historical_context_needed"] = "yes"
            requirement["alternatives"] = [
                {"sources": [{"part_id": source.part_id, "quote": source.text}]}
            ]
        recent = RetentionRecipe(None, None, 1, True)
        # The labels say which look-backs matter: one group for most turns, six
        # for the turn that returns to the first document.
        self.assertEqual(
            evidence_windows([subject], rows),
            {"input": (0,), "output": (0,), "retrieval": (1, 6)},
        )
        reach = [
            fit_recipe([subject], rows, self.spec, RetentionRecipe(None, None, w))
            for w in (5, 6)
        ]
        self.assertEqual([r["historical_retained"] for r in reach], [5, 6])
        self.assertIn(
            RetentionRecipe(None, None, 6, True),
            recipe_grid(7, extra_windows={"retrieval": (1, 6)}),
        )
        with self.assertRaises(ContractError):
            recipe_grid(7, extra_windows={"summary": (3,)})
        grid = recipe_grid(len(subject.runs))
        strict = fit_recipes([subject], rows, self.spec, grid, select="cheapest")
        scored = strict[grid.index(recent)]
        self.assertEqual(
            (scored["historical_retained"], scored["historical_required"]), (5, 6)
        )
        self.assertEqual(scored["admission"], "below_retention_floor")
        self.assertFalse(any(s["eligible"] for s in strict))
        traded = fit_recipes(
            [subject], rows, self.spec, grid, select="cheapest", tradeoff_floor=0.8
        )
        chosen = traded[grid.index(recent)]
        # One lost requirement costs one more request and the document it reopens.
        self.assertEqual(chosen["admission"], "admitted")
        self.assertGreater(chosen["recovery_tokens"], 0)
        self.assertEqual(
            chosen["expected_tokens"],
            chosen["projected_tokens"] + chosen["recovery_tokens"],
        )
        self.assertLess(chosen["expected_tokens"], chosen["canonical_tokens"])
        # A window of zero loses every requirement and is not worth what it saves.
        none = traded[grid.index(RetentionRecipe(None, None, 0, True))]
        self.assertEqual(none["admission"], "below_retention_floor")
        # Charging recovery to every turn stops a cut that only pays on paper.
        sparse = fit_recipes(
            [subject],
            rows[-1:],
            self.spec,
            [recent],
            select="cheapest",
            retention_floor=1,
            tradeoff_floor=0,
        )[0]
        self.assertEqual(
            sparse["expected_tokens"],
            sparse["projected_tokens"] + 7 * sparse["recovery_tokens"],
        )
        self.assertEqual(sparse["admission"], "saves_no_tokens")
        with self.assertRaises(ContractError):
            fit_recipes([subject], rows, self.spec, grid, tradeoff_floor=0.99)

    def test_workload_fit_skips_retrieval_windows_for_a_chat_without_retrievals(self):
        data = thread(n=6).to_dict()
        history = []
        for run in data["runs"]:
            run["retrievals"] = []
            run["output"]["text"] = "An observed final answer. " * 40
            run["generation"]["request_part_ids"] = history + [run["input"]["part_id"]]
            history = history + [run["input"]["part_id"], run["output"]["part_id"]]
        chat = Thread.from_dict(data)
        artifact = fit_workload([chat], [], self.spec, observation_end=END)
        self.assertEqual(
            len(artifact["recipes"]),
            len(
                recipe_grid(
                    6,
                    kinds={"input", "output"},
                    keep_under_tokens=size_ladder([chat], self.spec),
                )
            ),
        )
        self.assertTrue(
            all(row["recipe"]["retrieval_runs"] is None for row in artifact["recipes"])
        )

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
