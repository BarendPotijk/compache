import unittest

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.gate import Request, Source, handle
from compache.optimizer.integration import recover
from compache.optimizer.retention import (
    RetentionRecipe,
    fit_recipe,
    fit_recipes,
    matched_prefix_tokens,
    render_recipe,
    reopen_part,
)
from compache.traces import ContractError
from test_optimizer import labels_for, thread


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
        permissive = fit_recipe(
            [self.thread], rows[:-1], self.spec, recipe, retention_floor=0
        )
        self.assertEqual(permissive["unknown_changed_runs"], 1)
        self.assertFalse(permissive["eligible"])
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
