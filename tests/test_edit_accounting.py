import unittest
from collections import Counter
from dataclasses import replace

from compache.optimizer.contracts import ModelSpec
from compache.optimizer.edit_accounting import (
    EditErrorHistory,
    InputAnchor,
    PartMeter,
    estimate_input,
)
from compache.optimizer.parts import Request, Source
from compache.optimizer.retention import (
    RetentionRecipe,
    matched_prefix_tokens,
    render_recipe,
)
from compache.traces import ContractError


class EditAccountingTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec("model")
        self.calls = Counter()

        def measure(p):
            self.calls[p] += 1
            return len(p.wire.encode("utf-8")) + len(p.version)

        self.meter = PartMeter(self.spec, measure, max_entries=2)
        self.question = Source("u1", "input", 1, "first")
        self.doc = Source(
            "d1", "retrieval", 1, "old source 🍎 " * 120, "document", "v1", "chunk"
        )
        self.answer = Source("a1", "output", 1, "an answer")
        self.next = Source("u2", "input", 2, "next")
        self.parts = (self.question, self.doc)

    def view(self, parts, index=1, *, archived=(), header=11, tid="T", meter=True):
        req = Request(tid, f"{tid}:{index}", index, self.spec.model_id, tuple(parts))
        return render_recipe(
            req,
            self.spec,
            RetentionRecipe(),
            archived_ids=archived,
            cut=False,
            overhead_tokens=header,
            part_meter=self.meter if meter else None,
        )

    def test_addition_then_replacement_with_handle_and_changed_header(self):
        original = self.view(self.parts)
        self.meter.retain(original)
        anchor = InputAnchor(original, original.tokens + 4981)
        extended = self.view((*self.parts, self.answer, self.next), 2, header=19)
        new = sum(extended.part_tokens[2:])
        estimated = estimate_input(extended, anchor)
        self.assertEqual(
            (
                estimated.added_tokens,
                estimated.removed_tokens,
                estimated.overhead_delta,
            ),
            (new, 0, 8),
        )
        self.assertEqual(estimated.tokens, anchor.reported_input_tokens + new + 8)
        self.assertEqual(estimated.to_dict()["anchor_run_id"], "T:1")
        compact = self.view(
            (*self.parts, self.answer, self.next), 2, archived={"d1"}, header=19
        )
        estimated = estimate_input(compact, anchor)
        handle_size = compact.part_tokens[1]
        self.assertEqual(estimated.added_tokens, new + handle_size)
        self.assertEqual(estimated.removed_tokens, original.part_tokens[1])
        self.assertEqual(
            estimated.tokens,
            anchor.reported_input_tokens
            + new
            + handle_size
            - original.part_tokens[1]
            + 8,
        )
        self.assertEqual(self.calls[self.question], 1)
        self.assertEqual(self.calls[self.doc], 1)

    def test_reorder_costs_no_edits_but_breaks_the_cache_prefix(self):
        original = self.view(self.parts)
        self.meter.retain(original)
        moved = self.view(reversed(self.parts), 2)
        estimate = estimate_input(moved, InputAnchor(original, 1409))
        self.assertEqual(
            (estimate.added_tokens, estimate.removed_tokens, estimate.tokens),
            (0, 0, 1409),
        )
        self.assertEqual(
            matched_prefix_tokens(original, moved, stable_overhead_tokens=11), 11
        )

    def test_unchanged_count_validated_by_content_and_metadata(self):
        original = self.view(self.parts)
        self.meter.retain(original)
        changed = replace(self.doc, version="v10")
        updated = self.view((self.question, changed), 2)
        est = estimate_input(updated, InputAnchor(original, 3897))
        self.assertEqual(
            (est.tokens, est.removed_tokens, est.added_tokens),
            (3898, original.part_tokens[1], updated.part_tokens[1]),
        )
        edited = replace(changed, text="edited source")
        changed_again = self.view((self.question, edited), 2)
        self.assertLess(
            estimate_input(changed_again, InputAnchor(original, 3897)).tokens,
            est.tokens,
        )
        self.assertEqual(
            (self.calls[self.doc], self.calls[changed], self.calls[edited]), (1, 1, 1)
        )

    def test_unsafe_subtraction_falls_back_even_when_masked_by_new_content(self):
        original = self.view(self.parts)
        self.meter.retain(original)
        replacer = replace(self.doc, text="new source" * 300)
        edited = self.view((self.question, replacer), 2)
        bad = estimate_input(edited, InputAnchor(original, original.part_tokens[1] - 1))
        self.assertEqual(
            (bad.fallback_reason, bad.tokens),
            ("removed_more_than_reported", edited.tokens),
        )
        empty = estimate_input(edited, InputAnchor(original, original.part_tokens[1]))
        self.assertEqual(
            (empty.fallback_reason, empty.tokens),
            ("exhausted_anchor_with_surviving_blocks", edited.tokens),
        )
        self.assertGreater(bad.added_tokens, bad.removed_tokens)

    def test_no_measurements_local_fallback_scope_and_effective_views(self):
        original = self.view(self.parts)
        foreign = self.view(self.parts, tid="Other")
        estimated = estimate_input(original)
        self.assertEqual(
            (estimated.tokens, estimated.fallback_reason),
            (original.tokens, "no_anchor"),
        )
        self.assertEqual(
            estimate_input(foreign, InputAnchor(original, 88)).fallback_reason,
            "incompatible_scope",
        )
        anchor = InputAnchor(replace(original, tokens=18000), 18844)
        self.assertEqual(anchor.local_tokens, original.tokens)
        self.assertEqual(
            estimate_input(replace(original, tokens=9999), anchor).tokens, 18844
        )
        with self.assertRaises(ContractError):
            InputAnchor(original, True)

    def test_sent_parts_survive_lru_pressure_even_when_reconstructed(self):
        items = tuple(Source(f"item-{i}", "input", 1, f"S{i}") for i in range(12))
        original = self.view(items)
        self.meter.retain(original)
        self.view([replace(p) for p in reversed(items)])
        self.assertEqual(sum(self.calls.values()), 12)
        for p in items:
            self.meter.count(replace(p))
        self.assertEqual(sum(self.calls.values()), 12)

    def test_observed_answer_survives_foreground_retain_and_other_previews(self):
        request = self.view(self.parts)
        self.meter.observe_count(self.answer, 131)
        self.meter.retain(request)
        for i in range(5):
            self.meter.count(Source(str(i), "input", 1, "extra"))
        with_answer = self.view((*self.parts, self.answer, self.next), 2)
        self.assertEqual(with_answer.part_tokens[-2], 131)
        self.assertEqual(with_answer.wire[-2], self.answer.wire)
        self.assertEqual(
            with_answer.part_wire_bytes[-2], len(self.answer.wire.encode())
        )
        self.assertEqual(self.calls[self.answer], 0)
        self.meter.retain(with_answer)
        for i in range(3):
            self.meter.observe_count(Source(str(i), "output", 1, "other"), i + 1)
        self.assertEqual(self.meter.count(self.answer), 131)
        changed = replace(self.answer, text="revised")
        self.assertEqual(self.meter.count(changed), len(changed.wire.encode()))
        self.assertEqual(self.calls[changed], 1)
        self.assertEqual(self.calls[self.answer], 0)

    def test_refined_size_preserves_unchanged_input_and_response_provenance(self):
        original = self.view(self.parts)
        anchor = InputAnchor(original, 935, response_id="resp-1")
        self.meter.observe_count(self.question, 163)
        corrected = self.view(self.parts)
        est = estimate_input(corrected, anchor)
        self.assertNotEqual(corrected.tokens, original.tokens)
        self.assertEqual(
            (est.added_tokens, est.removed_tokens, est.tokens), (0, 0, 935)
        )
        self.assertEqual(est.to_dict()["anchor_response_id"], "resp-1")
        for invalid in ("", " ", 1):
            with self.assertRaises(ContractError):
                InputAnchor(original, 935, response_id=invalid)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ContractError):
                self.meter.observe_count(self.doc, invalid)

    def test_residual_history_separates_true_repeat_from_reorder_and_header_drift(self):
        original = self.view(self.parts)
        a = InputAnchor(original, 9999)
        errors = EditErrorHistory()
        keep = estimate_input(original, a)
        self.assertEqual(errors.classify(keep), "unchanged")
        self.assertEqual(errors.classify(keep, header_changed=True), "header_change")
        self.assertEqual(
            errors.classify(estimate_input(self.view(self.parts, header=19), a)),
            "header_change",
        )
        reordered = estimate_input(self.view(tuple(reversed(self.parts))), a)
        self.assertEqual(errors.classify(reordered), "rewrite")
        self.assertEqual(reordered.added_tokens, 0)
        self.assertEqual(reordered.removed_tokens, 0)
        append = estimate_input(self.view((*self.parts, self.answer), 2), a)
        self.assertEqual(errors.classify(append), "append")
        local = estimate_input(original)
        self.assertEqual(errors.classify(local, header_changed=True), "local")
        errors.observe("unchanged", keep, 10006)
        errors.observe("unchanged", keep, None)
        errors.observe("unchanged", local, 99999)
        self.assertEqual(errors.allowance("unchanged"), (1, 7))
        self.assertEqual(errors.allowance("append"), (0, 0))
        self.assertEqual(errors.allowance("local"), (0, 0))

    def test_one_sided_nearest_rank_p95_includes_zero_and_keeps_latest_64(self):
        original = self.view(self.parts)
        keep = estimate_input(original, InputAnchor(original, 10000))
        errors = EditErrorHistory()
        for offset in range(1, 21):
            errors.observe("unchanged", keep, 10000 + offset)
        self.assertEqual(errors.allowance("unchanged"), (20, 19))
        for _ in range(60):
            errors.observe("append", keep, 9980)
        for _ in range(4):
            errors.observe("append", keep, 10030)
        self.assertEqual(errors.allowance("append"), (64, 30))
        for _ in range(60):
            errors.observe("append", keep, 10000)
        self.assertEqual(errors.allowance("append"), (64, 30))
        errors.observe("append", keep, 10000)
        self.assertEqual(errors.allowance("append"), (64, 0))
        with self.assertRaises(ContractError):
            errors.observe("append", keep, False)

    def test_default_meter_preserves_existing_projection_and_counts(self):
        req = Request("T", "r1", 1, self.spec.model_id, self.parts)
        metered = render_recipe(
            req, self.spec, RetentionRecipe(), part_meter=PartMeter(self.spec)
        )
        expected = render_recipe(req, self.spec, RetentionRecipe())
        self.assertEqual(metered, expected)
        with self.assertRaises(ContractError):
            render_recipe(
                req,
                self.spec,
                RetentionRecipe(),
                part_meter=PartMeter(ModelSpec("other")),
            )
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ContractError):
                PartMeter(self.spec, lambda _, value=invalid: value).count(
                    self.question
                )


if __name__ == "__main__":
    unittest.main()
