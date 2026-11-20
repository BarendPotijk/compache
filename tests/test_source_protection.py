import copy
import unittest

from compache.optimizer import (
    ModelSpec,
    Policy,
    Request,
    Source,
    ThreadState,
    expand,
    handle,
    prepare,
)
from compache.optimizer.gate import protected_sources
from compache.traces import ContractError


def ret(pid, run, text, doc="register", version="v1", chunk=""):
    return Source(pid, "retrieval", run, text, doc, version, chunk)


def request(query, *parts, run=10):
    return Request(
        "thread", "current", run, "test", (*parts, Source("query", "input", run, query))
    )


class SourceProtectionTests(unittest.TestCase):
    def test_recall_lifecycle_and_reopen_handle(self):
        spec = ModelSpec("test", context_tokens=2048)
        policy = Policy(spec.to_dict(), "x", retrieval_grace_runs=0)
        original = ret("a", 1, "Case AB123 evidence " + "x" * 2000)
        state = ThreadState("thread", removed={"a"})
        parts = [
            Source("i1", "input", 1, "first"),
            original,
            Source("o1", "output", 1, "answer"),
            Source("i2", "input", 2, "other"),
        ]
        first = prepare(
            Request("thread", "r2", 2, "test", tuple(parts)), policy, state, 0
        )
        self.assertNotIn("a", first.full_part_ids)
        parts.extend(
            [
                Source("o2", "output", 2, "answer"),
                Source("i3", "input", 3, "Revisit AB123"),
            ]
        )
        req3 = Request("thread", "r3", 3, "test", tuple(parts))
        recalled = prepare(req3, policy, state, 1)
        self.assertEqual(recalled.recalled_parts, 1)
        self.assertEqual(recalled.wire[: len(first.wire)], first.wire)
        self.assertIn("a", recalled.full_part_ids)
        alias = state.expansions[3]["a"]
        self.assertEqual(recalled.parts[-1].part_id, alias.part_id)
        same = prepare(req3, policy, state, 2)
        self.assertEqual(same.recalled_parts, 0)
        self.assertEqual(len(state.expansions[3]), 1)
        parts.extend(
            [Source("o3", "output", 3, "answer"), Source("i4", "input", 4, "other")]
        )
        req4 = Request("thread", "r4", 4, "test", tuple(parts))
        compacted = prepare(req4, policy, state, 400, force_cold=True)
        self.assertIn(alias.part_id, compacted.removed_part_ids)
        self.assertNotIn("a", compacted.full_part_ids)
        compacted_alias = next(p for p in compacted.parts if p.part_id == alias.part_id)
        self.assertIn(handle("thread", original), compacted_alias.text)
        self.assertEqual(expand(req4, handle("thread", original))["part_id"], "a")
        parts.extend(
            [
                Source("o4", "output", 4, "answer"),
                Source("i5", "input", 5, "Revisit AB123"),
            ]
        )
        again = prepare(
            Request("thread", "r5", 5, "test", tuple(parts)), policy, state, 401
        )
        self.assertEqual(again.recalled_parts, 1)
        self.assertIn("a", again.full_part_ids)
        self.assertEqual(again.wire[: len(compacted.wire)], compacted.wire)

    def test_recall_rollback_and_unknown_input(self):
        spec = ModelSpec("test", context_tokens=2048)
        policy = Policy(spec.to_dict(), "x", retrieval_grace_runs=0)
        original = ret("a", 1, "Case AB123 " + "x" * 2000)
        state = ThreadState("thread", removed={"a"})
        req = request("Revisit AB123", original, run=2)
        saved = copy.deepcopy(state)
        result = prepare(req, policy, state, 0, overhead_tokens=3000)
        self.assertFalse(result.admitted)
        self.assertEqual(state, saved)
        masked = Request(
            "thread", "r2", 2, "test", (original, Source("i2", "input", 2, None))
        )
        unknown = prepare(masked, policy, state, 0, protected_ids={"a"})
        self.assertEqual(unknown.action, "observe_only")
        self.assertEqual(unknown.recalled_parts, 0)
        self.assertNotIn("a", unknown.full_part_ids)

    def test_explicit_protection_and_existing_priced_expansion(self):
        spec = ModelSpec("test", context_tokens=2048)
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        original = ret("a", 1, "useful " + "x" * 2000)
        req = request("other", original, run=2)
        state = ThreadState("thread", removed={"a"})
        result = prepare(req, policy, state, 0, protected_ids={"a"})
        self.assertTrue(result.admitted)
        self.assertEqual(result.recalled_parts, 1)
        self.assertIn("a", result.full_part_ids)
        priced = ret("oracle-alias", 2, original.text)
        other = ThreadState("thread", removed={"a"}, expansions={2: {"a": priced}})
        result = prepare(req, policy, other, 0, protected_ids={"a"})
        self.assertEqual(result.recalled_parts, 0)
        self.assertIn("a", result.full_part_ids)
        partial = ret("partial-alias", 1, "useful")
        other = ThreadState("thread", removed={"a"}, expansions={1: {"a": partial}})
        no_recall = prepare(req, policy, copy.deepcopy(other), 0)
        self.assertNotIn("a", no_recall.full_part_ids)
        result = prepare(req, policy, other, 0, protected_ids={"a"})
        self.assertEqual(result.recalled_parts, 1)
        self.assertIn("a", result.full_part_ids)

    def test_citation_cannot_pin_future_occurrence_and_unrelated_answers(self):
        a = ret("a", 1, "amber fact", "register", chunk="one")
        b = ret("b", 3, "blue fact", "register", chunk="two")
        output = Source("o", "output", 2, "Case AB123 was addressed [register v1].")
        self.assertEqual(
            protected_sources(request("Revisit case AB123", a, output, b)), {"a"}
        )
        self.assertEqual(
            protected_sources(request("Unrelated question", a, output, b)), set()
        )

    def test_identifier_selects_needed_chunk_without_generic_matches(self):
        a = ret("a", 1, "Case AB123 original terms, annual procurement allowance.")
        b = ret("b", 2, "Case CD456 original terms, annual procurement allowance.")
        c = ret("c", 10, "Case ZZ999 original terms, annual procurement allowance.")
        self.assertEqual(
            protected_sources(
                request(
                    "Compare the annual procurement allowance for AB123 and ZZ999",
                    a,
                    b,
                    c,
                )
            ),
            {"a"},
        )
        self.assertEqual(
            protected_sources(
                request("What is the annual procurement allowance?", a, b, c)
            ),
            set(),
        )

    def test_ambiguous_citation_keeps_same_run_chunks_and_filters_versions(self):
        a = ret("a", 2, "first", "register", "v1", "one")
        b = ret("b", 2, "second", "register", "v1", "two")
        c = ret("c", 2, "third", "register", "v2", "three")
        out = Source("o", "output", 2, "Answer [register v1]")
        self.assertEqual(
            protected_sources(request("What was said in turn 2?", a, b, c, out)),
            {"a", "b"},
        )
        out2 = Source("o", "output", 2, "Answer [register v1 two]")
        self.assertEqual(
            protected_sources(request("What was said in turn 2?", a, b, c, out2)), {"b"}
        )

    def test_repeated_chunk_citation_and_explicit_version(self):
        a = ret("a", 1, "older", "register", "v1", "one")
        b = ret("b", 3, "different occurrence", "register", "v1", "one")
        c = ret("c", 4, "revision", "register", "v2", "one")
        out = Source("o", "output", 5, "[register v1 one]")
        self.assertEqual(
            protected_sources(request("What was said in turn 5?", a, b, c, out)),
            {"a", "b"},
        )
        self.assertEqual(protected_sources(request("Show register v2", a, b, c)), {"c"})

    def test_exact_duplicate_is_available_but_version_or_text_change_is_not(self):
        a = ret("a", 1, "Case AB123 evidence", "register", "v1", "one")
        duplicate = ret("b", 10, a.text, "register", "v1", "one")
        changed = ret("c", 10, a.text, "register", "v2", "one")
        self.assertEqual(protected_sources(request("Case AB123?", a, duplicate)), set())
        self.assertEqual(protected_sources(request("Case AB123?", a, changed)), {"a"})
        self.assertEqual(protected_sources(request("Inspect a", a, duplicate)), {"a"})

    def test_calibrated_estimate_recomputed_after_each_cut(self):
        spec = ModelSpec("test", context_tokens=2048)
        policy = Policy(
            spec.to_dict(), "x", retrieval_grace_runs=0, protect_query_matches=False
        )
        req = request("new", ret("a", 1, "x " * 1700), ret("b", 2, "y " * 1700), run=3)
        calls = []

        def estimator(payload, overhead):
            calls.append(payload)
            return payload * 2 + overhead

        result = prepare(
            req, policy, ThreadState("thread"), 0, token_estimator=estimator
        )
        self.assertTrue(result.admitted)
        self.assertEqual(result.removed_part_ids, {"a", "b"})
        self.assertGreater(calls[0], calls[1])
        self.assertGreater(calls[1], calls[2])
        self.assertEqual(result.tokens, calls[-1] * 2)
        with self.assertRaises(ContractError):
            prepare(
                req, policy, ThreadState("thread"), 0, token_estimator=lambda *_: -1
            )


if __name__ == "__main__":
    unittest.main()
