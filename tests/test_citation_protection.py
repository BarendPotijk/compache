import unittest

from compache.optimizer import Request, Source
from compache.optimizer.parts import protected_sources


def retrieval(pid, run, text, doc="register", version="v1", chunk=""):
    return Source(pid, "retrieval", run, text, doc, version, chunk)


def ask(query, *parts):
    return Request("t", "r", 10, "model", (*parts, Source("query", "input", 10, query)))


class CitationResolutionTests(unittest.TestCase):
    def test_coalesced_unrelated_claim_does_not_fan_out(self):
        history = [retrieval(f"old{i}", i, f"Case ZZ{i} unrelated details", "pricing", "v2") for i in range(1, 6)]
        wanted = retrieval("wanted", 6, "Case AA111 warranty details", "warranty", "v1", "chunk-a")
        unrelated = retrieval("other", 6, "Case BB222 pricing details", "pricing", "v2")
        answer = Source("answer", "output", 6, "AA111 has a warranty. BB222 has pricing terms [warranty v1 chunk-a; pricing v2].")
        result = protected_sources(ask("Explain AA111", *history, wanted, unrelated, answer))
        self.assertEqual(result, {"wanted"})

    def test_explicit_chunk_and_occurrences_remain_protected(self):
        a = retrieval("first", 1, "older", chunk="section-x")
        b = retrieval("second", 3, "newer", chunk="section-x")
        c = retrieval("third", 3, "other", chunk="section-y")
        out = Source("out", "output", 3, "A finding [register v1 section-x]")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, c, out)), {"first", "second"})
        out = Source("out", "output", 3, "A finding [register v1 second]")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, c, out)), {"second"})

    def test_bare_occurrence_and_chunk_references(self):
        a = retrieval("source-a", 1, "AA111 details", chunk="chunk-x")
        b = retrieval("source-b", 3, "AA111 later details", chunk="chunk-x")
        future = retrieval("source-future", 5, "future", chunk="chunk-x")
        out = Source("out", "output", 3, "AA111 finding [source-a].")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, out, future)), {"source-a"})
        out = Source("out", "output", 3, "AA111 finding [chunk-x].")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, out, future)), {"source-a", "source-b"})
        collision = retrieval("other", 2, "other", "different", chunk="chunk-x")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, collision, out, future)), set())

    def test_bare_reference_respects_claim_relevance_and_prose(self):
        a = retrieval("source-a", 1, "generic evidence", "alpha")
        b = retrieval("source-b", 1, "generic evidence", "beta")
        out = Source("out", "output", 2, "AA111 result [source-a]. BB222 result [source-b].")
        self.assertEqual(protected_sources(ask("Explain AA111", a, b, out)), {"source-a"})
        out = Source("out", "output", 2, "AA111 mentions source-a in prose [unrelated note].")
        self.assertEqual(protected_sources(ask("Explain AA111", a, out)), set())

    def test_versions_are_scoped_to_each_document(self):
        a = retrieval("alpha1", 3, "first", "alpha", "v1")
        b = retrieval("alpha2", 3, "wrong", "alpha", "v2")
        c = retrieval("beta1", 3, "wrong", "beta", "v1")
        d = retrieval("beta2", 3, "second", "beta", "v2")
        out = Source("out", "output", 3, "Findings [alpha v1; beta v2].")
        self.assertEqual(protected_sources(ask("What did turn 3 say?", a, b, c, d, out)), {"alpha1", "beta2"})

    def test_document_only_citation_uses_answer_time_provenance(self):
        a = retrieval("old", 1, "older observation")
        b = retrieval("at_answer", 4, "new observation")
        c = retrieval("future", 6, "future observation")
        out = Source("out", "output", 4, "The observation uses register.")
        self.assertEqual(protected_sources(ask("What did turn 4 say?", a, b, out, c)), {"at_answer"})

    def test_natural_language_query_matches_source_content(self):
        a = retrieval("old", 1, "ordinary account information")
        b = retrieval("matched", 3, "volcano eruption evacuation plan")
        c = retrieval("other", 4, "different administrative facts")
        out = Source("out", "output", 4, "The volcano eruption evacuation plan is documented [register v1].")
        self.assertEqual(protected_sources(ask("Describe the volcano eruption evacuation plan", a, b, c, out)), {"matched"})

    def test_separate_claims_skip_unrelated_citation(self):
        a = retrieval("a", 2, "Case AA111 details", "alpha")
        b = retrieval("b", 2, "Case BB222 details", "beta")
        out = Source("out", "output", 2, "AA111 was resolved [alpha v1]. BB222 remains open [beta v1].")
        self.assertEqual(protected_sources(ask("Revisit AA111", a, b, out)), {"a"})


if __name__ == "__main__":
    unittest.main()
