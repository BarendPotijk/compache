import json
import unittest
from dataclasses import replace

from compache.optimizer import ModelSpec, Policy, Request, Source, ThreadState, expand, optimize, prepare
from compache.optimizer.solve import ingress_variants
from compache.traces import ContractError, Thread
from test_optimizer import END, thread


def retrieval(pid, run, text, doc="manual", version="v1"):
    return Source(pid, "retrieval", run, text, doc, version, pid)


def request(index, parts):
    return Request("thread", f"run-{index}", index, "test", tuple(parts))


def reconstruct(source, visible):
    delta = json.loads(source.text)
    base = visible[delta["copy_lines_from_part"]].text.splitlines(keepends=True)
    return "".join("".join(base[s[0]-1:s[1]]) if isinstance(s, list) else s
                   for s in delta["segments"])


class IngressTests(unittest.TestCase):
    def setUp(self):
        self.legacy = Policy(ModelSpec("test", context_tokens=5000, cache_ttl_seconds=1800).to_dict(),
                             "p", retrieval_grace_runs=0, protect_query_matches=False)
        self.policy = next(ingress_variants(self.legacy))
        self.first = ("original header\r\n" + "α" * 500 + "\r\n" + "original middle\n"
                      + "ø" * 500 + "\noriginal tail")
        self.second = ('revised header\r\n' + 'α' * 500 + '\r\n' + '["insert"]\t\\\n'
                       + 'ø' * 500 + '\nnew tail')

    def test_exact_reconstruction_append_stability_and_reopen(self):
        a = retrieval("a", 1, self.first)
        b = retrieval("b", 2, self.second)
        c = retrieval("c", 3, self.second + " — updated")
        state = ThreadState("thread")
        first = prepare(request(1, [a]), self.policy, state, 0)
        second = prepare(request(2, [a, b]), self.policy, state, 1)
        req = request(3, [a, b, c])
        third = prepare(req, self.policy, state, 2)
        self.assertEqual(first.wire, second.wire[:1])
        self.assertEqual(second.wire, third.wire[:2])
        self.assertEqual(third.referenced_part_ids, {"b", "c"})
        self.assertEqual(third.full_part_ids, {"a", "b", "c"})
        self.assertEqual(third.telemetry()["literal_retrievals"], 1)
        visible = {p.part_id: p for p in third.parts}
        for source in (b, c):
            projected = visible[source.part_id]
            self.assertEqual(reconstruct(projected, visible), source.text)
            encoded = json.loads(projected.text)
            self.assertEqual(encoded["copy_lines_from_part"], "a")
            self.assertEqual(expand(req, encoded["reopen_original"])["text"], source.text)

    def test_scope_and_changed_anchor(self):
        a = retrieval("a", 1, self.first)
        b = retrieval("b", 1, self.first, version="v2")
        c = retrieval("c", 1, self.first, doc="other")
        q = Source("q", "input", 1, self.first)
        state = ThreadState("thread")
        first = prepare(request(1, [a, b, c, q]), self.policy, state, 0)
        self.assertFalse(first.referenced_part_ids)
        self.assertEqual(first.parts, (a,b,c,q))
        d = retrieval("d", 2, self.second)
        original = prepare(request(2, [a,b,c,q,d]), self.policy, state, 1)
        self.assertIn("d", original.referenced_part_ids)
        changed = replace(a, text="all new")
        rewritten = prepare(request(2, [changed,b,c,q,d]), self.policy, state, 2)
        self.assertNotIn("d", rewritten.referenced_part_ids)
        self.assertEqual(rewritten.parts[-1], d)

    def test_anchor_and_dependent_eviction_and_admission_rollback(self):
        policy = replace(self.policy, model_spec=replace(self.policy.spec, context_tokens=1100).to_dict())
        a = retrieval("a", 1, self.first)
        b = retrieval("b", 1, self.second)
        state = ThreadState("thread")
        prior = prepare(request(1, [a,b]), policy, state, 0)
        self.assertTrue(prior.admitted)
        q = Source("q", "input", 2, "z" * 3000)
        blocked = prepare(request(2, [a,b,q]), policy, state, 3,
                          protected_ids={"b"})
        self.assertFalse(blocked.admitted)
        self.assertEqual(state.last_wire, prior.wire)
        self.assertFalse(state.removed)
        reduced = prepare(request(2, [a,b,q]), policy, state, 3)
        self.assertTrue(reduced.admitted)
        self.assertEqual(reduced.removed_part_ids, {"a", "b"})
        self.assertEqual(reduced.full_part_ids, {"q"})
        self.assertFalse(reduced.referenced_part_ids)

    def test_unknown_content_and_v1_backward_compatibility(self):
        short = replace(self.legacy, model_spec=replace(self.legacy.spec, cache_ttl_seconds=300).to_dict())
        self.assertFalse(list(ingress_variants(short)))
        self.assertNotIn("ingress_match_min_chars", short.to_dict())
        self.assertEqual(Policy.from_dict(short.to_dict()).hash, short.hash)
        self.assertEqual(Policy.from_dict(self.policy.to_dict()), self.policy)
        with self.assertRaises(ContractError):
            Policy.from_dict({**short.to_dict(), "ingress_match_min_chars": 256})
        a = retrieval("a", 1, self.first)
        b = retrieval("b", 1, self.second)
        q = Source("q", "input", 1, None)
        unknown = prepare(request(1, [a,b,q]), self.policy, ThreadState("thread"), 0)
        self.assertEqual(unknown.action, "observe_only")
        self.assertFalse(unknown.referenced_part_ids)

    def test_native_optimizer_trains_and_promotes_ingress_without_labels(self):
        fixtures = []
        for name in ("one", "two", "three"):
            raw = thread(tid=name,company=name).to_dict()
            for i, run in enumerate(raw["runs"]):
                source = run["retrievals"][0]
                source["document_id"] = "manual"
                source["text"] = self.second if i else self.first
            fixtures.append(Thread.from_dict(raw))
        result = optimize(fixtures, [], self.policy.spec, observation_end=END, max_pings=0)
        self.assertEqual(result["recommended_policy"]["schema_version"], "compache-policy-v2")
        self.assertFalse(result["selection"]["heldout_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
