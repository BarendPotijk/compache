from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

from compache.optimizer import (
    ModelSpec,
    Policy,
    Request,
    Source,
    ThreadState,
    prepare,
    fit_profile,
    handle,
    replay,
)
from compache.optimizer.contracts import digest
from compache.optimizer.integration import decode_response, recover
from compache.optimizer.replay import Cache
from compache.traces import ContractError, Thread
from test_optimizer import thread, labels_for, END

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_live_comparison as runner


class LiveTests(unittest.TestCase):
    def test_plain_answers_are_kept_but_malformed_tools_fail(self):
        self.assertEqual(
            runner.decode_live_response("A quoted policy [source]."),
            ({"answer": "A quoted policy [source]."}, "plain_text"),
        )
        with self.assertRaises(ContractError):
            runner.decode_live_response('{"tool":"retrieve",')
        self.assertEqual(
            runner.decode_live_response('{"answer":"Verbatim answer"}}'),
            ({"answer": "Verbatim answer"}, "json_trailing_brace"),
        )
        with self.assertRaises(ContractError):
            runner.decode_live_response(
                '{"tool":"retrieve","document_id":"d","version":"v"}}'
            )

    def test_overhead_is_reserved_and_cached_without_fake_content(self):
        spec = ModelSpec("test", context_tokens=100)
        req = Request("t", "r", 1, "test", (Source("u", "input", 1, "hi"),))
        policy = Policy(spec.to_dict(), "x")
        state = ThreadState("t")
        self.assertFalse(prepare(req, policy, state, 0, overhead_tokens=100).admitted)
        self.assertIsNone(state.last_started_at)
        cache = Cache(spec, overhead_tokens=100)
        first = cache.charge(req.parts, 0, 2)
        second = cache.charge((replace(req.parts[0], text="changed"),), 1, 2)
        self.assertEqual(first["input_tokens_estimated"], 107)
        self.assertEqual(second["cache_read_tokens"], 100)

    def test_zero_recovery_does_not_restore_or_erase_misses(self):
        t = thread(body_size=500)
        labels = labels_for(t, True)
        spec = ModelSpec("test", context_tokens=2048)
        profile = fit_profile([t], labels, spec, observation_end=END)
        policy = Policy(
            spec.to_dict(),
            digest(profile),
            retrieval_grace_runs=0,
            target=0.01,
            cold_trigger=0.02,
            protect_query_matches=False,
        )
        result = replay(
            [t], labels, policy, profile, observation_end=END, recovery_overhead="zero"
        )
        self.assertGreater(result["summary"]["miss_runs"], 0)
        self.assertEqual(result["summary"]["expansion_cost"], 0)
        self.assertTrue(all(not x["recovered_part_ids"] for x in result["runs"]))

    def test_recovery_cannot_read_future_or_cross_thread(self):
        t = thread()
        req = Request.from_thread(t, t.runs[0].run_id)
        src = next(p for p in req.parts if p.kind == "retrieval")
        call = {"tool": "reopen", "handle": handle(t.thread_id, src)}
        self.assertEqual(recover(req, call, allow_reopen=True)["text"], src.text)
        with self.assertRaises(ContractError):
            recover(req, call, allow_reopen=False)
        with self.assertRaises(ContractError):
            recover(
                req,
                {"tool": "retrieve", "document_id": "doc-2", "version": "v1"},
                allow_reopen=True,
            )
        with self.assertRaises(ContractError):
            decode_response('{"answer":"hello", "tool":"retrieve"}')

    def test_render_prefix_extends_and_no_fixture_answer_leaks(self):
        t = thread()
        req = Request.from_thread(t, t.runs[0].run_id)
        rows = runner.messages_from_sources(req.parts)
        view = runner.render("nonce", rows, "compache")
        self.assertNotIn(t.runs[0].output.text, view)
        self.assertIn('"name":"reopen"', view)
        self.assertNotIn('"name":"reopen"', runner.render("nonce", rows, "opencode"))
        updated = runner.render(
            "nonce", rows + [{"role": "assistant", "text": "actual answer"}], "compache"
        )
        self.assertTrue(updated.startswith(view.split("\nTASK\n")[0]))

    def test_fake_full_arm_uses_its_own_answers_and_counts_summary(self):
        data = thread(n=3).to_dict()
        for r in data["runs"]:
            r["generation"]["model"] = "gpt-6-luna"
        t = Thread.from_dict(data)
        labels = labels_for(t)
        spec = ModelSpec("gpt-6-luna", context_tokens=50000)
        profile = fit_profile([t], labels, spec, observation_end=END)
        policy = Policy(spec.to_dict(), digest(profile))
        prompts = []

        def fake(view, output_dir, **kwargs):
            prompts.append(view)
            is_summary = "<template>" in view
            text = (
                "A real summary from the fake model"
                if is_summary
                else json.dumps({"answer": f"Actual answer {len(prompts)}"})
            )
            return {
                "ok": True,
                "answer": text,
                "view_sha256": runner.hashlib.sha256(view.encode()).hexdigest(),
                "usage": {
                    "input_tokens": 20000,
                    "cached_input_tokens": 10000,
                    "output_tokens": 20,
                },
                "wall_seconds": 0.1,
                "errors": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            out = base / "results"
            out.mkdir()
            (source / "threads.jsonl").write_text(json.dumps(data) + "\n")
            (source / "labels.jsonl").write_text(
                "".join(json.dumps(l) + "\n" for l in labels)
            )
            runner.dump(out / "policy.json", policy.to_dict())
            runner.dump(out / "profile.json", profile)
            plan = {
                "heldout_thread_id": t.thread_id,
                "source_hashes": {},
                "context_tokens": 50000,
                "overhead_tokens_for_planning_only": 0,
            }
            result = runner.one_arm(
                source, out, plan, "opencode", 1, base, "unused", invoke_override=fake
            )
            self.assertEqual(result["answers"], 3)
            rows = runner.read_rows(out / "runs/opencode-1/ledger.jsonl")
            self.assertEqual(sum(r["role"] == "summary" for r in rows), 2)
            self.assertNotIn(t.runs[0].output.text, prompts[0])
            self.assertIn("Actual answer 1", prompts[1])
            self.assertEqual(
                json.loads((out / "runs/opencode-1/complete.json").read_text())[
                    "counts"
                ]["answer"],
                3,
            )
            called = len(prompts)
            runner.one_arm(
                source,
                out,
                plan,
                "opencode",
                1,
                base,
                "unused",
                invoke_override=fake,
                resume=True,
            )
            self.assertEqual(len(prompts), called)
            self.assertEqual(
                len(runner.read_rows(out / "runs/opencode-1/ledger.jsonl")), len(rows)
            )


if __name__ == "__main__":
    unittest.main()
