import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from compache.cli import main
from compache.onboarding import make_metadata_row, pseudonymize_thread_id
from compache.optimizer import WorkloadPolicy
from compache.optimizer.contracts import digest
from compache.workflow_presets import example_metadata, example_model_spec


def run(*args, success=True):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["setup", "--json", *map(str, args)])
    if success and code:
        raise AssertionError(err.getvalue())
    if not success and not code:
        raise AssertionError("invalid option combination succeeded")
    return json.loads((out if success else err).getvalue())


class SetupTests(unittest.TestCase):
    def test_install_default_does_not_touch_user_files_or_enable_optimizations(self):
        with patch.object(Path, "open", side_effect=AssertionError("private read")):
            preview = run("--workflow", "research")
            for bad in (("--labels", "x"), ("--metadata", "x"), ("--content-sample", "x"),
                        ("--demo", "--content-sample", "x", "--packets-dir", "y")):
                self.assertFalse(run(*bad, success=False)["valid"])
        self.assertFalse(any(preview["access"].values()))
        self.assertEqual(preview["active_actions"], {"compaction": "host", "keep_warm_pings": False})

    def test_demo_prices_the_mixed_policy_and_never_opens_customer_files(self):
        with patch.object(Path, "open", side_effect=AssertionError("file access")):
            demo = run("--workflow", "support", "--demo")
        c = demo["cache_comparison"]
        self.assertEqual(c["own"]["ping_calls"], 0)
        self.assertEqual(c["generic"]["ping_calls"], 13)
        self.assertAlmostEqual(c["savings_vs_generic_fraction"], 0.0548, places=4)
        self.assertFalse(any(demo["access"].values()))

    def test_single_thread_candidate_without_savings_cannot_overwrite_input(self):
        rows, cutoff = example_metadata("research")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            meta, spec, dest = (folder / s for s in ("counts", "spec", "policy"))
            meta.write_text(json.dumps(rows[0]) + "\n")
            spec.write_text(json.dumps(example_model_spec().to_dict()))
            options = ("--metadata", meta, "--model-spec", spec, "--observation-end", cutoff)
            with patch("compache.setup_dependencies._sample", side_effect=AssertionError("content read")):
                r = run(*options, "--policy-out", dest)
            self.assertEqual(r["cache_comparison"]["status"], "insufficient_evidence")
            self.assertFalse(WorkloadPolicy(json.loads(dest.read_text())).recipes)
            self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o600)
            self.assertFalse(run(*options, "--policy-out", dest, success=False)["valid"])
            meta.write_text(json.dumps(dict(rows[0], question="never echo this text")) + "\n")
            message = run(*options, success=False)["error"]
            self.assertNotIn("never echo", message)

    def test_combined_fit_excludes_content_threads_from_holdout(self):
        fx = Path(__file__).resolve().parents[1] / "examples/rag_workload"
        threads = [json.loads(s) for s in (fx / "threads.jsonl").read_text().splitlines()]
        labels = [json.loads(s) for s in (fx / "labels.jsonl").read_text().splitlines()]
        rows, cutoff = example_metadata("support")
        ids = list(dict.fromkeys(r["thread_id"] for r in rows))
        # Pick the thread that hash splitting would otherwise hold out.
        private_name = min((t["thread_id"] for t in threads), key=digest)
        sample = next(t for t in threads if t["thread_id"] == private_name)
        specs = json.loads((fx / "roomy-model.json").read_text())
        for item in rows:
            item["model_id"] = specs["model_id"]
            if item["thread_id"] == ids[0]:
                item["thread_id"] = private_name
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            p, lab, meta, out = (folder / s for s in ("sample", "labels", "counts", "out"))
            p.write_text(json.dumps(sample) + "\n")
            lab.write_text("".join(json.dumps(r) + "\n" for r in labels if r["thread_id"] == private_name))
            meta.write_text("".join(json.dumps(r) + "\n" for r in rows))
            opts = ("--content-sample", p, "--labels", lab, "--model-spec", fx / "roomy-model.json")
            assessment = run(*opts)
            self.assertNotIn("metadata", assessment)
            self.assertNotIn("cache_comparison", assessment)
            self.assertTrue(assessment["dependencies"]["recipe_results"])
            result = run(*opts, "--metadata", meta, "--observation-end", cutoff, "--policy-out", out)
            art = json.loads(out.read_bytes())
            self.assertEqual(result["mode"], "combined_fit")
            self.assertIn(digest(private_name), art["training_threads"])
            self.assertEqual(len(art["training_threads"]), result["metadata"]["threads"])
            self.assertEqual(len(art["recipes"]), 2)
            self.assertNotIn(sample["runs"][0]["input"]["text"], json.dumps(result))
            # A different alias for the same records cannot protect the held-out split.
            meta.write_text(meta.read_text().replace(private_name, "other-alias"))
            self.assertIn("same pseudonyms", run(*opts, "--metadata", meta, "--observation-end", cutoff, success=False)["error"])

    def test_host_helper_accepts_only_counters_and_opaque_keys(self):
        original = "private-customer"
        identifier = pseudonymize_thread_id(original, key=b"a" * 32)
        self.assertNotEqual(identifier, pseudonymize_thread_id(original, key=b"b" * 32))
        record = dict(next(iter(example_metadata("support")[0])), thread_id=identifier)
        self.assertEqual(make_metadata_row(**record), record)
        self.assertNotIn(original, json.dumps(record))
        with self.assertRaises(TypeError):
            make_metadata_row(**record, question=original)
