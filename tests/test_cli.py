"""The command line: fit, replay, profile, check, packets and models."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from compache.cli import main
from compache.threads import Thread, write_threads

from helpers import labels_for, thread
from test_api import SPEC, labels, threads


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main([str(a) for a in args])
    return status, out.getvalue(), err.getvalue()


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        sample = threads()
        write_threads(sample, self.base / "threads.jsonl")
        (self.base / "labels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in labels(sample))
        )
        held = [Thread.from_dict(thread(tid="held", n=6, gap=400).to_dict())]
        write_threads(held, self.base / "held.jsonl")
        (self.base / "held-labels.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n"
                for t in held
                for row in labels_for(t, need_old=True)
            )
        )
        (self.base / "model.json").write_text(json.dumps(SPEC.to_dict()))
        (self.base / "system.txt").write_text("Answer from the supplied notes.")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fit_then_replay(self):
        base = self.base
        status, out, _ = run(
            "fit",
            base / "threads.jsonl",
            "--labels",
            base / "labels.jsonl",
            "--model",
            base / "model.json",
            "--system-prompt",
            base / "system.txt",
            "-o",
            base / "policy.json",
        )
        self.assertEqual(status, 0)
        self.assertIn("Fitted on 4 threads, 24 turns (24 labelled) for test.", out)
        self.assertIn(
            "Keep all questions, no earlier answers and all retrieved text.", out
        )
        self.assertIn(f"Wrote {base / 'policy.json'}", out)
        policy = json.loads((base / "policy.json").read_text())
        self.assertEqual(
            policy["header"]["system_prompt"], "Answer from the supplied notes."
        )
        status, out, _ = run(
            "replay",
            base / "held.jsonl",
            "--policy",
            base / "policy.json",
            "--labels",
            base / "held-labels.jsonl",
            "-o",
            base / "report.json",
        )
        self.assertEqual(status, 0)
        self.assertIn("1 threads, 6 turns.", out)
        self.assertIn("per 1,000 turns with the policy", out)
        self.assertIn(
            "100% of the 5 labelled needs for earlier context are in view.", out
        )
        self.assertIn(
            "candidate", json.loads((base / "report.json").read_text())["detail"]
        )
        status, out, _ = run(
            "replay", base / "held.jsonl", "--policy", base / "policy.json", "--json"
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)["turns"], 6)

    def test_fit_with_a_known_model_prints_dollars(self):
        status, out, _ = run(
            "fit",
            self.base / "threads.jsonl",
            "--model",
            "claude-sonnet-5-5",
            "-o",
            self.base / "p.json",
            "--json",
        )
        self.assertEqual(status, 0)
        result = json.loads(out)
        self.assertEqual(result["cost_per_1000_turns"]["unit"], "USD")
        self.assertIsNone(result["keep"])

    def test_profile_prints_a_calculator_link(self):
        status, out, _ = run(
            "profile",
            self.base / "threads.jsonl",
            "--labels",
            self.base / "labels.jsonl",
            "--model",
            "sonnet",
            "-o",
            self.base / "profile.json",
        )
        self.assertEqual(status, 0)
        self.assertIn("4 threads, 24 turns.", out)
        self.assertIn(
            "earlier retrieved text: 100% of turns, median 3 back (95th percentile 5).",
            out,
        )
        self.assertIn(
            "https://barendpotijk.github.io/compache/#v=6&model=sonnet&scenario=", out
        )
        saved = json.loads((self.base / "profile.json").read_text())
        self.assertEqual(saved["schema_version"], "compache-profile-v1")
        self.assertEqual(saved["calculator"]["scenario"]["plannedTurns"], 6)

    def test_check_validates_threads_and_labels(self):
        status, out, _ = run(
            "check", self.base / "threads.jsonl", "--labels", self.base / "labels.jsonl"
        )
        self.assertEqual(status, 0)
        self.assertTrue(out.startswith("Valid.\n4 threads, 24 turns."))
        self.assertIn(
            "Labels: 24 turns. Of the 20 assessed after a thread's first turn, 100% need earlier context.",
            out,
        )
        bad = self.base / "bad-labels.jsonl"
        bad.write_text(
            json.dumps({"thread_id": "t0", "run": 2, "needs": [["t0-r2-out"]]}) + "\n"
        )
        status, out, err = run("check", self.base / "threads.jsonl", "--labels", bad)
        self.assertEqual((status, out), (1, ""))
        self.assertIn("error: invalid label for t0-r2", err)
        status, _, err = run("check", self.base / "missing.jsonl")
        self.assertEqual(status, 1)
        self.assertIn("error:", err)

    def test_packets_sample_runs_to_label(self):
        status, out, _ = run(
            "packets",
            self.base / "threads.jsonl",
            "-o",
            self.base / "packets.jsonl",
            "--sample",
            7,
        )
        self.assertEqual(status, 0)
        self.assertIn("Wrote 7 of 20 runs", out)
        packets = [
            json.loads(line)
            for line in (self.base / "packets.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(packets), 7)
        self.assertTrue(
            all(p["target_run_index"] > 1 and p["request_parts"] for p in packets)
        )
        again = run(
            "packets",
            self.base / "threads.jsonl",
            "-o",
            self.base / "again.jsonl",
            "--sample",
            7,
        )
        self.assertEqual(again[0], 0)
        self.assertEqual(
            (self.base / "again.jsonl").read_text(),
            (self.base / "packets.jsonl").read_text(),
        )

    def test_models_lists_the_price_sheet(self):
        status, out, _ = run("models")
        self.assertEqual(status, 0)
        self.assertIn("claude-sonnet-5-5", out)
        self.assertIn("input $2, cache read $0.1, output $10, cache 5 min", out)
        status, out, _ = run("models", "--json")
        self.assertIn("gpt-6-luna", json.loads(out))

    def test_traces_kept_in_mlflow_or_an_opentelemetry_export_become_a_threads_file(
        self,
    ):
        from test_mlflow import tool_loop
        from test_otel import export, genai_spans

        traces = [t for n in range(3) for t in tool_loop(session=f"s{n}")]
        (self.base / "traces.json").write_text(json.dumps({"traces": traces}))
        spans = [s for n in range(3) for s in genai_spans(session=f"visit-{n}")]
        (self.base / "spans.jsonl").write_text(json.dumps(export(spans)))
        for source, name in (("mlflow", "traces.json"), ("otel", "spans.jsonl")):
            pulled = self.base / f"{source}.jsonl.gz"
            status, out, err = run("pull", source, self.base / name, "-o", pulled)
            self.assertEqual((status, err), (0, ""))
            self.assertIn("model calls in 3 sessions.", out)
            self.assertIn("Wrote 3 threads, 6 turns", out)
            self.assertIn(f"Next: compache fit {pulled}", out)
            status, out, _ = run("check", pulled)
            self.assertIn("3 threads, 6 turns.", out)
        status, _, err = run("pull", "otel")
        self.assertEqual(status, 1)
        self.assertIn("compache pull otel spans.jsonl", err)

    def test_an_unknown_model_is_an_error_not_a_guess(self):
        status, _, err = run(
            "fit", self.base / "threads.jsonl", "--model", "no-such-model"
        )
        self.assertEqual(status, 1)
        self.assertIn("no price sheet for model 'no-such-model'", err)


if __name__ == "__main__":
    unittest.main()
