import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_optimizer import END, labels_for, thread

from compache.cli import main
from compache.optimizer.contracts import ModelSpec
from compache.optimizer.retention import recipe_grid
from compache.traces import Thread, write_threads


class WorkloadCLITests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.spec = ModelSpec(
            "test",
            context_tokens=100000,
            cache_ttl_seconds=300,
            cache_min_tokens=8,
            cache_block_tokens=4,
            bytes_per_token=1,
            read_multiplier=0.25,
            write_multiplier=1.25,
        )
        self.spec_path = self.write("model.json", self.spec.to_dict())
        self.header = self.write(
            "header.json",
            {
                "system_prompt": "Customer instructions <|endoftext|>",
                "default_tools": [{"name": "search"}, {"name": "open"}],
                "component_order": ["default_tools", "system_prompt"],
            },
        )
        self.header_args = [
            "planner",
            "headers",
            "--header",
            self.header,
            "--model-spec",
            self.spec_path,
        ]

    def write(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value))
        return path

    def call(self, args, expected=0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main([str(arg) for arg in args])
        self.assertEqual(status, expected, err.getvalue())
        return json.loads((err if expected else out).getvalue())

    def test_price_new_thread_header_overlap_with_order_and_expiry(self):
        report = self.call([*self.header_args, "--additional-input-tokens", 40])
        self.assertTrue(report["new_thread_used_other_thread"])
        self.assertEqual(
            report["header_token_ruler"],
            {"source": "estimated_from_bytes", "bytes_per_token": 1},
        )
        self.assertEqual(
            report["default_tool_tokens"], sum(report["default_tool_breakdown"])
        )
        self.assertEqual(len(report["default_tool_breakdown"]), 2)
        reads = report["static_tokens"] // 4 * 4
        self.assertEqual(report["modeled_reusable_header_tokens"], reads)
        self.assertEqual(
            report["reusable_default_tool_tokens"], report["default_tool_tokens"]
        )
        self.assertEqual(
            report["cold_input_cost"], (40 + report["static_tokens"]) * 1.25
        )
        self.assertEqual(report["input_savings"], reads)
        expired = self.call([*self.header_args, "--gap-seconds", "300"])
        self.assertEqual(expired["new_thread_prefix_status"], "expired")
        self.assertEqual(expired["input_savings"], 0)
        self.assertEqual(expired["new_thread_input_cost"], expired["cold_input_cost"])

    def test_explicit_and_model_encoding_choose_right_ruler(self):
        encoder = SimpleNamespace(
            name="test-encoding", encode_ordinary=lambda text: list(text.encode())
        )
        api = SimpleNamespace(
            encoding_for_model=Mock(return_value=encoder),
            get_encoding=Mock(return_value=encoder),
        )
        with patch.dict(sys.modules, tiktoken=api):
            explicit = self.call([*self.header_args, "--encoding", "test-encoding"])
            mapped = self.call([*self.header_args, "--encoding", "model"])
            api.encoding_for_model.assert_called_once_with("test")
            api.get_encoding.assert_called_once_with("test-encoding")
            self.assertEqual(explicit["count_source"], "native_token_ids")
            self.assertEqual(mapped["header_token_ruler"]["encoding"], "test-encoding")
            api.encoding_for_model.side_effect = KeyError("unknown")
            self.assertIn(
                "Pass --encoding",
                self.call([*self.header_args, "--encoding", "model"], 1)["error"],
            )
        with patch.dict(sys.modules, tiktoken=None):
            self.assertIn(
                "Install tiktoken",
                self.call([*self.header_args, "--encoding", "model"], 1)["error"],
            )

    def test_validates_header_input_and_nonnegative_workload(self):
        for header in (
            {},
            {"system_prompt": "s", "default_tools": [], "component_order": "wrong"},
            {"system_prompt": "s", "default_tools": [], "new-tools": {}},
        ):
            with self.subTest(header=header):
                self.header.write_text(json.dumps(header))
                self.assertFalse(self.call(self.header_args, 1)["valid"])
        self.header.write_text(json.dumps({"system_prompt": "s", "default_tools": []}))
        for extra in (
            ["--additional-input-tokens", "-1"],
            ["--additional-input-tokens", "100001"],
            ["--gap-seconds", "nan"],
        ):
            with self.subTest(extra=extra):
                self.assertFalse(self.call([*self.header_args, *extra], 1)["valid"])

    def test_fit_and_frozen_policy_heldout_replay(self):
        paths = {}
        for tid in ("train", "eval"):
            t = thread(tid=tid, n=3)
            docs = self.base / f"{tid}.jsonl"
            labels = self.base / f"{tid}-labels.jsonl"
            write_threads([t], docs)
            labels.write_text("".join(json.dumps(row) + "\n" for row in labels_for(t)))
            paths[tid] = ["--threads", docs, "--labels", labels]
        artifact, report = self.base / "policy.json", self.base / "replay.json"
        recipes = self.write(
            "recipes.json", [{"input_runs": 1, "output_runs": 1, "retrieval_runs": 1}]
        )
        route = ["--provider", "test-route", "--cache-key", "west"]
        train_id = thread(tid="train", n=3).runs[0].run_id
        counts = self.write(
            "train-calls.json",
            {
                "complete_run_ids": [train_id],
                "events": [
                    {
                        "run_id": train_id,
                        "call_id": str(i),
                        "role": role,
                        "status": "succeeded",
                    }
                    for i, role in enumerate(("foreground_step", "answer", "ping"))
                ],
            },
        )
        fit = self.call(
            [
                "planner",
                "fit",
                *route,
                *paths["train"],
                "--model-spec",
                self.spec_path,
                "--observation-end",
                END,
                "--header",
                self.header,
                "--output",
                artifact,
                "--recipes",
                recipes,
                "--calls-per-run",
                counts,
            ]
        )
        self.assertEqual(fit["training_threads"], 1)
        self.assertEqual(fit["retention_floor"], 0.95)
        self.assertEqual(fit["candidate_recipes"], 1)
        self.assertEqual(fit["recipe_selection"]["select"], "supplied")
        policy = json.loads(artifact.read_text())
        self.assertEqual(policy["demand"]["by_position"]["1"]["calls_per_run"], [2])
        self.assertIsNotNone(policy["header_traffic"])
        self.assertEqual(policy["header_traffic"]["provider"], "test-route")
        self.assertEqual(policy["header_token_ruler"]["source"], "estimated_from_bytes")
        params = [
            *route,
            "--observation-end",
            END,
            "--header",
            self.header,
            "--output",
            report,
            "--policy",
            artifact,
            "--lookahead",
            "2",
            "--branches",
            "1",
            "--max-pings",
            "0",
        ]
        result = self.call(["planner", "replay", *paths["eval"], *params])
        self.assertTrue(result["candidate_complete"] and result["baseline_complete"])
        replay = json.loads(report.read_text())
        self.assertEqual(replay["heldout_threads"], 1)
        self.assertEqual(
            replay["shared_cache_scope"],
            {k: policy["header_traffic"][k] for k in ("provider", "cache_key_digest")},
        )
        self.assertEqual(replay["candidate"]["answers"], 3)
        self.assertIsNotNone(result["savings_fraction"])
        self.assertEqual(replay["header_token_ruler"]["source"], "estimated_from_bytes")
        self.assertIn(
            "training thread",
            self.call(["planner", "replay", *paths["train"], *params], 1)["error"],
        )

    def test_fit_without_recipes_searches_the_grid(self):
        data = thread(tid="train", n=5).to_dict()
        for run in data["runs"]:
            run["output"]["text"] = "An observed final answer. " * 40
        t = Thread.from_dict(data)
        docs, labels = self.base / "train.jsonl", self.base / "train-labels.jsonl"
        write_threads([t], docs)
        labels.write_text(
            "".join(json.dumps(row) + "\n" for row in labels_for(t, need_old=True))
        )
        common = [
            "planner",
            "fit",
            "--threads",
            docs,
            "--labels",
            labels,
            "--model-spec",
            self.spec_path,
            "--observation-end",
            END,
        ]
        fit = self.call([*common, "--output", self.base / "grid.json"])
        self.assertEqual(fit["candidate_recipes"], len(recipe_grid(5)))
        self.assertEqual(
            fit["recipe_selection"],
            {
                "select": "cheapest",
                "unknowns": "excluded",
                "max_unknown_change_rate": None,
            },
        )
        self.assertEqual(fit["eligible_recipes"], 1)
        (admitted,) = fit["admitted"]
        self.assertIsNone(admitted["recipe"]["retrieval_runs"])
        self.assertEqual(admitted["coverage_lower_bound"], 1)
        self.assertLess(admitted["compact_fraction"], 1)
        rows = json.loads((self.base / "grid.json").read_text())["recipes"]
        self.assertEqual(sum(row["eligible"] for row in rows), 1)
        every = self.call(
            [*common, "--select", "supplied", "--output", self.base / "all.json"]
        )
        self.assertGreater(every["eligible_recipes"], 1)
        strict = self.call(
            [
                *common,
                "--unknowns",
                "worst_case",
                "--max-unknown-change-rate",
                "0",
                "--output",
                self.base / "strict.json",
            ]
        )
        self.assertEqual(
            strict["recipe_selection"],
            {
                "select": "cheapest",
                "unknowns": "worst_case",
                "max_unknown_change_rate": 0,
            },
        )
        self.assertEqual(strict["eligible_recipes"], 1)


if __name__ == "__main__":
    unittest.main()
