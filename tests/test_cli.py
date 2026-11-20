"""Exercise public commands as an installed-style executable."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_traces import sample

REPO = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        return subprocess.run(
            [sys.executable, "-m", "compache", *map(str, args)],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_trace_roundtrip_and_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "threads.jsonl"
            source.write_text(json.dumps(sample()) + "\n")
            rows, normalized, packet = (
                root / "rows.jsonl",
                root / "normalized.jsonl",
                root / "packet.json",
            )
            for args in [
                ("traces", "validate", source),
                ("traces", "export-langfuse", "--input", source, "--output", rows),
                ("traces", "normalize", "--input", rows, "--output", normalized),
                (
                    "dependencies",
                    "prepare",
                    "--threads",
                    normalized,
                    "--thread-id",
                    "t",
                    "--run-id",
                    "r",
                    "--output",
                    packet,
                ),
            ]:
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(normalized.read_text()), sample())
            prepared = json.loads(packet.read_text())
            self.assertEqual(prepared["target_run_id"], "r")
            self.assertEqual(prepared["request_coverage"], "complete")

    def test_errors_are_actionable_and_do_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad, output = root / "bad.jsonl", root / "no-output.json"
            for content in ["{bad json", "[]\n", "{}\n"]:
                bad.write_text(content)
                result = self.run_cli(
                    "traces", "normalize", "--input", bad, "--output", output
                )
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(json.loads(result.stderr)["valid"])
                self.assertFalse(output.exists())

    def test_missing_run_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "threads.jsonl", root / "packet.json"
            source.write_text(json.dumps(sample()) + "\n")
            result = self.run_cli(
                "dependencies",
                "prepare",
                "--threads",
                source,
                "--thread-id",
                "t",
                "--run-id",
                "missing",
                "--output",
                output,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            output.write_text("[]")
            labels = root / "labels.jsonl"
            labels.write_text("{}\n")
            result = self.run_cli(
                "dependencies", "validate", "--input", output, "--labels", labels
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("packet", result.stderr)



if __name__ == "__main__":
    unittest.main()
