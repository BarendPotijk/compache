"""The README's quickstart, the sample it runs on, and the figures drawn from committed data."""

import contextlib
import io
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import compache

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "quickstart"


class QuickstartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = compache.fit(
            SAMPLE / "threads.jsonl.gz",
            SAMPLE / "labels.jsonl",
            model="claude-sonnet-5-5",
        )

    def test_the_readme_shows_what_the_fit_prints(self):
        readme = (ROOT / "README.md").read_text("utf-8")
        guide = (ROOT / "docs" / "guide.md").read_text("utf-8")
        for line in self.policy.describe().splitlines():
            self.assertIn(line, readme)
            self.assertIn(line, guide)
        summary = self.policy.summary
        self.assertEqual(
            (summary["threads"], summary["turns"], summary["labelled_turns"]),
            (4, 71, 71),
        )
        self.assertEqual(summary["keep"]["retrieved"], 3)

    def test_the_sample_is_a_chat_log_with_simple_labels(self):
        threads = compache.load_threads(SAMPLE / "threads.jsonl.gz")
        self.assertEqual([len(t.runs) for t in threads], [10, 26, 10, 25])
        self.assertEqual(
            threads[0].runs[0].input.part_id, f"{threads[0].thread_id}:1:input"
        )

    def test_the_apply_example_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.policy.save(Path(tmp) / "policy.json")
            here, out = os.getcwd(), io.StringIO()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(out):
                    runpy.run_path(str(SAMPLE / "apply.py"), run_name="__main__")
            finally:
                os.chdir(here)
        rows = out.getvalue().splitlines()
        self.assertEqual(len(rows), 27)
        sent, full, archived = (int(x.replace(",", "")) for x in rows[-1].split()[1:4])
        self.assertGreaterEqual(self.policy.summary["coverage"], 0.95)
        self.assertLess(sent, full / 2)
        self.assertGreater(archived, 0)


class FigureTests(unittest.TestCase):
    def test_committed_figures_match_their_data(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import figures
        finally:
            sys.path.pop(0)
        for name, (target, source, draw) in figures.FIGURES.items():
            args = (
                [figures.json.loads((figures.DATA / source).read_text("utf-8"))]
                if source
                else []
            )
            with self.subTest(figure=name):
                self.assertEqual(
                    draw(*args), (figures.ASSETS / target).read_text("utf-8")
                )


if __name__ == "__main__":
    unittest.main()
