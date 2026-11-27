"""The README's quickstart, the sample it runs on, the figures drawn from committed data, and the links between pages."""

import contextlib
import importlib.metadata
import io
import json
import os
import re
import runpy
import sys
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import compache
from compache.cli import main

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
        self.assertEqual(summary["keep"]["retrieved"], 1)
        self.assertEqual(self.policy.min_retention, 0.8)
        self.assertGreaterEqual(summary["coverage"], self.policy.min_retention)

    def test_a_policy_carries_the_workload_for_the_calculator(self):
        profiled = compache.profile(
            SAMPLE / "threads.jsonl.gz",
            SAMPLE / "labels.jsonl",
            model="claude-sonnet-5-5",
            policy=self.policy,
        )["calculator"]
        carried = self.policy.artifact["workload"]
        decision = profiled["scenario"].pop("policy")
        self.assertEqual(
            decision,
            {
                "questions": 4,
                "answers": 9,
                "retrieved": 1,
                "keepUnder": 0,
                "maxPings": 1,
                "long": False,
            },
        )
        self.assertEqual(carried, profiled)
        with self.assertRaises(compache.ContractError):
            compache.profile(
                SAMPLE / "threads.jsonl.gz", model="gpt-6-luna", policy=self.policy
            )

    def test_the_sample_is_a_chat_log_with_simple_labels(self):
        threads = compache.load_threads(SAMPLE / "threads.jsonl.gz")
        self.assertEqual([len(t.runs) for t in threads], [10, 26, 10, 25])
        self.assertEqual(
            threads[0].runs[0].input.part_id, f"{threads[0].thread_id}:1:input"
        )

    def test_the_package_carries_the_sample(self):
        packaged = files("compache").joinpath("demo")
        for name in ("threads.jsonl.gz", "labels.jsonl"):
            self.assertEqual(
                packaged.joinpath(name).read_bytes(), (SAMPLE / name).read_bytes()
            )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["demo"]), 0)
        for line in self.policy.describe().splitlines():
            self.assertIn(line, out.getvalue())
        self.assertIn("Not kept", out.getvalue())
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = Path(tmp) / "policy.json"
            self.assertEqual(main(["demo", "-o", str(path)]), 0)
            self.assertEqual(compache.Policy.load(path).recipe, self.policy.recipe)

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
        self.assertLess(sent, full / 3)
        self.assertGreater(archived, 0)

    def test_the_held_out_example_reports_cost_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                runpy.run_path(str(SAMPLE / "fit_replay.py"), run_name="__main__")
        guide = (ROOT / "examples" / "README.md").read_text("utf-8")
        for line in out.getvalue().splitlines():
            self.assertIn(line, guide)
        self.assertIn("Historical requirements retained: 5/8", out.getvalue())

    def test_the_wrapped_example_changes_the_outgoing_request(self):
        try:
            version = importlib.metadata.version("anthropic")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("install the example's anthropic SDK")
        if version.split(".")[0] != "1":
            self.skipTest("this example uses the 1.x SDK and its httpx2 transport")
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            self.policy.save("policy.json")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                runpy.run_path(str(SAMPLE / "wrapped_chatbot.py"), run_name="__main__")
        self.assertIn("First question in application history: True", out.getvalue())
        self.assertIn("First question in final provider request: False", out.getvalue())
        self.assertIn("Recovery tool offered on final request: True", out.getvalue())

    def test_the_app_managed_example_runs_without_a_provider(self):
        example = runpy.run_path(str(SAMPLE / "app_managed.py"))
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.chdir(tmp),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.policy.save("policy.json")
            example["main"]()
        self.assertIn("It ships Friday.", out.getvalue())
        self.assertIn("Changes close Thursday.", out.getvalue())
        self.assertRegex(out.getvalue(), r"[4-9] physical calls recorded")


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


class SkillTests(unittest.TestCase):
    def test_the_marketplace_names_every_skill(self):
        market = json.loads(
            (ROOT / ".claude-plugin" / "marketplace.json").read_text("utf-8")
        )
        named = {path for plugin in market["plugins"] for path in plugin["skills"]}
        there = sorted(page.parent for page in ROOT.glob("skills/*/SKILL.md"))
        self.assertEqual(named, {f"./skills/{folder.name}" for folder in there})
        for folder in there:
            head = (folder / "SKILL.md").read_text("utf-8").split("---")[1]
            self.assertIn(f"\nname: {folder.name}\n", head)


class LinkTests(unittest.TestCase):
    LINK = re.compile(r"\]\((?!https?://|mailto:)([^)\s]+)\)")
    IMAGE = re.compile(r'\b(?:src|srcset|href)="(?!https?://|mailto:)([^"]+)"')

    @staticmethod
    def anchors(page):
        """The anchors GitHub gives a page's headings."""
        text = re.sub(r"```.*?```", "", page.read_text("utf-8"), flags=re.S)
        for heading in re.findall(r"^#+ +(.+?) *$", text, flags=re.M):
            yield re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")

    def test_every_link_between_the_pages_leads_somewhere(self):
        pages = [
            page
            for page in ROOT.rglob("*.md")
            if not any(
                part.startswith(".") or part == "node_modules"
                for part in page.relative_to(ROOT).parts
            )
        ]
        self.assertGreater(len(pages), 10)
        for page in pages:
            text = page.read_text("utf-8")
            for target in self.LINK.findall(text) + self.IMAGE.findall(text):
                path, _, anchor = target.partition("#")
                with self.subTest(page=str(page.relative_to(ROOT)), link=target):
                    found = (page.parent / path).resolve() if path else page
                    self.assertTrue(found.exists())
                    if anchor and found.suffix == ".md":
                        self.assertIn(anchor, set(self.anchors(found)))

    def test_the_readme_on_pypi_has_no_link_that_only_works_on_github(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import pypi_readme
        finally:
            sys.path.pop(0)
        text = pypi_readme.absolute((ROOT / "README.md").read_text("utf-8"), "v1.2.3")
        self.assertEqual(self.LINK.findall(text), [])
        self.assertEqual(self.IMAGE.findall(text), [])
        self.assertIn("/compache/v1.2.3/docs/assets/headline.svg)", text)
        self.assertIn('/compache/v1.2.3/docs/assets/logo.svg"', text)
        self.assertNotIn("logo-dark", text)
        self.assertIn("/compache/blob/v1.2.3/docs/guide.md)", text)
        html = pypi_readme.absolute(
            '<a href="#x">a</a> <a href="docs/b.md">b</a>', "v1"
        )
        self.assertIn('href="https://github.com/BarendPotijk/compache/tree/v1#x"', html)
        self.assertIn("/compache/blob/v1/docs/b.md", html)
        self.assertNotIn("git+https", text)

    def test_the_mark_is_the_same_wherever_it_is_drawn(self):
        marks = set()
        for path in (
            "site/mark.svg",
            "docs/assets/logo.svg",
            "docs/assets/logo-dark.svg",
            "docs/assets/icon.svg",
            "docs/assets/preview.svg",
        ):
            drawing = (ROOT / path).read_text("utf-8")
            marks.add(re.search(r'<path [^>]*d="([^"]+)"', drawing)[1])
        self.assertEqual(len(marks), 1)


if __name__ == "__main__":
    unittest.main()
