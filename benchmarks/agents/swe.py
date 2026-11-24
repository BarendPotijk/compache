#!/usr/bin/env python3
"""Issues of SWE-bench Lite as tasks: a project's own bug reports, graded by its own tests.

    python swe.py <folder>      fetch the dataset and sympy, and keep the issues that run here

sympy is pure Python with one dependency, so its issues need no container: a tree of files at
the issue's commit, a Python with mpmath and pytest, and the tests of the fix. An issue is kept
where those tests fail before the fix and pass after it on this machine.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Lite"
    "&config=default&split=test&offset={}&length=100"
)
PROMPT = """This is the source of sympy. Resolve the issue below by changing the library. Do not change existing
tests. Reproduce the problem first, find the cause, fix it, and run the tests of the modules you touched with
`python -m pytest`. When you are done, reply with one short sentence.

<issue>
{issue}
</issue>"""


def _changed(patch: str) -> list[str]:
    return [f for f in re.findall(r"^\+\+\+ b/(\S+)", patch, re.M) if f.endswith(".py")]


def _apply(tree: Path, patch: str) -> bool:
    done = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=tree,
        input=patch,
        text=True,
        capture_output=True,
    )
    return done.returncode == 0


def _run(python: str, tree: Path, files: list[str], limit: float = 600) -> dict:
    """Which tests of some files pass."""
    words = ["-m", "pytest", "-q", "--tb=no", "-rA", "-p", "no:cacheprovider", *files]
    done = subprocess.run(
        [python, *words], cwd=tree, capture_output=True, text=True, timeout=limit
    )
    found = re.findall(r"^(PASSED|FAILED|ERROR) (\S+)", done.stdout, re.M)
    return {name.split("::")[-1]: state == "PASSED" for state, name in found}


def extract(root: Path, row: dict, tree: Path) -> None:
    tree.mkdir(parents=True)
    packed = subprocess.run(
        ["git", "archive", row["base_commit"]],
        cwd=root / "repo",
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "x", "-C", str(tree)], input=packed.stdout, check=True)


class Issue:
    """One issue as a task for `run.py`."""

    def __init__(self, root: Path, name: str):
        self.root, self.name = Path(root), name
        rows = json.loads((self.root / "lite.json").read_text())
        self.row = next(r for r in rows if r["instance_id"] == name)
        self.kept = json.loads((self.root / "valid.json").read_text())[name]

    def make(self, tree: Path) -> None:
        extract(self.root, self.row, tree)

    def prompts(self, parts: int) -> list[str]:
        return [PROMPT.format(issue=self.row["problem_statement"])]

    def grade(self, tree: Path) -> dict:
        """1 of 1 where the tests of the fix pass and none that passed before fails."""
        for name in self.kept["tests"]:
            # What the agent wrote into the test files makes way for the project's own.
            subprocess.run(["git", "checkout", "-q", "HEAD", "--", name], cwd=tree)
        fits = _apply(tree, self.row["test_patch"])
        passed = _run(str(self.root / "venv/bin/python"), tree, self.kept["tests"])
        fixed = [n for n in self.kept["f2p"] if passed.get(n)]
        broke = [n for n in self.kept["p2p"] if not passed.get(n)]
        right = fits and len(fixed) == len(self.kept["f2p"]) and not broke
        return {
            "right": int(right),
            "of": 1,
            "fixed": len(fixed),
            "to_fix": len(self.kept["f2p"]),
            "broke": broke[:5],
        }


def fetch(root: Path) -> list[dict]:
    path = root / "lite.json"
    if not path.exists():
        context = ssl.create_default_context()
        rows: list[dict] = []
        while True:
            url = ROWS.format(len(rows))
            with urllib.request.urlopen(url, context=context, timeout=60) as answer:
                got = [r["row"] for r in json.load(answer)["rows"]]
            rows += got
            if len(got) < 100:
                break
        path.write_text(json.dumps(rows))
    return json.loads(path.read_text())


def validate(root: Path, least=(1, 4)) -> None:
    python = str(root / "venv/bin/python")
    rows = [
        r
        for r in fetch(root)
        if r["repo"] == "sympy/sympy"
        and tuple(map(int, r["version"].split("."))) >= least
    ]

    def one(row):
        tree = root / "trees" / row["instance_id"]
        extract(root, row, tree)
        tests = _changed(row["test_patch"])
        wanted = json.loads(row["FAIL_TO_PASS"])
        begun = time.time()
        try:
            fits = _apply(tree, row["test_patch"])
            before = _run(python, tree, tests)
            fits = _apply(tree, row["patch"]) and fits
            after = _run(python, tree, tests)
        except subprocess.TimeoutExpired:
            return row["instance_id"], None
        kept = [n for n in json.loads(row["PASS_TO_PASS"]) if after.get(n)]
        good = all(after.get(n) for n in wanted) and not any(map(before.get, wanted))
        return row["instance_id"], {
            "ok": fits and good,
            "version": row["version"],
            "seconds": round(time.time() - begun),
            "f2p": wanted,
            "p2p": kept,
            "tests": tests,
        }

    with ThreadPoolExecutor(3) as pool:
        found = {name: kept for name, kept in pool.map(one, rows) if kept}
    (root / "valid.json").write_text(json.dumps(found, indent=1))
    print(f"{sum(k['ok'] for k in found.values())} of {len(rows)} issues run here")


def issues(root: Path) -> list[str]:
    kept = json.loads((root / "valid.json").read_text())
    return sorted(name for name, k in kept.items() if isinstance(k, dict) and k["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if not (args.folder / "repo").exists():
        url = "https://github.com/sympy/sympy.git"
        words = ["clone", "-q", "--filter=blob:none", url, str(args.folder / "repo")]
        subprocess.run(["git", *words], check=True)
    if not (args.folder / "venv").exists():
        sys.exit("make a Python 3.9 with mpmath 1.3 and pytest at <folder>/venv first")
    validate(args.folder)
