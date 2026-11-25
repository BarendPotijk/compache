#!/usr/bin/env python3
"""Issues of SWE-bench Lite as tasks: a project's own bug reports, graded by its own tests.

    python swe.py <folder>      fetch the dataset and sympy, keep what runs here and make sessions of it

sympy is pure Python with one dependency, so its issues need no container: a tree of files at
the issue's commit, a Python with mpmath and pytest, and the tests of the fix. An issue is kept
where those tests fail before the fix and pass after it on this machine.

A session is several issues in one checkout, given one after the other as a user would. Its
issues are those that can all be fixed at the same commit, checked with the project's own fixes.
A day is all sessions one after the other, in one conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
NEXT = """Here is another issue of the same library. Resolve it in the same way.

<issue>
{issue}
</issue>"""
# The commits that sessions start from, by the issue they belong to, and how many issues one holds.
BASES = (
    "sympy__sympy-21612",
    "sympy__sympy-20590",
    "sympy__sympy-18835",
    "sympy__sympy-15345",
)
SIZE, LEAST = 6, 5


def _changed(patch: str) -> list[str]:
    return [f for f in re.findall(r"^\+\+\+ b/(\S+)", patch, re.M) if f.endswith(".py")]


def _apply(tree: Path, patch: str) -> bool:
    # In a folder of another checkout, git would leave out what is not in that one.
    alone = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)}
    done = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=tree,
        env=alone,
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


@contextmanager
def checkout(root: Path, row: dict) -> Iterator[Path]:
    """A tree for as long as it is needed: there are hundreds, of 30 MB each."""
    with tempfile.TemporaryDirectory(dir=root) as folder:
        tree = Path(folder) / "tree"
        extract(root, row, tree)
        yield tree


class Issue:
    """One issue as a task for `run.py`."""

    def __init__(self, root: Path, name: str):
        self.root, self.name = Path(root).resolve(), name
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


class Session:
    """Several issues of one checkout as a task, one prompt each."""

    def __init__(self, root: Path, name: str):
        self.root, self.name = Path(root).resolve(), name
        self.plan = json.loads((self.root / "sessions.json").read_text())[name]
        rows = {
            r["instance_id"]: r
            for r in json.loads((self.root / "lite.json").read_text())
        }
        self.rows = [rows[i["name"]] for i in self.plan["issues"]]

    def make(self, tree: Path) -> None:
        extract(self.root, {"base_commit": self.plan["base"]}, tree)

    def prompts(self, parts: int) -> list[str]:
        first, *rest = (r["problem_statement"] for r in self.rows)
        return [PROMPT.format(issue=first), *(NEXT.format(issue=r) for r in rest)]

    def grade(self, tree: Path) -> dict:
        """How many of the issues are fixed, and whether anything else broke."""
        files = sorted({f for i in self.plan["issues"] for f in i["tests"]})
        for name in files:
            subprocess.run(["git", "checkout", "-q", "HEAD", "--", name], cwd=tree)
        for row in self.rows:
            _apply(tree, row["test_patch"])
        passed = _run(str(self.root / "venv/bin/python"), tree, files)
        fixed = {
            i["name"]: all(passed.get(t) for t in i["f2p"]) for i in self.plan["issues"]
        }
        broke = [t for t in self.plan["p2p"] if not passed.get(t)]
        return {
            "right": sum(fixed.values()),
            "of": len(fixed),
            "fixed": fixed,
            "broke": broke[:5],
            "broken": len(broke),
        }


class Day:
    """All sessions in one conversation, each in a checkout of its own."""

    def __init__(self, root: Path, name: str):
        self.root, self.name = Path(root).resolve(), name
        plans = json.loads((self.root / "sessions.json").read_text())
        self.parts = {
            f"checkout{k}": Session(root, base) for k, base in enumerate(plans, 1)
        }

    def make(self, tree: Path) -> None:
        for name, session in self.parts.items():
            session.make(tree / name)
            for words in (["init", "-q"], ["add", "-A"]):
                subprocess.run(["git", *words], cwd=tree / name, check=True)
            who = ["-c", "user.email=a@b", "-c", "user.name=a"]
            words = ["commit", "-qm", "task"]
            subprocess.run(["git", *who, *words], cwd=tree / name, check=True)

    def prompts(self, parts: int) -> list[str]:
        said = []
        for name, session in self.parts.items():
            for k, prompt in enumerate(session.prompts(1)):
                where = f"Work in the folder {name}, a checkout of its own. "
                said.append(prompt if k else where + prompt)
        return said

    def grade(self, tree: Path) -> dict:
        found = {n: s.grade(tree / n) for n, s in self.parts.items()}
        return {
            "right": sum(f["right"] for f in found.values()),
            "of": sum(f["of"] for f in found.values()),
            "broken": sum(f["broken"] for f in found.values()),
            "fixed": {n: f["fixed"] for n, f in found.items()},
        }


def plan_sessions(root: Path, workers: int) -> None:
    """Find sets of issues that can be fixed at one commit, and that their own fixes do fix together."""
    python = str(root / "venv/bin/python")
    rows = {r["instance_id"]: r for r in fetch(root)}
    kept = json.loads((root / "valid.json").read_text())
    free, made = set(issues(root)), {}

    def fits(base: str, names: list[str]) -> dict:
        """Which of some issues are broken at a commit and fixed once all their fixes are in."""
        with checkout(root, rows[base]) as tree:
            return fixed(names, tree)

    def fixed(names: list[str], tree: Path) -> dict:
        files = sorted({f for n in names for f in kept[n]["tests"]})
        for n in names:
            if not _apply(tree, rows[n]["test_patch"]):
                return {}
        before = _run(python, tree, files)
        for n in names:
            if not _apply(tree, rows[n]["patch"]):
                return {}
        after = _run(python, tree, files)
        good = {
            n: not any(map(before.get, kept[n]["f2p"]))
            and all(map(after.get, kept[n]["f2p"]))
            for n in names
        }
        return {"good": good, "passes": [t for t, ok in after.items() if ok]}

    for base in BASES:
        # One by one first, then together: fixes of two issues may be in each other's way.
        candidates = sorted(free)
        with ThreadPoolExecutor(workers) as pool:
            alone = list(pool.map(lambda n: fits(base, [n]), candidates))
        able = [n for n, f in zip(candidates, alone) if f and f["good"][n]]
        chosen: list[str] = []
        for n in able:
            if len(chosen) == SIZE:
                break
            tried = fits(base, [*chosen, n])
            if tried and all(tried["good"].values()):
                chosen.append(n)
        if len(chosen) < LEAST:
            print(f"{base}: only {len(chosen)} issues fit together")
            continue
        last = fits(base, chosen)
        wanted = {t for n in chosen for t in kept[n]["f2p"]}
        made[base] = {
            "base": rows[base]["base_commit"],
            "issues": [
                {"name": n, "tests": kept[n]["tests"], "f2p": kept[n]["f2p"]}
                for n in chosen
            ],
            "p2p": sorted(set(last["passes"]) - wanted),
        }
        free -= set(chosen)
        print(f"{base}: {', '.join(n.rpartition('-')[2] for n in chosen)}")
    (root / "sessions.json").write_text(json.dumps(made, indent=1))


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


def validate(root: Path, workers: int, least=(1, 4)) -> None:
    python = str(root / "venv/bin/python")
    rows = [
        r
        for r in fetch(root)
        if r["repo"] == "sympy/sympy"
        and tuple(map(int, r["version"].split("."))) >= least
    ]

    def one(row):
        tests = _changed(row["test_patch"])
        wanted = json.loads(row["FAIL_TO_PASS"])
        begun = time.time()
        try:
            with checkout(root, row) as tree:
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

    with ThreadPoolExecutor(workers) as pool:
        found = {name: kept for name, kept in pool.map(one, rows) if kept}
    (root / "valid.json").write_text(json.dumps(found, indent=1))
    print(f"{sum(k['ok'] for k in found.values())} of {len(rows)} issues run here")


def issues(root: Path) -> list[str]:
    kept = json.loads((root / "valid.json").read_text())
    return sorted(name for name, k in kept.items() if isinstance(k, dict) and k["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=lambda name: Path(name).resolve())
    parser.add_argument(
        "--workers", type=int, default=2, help="Test runs at a time. Each wants 0.5 GB"
    )
    args = parser.parse_args()
    if not (args.folder / "repo").exists():
        url = "https://github.com/sympy/sympy.git"
        words = ["clone", "-q", "--filter=blob:none", url, str(args.folder / "repo")]
        subprocess.run(["git", *words], check=True)
    if not (args.folder / "venv").exists():
        sys.exit("make a Python 3.9 with mpmath 1.3 and pytest at <folder>/venv first")
    if not (args.folder / "valid.json").exists():
        validate(args.folder, args.workers)
    if not (args.folder / "sessions.json").exists():
        plan_sessions(args.folder, args.workers)
