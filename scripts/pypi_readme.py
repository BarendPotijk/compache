#!/usr/bin/env python3
"""The README as PyPI needs it: every link and image spelled out, as of one tag.

    python scripts/pypi_readme.py v0.15.1      # rewrites README.md, which the release workflow then builds

On GitHub the README's links are relative, so they work on every branch. PyPI has no
repository to resolve them against.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "BarendPotijk/compache"
# A link or image, `[text](target)` or `![alt](target)`, to a path in the repository.
LINK = re.compile(
    r"(!?)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\((?!https?://|mailto:)([^)\s]+)\)"
)
# The same in HTML: `<img src="path">`, `<a href="path">`.
IMAGE = re.compile(r'(<img [^>]*?src=")(?!https?://)')
ANCHOR = re.compile(r'(<a [^>]*?href=")(?!https?://|mailto:)([^"]+)')


def absolute(text: str, ref: str) -> str:
    raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/"

    def page(target):
        place = "tree" if target.startswith("#") else "blob"
        between = "" if target.startswith("#") else "/"
        return f"https://github.com/{REPOSITORY}/{place}/{ref}{between}{target}"

    def spell(match):
        image, label, target = match.groups()
        # The label of a link may itself hold an image.
        shown = raw + target if image else page(target)
        return f"{image}[{LINK.sub(spell, label)}]({shown})"

    text = LINK.sub(spell, text)
    # PyPI's page is light whatever the reader's system is.
    text = re.sub(r"\s*<source [^>]*>", "", text)
    text = IMAGE.sub(rf"\1{raw}", text)
    text = ANCHOR.sub(lambda m: m[1] + page(m[2]), text)
    # On the package's own page, the package comes from there.
    return text.replace(
        f"pip install git+https://github.com/{REPOSITORY}", "pip install compache"
    )


def main() -> None:
    readme = ROOT / "README.md"
    readme.write_text(absolute(readme.read_text("utf-8"), sys.argv[1]), "utf-8")
    print(f"README.md: links as of {sys.argv[1]}")


if __name__ == "__main__":
    main()
