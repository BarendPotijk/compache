#!/usr/bin/env python3
"""The README as PyPI needs it: every link and image spelled out, as of one tag.

    python scripts/pypi_readme.py v0.15.0      # rewrites README.md, which the release workflow then builds

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
# An image in HTML, `<img src="path">`.
IMAGE = re.compile(r'(<img [^>]*?src=")(?!https?://)')


def absolute(text: str, ref: str) -> str:
    def spell(match):
        image, label, target = match.groups()
        if image:
            base = f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/"
        elif target.startswith("#"):
            base = f"https://github.com/{REPOSITORY}/tree/{ref}"
        else:
            base = f"https://github.com/{REPOSITORY}/blob/{ref}/"
        # The label of a link may itself hold an image.
        return f"{image}[{LINK.sub(spell, label)}]({base}{target})"

    text = LINK.sub(spell, text)
    # PyPI's page is light whatever the reader's system is.
    text = re.sub(r"\s*<source [^>]*>", "", text)
    text = IMAGE.sub(rf"\1https://raw.githubusercontent.com/{REPOSITORY}/{ref}/", text)
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
