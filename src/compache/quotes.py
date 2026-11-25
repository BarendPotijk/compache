"""Which earlier results a reply repeats a line of.

An agent that edits a file repeats the lines it replaces, one that runs a test
repeats the path it read. A line of a reply that a tool returned before, and
that nobody else wrote, shows which result the reply relied on. No model has
to be asked, and only hashes of lines are held.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["Quotes", "spoken"]

# A line number that a tool put before the line of a file.
_NUMBERED = re.compile(r"^\s*\d+\s*[\t|:→]\s?")
# Shorter lines are said by anyone.
_LEAST = 24


def _lines(text: str):
    for raw in text.splitlines():
        line = _NUMBERED.sub("", raw).strip()
        if len(line) >= _LEAST and any(c.isalpha() for c in line):
            yield hash(line)


def spoken(text: str, calls: Any = ()) -> str:
    """What a reply says, in its text and in the arguments of its tool calls."""
    said = [text or ""]
    for arguments in calls or ():
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                pass
        values = arguments.values() if isinstance(arguments, dict) else [arguments]
        for value in values:
            text = value if isinstance(value, str) else json.dumps(value)
            said.append(text)
            if "\\n" in text:
                # Code that holds a text, as a patch in a script does.
                plain = text.replace("\\n", "\n").replace("\\t", "\t")
                said.append(plain.replace('\\"', '"').replace("\\'", "'"))
    return "\n".join(said)


class Quotes:
    """The lines of one conversation: which result each was last seen in, and which were said."""

    def __init__(self):
        self._seen: dict[int, Any] = {}
        self._own: set[int] = set()

    def result(self, name: Any, text: str) -> None:
        """A tool returned this."""
        for line in _lines(text):
            self._seen[line] = name

    def own(self, text: str) -> None:
        """Said by the user, or given as an instruction."""
        self._own.update(_lines(text))

    def said(self, text: str) -> list:
        """A reply. Returns the results it repeats, each where its line was last seen."""
        found: dict[Any, None] = {}
        for line in _lines(text):
            if line in self._own:
                continue
            if line in self._seen:
                found[self._seen[line]] = None
            self._own.add(line)
        return list(found)
