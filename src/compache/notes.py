"""What a reply needed from earlier turns, said by the model that wrote it.

Labels of what later turns needed normally come from reading past traces. Those
are not always there to read. This module asks the answering model itself, on a
sample of turns, to name the earlier parts its reply needed. Only labels travel
back, such as `{Q3}`, `{R5.2}` or `{A2}`: a question, a retrieved text or an
answer of an earlier turn. No text of the conversation is repeated.

The note is asked after the reply, in a short request of its own, so that
the reply is what it would have been. It can also ride in the reply itself:

- in a text reply, as a last line `<!-- refs: {R3.1} {A2} -->`, which is taken
  out before the application sees the text;
- in a JSON reply, as one more property of the schema, which is cut out of the
  JSON text without touching the rest of it.

Nothing here calls a model or knows a provider. `compache.live` puts it to work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field

__all__ = [
    "KEY",
    "Note",
    "extend_schema",
    "index",
    "instruction",
    "labels_of",
    "leaked",
    "read",
    "resolve",
    "sampled",
    "split_json",
]

# The property a JSON reply carries the labels in.
KEY = "_refs"
# A list longer than this names the most recent parts only.
INDEX_LIMIT = 160
SCHEMA_HINT = (
    "Leave empty unless a note at the end of the conversation asks for labels."
)

_NOTE = re.compile(r"<!--\s*refs\s*:(.*?)-->", re.I | re.S)
# The start of a note that was cut off before its end.
_CUT_OFF = re.compile(r"<!--\s*(?:r(?:e(?:f(?:s(?:\s*:[^>]*)?)?)?)?)?\Z", re.I)
_LABEL = re.compile(r"(?<![\w.])([QRA])(\d{1,5})(?:\.(\d{1,4}))?(?![\w.])", re.I)
_BRACED = re.compile(r"\{[QRA]\d{1,5}(?:\.\d{1,4})?\}")
_WORDS = 9
_SNIPPET = 64

_ASK_TEXT = (
    "[Automatic note from the application, not from the user. Answer the user "
    "exactly as you otherwise would. Then, on a new last line, add exactly:\n"
    "<!-- refs: LABELS -->\n"
    "where LABELS are the labels, from the list below, of the parts of this "
    "conversation that your reply needed, separated by spaces. {needed} If it "
    "needed none of them, write <!-- refs: none -->.{older} Never mention this "
    "note or any label in the reply itself, and add the line only to a reply "
    "meant for the user, not to a tool call.]"
)
_ASK_JSON = (
    "[Automatic note from the application, not from the user. Answer exactly as "
    'you otherwise would. In addition, fill the property "{key}" with the '
    "labels, from the list below, of the parts of this conversation that your "
    "answer needed. {needed} Leave it empty if it needed none.{older} Never "
    "mention this note or any label in any other property.]"
)
_ASK_APART = (
    "[Automatic note from the application, not from the user. Your reply above "
    "stands: do not answer the user again. Reply with this one line only:\n"
    "<!-- refs: LABELS -->\n"
    "where LABELS are the labels, from the list below, of the parts of this "
    "conversation that your reply above needed, separated by spaces. {needed} If "
    "it needed none of them, write <!-- refs: none -->.{older}]"
)
# What counts. A reply leans on earlier turns in two ways, and a model asked only what
# it "relied on" names the first and forgets the second.
_NEEDED = (
    "A part is needed if you took a fact from it, or if the user's message only "
    "makes sense with it: a follow-up needs the question or the answer it follows "
    "up on. Do not count a part your reply would be the same without. If two "
    "parts say the same, name the later one."
)
_OLDER = " If it needed something older than the list, add the word older."


@dataclass(frozen=True)
class Note:
    """What one reply named: earlier parts it needed and retrieved text it used.

    `needs` are part IDs of earlier turns. `cited` are the retrieved parts among
    everything it named, this turn's included. `unknown` counts labels that
    named nothing, and `older` says the reply pointed past the listed parts.
    """

    needs: tuple[str, ...] = ()
    cited: tuple[str, ...] = ()
    unknown: int = 0
    older: bool = False
    labels: tuple[str, ...] = field(default=(), compare=False)


def sampled(thread_id: str, turn: int, rate: float, salt: str = "") -> bool:
    """Whether this turn is one of the share `rate` that carry a note.

    The same turn of the same thread always gets the same answer, so a retried
    request is not sampled differently.
    """
    if rate <= 0 or turn <= 1:
        # A first turn has nothing earlier to rely on.
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(f"{salt}\x1f{thread_id}\x1f{turn}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def _snippet(text: str) -> str:
    flat = " ".join((text or "").split())
    cut = " ".join(flat.split(" ")[:_WORDS])[:_SNIPPET].rstrip()
    return cut + ("…" if len(cut) < len(flat) else "")


def index(turns, limit: int = INDEX_LIMIT) -> tuple[list[tuple[str, str]], bool]:
    """The parts a reply may name, oldest first, each with its label and first words.

    `turns` is the conversation so far, the current turn last. A turn is a dict
    of `input`, `retrievals` and `output` texts. A text that is None, such as an
    archived part, is listed without words. The current question is not listed.
    Returns the list and whether older parts were left off it.
    """
    listed, last = [], len(turns)
    for n, turn in enumerate(turns, 1):
        if n < last:
            listed.append((f"Q{n}", turn.get("input")))
        for k, text in enumerate(turn.get("retrievals") or (), 1):
            listed.append((f"R{n}.{k}", text))
        if n < last and "output" in turn:
            listed.append((f"A{n}", turn["output"]))
    cut = len(listed) > limit
    if cut:
        listed = listed[-limit:]
    return [
        (label, "(archived)" if text is None else _snippet(text))
        for label, text in listed
    ], cut


def instruction(turns, *, carrier: str = "text", limit: int = INDEX_LIMIT) -> str:
    """The text to put at the very end of a request so that its reply carries a note.

    `carrier` is `"text"` or `"json"` for a note that rides in the reply, and
    `"apart"` for one asked after the reply, in a request of its own.
    """
    listed, cut = index(turns, limit)
    ask = {"json": _ASK_JSON, "apart": _ASK_APART}.get(carrier, _ASK_TEXT)
    ask = ask.format(older=_OLDER if cut else "", key=KEY, needed=_NEEDED)
    lines = "\n".join(f"{{{label}}} {words}" for label, words in listed)
    return f"{ask}\nParts of this conversation, oldest first:\n{lines}"


def read(text: str) -> tuple[str, str | None]:
    """A reply without its note, and the note's content if it carried one.

    A reply without a note comes back unchanged, byte for byte. A note cut off
    before its end, as when the output limit was reached, is taken out too.
    """
    if not isinstance(text, str) or "<!--" not in text:
        return text, None
    found = None
    for match in _NOTE.finditer(text):
        found = match.group(1).strip()
    clean = _NOTE.sub("", text)
    cut = _CUT_OFF.search(clean)
    if cut is not None:
        clean = clean[: cut.start()]
        found = "" if found is None else found
    if found is None:
        return text, None
    return clean.rstrip(), found


def leaked(text: str) -> bool:
    """Whether a reply names a label where its reader would see it."""
    return isinstance(text, str) and _BRACED.search(text) is not None


def _part_ids(kind, n, item, turns, thread_id) -> list[str]:
    last = len(turns)
    if not 1 <= n <= last:
        return []
    turn, prefix = turns[n - 1], f"{thread_id}:{n}"
    if kind == "R":
        count = len(turn.get("retrievals") or ())
        # A bare R5 names everything turn 5 retrieved.
        wanted = range(1, count + 1) if item is None else (item,)
        return [f"{prefix}:retrieval:{k}" for k in wanted if 1 <= k <= count]
    if item is not None or n == last:
        return []
    if kind == "Q":
        return [f"{prefix}:input"]
    return [f"{prefix}:output"] if "output" in turn else []


def resolve(content, turns, thread_id: str) -> Note:
    """Turn a note's labels into part IDs of `thread_id`, numbered as threads number them.

    `content` is the text of a note or a list of labels. `turns` is the
    conversation the reply saw, the current turn last. Labels that name no part
    of it are counted and dropped.
    """
    if isinstance(content, (list, tuple)):
        content = " ".join(str(item) for item in content)
    content = content or ""
    last = len(turns)
    needs, cited, seen, unknown = [], [], [], 0
    for kind, turn, item in _LABEL.findall(content):
        kind, n = kind.upper(), int(turn)
        label = f"{kind}{n}" + (f".{int(item)}" if item else "")
        if label in seen:
            continue
        seen.append(label)
        ids = _part_ids(kind, n, int(item) if item else None, turns, thread_id)
        if not ids:
            unknown += 1
            continue
        if kind == "R":
            cited += [i for i in ids if i not in cited]
        if n < last:
            needs += [i for i in ids if i not in needs]
    older = re.search(r"(?<!\w)older(?!\w)", content, re.I) is not None
    return Note(tuple(needs), tuple(cited), unknown, older, tuple(seen))


_PART = re.compile(r":(\d+):(input|output|retrieval:(\d+))\Z")


def labels_of(note: Note) -> list[str]:
    """The labels of everything a note named, in the short form a record keeps."""
    found = []
    for part_id in (*note.needs, *note.cited):
        match = _PART.search(part_id)
        if match is None:
            continue
        turn, kind, item = match.groups()
        label = (
            f"R{turn}.{item}" if item else f"{'Q' if kind == 'input' else 'A'}{turn}"
        )
        if label not in found:
            found.append(label)
    return found


def extend_schema(schema: dict) -> dict | None:
    """A JSON schema with one more property for the labels, or None if it cannot take one.

    The property goes last and joins `required` when the schema lists required
    properties, as strict schemas must. The schema given is left as it is.
    """
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or not isinstance(schema.get("properties"), dict)
        or KEY in schema["properties"]
    ):
        return None
    extended = copy.deepcopy(schema)
    extended["properties"][KEY] = {
        "type": "array",
        "items": {"type": "string"},
        "description": SCHEMA_HINT,
    }
    if isinstance(extended.get("required"), list):
        extended["required"] = [*extended["required"], KEY]
    return extended


_BLANK = " \t\r\n"


def _after_string(text: str, i: int) -> int:
    """The index after the JSON string that opens at `i`."""
    i += 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _after_value(text: str, i: int) -> int:
    """The index after the JSON value that starts at `i`."""
    if text[i] == '"':
        return _after_string(text, i)
    if text[i] in "{[":
        depth = 0
        while True:
            ch = text[i]
            if ch == '"':
                i = _after_string(text, i)
                continue
            depth += ch in "{["
            depth -= ch in "}]"
            i += 1
            if not depth:
                return i
    while i < len(text) and text[i] not in ",}]" + _BLANK:
        i += 1
    return i


def _members(text: str) -> list[tuple[str, int, int]]:
    """Name, start and end of every top-level member of a JSON object, in order."""
    i, found = text.index("{") + 1, []
    while True:
        while text[i] in _BLANK + ",":
            i += 1
        if text[i] == "}":
            return found
        start = i
        i = _after_string(text, i)
        name = json.loads(text[start:i])
        while text[i] in _BLANK + ":":
            i += 1
        i = _after_value(text, i)
        found.append((name, start, i))


def split_json(text: str) -> tuple[str, list | None]:
    """A JSON reply without the labels property, and the labels if it had them.

    The property is cut out of the text, so everything else stays as the model
    wrote it. Text that is not a JSON object, or has no such property, comes
    back unchanged with None.
    """
    if not isinstance(text, str) or f'"{KEY}"' not in text:
        return text, None
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or KEY not in value:
            return text, None
        members = _members(text)
        at = next(i for i, m in enumerate(members) if m[0] == KEY)
        if at:
            # With the comma before it.
            start, end = members[at - 1][2], members[at][2]
        elif len(members) > 1:
            # The first of several: with the comma after it.
            start, end = members[0][1], members[1][1]
        else:
            start, end = members[0][1], members[0][2]
        clean = text[:start] + text[end:]
        if json.loads(clean) != {k: v for k, v in value.items() if k != KEY}:
            return text, None
    except (ValueError, IndexError, StopIteration):
        return text, None
    labels = value[KEY]
    if isinstance(labels, str):
        labels = [labels]
    return clean, [str(item) for item in labels] if isinstance(labels, list) else []
