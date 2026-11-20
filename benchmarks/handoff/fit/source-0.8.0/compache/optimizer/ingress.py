"""Lossless references to whole lines of an earlier, visibly retained retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class Delta:
    anchor_id: str
    text: str


def make_delta(source, anchors, source_handle, min_chars):
    """Copy ranges use one-based inclusive line numbers; strings are literal inserts."""
    lines = source.text.splitlines(keepends=True)
    best_size = len(source.text.encode())
    best = None
    if best_size < min_chars:
        return None
    # Bound matching work on large, rapidly changing document families.
    candidates = dict.fromkeys([anchors[0], *anchors[-3:]]) if anchors else ()
    for anchor in candidates:
        base = anchor.text.splitlines(keepends=True)
        blocks = SequenceMatcher(None, base, lines).get_matching_blocks()
        segments = []
        cursor = 0
        for start, at, count in blocks:
            if not count or sum(len(line) for line in lines[at : at + count]) < min_chars:
                continue
            if cursor < at:
                segments.append("".join(lines[cursor:at]))
            segments.append([start + 1, start + count])
            cursor = at + count
        if not segments:
            continue
        if cursor < len(lines):
            segments.append("".join(lines[cursor:]))
        text = json.dumps({
            "copy_lines_from_part": anchor.part_id,
            "instructions": "Read segments in order. A pair copies inclusive, 1-based lines of the cited part verbatim. A string is verbatim text of this source.",
            "segments": segments,
            "reopen_original": source_handle,
        }, ensure_ascii=False, separators=(",", ":"))
        size = len(text.encode())
        if size < best_size:
            best_size = size
            best = Delta(anchor.part_id, text)
    return best
