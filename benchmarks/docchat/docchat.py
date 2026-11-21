#!/usr/bin/env python3
"""DocChat: long chats over public documents, written by one model and checked by another.

The build runs in stages. Each stage stores its result, so a build can stop and resume.

  fetch     download the pinned public files
  corpus    parse them into documents and chunks
  plan      lay out each thread: who asks, about which excerpt, building on which earlier turn
  write     the writer model writes each user message and its reference answer
  verify    the writer answers again with parts of the history withheld, and the judge grades
  assemble  threads, evidence labels, grading notes, a split and a manifest

The plan only proposes which turns need history. The labels come from the verify stage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
SCHEMA = "compache-docchat-v1"
WRITER, JUDGE = "claude-haiku-5-5", "claude-sonnet-5-5"

# Declared shape of a thread. Turn counts are drawn inside a tier's range.
TIERS = {"short": (8, 12), "medium": (25, 35), "long": (70, 90)}
# How far back a turn that builds on history reaches, as (turn range, share).
LAGS = (((1, 5), 0.5), ((6, 20), 0.3), ((21, 10**6), 0.2))
CHUNK_CHARS = 2800
EVIDENCE_CHARS = 700
SPARE_EXCERPTS = 6

# Idle time before a user message: lognormal around a median. The first three are
# declared scenarios. "tracelab" follows the quantiles published for coding-agent
# sessions (median 1.4 minutes, 90th percentile 20.6), which are not document chats.
CLOCKS = {
    "fast": {"median_idle_seconds": 30, "sigma": 1.2},
    "mixed": {"median_idle_seconds": 240, "sigma": 1.2},
    "slow": {"median_idle_seconds": 1800, "sigma": 1.2},
    "tracelab": {"median_idle_seconds": 84, "sigma": 2.1},
}
FIRST_TOKEN_SECONDS = 1.0
OUTPUT_TOKENS_PER_SECOND = 60.0
EPOCH = datetime(2026, 1, 5, 9, tzinfo=timezone.utc)

TURN_TYPES = {
    "lookup": "Ask something new that the NEW EXCERPT answers. The message must make sense on its own.",
    "follow_up": (
        "Ask a follow-up to the previous turn the way people do in chat: short, leaning on what was just "
        "said without repeating it ('and what about...', 'does that also...'). It must be answered by a "
        "detail in the earlier turn's excerpt that the assistant has not stated yet."
    ),
    "recall": (
        "Return to the topic of the earlier turn shown below, which came up a while ago. Ask about a detail "
        "in that turn's excerpt that the assistant has not stated yet. Point back to it the way a person "
        "would ('going back to...', 'earlier you said...') without restating what it said."
    ),
    "compare": (
        "Ask something that needs both the assistant's earlier answer shown below and the NEW EXCERPT: a "
        "difference, a combined figure, or whether the two fit together. Point back to the earlier answer "
        "without restating its numbers or wording."
    ),
    "user_context": (
        "Ask about the NEW EXCERPT in a way whose correct answer depends on what you said earlier about "
        "your own situation: '{fact}'. Do not restate it. Say 'in my case' or 'for me' instead."
    ),
    "reuse_output": (
        "Ask the assistant to do something with its earlier answer shown below: put it in a table, shorten "
        "it, rank it, turn it into a short note for a colleague, or redo a figure with one changed "
        "assumption that you give. No new lookup is needed."
    ),
    "synthesis": (
        "Ask for a short summary that pulls together the earlier turns shown below. Name their topics, not "
        "their content."
    ),
}
PLANT = (
    "PLANT A FACT: later the user asks about the excerpt below. In this message, also mention in passing "
    "one concrete fact about your own situation that will change the answer to a question about it, and "
    "return that fact in planted_fact.\n{text}"
)
WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "skip": {"type": "boolean"},
        "skip_reason": {"type": "string"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "planted_fact": {"type": "string"},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["skip", "skip_reason", "question", "answer", "planted_fact", "quotes"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_supported": {"type": "boolean"},
        "problem": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "correct": {"type": "boolean"},
                },
                "required": ["id", "correct"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reference_supported", "problem", "key_facts", "verdicts"],
    "additionalProperties": False,
}
PART_ORDER = {"input": 0, "retrieval": 1, "output": 2}
PART_NAME = {"input": "user", "retrieval": "retrieved excerpt", "output": "assistant"}


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def unit(*key):
    """A reproducible number strictly between zero and one for this key."""
    raw = hashlib.sha256(compact(list(key)).encode()).digest()
    return (int.from_bytes(raw[:8], "big") + 0.5) / 2**64


def pick(items, *key):
    return items[int(unit(*key) * len(items))]


def weighted(table, *key):
    left = unit(*key) * sum(table.values())
    for name, weight in table.items():
        left -= weight
        if left < 0:
            break
    return name


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z,
    ):
        for row in rows:
            z.write((compact(row) + "\n").encode())


# Sources: pinned public files to documents to chunks.

EDGAR_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for the Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
}
HTML_TAGS = r"</?(?:a|div|span|br|p|table|thead|tbody|tr|td|th|ul|ol|li|img|strong|em|b|i|sup|sub|details|summary|h[1-6])\b[^>]*>"


def fetch(doc, cache):
    """Download a document's files once, with curl, and return their paths."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in doc["files"]:
        path = cache / item["name"]
        if not path.is_file():
            partial = path.with_name(path.name + ".part")
            subprocess.run(
                [
                    "curl",
                    "-sSfL",
                    "-A",
                    "compache-docchat/1",
                    "-o",
                    str(partial),
                    item["url"],
                ],
                check=True,
            )
            partial.rename(path)
        paths.append(path)
    return paths


def parse_edgar(path, filename):
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if f'"{filename}"' in line[:300]:
                row = json.loads(line)
                return [
                    (f"Item {key}. {name}", row[f"section_{key}"])
                    for key, name in EDGAR_ITEMS.items()
                    if row.get(f"section_{key}", "").strip()
                ]
    raise SystemExit(f"{filename} is not in {path.name}")


def parse_markdown(paths):
    def tooltip(match):
        found = re.search(r'text="([^"]*)"', match.group(1)) or re.search(
            r'term_id="([^"]*)"', match.group(1)
        )
        return found.group(1).replace("-", " ") if found else ""

    sections = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ")
        if text.startswith("---"):
            head, _, text = text[3:].partition("\n---")
            found = re.search(r"^title:\s*(.+)$", head, re.M)
            title = found.group(1).strip("\"' ") if found else title
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        text = re.sub(r"\{\{<\s*glossary_tooltip([^>]*)>\}\}", tooltip, text)
        text = re.sub(r"\{\{[<%].*?[>%]\}\}|\{%.*?%\}", "", text, flags=re.S)
        text = re.sub(HTML_TAGS, "", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        heading, body, fenced = title, [], False
        for line in text.split("\n"):
            fenced ^= line.lstrip().startswith("```")
            found = None if fenced else re.match(r"#{1,4}\s+(.*)", line)
            if found is None:
                body.append(line)
                continue
            sections.append((heading, "\n".join(body)))
            name = re.sub(r"\s*\{#[^}]*\}", "", found.group(1)).strip(" #")
            heading, body = f"{title} › {name}", []
        sections.append((heading, "\n".join(body)))
    return [(heading, body) for heading, body in sections if body.strip()]


def parse_openfda(path):
    row = read_json(path)["results"][0]
    return [
        (key.replace("_", " ").capitalize(), "\n".join(value))
        for key, value in row.items()
        if isinstance(value, list)
        and value
        and isinstance(value[0], str)
        and not key.endswith("_table")
        and not key.startswith(("spl_product", "package_label"))
    ]


def parse(doc, paths):
    kind = doc["parser"]
    if kind == "edgar":
        return parse_edgar(paths[0], doc["files"][0]["select"])
    if kind == "markdown":
        return parse_markdown(paths)
    if kind == "openfda":
        return parse_openfda(paths[0])
    if kind == "text":
        return [
            (
                "",
                re.sub(
                    r"[ \t]{3,}",
                    "\n",
                    paths[0].read_text(encoding="utf-8", errors="replace"),
                ),
            )
        ]
    raise SystemExit(f"unknown parser: {kind}")


def split(text, limit):
    """Pieces of at most `limit` characters, each cut at the coarsest boundary that fits."""
    pieces = [text]
    for boundary in (r"\n\s*\n", r"\n", r"(?<=[.;:?!])\s+", r"\s+"):
        pieces = [
            part
            for piece in pieces
            for part in (re.split(boundary, piece) if len(piece) > limit else [piece])
            if part.strip()
        ]
    return [
        piece[i : i + limit] for piece in pieces for i in range(0, len(piece), limit)
    ]


def chunk(doc, sections):
    """Pack each section's pieces into chunks. A chunk can be evidence when it is mostly prose."""
    bodies = []
    for heading, text in sections:
        body = ""
        for piece in split(text.strip(), CHUNK_CHARS) + [None]:
            if piece is None or len(body) + len(piece) + 1 > CHUNK_CHARS:
                if body.strip():
                    bodies.append((heading, body.strip()))
                body = ""
            if piece is not None:
                body = f"{body}\n{piece}" if body else piece
    # Short neighbours on one page share a chunk, so a retrieval is not a lone heading.
    merged = []
    for heading, body in bodies:
        last = merged[-1] if merged else ("\0", "")
        same_page = last[0].split(" › ")[0] == heading.split(" › ")[0]
        if (
            same_page
            and min(len(body), len(last[1])) < EVIDENCE_CHARS
            and len(body) + len(last[1]) < CHUNK_CHARS
        ):
            title = heading.split(" › ", 1)[-1] + "\n" if heading != last[0] else ""
            merged[-1] = (last[0], f"{last[1]}\n\n{title}{body}")
        else:
            merged.append((heading, body))
    rows = []
    for n, (heading, body) in enumerate(merged):
        letters = sum(c.isalpha() for c in body) / max(
            1, sum(not c.isspace() for c in body)
        )
        rows.append(
            {
                "chunk_id": f"{doc['id']}-c{n:03d}",
                "document_id": doc["id"],
                "version": doc["version"],
                "text": (f"{doc['title']} › {heading}" if heading else doc["title"])
                + "\n"
                + body,
                "evidence": len(body) >= EVIDENCE_CHARS and letters >= 0.55,
            }
        )
    return rows


def load_corpus(doc, cache, work, lock=False):
    """A document's chunks, checked against its pinned hash."""
    stored = work / "corpus" / f"{doc['id']}.json"
    if stored.is_file() and not lock:
        return read_json(stored)
    sections = parse(doc, fetch(doc, cache))
    found = digest(sections)
    if lock:
        doc["sha256"] = found
    elif found != doc.get("sha256"):
        raise SystemExit(
            f"{doc['id']}: the source no longer matches its pinned hash. Run `lock` to accept the new text."
        )
    chunks = chunk(doc, sections)
    write_json(stored, chunks)
    return chunks


# Retrieval: plain BM25 over one document's chunks, as a simple chat pipeline would run it.

STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will "
    "with what which who how does do can i my me we our you your about if any there their".split()
)


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.counts = [Counter(words(c["text"])) for c in chunks]
        self.sizes = [sum(c.values()) for c in self.counts]
        self.mean = sum(self.sizes) / len(self.sizes)
        seen = Counter(w for c in self.counts for w in c)
        self.idf = {
            w: math.log(1 + (len(chunks) - n + 0.5) / (n + 0.5))
            for w, n in seen.items()
        }
        self.k1, self.b = k1, b

    def search(self, query, k, must=None):
        """The k best chunks for a query. `must` takes the last place if it did not rank."""
        terms = set(words(query))
        scores = []
        for counts, size in zip(self.counts, self.sizes):
            norm = self.k1 * (1 - self.b + self.b * size / self.mean)
            scores.append(
                sum(
                    self.idf[t] * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                    for t in terms
                    if t in counts
                )
            )
        hits = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        if must is not None and must not in hits:
            hits[-1] = must
        return hits


# Plan: a deterministic outline of each thread.


def itinerary(evidence, *key):
    """Evidence chunks in the order a reader moves through a document: mostly nearby, sometimes a jump."""
    left, position, n = list(evidence), pick(evidence, *key, "start"), 0
    while True:
        left = left or list(evidence)
        near = [i for i in left if abs(i - position) <= 4]
        position = pick(
            near if near and unit(*key, n, "stay") < 0.7 else left, *key, n, "step"
        )
        left.remove(position)
        n += 1
        yield position


def earlier_turn(turn, candidates, *key):
    """One earlier turn, at a distance drawn from the declared lag bands."""
    drawn = weighted({i: share for i, (_, share) in enumerate(LAGS)}, *key, "band")
    for band in sorted(range(len(LAGS)), key=lambda i: abs(i - drawn)):
        low, high = LAGS[band][0]
        pool = [c for c in candidates if low <= turn - c <= high]
        if pool:
            return [pick(pool, *key, "turn")]
    return []


def plan_thread(tid, domain, cfg, doc, chunks, tier, seed):
    low, high = TIERS[tier]
    count = low + int(unit(seed, tid, "length") * (high - low + 1))
    walk = itinerary([i for i, c in enumerate(chunks) if c["evidence"]], seed, tid)
    turns, grounded, own = [], set(), set()
    for t in range(1, count + 1):
        intent = "lookup" if t == 1 else weighted(cfg["mix"], seed, tid, t, "intent")
        key = (seed, tid, t)
        if intent == "follow_up":
            refs = [t - 1] if t - 1 in grounded else []
        elif intent == "recall":
            refs = earlier_turn(t, [i for i in grounded if t - i >= 3], *key)
        elif intent == "compare":
            refs = earlier_turn(t, [i for i in grounded if t - i >= 2], *key)
        elif intent == "user_context":
            refs = earlier_turn(
                t,
                [i for i in own if t - i >= 2 and turns[i - 1]["plant_for"] is None],
                *key,
            )
        elif intent == "reuse_output":
            refs = earlier_turn(t, list(range(1, t)), *key)
        elif intent == "synthesis":
            refs = (
                sorted({pick(list(range(1, t)), *key, n) for n in range(3)})
                if t >= 6
                else []
            )
        else:
            refs = []
        if not refs:
            intent = "lookup"
        fresh = intent in ("lookup", "compare", "user_context")
        if intent == "user_context":
            turns[refs[0] - 1]["plant_for"] = t
        if fresh:
            own.add(t)
        if fresh or (intent != "synthesis" and refs[0] in grounded):
            grounded.add(t)
        turns.append(
            {
                "turn": t,
                "intent": intent,
                "refs": refs,
                "chunk": next(walk) if fresh else None,
                "plant_for": None,
            }
        )
    return {
        "thread_id": tid,
        "domain": domain,
        "document": doc["id"],
        "title": doc["title"],
        "tier": tier,
        "persona": pick(cfg["personas"], seed, tid, "persona"),
        "turns": turns,
        "spares": [next(walk) for _ in range(SPARE_EXCERPTS)],
    }


# Models: cached headless calls.


def claude_cli(model, system, prompt, schema, effort=None):
    """One call through the Claude Code command line, with no tools and nothing from this project."""
    command = [
        "claude",
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--system-prompt",
        system,
    ]
    if schema:
        command += ["--json-schema", json.dumps(schema)]
    if effort:
        command += ["--effort", effort]
    error = None
    for attempt in range(4):
        try:
            done = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=tempfile.gettempdir(),
                timeout=600,
            )
            reply = json.loads(done.stdout)
            value = reply.get("structured_output") if schema else reply.get("result")
            if reply.get("is_error") or not isinstance(value, dict if schema else str):
                raise ValueError(str(reply.get("result"))[:300])
            return value, {
                "model": model,
                "usd": reply.get("total_cost_usd"),
                "ms": reply.get("duration_ms"),
            }
        except (ValueError, subprocess.TimeoutExpired) as problem:
            error = problem
            time.sleep(10 * 2**attempt)
    raise RuntimeError(f"{model} call failed: {error}")


class Models:
    """Every answer is stored under the hash of what was asked, so a rerun asks nothing twice."""

    def __init__(self, work, runner=claude_cli):
        self.store, self.runner, self.lock = work / "calls", runner, threading.Lock()
        self.ledger = work / "calls.jsonl"
        self.new = Counter()

    def ask(self, model, system, prompt, schema=None, effort=None):
        key = digest([model, system, prompt, schema, effort])
        path = self.store / key[:2] / f"{key}.json"
        if path.is_file():
            return read_json(path)
        value, note = self.runner(model, system, prompt, schema, effort)
        write_json(path, value)
        with self.lock:
            self.new[model] += 1
            with self.ledger.open("a", encoding="utf-8") as out:
                out.write(compact(note | {"key": key}) + "\n")
        return value


# Write: the user message and reference answer for each planned turn.


def origin(done, turn):
    """The turn and chunk a turn's evidence came from, following what it builds on."""
    seen = set()
    while turn in done and turn not in seen:
        seen.add(turn)
        if done[turn]["chunk"] is not None:
            return turn, done[turn]["chunk"]
        if not done[turn]["refs"]:
            break
        turn = done[turn]["refs"][0]
    return None


def writer_prompt(plan, step, intent, refs, chunk, fact, done, chunks):
    recent = [
        f"User: {done[t]['question']}\nAssistant: {done[t]['answer']}"
        for t in sorted(done)[-2:]
    ]
    parts = [
        f"USER: {plan['persona']['who']}. They want to {plan['persona']['goal']}.",
        f"DOCUMENT: {plan['title']}",
        "LATEST TURNS:\n" + ("\n".join(recent) or "(this is the first message)"),
        "TURN TYPE: " + TURN_TYPES[intent].format(fact=fact),
    ]
    for ref in refs:
        block = f"EARLIER TURN {ref}\nUser: {done[ref]['question']}\nAssistant: {done[ref]['answer']}"
        source = origin(done, ref)
        if intent in ("follow_up", "recall") and source:
            block += f"\nExcerpt that turn was about:\n{chunks[source[1]]['text']}"
        parts.append(block)
    if chunk is not None:
        parts.append(
            "NEW EXCERPT, retrieved for this message:\n" + chunks[chunk]["text"]
        )
    if step["plant_for"]:
        parts.append(
            PLANT.format(
                text=chunks[plan["turns"][step["plant_for"] - 1]["chunk"]]["text"]
            )
        )
    return "\n\n".join(parts)


def related(index, done, ref, chunks):
    """An unused evidence chunk on the subject of an earlier turn, so a comparison has a point."""
    used = {r["chunk"] for r in done.values()} | {(origin(done, ref) or (0, None))[1]}
    query = f"{done[ref]['question']} {done[ref]['answer']}"
    hits = index.search(query, 12)
    return next((i for i in hits if chunks[i]["evidence"] and i not in used), None)


def write_thread(plan, chunks, index, cfg, models, system):
    """Write the planned turns in order. A turn the material cannot carry becomes a plain lookup."""
    done, spares = {}, list(plan["spares"])
    for step in plan["turns"]:
        intent, refs, chunk = (
            step["intent"],
            [r for r in step["refs"] if r in done],
            step["chunk"],
        )
        fact = (
            done[refs[0]]["planted_fact"] if intent == "user_context" and refs else ""
        )
        broken = (
            len(refs) < len(step["refs"])
            or (intent == "user_context" and not fact)
            or (
                intent in ("follow_up", "recall", "compare")
                and origin(done, refs[0]) is None
            )
        )
        if intent != "lookup" and broken:
            intent, refs = "lookup", []
        if intent == "compare":
            chunk = related(index, done, refs[0], chunks) or chunk
        for _ in range(3):
            if intent == "lookup" and chunk is None:
                if not spares:
                    break
                chunk = spares.pop(0)
            reply = models.ask(
                WRITER,
                system,
                writer_prompt(plan, step, intent, refs, chunk, fact, done, chunks),
                WRITER_SCHEMA,
            )
            if (
                not reply["skip"]
                and reply["question"].strip()
                and reply["answer"].strip()
            ):
                done[step["turn"]] = {
                    "turn": step["turn"],
                    "planned": step["intent"],
                    "intent": intent,
                    "refs": refs,
                    "chunk": chunk,
                    "question": reply["question"].strip(),
                    "answer": reply["answer"].strip(),
                    "quotes": [q for q in reply["quotes"] if q.strip()],
                    "planted_fact": reply["planted_fact"].strip()
                    if step["plant_for"]
                    else "",
                    "retrieved": index.search(
                        reply["question"], cfg["top_k"], must=chunk
                    ),
                }
                break
            intent, refs, chunk = "lookup", [], None
    return [done[t] for t in sorted(done)]


# Verify: which earlier parts a turn needs, found by withholding them.


def candidates(record, done):
    """Named sets of earlier parts to show the answerer. Each part is (turn, kind)."""
    sets = {"cur": []}
    before = [t for t in done if t < record["turn"]]
    if before and max(before) not in record["refs"]:
        sets["prev"] = [(max(before), "input"), (max(before), "output")]
    if record["refs"]:
        sets["in"] = [(r, "input") for r in record["refs"]]
        sets["out"] = [(r, "output") for r in record["refs"]]
        sources = sorted({origin(done, r) for r in record["refs"]} - {None})
        if sources:
            sets["ret"] = [(turn, "retrieval") for turn, _ in sources]
        full = sorted(set(sets["in"] + sets["out"] + sets.get("ret", [])))
        sets["full"] = full
    return sets


def part_text(part, done, chunks):
    turn, kind = part
    if kind == "retrieval":
        return chunks[done[turn]["chunk"]]["text"]
    return done[turn]["question" if kind == "input" else "answer"]


def answer_prompt(record, shown, done, chunks):
    lines = ["EARLIER IN THIS CONVERSATION (only part of it is shown):"]
    for part in sorted(shown, key=lambda p: (p[0], PART_ORDER[p[1]])):
        lines.append(
            f"[turn {part[0]}] {PART_NAME[part[1]]}: {part_text(part, done, chunks)}"
        )
    if not shown:
        lines.append("(nothing from earlier turns is shown)")
    lines.append("\nRETRIEVED FOR THE LATEST MESSAGE:")
    lines += [
        f"({n}) {chunks[i]['text']}" for n, i in enumerate(record["retrieved"], 1)
    ]
    lines.append("\nLATEST USER MESSAGE:\n" + record["question"])
    return "\n".join(lines)


def judge_prompt(record, answers, done, chunks):
    material = []
    if record["chunk"] is not None:
        material.append(
            "Excerpt retrieved for this message:\n" + chunks[record["chunk"]]["text"]
        )
    for ref in record["refs"]:
        material.append(
            f"Earlier turn {ref}\nUser: {done[ref]['question']}\nAssistant: {done[ref]['answer']}"
        )
        source = origin(done, ref)
        if source:
            material.append(
                f"Excerpt turn {ref} was about:\n{chunks[source[1]]['text']}"
            )
    listed = "\n\n".join(
        f"Candidate {letter}:\n{text}" for letter, text in answers.items()
    )
    return (
        f"LATEST USER MESSAGE:\n{record['question']}\n\nMATERIAL:\n"
        + "\n\n".join(material)
        + f"\n\nREFERENCE ANSWER:\n{record['answer']}\n\nCANDIDATES:\n{listed}"
    )


def verify_turn(tid, record, done, chunks, models, answerer, judge):
    sets = candidates(record, done)
    order = sorted(sets, key=lambda name: unit(tid, record["turn"], name))
    letters = {chr(65 + n): name for n, name in enumerate(order)}
    answers = {
        letter: models.ask(
            WRITER,
            answerer,
            answer_prompt(record, sets[name], done, chunks),
            effort="low",
        )
        for letter, name in letters.items()
    }
    grade = models.ask(
        JUDGE, judge, judge_prompt(record, answers, done, chunks), JUDGE_SCHEMA
    )
    marks = {v["id"].strip().upper()[-1:]: v["correct"] for v in grade["verdicts"]}
    return {
        "turn": record["turn"],
        "supported": grade["reference_supported"],
        "problem": grade["problem"],
        "key_facts": grade["key_facts"],
        "passes": {name: bool(marks.get(letter)) for letter, name in letters.items()},
        "sets": sets,
    }


def needed(check):
    """Whether history is needed, and the sets of earlier parts that each suffice."""
    passes = check["passes"]
    if not check["supported"]:
        return "unknown", []
    if passes["cur"]:
        return "no", [[]]
    names = [n for n in ("prev", "in", "out", "ret") if passes.get(n)] or [
        n for n in ("full",) if passes.get(n)
    ]
    return ("yes", [check["sets"][n] for n in names]) if names else ("unknown", [])


# Assemble: threads in the canonical format, labels, grading notes, split and manifest.


def idle_seconds(clock, *key):
    spec = CLOCKS[clock]
    return max(
        1,
        round(
            spec["median_idle_seconds"]
            * math.exp(spec["sigma"] * NormalDist().inv_cdf(unit(*key)))
        ),
    )


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def assemble_thread(plan, written, checks, chunks, *, clock, seed, model):
    tid = plan["thread_id"]
    done = {r["turn"]: r for r in written}
    run_of = {r["turn"]: n for n, r in enumerate(written, 1)}
    part = {}
    for record in written:
        n = run_of[record["turn"]]
        part[record["turn"], "input"] = f"{tid}-u{n:03d}"
        part[record["turn"], "output"] = f"{tid}-a{n:03d}"
    runs, labels, notes, history, visible = [], [], [], [], {}
    cursor = EPOCH + timedelta(days=int(unit(seed, tid, "day") * 28))
    for record, check in zip(written, checks):
        n = run_of[record["turn"]]
        run_id = f"{tid}-run{n:03d}"
        started = cursor + timedelta(
            seconds=idle_seconds(clock, seed, tid, n) if n > 1 else 0
        )
        cursor = started + timedelta(
            seconds=max(
                1,
                round(
                    FIRST_TOKEN_SECONDS
                    + len(record["answer"]) / 4 / OUTPUT_TOKENS_PER_SECOND
                ),
            )
        )
        retrievals = [
            {
                "part_id": f"{tid}-r{n:03d}-{k}",
                "text": chunks[i]["text"],
                "document_id": chunks[i]["document_id"],
                "version": chunks[i]["version"],
                "chunk_id": chunks[i]["chunk_id"],
                "exposed_to_model": True,
            }
            for k, i in enumerate(record["retrieved"], 1)
        ]
        gold = next(
            (
                r["part_id"]
                for r, i in zip(retrievals, record["retrieved"])
                if i == record["chunk"]
            ),
            None,
        )
        part[record["turn"], "retrieval"] = gold
        ids = [part[record["turn"], "input"], *(r["part_id"] for r in retrievals)]
        runs.append(
            {
                "run_id": run_id,
                "run_index": n,
                "started_at": iso(started),
                "ended_at": iso(cursor),
                "history_policy": "full_canonical",
                "input": {"part_id": ids[0], "text": record["question"]},
                "retrievals": retrievals,
                "output": {
                    "part_id": part[record["turn"], "output"],
                    "text": record["answer"],
                },
                "generation": {
                    "observation_id": run_id + "-g",
                    "model": model,
                    "started_at": iso(started),
                    "ended_at": iso(cursor),
                    "usage": None,
                    "request_part_ids": history + ids,
                },
            }
        )
        current = {
            "input": [runs[-1]["input"]],
            "retrieval": retrievals,
            "output": [runs[-1]["output"]],
        }
        for kind in ("input", "retrieval"):
            for p in current[kind]:
                visible[p["part_id"]] = {
                    "part_id": p["part_id"],
                    "origin_run_id": run_id,
                    "origin_run_index": n,
                    "kind": kind,
                    "text": p["text"],
                    "content_status": "available",
                }

        def source(part_id, quotes):
            text = visible[part_id]["text"]
            return {
                "part_id": part_id,
                "quote": next((q for q in quotes if q in text), text[:200]),
            }

        need, sets = needed(check)
        known = {(origin(done, r) or (0, None))[1] for r in record["refs"]}
        again = next(
            (
                r["part_id"]
                for r, i in zip(retrievals, record["retrieved"])
                if i in known
            ),
            None,
        )
        # The excerpt the turn was written from is part of every set that made the answer right.
        here = [source(gold or again or ids[0], record["quotes"])] if gold else []
        labels.append(
            {
                "thread_id": tid,
                "run_id": run_id,
                # Each alternative let a model answer correctly with the rest of the history withheld.
                "needs": None
                if need == "unknown"
                else [
                    [
                        source(part[tuple(p)], done[p[0]]["quotes"] + record["quotes"])
                        for p in earlier
                    ]
                    + here
                    for earlier in sets
                    if earlier
                ],
            }
        )
        notes.append(
            {
                "run_id": run_id,
                "intent": record["intent"],
                "planned_intent": record["planned"],
                "builds_on": [f"{tid}-run{run_of[r]:03d}" for r in record["refs"]],
                "history_needed": need,
                "key_facts": check["key_facts"],
                "passes": check["passes"],
                "reference_supported": check["supported"],
                "problem": check["problem"],
                "lags": sorted({n - run_of[p[0]] for earlier in sets for p in earlier}),
            }
        )
        visible[runs[-1]["output"]["part_id"]] = {
            "part_id": runs[-1]["output"]["part_id"],
            "origin_run_id": run_id,
            "origin_run_index": n,
            "kind": "output",
            "text": record["answer"],
            "content_status": "available",
        }
        history += [*ids, runs[-1]["output"]["part_id"]]
    thread = {
        "thread_id": tid,
        "company_id": plan["domain"],
        "scenario_id": f"docchat-{clock}",
        "provenance": "DocChat: public document, model-written turns, declared clock; usage unavailable",
        "runs": runs,
    }
    return thread, labels, notes


def check_contract(thread, labels):
    """The thread format, and every label against what its turn could see."""
    from compache.labels import label_index
    from compache.threads import Thread

    label_index([Thread(thread)], labels)


def split_documents(sources, salt="docchat-split-v1"):
    """A third of each domain's documents, chosen by hash of their ID, is the test set."""
    test = set()
    for domain in sources["domains"].values():
        ordered = sorted(
            (d["id"] for d in domain["documents"]), key=lambda i: digest(f"{salt}:{i}")
        )
        test.update(
            ordered[: max(1, round(len(ordered) / 3))] if len(ordered) > 1 else []
        )
    return test


def build(args, models=None):
    sources, domains = (
        read_json(HERE / "sources.json"),
        read_json(HERE / "domains.json")["domains"],
    )
    work, output = Path(args.work), Path(args.output)
    prompts = {
        name: (HERE / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip()
        for name in ("writer", "answerer", "judge")
    }
    test = split_documents(sources)
    jobs = []
    for domain in args.domains:
        # Train and test documents in turn, so a small build still has both.
        ranked = sorted(
            sources["domains"][domain]["documents"], key=lambda d: digest(d["id"])
        )
        sides = [
            [d for d in ranked if (d["id"] in test) == held] for held in (False, True)
        ]
        mixed = [
            d for i in range(len(ranked)) for side in sides for d in side[i : i + 1]
        ]
        for doc in mixed[: args.docs]:
            for tier, count in (
                ("short", args.short),
                ("medium", args.medium),
                ("long", args.long),
            ):
                jobs += [
                    (domain, doc, tier, f"{doc['id']}-{tier[0]}{n}")
                    for n in range(1, count + 1)
                ]
    corpora = {
        doc["id"]: load_corpus(doc, Path(args.cache), work)
        for doc in {d["id"]: d for _, d, _, _ in jobs}.values()
    }
    plans = [
        plan_thread(
            tid, domain, domains[domain], doc, corpora[doc["id"]], tier, args.seed
        )
        for domain, doc, tier, tid in jobs
    ]
    turns = sum(len(p["turns"]) for p in plans)
    print(
        compact(
            {
                "threads": len(plans),
                "planned_turns": turns,
                "writer_calls_about": turns * 5,
                "judge_calls": turns,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return None
    models = models or Models(work)
    indexes = {name: Index(chunks) for name, chunks in corpora.items()}

    def write(plan):
        return write_thread(
            plan,
            corpora[plan["document"]],
            indexes[plan["document"]],
            domains[plan["domain"]],
            models,
            prompts["writer"],
        )

    def verify(task):
        plan, done, record = task
        answerer = domains[plan["domain"]]["assistant"] + " " + prompts["answerer"]
        return verify_turn(
            plan["thread_id"],
            record,
            done,
            corpora[plan["document"]],
            models,
            answerer,
            prompts["judge"],
        )

    with ThreadPoolExecutor(args.workers) as pool:
        written = list(pool.map(write, plans))
        print(compact({"written_turns": sum(map(len, written))}), flush=True)
        tasks = [
            (plan, {r["turn"]: r for r in records}, record)
            for plan, records in zip(plans, written)
            for record in records
        ]
        checks = iter(list(pool.map(verify, tasks)))
    results = [
        {"plan": plan, "written": records, "checks": [next(checks) for _ in records]}
        for plan, records in zip(plans, written)
    ]
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Choose a new or empty output directory")
    entries = []
    for result in results:
        plan = result["plan"]
        thread, labels, notes = assemble_thread(
            plan,
            result["written"],
            result["checks"],
            corpora[plan["document"]],
            clock=args.clock,
            seed=args.seed,
            model=args.model,
        )
        check_contract(thread, labels)
        tid = plan["thread_id"]
        write_gz(output / "traces" / f"{tid}.jsonl.gz", [thread])
        write_gz(output / "labels" / f"{tid}.jsonl.gz", labels)
        write_gz(output / "grading" / f"{tid}.jsonl.gz", notes)
        need = Counter(n["history_needed"] for n in notes)
        entries.append(
            {
                "thread_id": tid,
                "domain": plan["domain"],
                "document": plan["document"],
                "tier": plan["tier"],
                "split": "test" if plan["document"] in test else "train",
                "turn_count": len(notes),
                "history_yes": need["yes"],
                "history_no": need["no"],
                "history_unknown": need["unknown"],
                "planning_tokens": sum(
                    math.ceil(len(p["text"].encode()) / 4)
                    for r in thread["runs"]
                    for p in [r["input"], r["output"], *r["retrievals"]]
                ),
                "trace_path": f"traces/{tid}.jsonl.gz",
                "labels_path": f"labels/{tid}.jsonl.gz",
                "grading_path": f"grading/{tid}.jsonl.gz",
            }
        )
    used = {e["document"] for e in entries}
    manifest = {
        "schema_version": SCHEMA,
        "writer": WRITER,
        "judge": JUDGE,
        "prompts_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in prompts.items()
        },
        "seed": args.seed,
        "model": args.model,
        "clock": {"scenario": args.clock, **CLOCKS[args.clock]},
        "split": "A third of each domain's documents, by SHA-256 of 'docchat-split-v1:<document id>', is the test set. Threads follow their document.",
        "labels": "A turn needs history when a model answering with no earlier turns shown was graded wrong and at least one set of earlier parts made it right.",
        "sources": {
            name: {k: v for k, v in domain.items() if k != "documents"}
            | {"documents": [d for d in domain["documents"] if d["id"] in used]}
            for name, domain in sources["domains"].items()
            if name in args.domains
        },
        "thread_count": len(entries),
        "turn_count": sum(e["turn_count"] for e in entries),
        "threads": entries,
    }
    write_json(output / "manifest.json", manifest)
    print(
        compact(
            {k: v for k, v in manifest.items() if k in ("thread_count", "turn_count")}
            | {"new_calls": dict(models.new), "output": str(output)}
        )
    )
    return manifest


def lock(args):
    """Fetch and parse every source and record the hash of its text."""
    path = HERE / "sources.json"
    sources = read_json(path)
    for domain in sources["domains"].values():
        for doc in domain["documents"]:
            chunks = load_corpus(doc, Path(args.cache), Path(args.work), lock=True)
            print(
                doc["id"],
                len(chunks),
                "chunks,",
                sum(c["evidence"] for c in chunks),
                "usable as evidence",
            )
    write_json(path, sources)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    locking, command = commands.add_parser("lock"), commands.add_parser("build")
    for each in (locking, command):
        each.add_argument(
            "--cache", required=True, help="Folder for downloaded source files"
        )
        each.add_argument(
            "--work",
            required=True,
            help="Folder for parsed documents and stored model answers",
        )
    command.add_argument("--output", required=True)
    command.add_argument(
        "--domains",
        nargs="+",
        default=list(read_json(HERE / "domains.json")["domains"]),
    )
    command.add_argument("--docs", type=int, default=10, help="Documents per domain")
    for tier, default in (("short", 1), ("medium", 1), ("long", 1)):
        command.add_argument(
            f"--{tier}",
            type=int,
            default=default,
            help=f"{tier.capitalize()} threads per document",
        )
    command.add_argument("--clock", choices=sorted(CLOCKS), default="mixed")
    command.add_argument("--seed", type=int, default=20261120)
    command.add_argument(
        "--model", default="replay", help="Model ID written to every generation"
    )
    command.add_argument("--workers", type=int, default=8)
    command.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the threads and stop before any model call",
    )
    args = parser.parse_args(argv)
    (lock if args.command == "lock" else build)(args)


if __name__ == "__main__":
    main()
