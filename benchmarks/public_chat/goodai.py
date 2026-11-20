"""Prepare and score a fixed-schedule GoodAI LTM pilot from a pinned checkout."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
from pathlib import Path
import re

REVISION = "188e7618413775f1ce783763d5ee0b5ccd4c31c9"
REPO = f"https://github.com/GoodAI/goodai-ltm-benchmark/tree/{REVISION}"
SUITE = Path("data/tests/Benchmark 3 - 32k")
LOG = SUITE / "results/LLMChatSession - gpt-4o - 128000/master_log.jsonl"
LOG_BLOB = "f601ed36f0adf2ede483d42f88982725392c45a7"
DEFINITIONS = {
    "Colours": ("a35f536a54941584292e08d891ca3e2f7c2e9677", "23efadea2c861970e64639274f9a272aa07e0662", "1353a5e190b7e21607f1d44e65dba805a25521c6"),
    "NameList": ("4ae83987243cd70f22d9a244299c211c5d1560ed", "f5678f99c009f4ce7fe4eb9e47bc3318fe6435ea", "58048853b716c386e5e46aef3cff56b2e9f9ef68"),
    "Shopping": ("77e7e86a8d5c608fd1cc3e982dd8365d3cf6eced", "b80551038cbcf154bb655b5f948ea2ba89477f05", "fabc34586ccbde65ee54195faba997bc532ec21b"),
}


def verified(path, blob):
    raw = path.read_bytes()
    if sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != blob:
        raise ValueError(f"GoodAI source differs from the pinned revision: {path.name}")
    return raw


def message_blocks(records):
    """Group native user/response pairs by published reset marker, ignoring simulator events."""
    blocks, block, user = [], [], None
    for record in records:
        kind = record["type"]
        if kind == 7:
            if user is not None:
                raise ValueError("Reset inside one exchange")
            blocks.append(block)
            block = []
        elif kind in (0, 3):
            if user is not None:
                raise ValueError("Missing source reply")
            user = record
        elif kind in (4, 5):
            expected_user_kind = 0 if kind == 4 else 3
            if user is None or record["test_id"] != user["test_id"] or user["type"] != expected_user_kind:
                raise ValueError("Invalid source exchange")
            block.append((user, record))
            user = None
    if user is not None:
        raise ValueError("Source ends before reply")
    return [*blocks, block]


def select_block(block, definitions, index):
    messages = defaultdict(list)
    for user, _ in block:
        messages[user["test_id"]].append(user["data"]["message"])
    selected = {}
    for task, definition in definitions.items():
        script = definition["script"]
        if messages[task][:len(script)] == script:
            selected[task] = definition
    if not selected:
        return None
    # Keep the chronological history up to and including the final real check.
    # Do not send trailing reset or filler after the last measured turn.
    final = max(i for i, (u, _) in enumerate(block) if u["test_id"] in selected and u["data"]["is_question"])
    counts = Counter()
    pending = []
    turns = []
    for i, (user, reply) in enumerate(block[:final + 1], 1):
        task = user["test_id"]
        if user["type"] == 3:
            pending.extend([{"role": "user", "content": user["data"]["message"]},
                            {"role": "assistant", "content": reply["data"]["message"]}])
        elif task in selected:
            question = user["data"]["message"]
            record = {"question": question, "contexts": [], "source_turn": i, "source_task": task}
            if pending:
                record["prefix"] = pending
                pending = []
            if user["data"]["is_question"]:
                definition = selected[task]
                if question != definition["script"][-1]:
                    raise ValueError("Unexpected task question position")
                record["score"] = {"family": task.split(" - ")[0], "expected": definition["expected_responses"]}
            turns.append(record)
            counts[task] += 1
    return {"id": f"goodai-32k-block-{index}", "turns": turns,
            "complete_source_tasks": sorted(counts),
            "source_exchanges_in_reset_block": len(block),
            "frozen_filler_exchanges": sum(len(r.get("prefix", [])) // 2 for r in turns),
            "instructions": ("This is a conversational memory benchmark. Answer the latest user message extremely briefly. "
                             "For updates, provide a short acknowledgement. Follow the requested response format.")}


def prepare(checkout, *, master_log=None):
    expected = {}
    for family, blobs in DEFINITIONS.items():
        for i, blob in enumerate(blobs):
            data = verified(checkout / SUITE / "definitions" / family / f"{i}.def.json", blob)
            expected[f"{family} - {i}"] = json.loads(data)
    log = verified(master_log or checkout / LOG, LOG_BLOB)
    blocks = message_blocks(json.loads(line) for line in log.splitlines())
    cases = [case for i, block in enumerate(blocks)
             if (case := select_block(block, expected, i))]
    return {"benchmark": "goodai", "revision": REVISION, "source": REPO,
            "master_log": str(LOG), "master_log_sha256": sha256(log).hexdigest(),
            "design": ("Paired fixed-schedule adaptation using complete procedural task instances wholly inside a reset "
                       "block. Original unscored trivia question/answer pairs appear as frozen history. "
                       "Each arm generates all retained task updates and probes. Other task families are omitted. "
                       "32k denotes the upstream configuration, not the resulting context length."),
            "cases": cases}


def _answer_list(text):
    """Accept JSON lists, optionally wrapped in a Markdown JSON fence."""
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    value = json.loads(fence.group(1) if fence else stripped)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return value


def grade_turn(spec, answer):
    """Use task-rubric fractional credit and separately test exact answer completeness."""
    kind, expected = spec["family"], spec["expected"]
    if kind == "Colours":
        # GoodAI's original evaluator uses a case-insensitive substring.
        fraction = float(expected[0].lower() in answer.lower())
        return {"fraction": fraction, "passed": bool(fraction), "family": kind}
    if kind not in ("NameList", "Shopping"):
        raise ValueError("Unknown GoodAI task family")
    try:
        values = _answer_list(answer)
        if kind == "NameList":
            wanted = Counter(s.lower() for s in expected)
            got = Counter(s.lower() for s in values)
            fraction = sum((wanted & got).values()) / max(len(expected), len(values))
            passed = got == wanted
        else:
            wanted = {name.lower(): quantity for name, quantity in expected}
            matched = []
            correct_keys = []
            extras = 0
            for item in values:
                name, count = item["item"].lower(), item["quantity"]
                if type(count) is not int:
                    raise ValueError("Invalid shopping quantity")
                base = next((key for key in wanted if key in name), None)
                if base is None:
                    extras += 1
                else:
                    matched.append(base)
                    if count == wanted[base]:
                        correct_keys.append(base)
            # Unique matches prevent the upstream substring rule rewarding duplicate rows.
            # Preserve the upstream thirds for item presence, quantity, and no additions.
            fraction = (len(set(matched)) / len(wanted) + len(set(correct_keys)) / len(wanted) + int(extras == 0)) / 3
            passed = sorted(matched) == sorted(wanted) and len(correct_keys) == len(wanted) and extras == 0
    except (ValueError, AttributeError, KeyError, TypeError):
        fraction, passed = 0.0, False
    return {"fraction": fraction, "passed": passed, "family": kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--master-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.checkout, master_log=args.master_log)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
