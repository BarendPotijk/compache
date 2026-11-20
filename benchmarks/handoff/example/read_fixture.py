"""Read-only synthetic RAG adapter example. Uses only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINGERPRINTS = {
    "fit.json": "ebe12e188e49a25e96d573f8555ba3a4c2d1f9dbd05f519291934bce14c745ba",
    "serve.json": "a50ed7150c6824812619021f69859fc581bc7069050fd5d437dd2a1d774824b2",
    "grading.json": "0dc44ff265f59ae25692c17662d4a073d74fc3f387b9da08012bff9bbcf1ed6f",
}


def load(name):
    data = (HERE / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != FINGERPRINTS[name]:
        raise ValueError(f"Fixture fingerprint mismatch: {name}")
    return json.loads(data)


def sources(trace, turn):
    """Emit only information introduced by the requested foreground call."""
    if not 1 <= turn <= len(trace["turns"]):
        raise ValueError(f"Turn {turn} is outside the thread")
    parts = []
    for number, record in enumerate(trace["turns"][:turn], start=1):
        prefix = f'{trace["thread"]}/{number}'
        parts.append({"id": f"{prefix}/in", "kind": "input", "text": record["input"]})
        for chunk in record["retrieved"]:
            parts.append({"id": chunk["id"], "kind": "retrieval", "text": chunk["text"]})
        if number < turn:
            parts.append({"id": f"{prefix}/out", "kind": "output", "text": record["output"]})
    return parts


def idle_gaps(trace):
    turns = trace["turns"]
    if any(t["end_s"] < t["start_s"] for t in turns):
        raise ValueError("Reversed foreground interval")
    gaps = [b["start_s"] - a["end_s"] for a, b in zip(turns, turns[1:])]
    if any(gap < 0 for gap in gaps):
        raise ValueError("Overlapping turns")
    return gaps


def retained(requirements, visible):
    return sum(any(set(group) <= visible for group in req["alternatives"]) for req in requirements)


def check():
    fit, serving, grading = load("fit.json"), load("serve.json"), load("grading.json")
    if fit["family"] == serving["family"] or fit["thread"] == serving["thread"]:
        raise ValueError("Development and evaluation identities overlap")
    for data in (fit, serving):
        idle_gaps(data)
        for turn in range(1, len(data["turns"]) + 1):
            parts = sources(data, turn)
            ids = [part["id"] for part in parts]
            if len(ids) != len(set(ids)) or any(not part["text"] for part in parts):
                raise ValueError("Duplicate source ID or empty message")
            record = data["turns"][turn - 1]
            # Alter the current answer and add evaluator-only metadata in memory.
            # Neither is available to the current request. Past recorded answers are.
            altered = {**record, "output": "PRESENT_ANSWER_SENTINEL", "target": "KEY_SENTINEL"}
            poisoned = {**data, "turns": data["turns"][:turn - 1] + [altered]}
            if sources(poisoned, turn) != parts:
                raise ValueError("Current answer, evaluator-only fields or future turns affect serving")
    if serving["thread"] != grading["thread"]:
        raise ValueError("Wrong scorecard")
    for label in fit["training_labels"]:
        fit_visible = {p["id"] for p in sources(fit, label["turn"])}
        if any(not group or not set(group) <= fit_visible for group in label["alternatives"]):
            raise ValueError("Development label references an unavailable source")
    visible = {p["id"] for p in sources(serving, grading["turn"])}
    required = grading["requirements"]
    for req in required:
        for group in req["alternatives"]:
            if not group or not set(group) <= visible:
                raise ValueError("Invalid evaluator evidence group")
    kept = retained(required, visible)
    without_direct = retained(required, visible - {"harbor/2/r2"})
    missing_both = retained(required, visible - {"harbor/2/r2", "harbor/2/r3"})
    if (kept, without_direct, missing_both) != (3, 3, 2):
        raise ValueError("Evidence-set grading regression")
    return {"fit_turns": len(fit["turns"]), "evaluation_turns": len(serving["turns"]),
            "idle_gaps_s": idle_gaps(serving), "requirements_kept": kept,
            "kept_with_alternative": without_direct, "kept_after_losing_both_paths": missing_both}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("fit-input")
    serve = commands.add_parser("serve")
    serve.add_argument("--turn", type=int, required=True)
    score = commands.add_parser("score")
    score.add_argument("--answer", required=True)
    score.add_argument("--visible", nargs="*", required=True, metavar="SOURCE_ID")
    args = parser.parse_args()
    if args.command == "check":
        result = check()
    elif args.command == "fit-input":
        result = load("fit.json")
    elif args.command == "serve":
        result = sources(load("serve.json"), args.turn)
    else:
        grading = load("grading.json")
        visible = set(args.visible)
        allowed = {p["id"] for p in sources(load("serve.json"), grading["turn"])}
        if not visible <= allowed:
            parser.error("A reported visible source was never introduced by this turn")
        result = {"exact_answer": args.answer.strip() == grading["expected_answer"],
                  "requirements_kept": retained(grading["requirements"], visible),
                  "requirements_assessed": len(grading["requirements"])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
