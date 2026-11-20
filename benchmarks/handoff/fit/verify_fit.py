"""Verify the frozen policies and, optionally, the aggregate proof of a regenerated corpus."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_ids(first, last, episode_count):
    return [f"scaled-{company:02d}-{episode:02d}" for company in range(first, last + 1)
            for episode in range(1, episode_count + 1)]


def verify_fit(proof):
    first, last = proof["fit_company_range_inclusive"]
    selected = case_ids(first, last, proof["episodes_per_company"])
    # The optimizer fingerprints a canonical JSON string, including its quotes.
    fingerprints = sorted(hashlib.sha256(json.dumps(tid).encode()).hexdigest() for tid in selected)
    for minutes in (5, 30):
        name = f"policy-a-{minutes}m.json.gz"
        content = gzip.decompress((HERE / name).read_bytes())
        require(hashlib.sha256(content).hexdigest() == proof["policy_json_sha256"][str(minutes)],
                f"Changed policy payload: {name}")
        policy = json.loads(content)
        require(policy["schema_version"] == "compache-workload-planner-v1", "Policy schema")
        require(policy["training_threads"] == fingerprints, "Training thread selection")
        require(policy["demand"]["training_runs"] == proof["fit_turns"], "Policy training size")
        require(policy["model_spec"]["cache_ttl_seconds"] == 60 * minutes, "Cache window")
        require(policy["retention_floor"] == 0.95, "Retention gate")
        require(len(policy["recipes"]) == 2 and all(not r["eligible"] for r in policy["recipes"]),
                "Recipe admission")
    print(f"Verified the two frozen policies, {len(selected)} training threads and "
          f"{proof['fit_turns']:,} training turns.")


def verify_generated(directory, proof):
    manifest_path = directory / "manifest.json"
    require(file_hash(manifest_path) == proof["manifest_sha256"], "Full manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["seed"] == proof["seed"] and manifest["schema_version"] == proof["corpus_schema"],
            "Corpus generation settings differ")
    expected = case_ids(1, proof["company_count"], proof["episodes_per_company"])
    rows = manifest["threads"]
    require([r["thread_id"] for r in rows] == expected, "Case selection or order differs")
    require(sum(r["turn_count"] for r in rows) == proof["total_turns"], "Corpus size differs")

    files = sorted(p for group in ("labels", "traces") for p in (directory / group).glob("*.jsonl.gz"))
    require(len(files) == 2 * len(rows), "Unexpected number of shards")
    # Paths and compressed-file SHA-256 digests enter in relative-path lexical order.
    # The public proof stores only this single final root.
    combined = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode("ascii")
        combined.update(name + b"\0" + bytes.fromhex(file_hash(path)))
    require(combined.hexdigest() == proof["shards_sha256"], "Combined shard SHA-256 differs")
    print(f"Verified the full authoring manifest and all {len(files)} generated shards "
          f"for {len(rows)} conversations and {proof['total_turns']:,} turns.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="Regenerated full 300-thread corpus directory")
    args = parser.parse_args()
    proof = json.loads((HERE / "proof.json").read_bytes())
    verify_fit(proof)
    if args.corpus is not None:
        verify_generated(args.corpus, proof)


if __name__ == "__main__":
    main()
