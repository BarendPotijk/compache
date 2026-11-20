"""Contract and semantic checks for the authored scaled workload."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_scaled_rag_workload import LENGTHS, build, make_thread, profile
from scaled_rag_dataset import iter_threads, load_thread, export_thread
from compache.traces import Thread, load_threads
from compache.dependencies import build_packet, validate_label


def test_deterministic_shards_and_export(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma = build(a, selected=[(26, 6)])
    mb = build(b, selected=[(26, 6)])
    assert ma == mb
    entry = ma["threads"][0]
    for key, hashkey in [
        ("trace_path", "trace_sha256"),
        ("labels_path", "labels_sha256"),
    ]:
        assert (a / entry[key]).read_bytes() == (b / entry[key]).read_bytes()
        assert (
            hashlib.sha256((a / entry[key]).read_bytes()).hexdigest() == entry[hashkey]
        )
    t, labels, meta = load_thread(a, entry["thread_id"])
    assert next(iter_threads(a)) == (t, labels, meta)
    export_thread(a, entry["thread_id"], tmp_path / "export")
    assert (
        load_threads(tmp_path / "export" / "threads.jsonl")[0].thread_id
        == entry["thread_id"]
    )
    assert len((tmp_path / "export" / "labels.jsonl").read_text().splitlines()) == 96
    assert json.loads((tmp_path / "export" / "manifest.json").read_text())[
        "heldout_thread_ids"
    ] == [entry["thread_id"]]


def test_truth_visibility_and_late_dependencies():
    thread, labels, meta = make_thread(29, 6, profile(29))
    parsed = Thread.from_dict(thread)
    assert len(thread["runs"]) == 96
    assert meta["retrieval_planning_tokens"]["min"] >= 15000
    assert meta["retrieval_planning_tokens"]["max"] <= 50000
    assert all(r["generation"]["usage"] is None for r in thread["runs"])
    for label in labels:
        assert not validate_label(label, build_packet(parsed, label["target_run_id"]))
    designs = meta["run_design"]
    assert sum(bool(x["lag"]) for x in designs) >= 16
    assert sum(bool(x["lag"]) for x in designs[50:]) >= 8
    assert sum(x["lag"] >= 21 for x in designs[80:]) >= 3
    assert {
        "input",
        "history",
        "prior_output",
        "mixed",
        "version",
        "correction",
        "fresh",
        "abstention",
    } <= set(meta["families"])
    for i, d in enumerate(designs):
        if not d["lag"]:
            continue
        source = i - d["lag"]
        fact = designs[source]["current_expected"]
        for r in thread["runs"][source + 1 : i]:
            assert fact not in r["input"]["text"]
            assert fact not in r["output"]["text"]
            assert all(fact not in p["text"] for p in r["retrievals"])
        if d["family"] == "version":
            current = thread["runs"][i]["retrievals"][0]["text"]
            assert "superseding the original" in current
            assert fact not in current
        if d["family"] == "fresh":
            assert fact in thread["runs"][i]["retrievals"][0]["text"]


def test_split_and_length_definition():
    assert len(LENGTHS) == 10 and sum(LENGTHS) * 30 == 23760
    assert profile(1) != profile(2)
    assert profile(26) == profile(26)


def test_all_company_bundles_use_package_planning_metric():
    from compache.optimizer.contracts import ModelSpec

    model = ModelSpec(model_id="planning")
    for n in range(1, 31):
        thread, _, meta = make_thread(n, 6, profile(n))
        sizes = [model.part_tokens(r["retrievals"][0]["text"]) for r in thread["runs"]]
        assert min(sizes) >= 15000 and max(sizes) <= 50000
        assert meta["retrieval_planning_tokens"] == {
            "min": min(sizes),
            "max": max(sizes),
        }


def test_domain_questions_opaque_cases_and_separate_corrections():
    thread, labels, meta = make_thread(29, 6, profile(29))
    designs = meta["run_design"]
    assert len({d["domain"] for d in designs}) >= 4
    assert len({d["answer_position"] for d in designs}) > 20
    assert len(set(meta["case_ids"])) == len(meta["case_ids"])
    from build_scaled_rag_workload import opaque

    assert all(
        c != opaque(thread["thread_id"], i) for i, c in enumerate(meta["case_ids"], 1)
    )
    assert (
        sum("my initial allocation" in r["input"]["text"] for r in thread["runs"]) < 75
    )
    corrections = [
        (i, l)
        for i, (d, l) in enumerate(zip(designs, labels))
        if d["family"] == "correction"
    ]
    assert corrections
    for i, label in corrections:
        sources = label["task_requirements"][1]["alternatives"][0]["sources"]
        assert len(sources) == 2 and sources[0]["part_id"] != sources[1]["part_id"]
        assert "initial allocation" in sources[0]["quote"]
        assert "Correction for procurement case" in sources[1]["quote"]
        assert (
            sources[0]["part_id"] in thread["runs"][i]["generation"]["request_part_ids"]
        )
        assert (
            sources[1]["part_id"] in thread["runs"][i]["generation"]["request_part_ids"]
        )
    for i, d in enumerate(designs):
        if d["family"] == "fresh":
            assert (
                "turn "
                not in thread["runs"][i]["input"]["text"].split("Also report")[-1]
            )
            assert (
                d["historical_expected"] in thread["runs"][i]["retrievals"][0]["text"]
            )
    assert all(
        d["actual_prior_output_required"] == (d["family"] == "prior_output")
        for d in designs
    )


# The repository uses unittest discovery; keep the standalone helpers above
# callable while exposing each check as an individual discovered test.
import tempfile
import unittest


class ScaledRagWorkloadTests(unittest.TestCase):
    def test_deterministic_shards_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            test_deterministic_shards_and_export(Path(directory))

    def test_truth_visibility_and_late_dependencies(self):
        test_truth_visibility_and_late_dependencies()

    def test_split_and_length_definition(self):
        test_split_and_length_definition()

    def test_all_company_bundles_use_package_planning_metric(self):
        test_all_company_bundles_use_package_planning_metric()

    def test_domain_questions_opaque_cases_and_separate_corrections(self):
        test_domain_questions_opaque_cases_and_separate_corrections()
