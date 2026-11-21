"""A small synthetic thread and matching labels, shared by the tests."""

from datetime import datetime, timedelta, timezone

from compache.labels import build_packet
from compache.threads import Thread


def thread(tid="t", company="a", n=3, gap=600, body_size=1000):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runs = []
    history = []
    for i in range(1, n + 1):
        rid = f"{tid}-r{i}"
        start = origin + timedelta(seconds=(i - 1) * gap)
        finish = start + timedelta(seconds=30)
        inp = {"part_id": rid + "-in", "text": "Describe this unrelated new item."}
        ret = {
            "part_id": rid + "-doc",
            "text": f"Unique document {i}. " + "x " * body_size,
            "document_id": f"doc-{i}",
            "version": "v1",
            "chunk_id": "one",
            "exposed_to_model": True,
        }
        out = {"part_id": rid + "-out", "text": "An observed final answer."}
        ids = history + [inp["part_id"], ret["part_id"]]
        runs.append(
            {
                "run_id": rid,
                "run_index": i,
                "started_at": start.isoformat(),
                "ended_at": finish.isoformat(),
                "history_policy": "full_canonical",
                "input": inp,
                "retrievals": [ret],
                "output": out,
                "generation": {
                    "observation_id": rid + "-g",
                    "model": "test",
                    "started_at": start.isoformat(),
                    "ended_at": finish.isoformat(),
                    "usage": None,
                    "request_part_ids": ids,
                },
            }
        )
        history = ids + [out["part_id"]]
    return Thread.from_dict({"thread_id": tid, "company_id": company, "runs": runs})


def labels_for(t, need_old=False):
    rows = []
    for r in t.runs:
        p = build_packet(t, r.run_id)
        src = next(
            s
            for s in p["request_parts"]
            if s["kind"] == "retrieval"
            and s["origin_run_index"]
            == (1 if need_old and r.run_index > 1 else r.run_index)
        )
        task = {
            "requirement_id": "t1",
            "description": "Required source content",
            "source_kind": "retrieval",
            "necessity": "inferred",
            "alternatives": [
                {"sources": [{"part_id": src["part_id"], "quote": src["text"]}]}
            ],
            "evidence_note": "Test source requirement",
            "question_quote": r.input.text,
            "fulfillment": "fulfilled",
            "historical_context_needed": "yes"
            if src["origin_run_index"] < r.run_index
            else "no",
        }
        rows.append(
            {
                "schema_version": "compache-dependency-label-v1",
                "thread_id": t.thread_id,
                "target_run_id": r.run_id,
                "status": "complete",
                "coverage_notes": [],
                "task_requirements": [task],
                "claims": [
                    {
                        "claim_id": "c1",
                        "target_quote": r.output.text,
                        "relation": "uncertain",
                        "requirements": [],
                    }
                ],
                "historical_dependencies": [],
                "unresolved": [],
            }
        )
    return rows


END = "2026-01-10T00:00:00+00:00"
