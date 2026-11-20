"""Exercise the frozen synthetic-RAG policies without a model or credentials."""
import gzip
import hashlib
import json
from pathlib import Path

from compache.optimizer import CacheBreakChecker, CacheScope, Request, Source, WorkloadPolicy, WorkloadSession


HERE = Path(__file__).resolve().parent
SHA256 = {
    5: "89a4c2cecc5fd2c6e53f0461d3a4c6e530212202dc916a87a60993582a5c2ace",
    30: "96b16e6026698aef2845197377cf80d0d96596a43b5f32c3e50dd8d4286c4248",
}


def exercise(minutes):
    data = gzip.decompress((HERE / f"policy-a-{minutes}m.json.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != SHA256[minutes]:
        raise ValueError(f"The {minutes}-minute policy does not match its recorded fit")
    artifact = json.loads(data)
    policy = WorkloadPolicy(artifact)
    model = policy.spec.model_id
    scope = CacheScope("demo", "offline", model, "dispatch")
    header = {"system_prompt": "Use only the supplied dispatch notices.", "tools": []}
    question = Source("dispatch/1/in", "input", 1, "When does the desk close?")
    notice = Source("dispatch/1/notice", "retrieval", 1, "Desk closes at 17:40.", "notice", "1", "body")
    answer = Source("dispatch/1/out", "output", 1, "17:40.")
    followup = Source("dispatch/2/in", "input", 2, "Repeat the closing time.")
    with CacheBreakChecker() as guard:
        session = WorkloadSession(policy, scope, guard)
        first = session.prepare(Request("dispatch", "dispatch/1", 1, model, (question, notice)),
                                started_at=0, **header)
        session.complete(first, answer, ended_at=12)
        next_request = Request("dispatch", "dispatch/2", 2, model, (question, notice, answer, followup))
        second = session.prepare(next_request, started_at=372, **header)
        return {
            "cache_minutes": minutes,
            "fitting_threads": len(policy.training_threads),
            "fitting_turns": artifact["demand"]["training_runs"],
            "candidate_recipes": len(artifact["recipes"]),
            "admitted_recipes": len(policy.recipes),
            "return_gap_seconds": 360,
            "actions": [first.decision["action"], second.decision["action"]],
            "second_estimated_input_tokens": second.view.tokens,
        }


if __name__ == "__main__":
    print(json.dumps([exercise(minutes) for minutes in SHA256], indent=2))
