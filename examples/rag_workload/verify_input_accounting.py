"""Run with Python 3.11+ and Compache 0.6 installed. Uses a local two-turn fixture."""

import json
from pathlib import Path

from compache.optimizer import (
    CacheBreakChecker,
    CacheScope,
    Request,
    Source,
    WorkloadPolicy,
    WorkloadSession,
)


def main():
    artifact = json.loads(
        Path(__file__).with_name("input-accounting-example-policy.json").read_text()
    )
    policy = WorkloadPolicy(artifact, lookahead=1, max_pings=0, reserve_tokens=100)
    q1 = Source("question1", "input", 1, "Q" * 30)
    d1 = Source("retrieval1", "retrieval", 1, "D" * 1000)
    a1 = Source("answer1", "output", 1, "A" * 40)
    q2 = Source("question2", "input", 2, "q" * 20)
    d2 = Source("retrieval2", "retrieval", 2, "d" * 80)

    # Deliberately simple ruler with checkable integers, specific to this example.
    def count_block(source):
        return 12 if source.text.startswith("[Compacted passage") else len(source.text)

    with CacheBreakChecker() as guard:
        session = WorkloadSession(
            policy,
            CacheScope("demo", "local", "test", "thread"),
            guard,
            count_part_tokens=count_block,
            count_tokens=len,
        )
        header = dict(system_prompt="0123456789", tools=b"")
        first = session.prepare(
            Request("thread", "run-1", 1, "test", (q1, d1)), started_at=0, **header
        )
        meter = {
            "id": "fixture-response-1",
            "status": "completed",
            "usage": {
                "input_tokens": 1423,
                "input_tokens_details": {"cached_tokens": 128},
                "output_tokens": 900,
            },
        }
        completed = session.complete(first, a1, ended_at=12, response=meter)
        second = session.prepare(
            Request("thread", "run-2", 2, "test", (q1, d1, a1, q2, d2)),
            started_at=100,
            **header,
        )
        result = {
            "local_first_input": first.view.tokens,
            "first_response": completed["input_reconciliation"],
            "ping_projection": completed["ping_input_accounting"],
            "input_cap": 1500,
            "candidate_input_accounting": second.decision["candidate_input_accounting"],
            "candidate_costs": second.decision["costs"],
            "chosen": second.decision["action"],
            "chosen_input": second.view.tokens,
        }
        assert (
            result["local_first_input"],
            result["ping_projection"]["estimated_input_tokens"],
        ) == (1040, 1463)
        assert (
            result["candidate_input_accounting"]["keep"]["estimated_input_tokens"]
            == 1563
        )
        assert result["chosen"] == "cut-0" and result["chosen_input"] == 575
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
