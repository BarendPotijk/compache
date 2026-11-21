"""Apply a fitted policy turn by turn.

Fit first, from the repository root:

    compache fit examples/quickstart/threads.jsonl.gz \
        --labels examples/quickstart/labels.jsonl --model claude-sonnet-5-5

Then run this file. It replays one recorded conversation as if it were live, so
it needs no provider key: where it takes the recorded answer, your application
sends `turn.messages` to the model.
"""

from datetime import datetime
from pathlib import Path

import compache

HERE = Path(__file__).resolve().parent

policy = compache.Policy.load("policy.json")
recorded = compache.load_threads(HERE / "threads.jsonl.gz")[1]
session = policy.session("live-conversation")
spec = policy.spec

print(f"{'turn':>4}  {'sent':>8}  {'full history':>12}  {'archived':>8}  next ping")
full = 0
for run in recorded.runs:
    start = datetime.fromisoformat(run.started_at.replace("Z", "+00:00")).timestamp()
    end = datetime.fromisoformat(run.ended_at.replace("Z", "+00:00")).timestamp()
    retrieved = [part.text for part in run.retrievals]

    turn = session.prepare(run.input.text, retrieved, at=start)
    # Your model call goes here: send turn.messages, read the answer and its usage.
    answer = run.output.text
    due = session.complete(turn, answer, at=end)

    full += sum(spec.part_tokens(text) for text in (run.input.text, *retrieved))
    wait = "none" if due is None else f"in {due - end:.0f} s"
    print(
        f"{run.run_index:>4}  {turn.tokens:>8,}  {full:>12,}  {len(turn.archived_ids):>8}  {wait}"
    )
    full += spec.part_tokens(answer)
