# compache

**Cut the cost of long LLM conversations by fitting compaction and prompt caching to your own traffic.**

| Cost&nbsp;per&nbsp;1,000&nbsp;turns<br><sub>OpenClaw&nbsp;→&nbsp;compache</sub> | 8&#8209;turn&nbsp;threads | 30&#8209;turn&nbsp;threads | 80&#8209;turn&nbsp;threads |
|:--|:-:|:-:|:-:|
| **Claude&nbsp;Fable&nbsp;5.2**<br><sub>5&#8209;minute&nbsp;cache</sub> | **−50%**<br><sub>$852&nbsp;→&nbsp;$428</sub> | **−52%**<br><sub>$1,011&nbsp;→&nbsp;$484</sub> | **−55%**<br><sub>$1,124&nbsp;→&nbsp;$507</sub> |
| **Claude&nbsp;Opus&nbsp;5.5**<br><sub>5&#8209;minute&nbsp;cache</sub> | **−44%**<br><sub>$432&nbsp;→&nbsp;$241</sub> | **−47%**<br><sub>$515&nbsp;→&nbsp;$274</sub> | **−50%**<br><sub>$573&nbsp;→&nbsp;$289</sub> |
| **Claude&nbsp;Sonnet&nbsp;5.5**<br><sub>5&#8209;minute&nbsp;cache</sub> | **−44%**<br><sub>$216&nbsp;→&nbsp;$121</sub> | **−47%**<br><sub>$257&nbsp;→&nbsp;$137</sub> | **−50%**<br><sub>$287&nbsp;→&nbsp;$144</sub> |
| **Claude&nbsp;Haiku&nbsp;5.5**<br><sub>5&#8209;minute&nbsp;cache</sub> | **−60%**<br><sub>$46&nbsp;→&nbsp;$18</sub> | **−62%**<br><sub>$62&nbsp;→&nbsp;$23</sub> | **−30%**<br><sub>$72&nbsp;→&nbsp;$50</sub> |
| **GPT&#8209;6&nbsp;Astra**<br><sub>30&#8209;minute&nbsp;cache</sub> | **−28%**<br><sub>$856&nbsp;→&nbsp;$613</sub> | **−61%**<br><sub>$2,271&nbsp;→&nbsp;$887</sub> | **−62%**<br><sub>$2,682&nbsp;→&nbsp;$1,020</sub> |
| **GPT&#8209;5.6&nbsp;Terra**<br><sub>30&#8209;minute&nbsp;cache</sub> | **−28%**<br><sub>$172&nbsp;→&nbsp;$124</sub> | **−61%**<br><sub>$456&nbsp;→&nbsp;$179</sub> | **−62%**<br><sub>$538&nbsp;→&nbsp;$205</sub> |
| **GPT&#8209;6&nbsp;Luna**<br><sub>30&#8209;minute&nbsp;cache</sub> | **−28%**<br><sub>$8.56&nbsp;→&nbsp;$6.13</sub> | **−61%**<br><sub>$22.71&nbsp;→&nbsp;$8.87</sub> | **−62%**<br><sub>$27&nbsp;→&nbsp;$10</sub> |

![Cost per 1,000 turns on Claude Sonnet 5.5. OpenClaw: $216, $257 and $287 for 8-, 30- and 80-turn threads. compache: $121, $137 and $144.](docs/assets/openclaw-cost.svg)

Modeled with the [cost calculator](https://barendpotijk.github.io/compache/) on its default workload at three thread lengths: 25k retrieved tokens per turn, users back after a median of 4 minutes. compache's policy is fitted on one set of threads and priced on another. Both sides fetch back any earlier source a turn needs, and that call is in the price. Reproduce with `node scripts/openclaw_comparison.mjs`.

OpenClaw follows its documented defaults. With Claude it clears old tool results above 30% of the context window, and both sides end with every needed source in view. With OpenAI it is priced on its client-side summary path, which on 80-turn threads leaves 27% of what later turns need to a summary.

## Why it is cheaper

Two things drive the bill for a long thread: how much context every turn sends again, and whether the provider's cache has gone cold by the time the user returns. The right trade depends on how your users come back and how your agent uses its history. compache learns both from your recorded threads.

- **Fitted, not configured.** There are no thresholds to tune. compache works out what to keep, when to compact and when a keep-warm ping pays.
- **Chosen by cost.** Every candidate policy is replayed through the provider's cache rules at your threads' own timestamps and prices. The cheapest wins, after paying for anything it would have to fetch back.
- **Keeps what your agent uses.** A policy is admitted on evidence of which earlier parts later turns relied on. Where almost every turn builds on earlier ones, compache keeps the history and tunes only the pings.
- **Deterministic at runtime.** Applying a policy calls no model. The same history always gives the same request, so the cached prefix stays stable.

**Continuing the benchmark work?** Start with the [agent handover](docs/benchmark-handover.md) for the recorded traces, the original 5- and 30-minute policy fits, and the next offline experiment.

## Quickstart

Python 3.11 or later, installed from source.

```sh
git clone https://github.com/BarendPotijk/compache.git && cd compache
python -m pip install .

# Four synthetic demo conversations. Your own traces go here instead.
python scripts/build_scaled_rag_workload.py --select 1:1 --select 2:1 --select 4:1 --select 5:1 --output demo
gunzip -c demo/traces/*.gz > threads.jsonl
gunzip -c demo/labels/*.gz > labels.jsonl

compache planner solve --threads threads.jsonl --labels labels.jsonl \
  --model-spec examples/quickstart/model.json --header examples/quickstart/header.json \
  --observation-end 2028-01-01T00:00:00Z --output policy.json
```

In a few seconds it prints the policy it chose and what that saves on these threads (abridged):

```json
{"chosen": {"recipe": {"input_runs": 11, "output_runs": 2, "retrieval_runs": 0, "keep_under_tokens": 45},
            "max_pings": 0, "coverage": 0.944},
 "saving_fraction": 0.814}
```

Keep 11 turns of questions, 2 of answers and no earlier retrieved documents, and send no pings: 94% of later turns still see the evidence they need, at 81% lower modeled cost.

Then apply the policy on every request:

```python
import json
from compache.optimizer import CacheBreakChecker, CacheScope, Request, Source, WorkloadPolicy, WorkloadSession

policy = WorkloadPolicy(json.load(open("policy.json")))
model = policy.spec.model_id
with CacheBreakChecker() as checker:
    session = WorkloadSession(policy, CacheScope("my-app", "my-provider", model, "thread-1"), checker)
    question = Source("q1", "input", 1, "When does the desk close?")
    notice = Source("d1", "retrieval", 1, "Desk closes at 17:40.", "notices", "v1", "c1")
    plan = session.prepare(
        Request("thread-1", "run-1", 1, model, (question, notice)),
        started_at=0, system_prompt="Answer from the notices.", tools=[],
    )
    # Send plan.view.wire to your provider, then record what came back.
    session.complete(plan, Source("a1", "output", 1, "17:40."), ended_at=12)
    print(plan.decision["action"], session.next_ping_at)
```

## How it works

![Threads and labels feed offline fitting; a frozen policy drives each request and each run end](docs/assets/workflow.svg)

1. **Record.** Export your threads in the [trace format](docs/trace-contracts.md) and [label](docs/dependency-targets.md) which earlier parts later turns relied on. A [coding-agent skill](skills/compache-label-dependencies/SKILL.md) prepares the labels.
2. **Solve.** `planner solve` fits when users return and how far back evidence lies, replays the candidate policies and keeps the cheapest. `planner replay` prices that policy on threads it has not seen.
3. **Apply.** `WorkloadSession.prepare` returns the request to send. `complete` records the answer and says when a ping is due. Your application sends the requests and the pings.

[Workload planner guide](docs/workload-planner.md)

## On long-chat benchmarks

Modeled cost on held-out conversations against keeping the full history, across three return patterns.

| Dataset | What compache chose | Saving |
|:--|:--|:-:|
| [DocChat](benchmarks/docchat/README.md)<br><sub>document&nbsp;chats&nbsp;in&nbsp;five&nbsp;domains</sub> | Keep questions and answers, keep retrieved text for one turn, with 0 to 3 pings depending on how users return | **43%&nbsp;to&nbsp;85%** |
| [BEAM](docs/beam-workload.md)<br><sub>130k&nbsp;and&nbsp;500k&nbsp;token&nbsp;chats</sub> | Keep every user message and drop earlier assistant text, with 0 to 3 pings depending on how users return | **57%&nbsp;to&nbsp;91%** |
| LongDS<br><sub>26k&nbsp;token&nbsp;analysis&nbsp;tasks</sub> | Keep everything and tune the pings in five of six cases, and cut to the last 20 turns in one | **0%&nbsp;to&nbsp;32%** |

On DocChat, Claude Sonnet 5.5 answered 24 of 30 held-out turns correctly from the fitted policy's view and 25 from the full history, on a fifth of the tokens. On BEAM and LongDS the solver's choice was the cheapest candidate on the held-out conversations in 16 of 18 cases and within 0.03% in the other two. The LongDS conversion is not in the repository yet.

[Docs](docs/README.md) · [Methodology](https://barendpotijk.github.io/compache/methodology.html) · [Other tools compared](docs/comparisons.md) · [Issues](https://github.com/BarendPotijk/compache/issues)
