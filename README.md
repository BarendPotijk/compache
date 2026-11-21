# compache

**Cut the cost of long LLM conversations by fitting compaction and prompt caching to your own traffic.**

![Cost per 1,000 turns on Claude Sonnet 5.5 at three thread lengths: OpenClaw's defaults against compache](docs/assets/openclaw-cost.svg)

Modeled with the [cost calculator](https://barendpotijk.github.io/compache/) on its default workload: 25k retrieved tokens per turn, users back after a median of four minutes. OpenClaw follows its documented defaults. Both sides fetch back any earlier source a turn needs, and that call is in the price. `compache profile` opens the calculator on your own traffic.

## Why fitted beats default

Two things drive the bill of a long thread: how much context every turn sends again, and whether the provider's cache has gone cold by the time the user returns. Each changes what the other is worth, and both depend on your traffic.

![Historical policy sweep for an 80% admission floor, for users who return after 30 seconds, 4 minutes and 30 minutes](docs/assets/policy-map.svg)

- **A default is one point for every workload.** Keeping everything costs 1.8 times the fitted policy when users return within seconds and 6.6 times when they take half an hour. OpenClaw's thresholds and Headroom's profiles are fixed points too.
- **The two dials interact.** With users back after four minutes, pings save $50 per 1,000 turns while the whole history is kept and $4 once it is cut. The right ping schedule depends on the cut, and the other way round.
- **compache prices the map on your threads and takes the cheapest point.** Keeping everything, with and without pings, is always on the map, so the fitted policy never costs more than that default on the threads it was fitted to. On the held-out threads above it was the cheapest of the 21 policies priced in two cases and within 3% in the third.

## Quickstart

```sh
git clone https://github.com/BarendPotijk/compache && cd compache
pip install .
compache fit examples/quickstart/threads.jsonl.gz \
  --labels examples/quickstart/labels.jsonl --model claude-sonnet-5-5
```

In about two seconds, on four sample handbook chats:

```text
Fitted on 4 threads, 71 turns (71 labelled) for claude-sonnet-5-5.

Keep all questions, all answers and retrieved text from the last 3 turns.
Parts of at most 422 tokens always stay.
Send up to 3 keep-warm pings while a user is away.
95% of labelled historical requirements retain their evidence in view.
$27.22 per 1,000 turns against $50.25 keeping everything: 46% less.

Wrote policy.json
```

Then apply the policy on every request:

```python
import compache

policy = compache.Policy.load("policy.json")
session = policy.session("thread-17")        # one per conversation

turn = session.prepare(question, retrieved)   # the request to send: turn.messages
session.complete(turn, answer, usage=response.usage)
session.next_ping_at                          # when a keep-warm ping is due, if one pays
```

Applying a policy calls no model. The same history always gives the same request, so the cached prefix stays stable.

## What it needs from you

| You give | compache fits |
|---|---|
| Your chat log: each message with its role, text and time | When a keep-warm ping pays |
| Plus labels for a sample of turns: which earlier parts the turn needed | What stays in view, separately for questions, answers and retrieved text |

No thresholds to tune and no model to call. The log goes in as you store it, or straight from Langfuse with `compache pull langfuse`. A [coding-agent skill](skills/compache-label-dependencies/SKILL.md) writes the labels, and prices for current Claude and GPT models are built in. [Guide](docs/guide.md)

![Record threads, fit a policy, apply it on every request](docs/assets/workflow.svg)

## On long-chat benchmarks

These archived runs compared the fitted policy with keeping the full history on held-out conversations. They used the earlier 80% admission floor. New fits default to 95%.

| Dataset | What compache chose | Saving |
|:--|:--|:-:|
| [DocChat](benchmarks/docchat/README.md)<br><sub>document&nbsp;chats&nbsp;in&nbsp;five&nbsp;domains</sub> | Keep questions and answers, keep retrieved text for one turn, with 0 to 3 pings depending on how users return | **43%&nbsp;to&nbsp;85%** |
| [BEAM](benchmarks/beam/README.md)<br><sub>130k&nbsp;and&nbsp;500k&nbsp;token&nbsp;chats</sub> | Keep every user message and drop earlier assistant text, with 0 to 3 pings depending on how users return | **57%&nbsp;to&nbsp;91%** |
| LongDS<br><sub>26k&nbsp;token&nbsp;analysis&nbsp;tasks</sub> | Keep everything and tune the pings in five of six cases, and cut to the last 20 turns in one | **0%&nbsp;to&nbsp;32%** |

Three workloads, three different answers: that is the point of fitting. On DocChat, Claude Sonnet 5.5 answered 24 of 30 held-out turns correctly from the fitted policy's view and 25 from the full history, on a fifth of the tokens. [All results](benchmarks/README.md)

[Guide](docs/guide.md) · [How it works](docs/how-it-works.md) · [Calculator](https://barendpotijk.github.io/compache/) · [Other tools compared](docs/comparisons.md) · [Issues](https://github.com/BarendPotijk/compache/issues)
