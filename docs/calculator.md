# Calculator

The [calculator](https://barendpotijk.github.io/compache/) prices one chat workload twice: with a fitted compache policy and with OpenClaw's default rules. It runs in the browser from `site/`, with no keys and no provider requests.

## Open it on your workload

```sh
compache profile threads.jsonl --labels labels.jsonl --model claude-sonnet-5-5
```

The command measures your threads and prints a link that opens the calculator with every workload input filled in. `-o profile.json` writes the same for the calculator's Import button.

| Measured from | Calculator input |
|---|---|
| Text of each turn | Question and answer tokens per turn, the share of turns that retrieve anything, their retrieved tokens with the 95th percentile, and the notice that archiving a turn's retrieved parts leaves behind |
| Turns per thread | Thread length: median and 95th percentile, or the chance of another turn after the first turn and in long threads and how fast it moves between them, whichever describes the threads better |
| Time between an answer and the next question | Return delay: median and 95th percentile, how much longer the wait after the first turn is, and the share and length of long breaks |
| Start and end of each turn | How long an answer takes. The cache lifetime runs from when the request was sent |
| Assistant messages per turn | Model calls per turn |
| Reported usage | Billed output that is not kept in the history, such as reasoning, and how much of a request still in the cache the next turn read |
| System prompt and tools | Tokens sent on every request, and how often another thread had them cached at a cold start |
| Labels | For questions, retrieved text and answers: the share of turns that need an earlier one, and how many turns back, median and 95th percentile |
| `--model` | Prices and cache rules of a built-in model, or your own |

Without labels the calculator keeps its defaults for what later turns need, which are measured on [DocChat](../benchmarks/docchat/README.md).

## How close it is

Opened on the three [DocChat](../benchmarks/docchat/README.md) profiles, the calculator puts the saving at 45%, 83% and 81% for users back after 30 seconds, 4 minutes and 30 minutes. The package's replay on DocChat's held-out threads gives 43%, 83% and 85%.

The calculator's compache side is a simplified policy, so treat its figure as an estimate and `compache fit` with `compache replay` as the number for your threads.

## What it models

| Part | Rule |
|---|---|
| Workload | Seeded threads drawn from the inputs. Lengths, retrieved sizes, return delays and how far back a turn reaches are lognormal with the given median and 95th percentile |
| Prices | List prices and cache rules of seven models, checked on 2026-11-20. The same sheet as the package: `src/compache/models.json` |
| compache | Archives old retrieved text when the cache is cold, or at a higher mark while it is warm, and sends keep-warm pings when a return is likely. The cheapest of 147 candidate policies that keeps the chosen share in view is fitted on one set of threads |
| OpenClaw | Its documented defaults at revision [`c49b553`](https://github.com/openclaw/openclaw/commit/c49b553b85b82bc700465d4607949574f892366c), with no keep-warm requests. Claude: tool results cleared above 30% of the window, keeping the newest three. OpenAI: a client-side summary at 75% of the window. Every threshold is an input |
| Fetching back | A needed source that was archived or cleared costs one extra call, on both sides. A source OpenClaw summarized is not fetched again |
| Reported cost | Both sides on a second set of threads, drawn with the next seed |

## What it leaves out

- No model is called. Answer quality is not measured, and both sides are assumed to know which earlier sources a turn needs.
- The compache side archives retrieved text only. For chats without retrieval, where the package drops other parts, it understates the saving.
- For OpenAI models OpenClaw's own default calls OpenAI's compaction at the same 75% mark. That is not modeled.

## Run it locally

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site   # then open http://127.0.0.1:8765/
node --test site/*.test.mjs
node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json && python scripts/figures.py openclaw
```

The full method is on the [methodology page](https://barendpotijk.github.io/compache/methodology.html).
