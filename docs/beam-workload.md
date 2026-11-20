# BEAM workload

[BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) holds 90 long single-thread chats of roughly 130k, 500k and 1M tokens. Each has 20 rubric-graded questions and the IDs of the messages that hold their evidence. The adapter turns it into Compache threads, evidence labels and a fixed train/test split.

Nothing from BEAM is bundled. It is licensed CC BY-SA 4.0, so converted files inherit that licence. The dialogue is LLM-generated.

## Build

```sh
REV=3205395e897e7318c7b094ef4e6047b9b82dbb03
mkdir beam
for f in 100K 500K 1M; do
  curl -L -o beam/$f-00000-of-00001.parquet \
    https://huggingface.co/datasets/Mohammadta/BEAM/resolve/$REV/data/$f-00000-of-00001.parquet
done
pip install pyarrow
PYTHONPATH=src python3 scripts/build_beam_workload.py --source beam --output beam-mixed --clock mixed
```

The adapter checks each file against its pinned SHA-256, every thread against the trace contract and every label against what its run could see. All 90 conversations convert in under a minute into about 150 MB. The output has the layout of the scaled RAG corpus: `traces/`, `labels/` and `manifest.json`.

| Size | Threads (train / test) | Chat runs per thread | Tokens per thread |
|---|---|---|---|
| 100K | 13 / 7 | 94 to 196 | 100k to 225k |
| 500K | 23 / 12 | 386 to 712 | 323k to 857k |
| 1M | 23 / 12 | 778 to 1,504 | 711k to 1.85M |

## Choices

| Topic | Choice |
|---|---|
| Runs | One user message and its reply. A thread has no retrievals. |
| Questions | Appended as the last 20 runs, with BEAM's reference answer as the output. BEAM asks each one independently, so fit demand on the first `chat_runs` runs only and fork per question in a live run. |
| Evidence | A question is assessed when each evidence ID names exactly one message. All named messages are required together. BEAM gives no span, so each quote is the first 200 characters of its message. |
| Unknown | 191 questions name no evidence, 180 of them abstention questions. Four 1M conversations restart their message numbering, which leaves 41 more questions ambiguous. Both are labelled unknown. 1,568 questions are assessed. |
| Split | Inside each size bucket, threads are ordered by SHA-256 of `beam-split-v1:<thread_id>` and the first third is the test set. It depends on nothing else. |
| Model | `--model` is written to every generation. The model spec used for fitting must carry the same ID. |

## Clock

BEAM dates each session and times nothing inside it. Sessions are a median of 26 days apart. The adapter starts a session at 09:00 UTC on its date, or eight hours after the previous one ended if that is later. Inside a session the idle gap is a declared scenario: lognormal with sigma 1.2 around the median below, drawn from a hash of the seed, thread and position.

| `--clock` | Median idle gap | Within 5 minutes | Within 30 minutes |
|---|---|---|---|
| `fast` | 30 s | 97% | 100% |
| `mixed` | 4 min | 57% | 95% |
| `slow` | 30 min | 7% | 50% |

An answer takes one second plus its length at 60 tokens a second. These are assumptions, not measurements.

## What the evidence looks like

- **Recency does not help.** For about 1% of assessed questions the whole evidence lies in the last tenth of the runs, and for about 10% in the last half.
- **Kind does.** 83% of assessed questions cite user messages only, so keeping every user message keeps them answerable with no assistant text at all.

## First fit

Fitted on the 59 training conversations (1,033 assessed questions, 143 candidate cuts) with `--retention-floor 0.95 --tradeoff-floor 0.8`, then scored on the 31 test conversations. Tokens are a share of the full history. Expected tokens add one more request for every question a cut leaves unanswerable.

| Cut | Train: answerable | Train: tokens (expected) | Test: answerable | Test: tokens (expected) |
|---|---|---|---|---|
| Keep every user message, drop earlier assistant text (admitted) | 82.5% | 8.5% (10.8%) | 83.0% | 8.2% (10.4%) |
| Also keep 200 runs of assistant text | 86.8% | 44% (51%) | | |
| Also keep 500 runs of assistant text | 90.1% | 78% (89%) | | |
| Also keep 1,000 runs of assistant text | 97.6% | 98% (104%) | 97.4% | 97% (102%) |

- At a 95% floor alone the only passing cut is the last row, which costs more than it saves once recovery is priced.
- The admitted cut holds between 80% and 85% in every size bucket of both splits. Its weakest conversation keeps 72%.
- These are retention and token counts on recorded text. What a model recovers by reopening an archived message needs a live run.

## Solved policies

`planner solve` ran on the training conversations of one size and one return scenario, and its policy was then priced on that size's test conversations. Prices are 1.25 for a cache write, 0.1 for a read and 5 for output, on a 1M-token window. The saving is against keeping everything and includes the modeled cost of recovering what the cut loses.

| Size | Cache lifetime | Returns | Pings chosen | Keep everything, per turn | Saving on test |
|---|---|---|---|---|---|
| 500K | 5 min | fast | up to 3 | 52k | 79% |
| 500K | 5 min | mixed | up to 3 | 188k | 91% |
| 500K | 5 min | slow | none | 349k | 89% |
| 500K | 30 min | fast | up to 1 | 41k | 76% |
| 500K | 30 min | mixed | up to 3 | 58k | 81% |
| 500K | 30 min | slow | up to 3 | 206k | 91% |
| 100K | 5 min | fast | up to 1 | 19k | 61% |
| 100K | 5 min | mixed | up to 3 | 59k | 83% |
| 100K | 5 min | slow | none | 102k | 85% |
| 100K | 30 min | fast | none | 17k | 57% |
| 100K | 30 min | mixed | up to 3 | 21k | 63% |
| 100K | 30 min | slow | up to 3 | 62k | 84% |

- Every solved policy uses the same cut: keep every user message and drop earlier assistant text. The ping count is what changes with behaviour.
- The solver's choice was the cheapest candidate on the test conversations in 10 of the 12 cases and within 0.03% in the other two.
- A size threshold, which never archives a small part, lifts the middle of the retention frontier by about three points at equal tokens. It is never the cheapest policy here.
