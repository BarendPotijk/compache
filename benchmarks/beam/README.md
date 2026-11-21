# BEAM

Long personal chats with no retrieval, where almost all the tokens are assistant text. compache keeps every user message, drops earlier assistant text and saves **57% to 91%** against keeping everything.

[BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) holds 90 single-thread chats of roughly 130k, 500k and 1M tokens. Each has 20 rubric-graded questions and the IDs of the messages that hold their evidence. The adapter turns it into threads, evidence labels and a fixed train/test split. Nothing from BEAM is bundled: it is licensed CC BY-SA 4.0, so converted files inherit that licence. The dialogue is LLM-generated.

## Result

[`run.py`](run.py) fits on the training conversations of one size and one return pattern and prices the policy on that size's test conversations. Prices are 1.25 for a cache write, 0.1 for a read and 5 for output, on a 1M-token window. The saving includes the cost of fetching back what the cut loses.

| Size | Cache lifetime | Returns | Pings chosen | Keep everything, tokens per turn | Saving on test |
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

- **One cut, different pings.** Every fitted policy keeps every user message and drops earlier assistant text. The ping count is what changes with how users return.
- **Held out.** The fitted policy was the cheapest candidate on the test conversations in 10 of the 12 cases and within 0.03% in the other two.
- **What it keeps answerable.** That cut leaves 83% of the assessed test questions answerable from the view, at 8% of the tokens. Recency would not: for about 1% of questions the whole evidence lies in the last tenth of the chat, while 83% cite user messages only.

| Cut | Train: answerable | Train: tokens (with fetch-back) | Test: answerable | Test: tokens (with fetch-back) |
|---|---|---|---|---|
| Keep every user message, drop earlier assistant text | 82.5% | 8.5% (10.8%) | 83.0% | 8.2% (10.4%) |
| Also keep 200 turns of assistant text | 86.8% | 44% (51%) | | |
| Also keep 500 turns of assistant text | 90.1% | 78% (89%) | | |
| Also keep 1,000 turns of assistant text | 97.6% | 98% (104%) | 97.4% | 97% (102%) |

These are counts on recorded text. What a model recovers by reopening an archived message needs a live run.

## Build

```sh
REV=3205395e897e7318c7b094ef4e6047b9b82dbb03
mkdir beam
for f in 100K 500K 1M; do
  curl -L -o beam/$f-00000-of-00001.parquet \
    https://huggingface.co/datasets/Mohammadta/BEAM/resolve/$REV/data/$f-00000-of-00001.parquet
done
pip install pyarrow
python benchmarks/beam/beam.py --source beam --output beam-mixed --clock mixed

# one row of the table: fit on the training chats of a size, price on its test chats
PYTHONPATH=benchmarks/beam python benchmarks/beam/run.py --data beam-mixed --size 100K --ttl 300
```

The adapter checks each file against its pinned SHA-256, every thread against the thread format and every label against what its turn could see. All 90 conversations convert in under a minute into about 150 MB: `traces/`, `labels/` and `manifest.json`, whose `threads` list names each thread's split.

| Size | Threads (train / test) | Chat turns per thread | Tokens per thread |
|---|---|---|---|
| 100K | 13 / 7 | 94 to 196 | 100k to 225k |
| 500K | 23 / 12 | 386 to 712 | 323k to 857k |
| 1M | 23 / 12 | 778 to 1,504 | 711k to 1.85M |

## Choices

| Topic | Choice |
|---|---|
| Turns | One user message and its reply. A thread has no retrievals |
| Questions | Appended as the last 20 turns, with BEAM's reference answer as the output. BEAM asks each one independently, so fit returns on the first `chat_runs` turns only |
| Evidence | A question is assessed when each evidence ID names exactly one message. All named messages are required together. BEAM gives no span, so each quote is the first 200 characters of its message |
| Unknown | 191 questions name no evidence, 180 of them abstention questions. Four 1M conversations restart their message numbering, which leaves 41 more ambiguous. Both are labelled unknown. 1,568 questions are assessed |
| Split | Inside each size, threads are ordered by SHA-256 of `beam-split-v1:<thread_id>` and the first third is the test set |
| Clock | BEAM dates each session and times nothing inside it. A session starts at 09:00 UTC on its date. Inside it the idle gap is a declared scenario, lognormal with sigma 1.2: `fast` 30 s, `mixed` 4 min or `slow` 30 min at the median. An answer takes one second plus its length at 60 tokens a second |
