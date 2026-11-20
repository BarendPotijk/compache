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
- **Measured on the 13 smallest training threads.** Keeping every user message and no earlier assistant text keeps 80% of assessed questions answerable at 11% of the tokens. Keeping the last 100 runs of both keeps 33% at 85%. Neither reaches a 95% floor, and the fit's default grid cannot yet propose the first cut because it gives inputs and outputs one shared window.
