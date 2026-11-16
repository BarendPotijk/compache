# Standard RAG benchmark v1

This is a deterministic, fictional corpus for testing trace normalization, source visibility and dependency annotation. It contains **45 threads, 207 runs, 30 document chunks and 45 authored final-run expectations**: every combination of three companies and 15 question families. Each thread has 3–6 sequential runs, with one generation per run and complete canonical history.

## Files and use

| File | Purpose |
|---|---|
| `threads.jsonl` | Canonical run inputs, retrieved occurrences, outputs and exact generation visibility |
| `corpus.jsonl` | Fictional policy documents; never extra evidence for a labeling agent |
| `ground_truth.jsonl` | Authored expectations for each final run; evaluator-only, not model input |
| `question-design.json` | Question families, examples and expected reasoning challenges |
| `manifest.json` | Counts, seed, content hashes, timing assumptions and exclusions |
| `review/` | Preserved model-review judgments, findings and correction record |

From the repository root, after installing Compache:

```sh
python3 scripts/build_rag_benchmark.py
compache traces validate benchmarks/rag_v1/threads.jsonl
compache traces export-langfuse --input benchmarks/rag_v1/threads.jsonl --output /tmp/rag-observations.jsonl
compache dependencies prepare --threads benchmarks/rag_v1/threads.jsonl --thread-id rag-001 --run-id rag-001-run-04 --output /tmp/rag-packet.json
```

The builder rewrites the five generated data files with seed `681204`. It does not rewrite review records. Rebuilds must match the manifest hashes; changing content requires a new review disposition. Export produces 621 observation rows in Compache's scoped offline envelope. Live Langfuse ingestion is not implemented.

Use prepared packets with the [coding-agent skill](../../skills/compache-label-dependencies/SKILL.md). A packet excludes future runs, scenario IDs, hidden candidates and expected answers. The author expectations have a deliberately separate format from model-produced dependency labels. They are review targets, not labels to pass directly to `dependencies validate` or a precomputed accuracy score.

## What varies

| Axis | Coverage in this release |
|---|---|
| Company | Three fictional businesses, different policy values, wording and cadence parameters |
| Thread length | 3–6 runs; distinct intervening topics |
| Historical source | Earlier user instruction, retrieved document or assistant output |
| Reference distance | Immediate and older turns, ellipsis, return to an earlier topic |
| Evidence structure | Single source, joint sources, fresh duplicates, alternative sufficient source sets |
| Retrieval state | Inserted chunks, unselected decoys, policy versions, missing/masked text |
| Answer quality | Supported answers, unsupported extrapolation, irrelevant citation and abstention |
| Timing | Seeded 5–14 second durations and 0–54 second start jitter around company cadence parameters |
| Usage | Explicitly unknown (`null`), including token and cache counters |

Fifteen balanced families exercise behavior; they do not estimate real workload prevalence. The durations and cadence are simulation inputs. Short authored documents do not stress context windows or establish token, cache-hit, latency or cost performance. A later workload simulator should vary long-thread lengths, correlated document/token sizes, bursty return intervals, abandonment and retrieval reuse, with parameters calibrated separately from this correctness set.

## Reading the expectations

Separate requirements are AND. Each requirement's alternatives are OR; every source inside one alternative is needed together. These describe sufficient visible content for the task. A prior assistant answer can supply conversational information; that does not independently verify the underlying policy. Version-specific questions must preserve document applicability.

Do not treat the authored alternatives as experimental evidence of causal use, or grade unseen reasonable alternatives as wrong automatically. Semantically equivalent source sets need adjudication. A correct claim can coexist with an incomplete answer; the label contract separates task fulfillment from claim support. Missing text remains unknown, even if the author knows what a hidden policy might say.

The [review record](review/README.md) preserves initial findings and subsequent corrections. Model review and deterministic validation improve the fixtures; neither is human certification or production validation. The three companies share templates, so random splits of these rows would leak structure. Use separately authored topics/templates and held-out companies for future labeler evaluation.
