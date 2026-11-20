# Revised scaled RAG workload

The revised workload is a deterministic, authored stress fixture for long RAG conversations. It is not a sample of production traffic, a public benchmark, or evidence that a model causally used a source. All companies, case records, outcomes, timestamps, answers and dependencies are fictional. Generated case registers are synthetic, and case-specific terms may differ from general handbook guidance.

The frozen revision uses seed `20261116` and contains 30 companies, 300 conversations and 23,760 turns. Each company has one conversation of each length: 24, 36, 48, 60, 72, 84, 96, 108, 120 and 144 turns. Twenty companies contribute 160 training and 40 validation conversations; ten entirely unseen companies contribute 100 held-out conversations. Companies have four to six domains drawn from returns, delivery, pricing, hosting, people and warranty. Their cadence, return timing, dependency frequency and bundle sizes vary deterministically.

The 11 authored patterns are direct lookup, historical input, historical case return, prior-output quotation, mixed-source synthesis, fresh duplicate, version revision, separate-event correction, arithmetic, scoped negative and abstention. Family frequencies are design choices, not population estimates. Fresh-duplicate evidence repeats a synthetic case fact in a later retrieval; it is not an independent corroborating document. Authored alternatives identify sufficient source sets; they do not establish which sources a model actually used.

Each turn has fabricated output scaffolding, and every usage field is `null`. Replace scaffolding with actual model responses and collect provider usage for live evaluation. For a prior-output quotation, grade against the actual earlier answer from the same arm. Other labels can also offer a prior output as an alternative: validate the cited text against the actual earlier answer or discard that alternative if it is absent. Keep labels and manifest gold out of model inputs.

Each thread is stored as a compressed JSONL shard, with a matching per-run label shard. The manifest records the split, company profiles, hashes, counts and evaluator-only answer metadata. The frozen revision is approximately 466 MB compressed. Its retrieval bundles range from **17,485 to 46,604 planning tokens** using the default `ModelSpec.part_tokens` ruler, `ceil(UTF-8 bytes / 4) + 6` per part. These are planning estimates, not native tokenizer or provider measurements; the manifest also records a separate word-based estimate.

An independent semantic audit of the three selected 96-turn pilot threads inspected the domain-specific questions, literal evidence, correction and version histories, fresh duplicates and source positions, and found no material answer or label mismatch. The final pilot changed only the case-ID assignment to an independently seeded random stream after that audit. Structural validation then checked all 23,760 runs and labels for trace contract, request visibility and exact source quotes, and verified all 600 shard hashes. This audit is not a model-quality or production-traffic validation.

## Generate

Run from the repository root and explicitly choose an output directory (the default is `work/scaled-rag-dataset` relative to your current directory):

```sh
PYTHONPATH=src:scripts python3 scripts/build_scaled_rag_workload.py \
  --output ./work/scaled-rag-dataset
```

The build writes one thread and one label gzip shard per conversation, streaming a conversation at a time. Gzip headers omit filename and timestamp for reproducibility. To build only the three selected pilot threads, use the same generator with explicit selections; the thread ordinal is one-based:

```sh
PYTHONPATH=src:scripts python3 scripts/build_scaled_rag_workload.py \
  --output ./work/scaled-rag-pilot \
  --select 26:7 --select 24:7 --select 29:7
```

## Read or export a thread

The helper loads one compressed conversation at a time. From the repository root:

```sh
PYTHONPATH=scripts python3 - <<'PY'
from scaled_rag_dataset import iter_threads, export_thread

for thread, labels, metadata in iter_threads('./work/scaled-rag-dataset'):
    print(metadata['thread_id'], len(thread['runs']), len(labels))

export_thread('./work/scaled-rag-dataset', 'scaled-26-07', './work/single-thread')
PY
```

The export contains `threads.jsonl`, `labels.jsonl` and `manifest.json` in the ordinary single-thread format. Its manifest excludes per-run `run_design` gold. The labels remain evaluator-only; the inference request must include only the relevant visible source parts.

Run the generator checks with the repository's unittest suite:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_scaled_rag_workload.py' -v
```
