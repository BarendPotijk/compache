# Offline trace and dependency contracts

Implemented in this milestone: validated RAG thread objects, JSONL loading/writing, a scoped Langfuse-shaped envelope adapter, dependency packet preparation, and label validation. The runtime gate/planner, live Langfuse client and OpenAI annotation worker are not implemented.

## Scope

The initial contract has one generation per sequential run and `full_canonical` history. It requires the exact ordered prior user inputs, selected retrieval chunks and assistant outputs, then current input and selected retrieval chunks. Arbitrary native provider payloads, multiple generations per run, overlapping runs and compacted histories require adapters/contracts beyond this milestone. Unsupported formats fail explicitly.

Use `Thread.from_dict`, `load_threads`, `write_threads`, `export_langfuse` and `normalize_langfuse` from `compache.traces`. The immutable typed part/run/generation objects describe data that has passed validation. Source IDs identify occurrences; a newly retrieved copy has a new `part_id` but can retain its document/version/chunk identity. Source kinds come from the typed containers, not naming conventions.

## Missingness and usage

Text is a string or null. `content_status` is `available`, `masked` or `missing`; unavailable statuses require null text. The contract rejects residual text in masked/missing records. Missing content remains visible as a coverage gap when the original model is known to have seen that source. Neither normalization nor labeling fills it from a corpus or expected answer.

Canonical usage is null, or exactly four nonnegative integer counters:

```json
{"input": 10000, "output": 400, "cache_read": 7000, "cache_write": 2000}
```

`input` is inclusive. Cache reads plus writes cannot exceed it. Langfuse's flat buckets are exclusive, so this becomes fresh `input:1000`, `cache_read_input_tokens:7000`, `cache_creation_input_tokens:2000`, and `output:400`. This adapter supports those defined keys; other populated usage dialects are rejected rather than guessed. Missing or null exported `usageDetails` means unknown canonical usage, never zero cost or zero tokens. No pricing or token estimation is implemented here.

## Offline Langfuse representation

The adapter accepts `compache-rag-message-envelope-v1` generation inputs, one named root span, one retrieval span and one generation per run. Generation inputs contain ordered role/content messages with source-part IDs, and retrieval rows carry document/version/chunk identities and selection flags. These are export-shaped evidence envelopes, not provider wire requests or a Langfuse ingestion protocol.

Rows may arrive out of order. Exact duplicate observation rows are deduplicated; conflicting duplicates are rejected. Session, run, company, parent identity and timestamp containment must agree. JSON-string-encoded I/O is decoded explicitly. No session ID is inferred from unrelated content.

Export and normalization preserve the canonical evidence projection. Optional `scenario_id` is benchmark metadata and is intentionally excluded from the export and from labeling packets. Keep it in the benchmark manifest/sidecar; do not claim a byte-identical round trip for that non-evidence field. Exported cache/source content remains identical, including nulls and source versions. Export uses deterministic namespaced identities; live ingestion/read-back is a later milestone.

## Dependency labels

`compache.dependencies.build_packet(thread, run_id)` includes only the target generation's actual visible parts, original source identities, the current question and observed answer. It excludes future runs, scenario labels, the corpus and synthetic truth. The package and coding-agent skill use the same schema and validator.

Task requirements assess the user's request even when the answer fails. Claims assess the observed output. Requirement alternatives are OR; sources within one alternative are AND; separate requirements are AND. A fresh retrieval or earlier answer can make an older document occurrence nonessential. Citation presence does not establish support, and observational support does not prove causal necessity.

`validate_label` checks structure, identities, exact quotes, source kinds, temporal boundaries, visibility and coverage. It rejects `necessity:tested` until an intervention artifact/runner exists. It does not establish semantic accuracy. Unknown evidence, refusals and incomplete model output must not become negative dependency examples. Model labels are frozen evidence artifacts for later deterministic fitting; rerunning a model need not reproduce the labels.

## Commands

```sh
compache traces validate benchmarks/rag_v1/threads.jsonl
compache traces export-langfuse --input benchmarks/rag_v1/threads.jsonl --output /tmp/observations.jsonl
compache traces normalize --input /tmp/observations.jsonl --output /tmp/threads.jsonl
compache dependencies prepare --threads benchmarks/rag_v1/threads.jsonl --thread-id rag-001 --run-id rag-001-run-03 --output /tmp/packet.json
compache dependencies validate --input /tmp/packet.json --labels /tmp/labels.jsonl
```

The example run ID must exist in the selected benchmark release; inspect its normalized rows when choosing another case. Every command operates on local files. API keys are not needed for this milestone.

## Validation

Run `python -m unittest discover -s tests -v` after installing the package. Tests cover malformed input, future/unexposed sources, exact duplicates and conflicts, role/content corruption, history ordering, masked residual text, cache-bucket conversion, evidence labels, CLI failures and benchmark reproducibility. The benchmark review record distinguishes authored expected requirements from independent model judgments and unresolved ambiguity.
