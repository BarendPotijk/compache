# Live comparison accounting

The subscription experiment supplies each full projected snapshot as developer instructions on a fresh ephemeral Codex thread. The user turn is a fixed request to complete the snapshot task. This placement changes the instruction hierarchy: the synthetic conversation is described as untrusted evidence, but it is carried in a higher-priority message. The CLI requested a context window of 872,000 tokens; the runtime reported 828,400 usable tokens. This is distinct from the API's 1,050,000-token context window. The OpenCode arm is an adapter derived from pinned source; the native OpenCode executable was not run.

Every model call has reported input, cached input, and output counters. Input includes Codex runtime overhead, whose exact separation from the snapshot is unavailable. The physical cache counters are measured on fresh threads and cannot establish the cache locality of persistent conversations. The standard-rate credit equivalents are a conditional conversion of reported counters, not a subscription billing statement.

The 300- and 1,800-second prefix-cache scenarios reprice the same observed calls and outputs using ordered snapshot-block hashes, reported input sizes, logical timing and the assumptions in `reprice_prefix`. They are modeled counterfactuals, not observed cache hits, independently optimized trajectories, or measured billing savings. Recovery calls are included in primary totals; the zero-recovery alternative excludes only their direct cost and does not model later carryover. Controller and platform overhead are not included in the respondent totals.

Run `PYTHONPATH=src python scripts/report_live_comparison.py OUTPUT_DIR` after all six runs are complete. It verifies completion markers, call and answer counts, order, and usage coverage before writing `report.json`, `report.md`, and `timeline.tsv`. Incomplete experiments fail without producing a complete report.

The 300-second TTL is a hypothetical sensitivity, not a claim that the current provider exposes that live cache option. The fixed policy was trained with the 300-second assumption; neither TTL scenario reoptimizes it. The API transport was prepared but not used for this subscription comparison. Public benchmark context is documented in [public benchmarks](public-benchmarks.md).

## Quality boundary

The custom workload has 72 turns but its explicitly historical dependency questions are concentrated in the first 37 turns. Later turns primarily ask direct lookups. It therefore exercises large-context compaction and cost accounting, but does not establish equivalent answer quality after OpenCode's first summary. The next workload should distribute old-source, version, and prior-answer questions throughout the post-compaction portion and include external benchmark evidence. A zero authored source-miss count is a context-availability check, not an answer-quality score.

A failed attempt that uses an external tool is rejected before its answer enters canonical history. Its raw events and every available model-call usage counter are retained separately as Codex retry overhead. Resumption reuses completed calls only when the reconstructed prompt hash, role, and run identity match; it does not regenerate favorable answers. Tool-isolation changes and resulting runtime-overhead changes are recorded in continuation metadata.

## Reproduce

```sh
python3 scripts/build_long_rag_workload.py --output /tmp/compache-long-fixtures
python3 scripts/run_live_comparison.py train --data /tmp/compache-long-fixtures --output /tmp/compache-long-run --transport codex
mkdir -p /tmp/compache-empty-cwd
python3 scripts/run_live_comparison.py run --data /tmp/compache-long-fixtures --output /tmp/compache-long-run --working-dir /tmp/compache-empty-cwd --codex codex
PYTHONPATH=src python3 scripts/report_live_comparison.py /tmp/compache-long-run
```

Training is offline. The run command makes subscription model requests through the user's authenticated Codex CLI. The optional API transport is not exercised by these commands. A current installed Codex catalog must support the requested model/window; this experiment used CLI 0.157.0.

The ledger's `usage` records the final provider invocation used for context-pressure decisions. The report audits raw fresh-thread cumulative usage, expands all provider invocations, and includes their sum in measured totals. Native correction tails are not exposed, so the prefix model assumes those additional invocations share the caller's snapshot and logical timestamp; this assumption is separate from the measured token totals.

See [the completed six-run result](live-comparison-result.md). The selected long-workload candidate lost to the OpenCode baseline under both TTL assumptions and is not recommended as a default policy.

The timeline allocates each measured logical-call wall duration to its final provider invocation and zero to earlier internal invocations to avoid double counting. These allocated zeros are not measured zero-latency responses. Original logical-call UTC bounds and wall durations remain in the raw ledger; individual internal invocation latencies are unavailable.
