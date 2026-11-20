# Original synthetic RAG fit

The two compressed policy files have the exact archived Fold A JSON payloads for the Compache lane labeled `generic` in the Luna public-chat comparison. Both used 200 synthetic conversations, 20 fictional companies and 15,840 user turns. The same gzip files also appear in the planner implementation examples. Each is about 108 KiB. Each has a 95% training retention floor and zero admitted custom cuts. The public-chat harness adjusted the operating model specification for the benchmark host and provider and aligned the turn index. The archived files preserve the original training artifact bytes.

## Verify and regenerate offline

Run from the repository root using Python 3.11 or later. The verifier and generator use the standard library. The optional one-conversation example previews the format. The aggregate fingerprint check uses the full corpus. They do not require a model, a tokenizer download, or an API key. Choose a fresh output directory.

```sh
python3 benchmarks/handoff/fit/verify_fit.py
python3 benchmarks/handoff/fit/generator/build_scaled_rag_workload.py --select 1:1 --output synthetic-example
```

To regenerate all 300 conversations, the 600 gzip-compressed trace and label shards plus the manifest require approximately 487 MB on disk.

```sh
python3 benchmarks/handoff/fit/generator/build_scaled_rag_workload.py --companies 30 --output synthetic-corpus
python3 benchmarks/handoff/fit/verify_fit.py --corpus synthetic-corpus
```

The generator writes `traces/<thread>.jsonl.gz` (one normalized conversation object per shard), `labels/<thread>.jsonl.gz` (one evaluator record per target turn), and `manifest.json` (case design, clocks, split labels, shard paths and checksums). Only model-facing fields from the current and previous turns may enter inference. The source labels, answer keys and future turns belong to the evaluator.

An independent full generation with Python 3.14.3 and zlib 1.2.12 matched the 14,245,930-byte original authoring manifest and all 600 compressed synthetic shards. The [aggregate proof](proof.json) records the generator revision and seed, the company/episode selection rule, aggregate counts, the manifest SHA-256, one combined SHA-256 for all shards and the two uncompressed policy hashes. The verifier computes the IDs and file digests from the local regeneration. It sorts relative paths lexicographically and adds each path, a zero byte and its raw 32-byte compressed-file digest to SHA-256. Different gzip implementations can emit different bytes for the same source text. Use the recorded runtime for this exact byte-level check. [Generator](generator/build_scaled_rag_workload.py) · [Original source snapshot](https://github.com/BarendPotijk/compache/tree/24e6ff652b5181ceaca1005fda2a7c3cc7f58025/scripts)

## Training membership and labels

The generator seed was `20261116`. Each company contributed one thread of each length from the sequence 24, 36, 48, 60, 72, 84, 96, 108, 120 and 144. All 300 threads contain 23,760 turns. The author's original labels assign 160 threads to training, 40 to validation and 100 to held-out companies. The later company-based Fold A pooled **all 200 threads from companies 01–20**, including those original 40 validation threads. Companies 21–30 form a separate 100-thread synthetic cohort. GoodAI and MemoryArena have separate data lineages. A future evaluation must not treat those original 40 threads as unseen by this fit.

The verifier generates the Fold A thread IDs from the company and episode ranges, then computes SHA-256 of each UTF-8, compact JSON string containing the thread ID. It compares their sorted set with each frozen policy. Hashing the unquoted ID produces different hashes. The default check also verifies both policy file payloads, window lengths, floor and admission outcomes.

The generator authors targets alongside turns. Historical requirements identify acceptable evidence sets of earlier input, output or retrieval part IDs and quotations. Several alternatives may each suffice. Gold uses scenario constructions including direct lookups, return visits, prior answers, revisions, corrections, arithmetic and abstentions. Frequencies and calendars are illustrative. A live prior-output target has to bind to the answer actually produced in that trajectory. Token planning uses a UTF-8 byte heuristic for retrieval parts. Usage and output scaffolding in this synthetic training corpus supply no provider measurements.

## Policy artifacts and reconstruction boundaries

| Original file | SHA-256 |
|---|---|
| [Five-minute fit](policy-a-5m.json.gz) | `89a4c2cecc5fd2c6e53f0461d3a4c6e530212202dc916a87a60993582a5c2ace` |
| [Thirty-minute fit](policy-a-30m.json.gz) | `96b16e6026698aef2845197377cf80d0d96596a43b5f32c3e50dd8d4286c4248` |

The top-level schema is `compache-workload-planner-v1`. `demand` holds the position-conditioned empirical growth, return and terminal observations. `model_spec` defines the fit’s illustrative token economics and 828,400-token context. `header_traffic` contains synthetic scope and timing statistics, including 374 system tokens and 755 default tool tokens; no training run had reported input usage. `training_threads` identifies the fitted cases, and `recipes` records both candidates’ aggregate scores and eligibility. See [the exact static synthetic header](synthetic-header.json).

The original fitter used Compache 0.8.0, two shipped recipes, a retention floor of 0.95, zero allowed changes on unassessable turns, and a closing observation cutoff of 2028-01-01 UTC. It fitted company-sized shards, then combined their sufficient statistics in company order for both windows. The [portable refit](refit.py) preserves those calculations and changes the original script’s workspace paths, progress handling and tokenizer dependency for offline use.

After generating the full corpus, use the first command for a one-company smoke test and the second for all 200 training conversations. Use a fresh output directory for each command. The full run checks both final policies against the original hashes. The intermediate company shards remain local. Increase `--workers` if sufficient memory is available.

```sh
python3 -B benchmarks/handoff/fit/refit.py --corpus synthetic-corpus --out synthetic-fit-smoke --company 1
python3 -B benchmarks/handoff/fit/refit.py --corpus synthetic-corpus --out synthetic-fit
```

The [historical fitting source](source-0.8.0/compache/optimizer/workload.py) contains the Compache 0.8.0 modules that the original fit imports. Their bytes were copied from the verified first-party wheel with SHA-256 `5cf9482d95972239b4fbcf07302b8e4f9890c1c1244cf10a3c1a1c3ed6b00f7d`. This directory supplies the historical fitting modules. Use the planner implementation for host integration. The original fit used `tiktoken` 0.14.0 with `o200k_base` for the fixed synthetic header only. A [token-ID lookup](header-token-ids.json) stores the exact outputs for the seven byte-distinct header prefixes, indexed by their SHA-256. A changed header requires its own tokenizer.

| Fitted candidate | Historical requirements retained | Unassessable turns changed | Admitted |
|---|---:|---:|---|
| Keep ten prior input groups, ten output groups and three retrieval groups | 4,138 / 8,181 | 455 / 455 | No |
| Keep all prior inputs and outputs and ten retrieval groups | 8,181 / 8,181 | 381 / 455 | No |

Both recipes protected query matches. Keeping every assessed historical requirement was insufficient for the second recipe because changing an unassessable turn also failed the configured gate. The admitted set and the original training distributions can be inspected directly without running a model. [Return to the evidence inventory](../README.md)
