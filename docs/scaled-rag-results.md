# Expanded synthetic RAG benchmark

The expanded corpus is complete, and it exposes an admission limitation in the frozen Compache 0.4.0 policy. The policy failed admission in 155 of 300 offline conversations. In the measured sample, all three OpenCode cases completed 96 turns, while Compache stopped after 19, 18 and 16 answers. These results do not support a full-thread savings or 95% coverage claim for this workload.

Only `dataset-revised` is the final corpus. The initial draft and interrupted diagnostic replay are preserved separately and excluded from these results.

| Corpus property | Value |
|---|---|
| Fictional companies / conversations / turns | 30 / 300 / 23,760 |
| Conversation lengths | 24, 36, 48, 60, 72, 84, 96, 108, 120, 144 turns |
| Training / validation / unseen-company test | 160 / 40 / 100 conversations |
| Question patterns | 11 across six business domains |
| Authored historical follow-ups | 12,498; 6,102 after turn 50; 2,800 after turn 80 |
| Retrieval size | 17,485–46,604 package planning tokens; not provider measurements |

The patterns cover current lookup, history, user constraints, prior-output quotation, mixed evidence, fresh duplicates, versions, separate corrections, arithmetic, scoped negatives and abstention. Companies vary in return cadence, dependency frequency/distance, document size and output style. The sources are long fictional procedures and case registers, with evidence at varied positions. Known-company validation contains the 120/144-turn lengths; training contains 24–108. Unseen companies span all lengths but share the generator and templates.

Generated answers and labels are authored scaffolding, and their usage fields are null. The independent semantic audit found no material mismatch in its sample; this is not a production prevalence sample or public benchmark. Template answers do not reproduce the live citation pattern: actual citations activated the guard more broadly in the measured sample. This is an important boundary when using synthetic answers to predict live behavior.

## Frozen-policy offline replay

No new policy fitting or model calls were used for this replay. Recovery cost was zero and missing sources were not restored. Counts below distinguish admitted requests, blocked requests and later skipped requests.

| Split | Conversations | Finished | Admission stops | Admitted turns | Blocked turns | Later skipped turns |
|---|---:|---:|---:|---:|---:|---:|
| train | 160 | 92 | 68 | 7,988 | 68 | 2,504 |
| validation | 40 | 13 | 27 | 2,952 | 27 | 2,301 |
| heldout | 100 | 40 | 60 | 4,767 | 60 | 3,093 |
| all | 300 | 145 | 155 | 15,707 | 155 | 7,898 |

All 7,598 known historical requirements assessed on admitted requests remained available. Another 121 were present in blocked projections, and 4,779 fell in later skipped turns. Thus only 7,598 of 12,498 planned historical requirements reached an admitted request; zero observed misses does not establish end-to-end 95% coverage. Unknown requirements remain separate: 430 were encountered and 217 were skipped.

The replay made 2,343 rewrites; 595 reached the 20% target. There were 2,369 warm deferrals and 123 emergency rewrites. Estimated costs are in ../examples/rag_scaled/results/offline-summary.json, with a cache-only comparator restricted to the same admitted prefixes. The full cache-only workload cost is not comparable with a stopped Compache trajectory, and this comparator is not OpenCode.

## Measured subscription sample

Three distinct 96-turn cases from unseen companies were selected before outcomes and paired through both arms. All calls used subscription Luna at low reasoning and the configured 828,400-token usable window. Calls were immediate within each run; five-minute and thirty-minute cache prices are separate assumptions over the frozen return cadence. No API calls were made.

| Case | OpenCode answers | Compache answers | Compache blocked turn | Historical requirements on unanswered Compache turns |
|---|---:|---:|---:|---:|
| scaled-26-07 | 96/96 | 19/96 | 20 | 42 |
| scaled-24-07 | 96/96 | 18/96 | 19 | 49 |
| scaled-29-07 | 96/96 | 16/96 | 17 | 52 |

Captured 341 actual answers and 382 reported provider invocations, including 23 internal Codex invocations. OpenCode made 14 summary calls and four recovery calls. Compache made no recovery calls before stopping. Successful calls reported 143,163,093 input tokens and 35,417 output tokens; runtime input scaffolding is included. Controller and construction-agent usage is unavailable.

Three OpenCode requests encountered network timeouts. Their raw events and 900-second request durations are preserved under live/unreported-transport-attempts; provider token usage was not reported and is not assumed zero. The cases resumed from saved answers without repeating successful calls. One Compache answer had extra closing JSON delimiters; its exact complete answer was extracted without a model retry, and the original raw text is preserved.

Full-thread savings are unavailable because Compache did not complete equivalent work, and total attempt cost is also incomplete because the network timeouts lack usage. live/timeline.tsv records successful per-invocation token counts, timing and both cache scenarios. live-summary.json includes matched-prefix diagnostics only; do not treat the cost of the shortened Compache runs as full-thread savings.

Live output dependencies are rebound to each arm’s actual earlier answer. Output-only quotation uses its actual first sentence; other authored output alternatives are removed when their literal fact quote is absent. This conservative check can exclude valid paraphrases. The literal quality screen is not semantic answer accuracy, and source availability does not establish causal source use.

## What the larger workload revealed

Every earlier retrieval was protected in the three stopped Compache cases: 19/19, 18/18 and 16/16. The citation guard matches document IDs reused across chunks and occurrences, so an old citation can protect a later occurrence from the same document. Broad lexical matches protect many of the same chunks. The resulting request exceeds the admission limit with no eligible retrieval left to compact.

The last successful provider input exceeded the planner estimate by approximately 39,785–43,248 tokens. The blocked requests were never sent, so their provider token counts are unknown. These findings concern this synthetic workload and harness; they are not a production failure-rate estimate.

The next implementation work is to resolve citation protection at the source-occurrence level, narrow generic lexical protection, and calibrate token estimates before refitting. Any relaxation must be checked against dependency retention and admission together. The current dataset and policy are frozen; no policy was tuned to these results.

## Reproduction and evidence

See [workload generation](scaled-rag-workload.md), [offline evaluation](scaled-rag-evaluation.md) and the [compact result bundle](../examples/rag_scaled/README.md). The frozen manifest, policy, profile, counts, live plan, per-invocation timeline, admission reconstruction and audits are committed. Large generated shards, raw requests, answers and events are delivered separately rather than checked into Git.

All 600 shard hashes were verified. The builder checked all 23,760 runs for contracts, visibility and exact evidence quotes. Final source passes 115 tests on Python 3.11.13 and 3.14.3 plus Ruff E9/F. The three selected conversations regenerate with matching compressed hashes across both Python versions.

The frozen policy is unchanged. This is a diagnostic expansion, not a newly fitted or approved production policy.
