# Compache 0.4.0: three live long-thread reruns

Completed 216 new answers: three concurrent 72-turn conversations through subscription GPT-6 Luna at low reasoning. Each replica executed its next call immediately, without artificial TTL sleeps. No API requests were made. The configured usable window was 828,400 tokens (872,000 requested).

The comparison reuses the three unchanged historical OpenCode recordings; they were not rerun concurrently with this candidate. The same synthetic questions, retrieved documents, trained profile and return cadence were used. Policy selection was frozen before execution: 20% target, 22% cold trigger, 50% warm trigger and 95% training dependency floor. This is a previously used diagnostic conversation, not a new unseen evaluation.

Costs below are totals across three conversations, in credit equivalents at frozen standard rates. They use actual provider input/output counts with counterfactual prefix reuse at the assumed return times. They are not invoice totals or measured persistent-session cache savings.

| Cache window | Historical OpenCode | Previous Compache 0.3.0 | Updated Compache 0.4.0 | Savings vs OpenCode | Reduction vs old Compache |
|---|---:|---:|---:|---:|---:|
| 5 minutes | 36.5457 | 100.2590 | 28.4093 | 22.26% | 71.66% |
| 30 minutes | 32.2125 | 100.2590 | 28.4093 | 11.81% | 71.66% |

The 30-minute scenario reprices the same calls and decisions selected under the 5-minute policy; it does not rerun or refit the policy. Rates per million tokens: input 2.5 credits, cached input 0.25, output 12.5. Matching prefixes are tracked by hashes, with 128-token blocks and a 128-token framing guard. A 7-minute return after turn 35 and 40-minute return after turn 66 come from the frozen workload, not execution sleeps.

| Fresh run | Provider invocations | Extra Codex invocations | Input tokens | Output tokens | Model-request wall seconds | Rewrites | Historical requirements retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compache 1 | 80 | 8 | 19,090,191 | 5,384 | 457.0 | 5 | 5/5 |
| Compache 2 | 78 | 6 | 18,697,402 | 4,790 | 811.3 | 5 | 5/5 |
| Compache 3 | 77 | 5 | 18,725,954 | 5,018 | 442.7 | 5 | 5/5 |

The fresh runs contain 0 recovery calls, 0 keep-warm pings, and 19 extra internal Codex invocations. All reported usage from successful logical calls is included, including internal invocations and runtime input scaffolding. Raw cumulative usage counters reconcile with every provider invocation. Controller and platform overhead cannot be measured here and is excluded. Historical discarded OpenCode transport attempts are retained and priced separately in report.md; they are excluded from the primary modeled comparison.

The slowest request was replica 2, turn 57: 309.9 seconds. It completed successfully without a manual retry; the cause of that latency was not diagnosed. Wall-time comparisons should not be treated as an isolated package-speed effect.

The five historical requirements are the same five questions repeated three times, not fifteen independent dependency cases. Retention means a complete authored source alternative remained available before recovery; it does not establish answer correctness or a population 95% guarantee. Those dependency questions occur before the historical OpenCode summary, so post-summary dependency quality remains untested.

| Replica 1 compaction turn | Cache state under 5m policy | Estimated context after cut | Reported input after cut |
|---|---|---:|---:|
| 20 | warm | 19.59% | 137,563 (16.61%) |
| 36 | cold | 37.28% | 268,404 (32.40%) |
| 44 | warm | 19.76% | 141,018 (17.02%) |
| 57 | warm | 19.60% | 152,540 (18.41%) |
| 67 | cold | 19.77% | 154,692 (18.67%) |

Policy thresholds use the package's estimated token ruler; reported provider tokens differ. The target is not forced by dropping protected evidence. Warm partial cuts are deferred when the target cannot be reached and the current view fits admission limits. Cold cuts may retain more than the target.

The cost decomposition is reproducible: net savings equal the difference in all-cold token/call costs plus Compache's cache discount minus OpenCode's cache discount. summary.json includes each term. This is arithmetic on fixed observed trajectories, not causal attribution to individual package features.

Historical baseline runtime configuration differs: OpenCode 1/2 and the first portion of OpenCode 3 exposed external-tool declarations; the new candidate disables them. Their scaffolding tokens remain in measured totals. Fresh ephemeral snapshot threads also differ from a persistent production session, which is why physical cached counters and modeled cache reuse are reported separately.

[Compact results and per-provider timeline](../examples/rag_long/live-0.4.0-results/) are committed with this report. The full local evidence bundle `compache-updated-live-comparison` retains raw answers, prompts, events, fixtures, execution sources and reproduction commands. Raw events are not duplicated in Git.
