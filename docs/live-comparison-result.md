# Long-context subscription result

The frozen Compache candidate **did not save cost** against the OpenCode adapter. Across three repeats of the same 72-turn conversation, its modeled cost was 174.3% higher with a 5-minute window and 211.2% higher with a 30-minute window. This candidate is not recommended as a default.

Each Compache repeat rewrote the prefix 17 times while retaining 29 turns of retrievals; OpenCode summarized once. The retention requirement prevented compaction from reaching its target size, causing repeated warm-prefix rewrites. This is a policy failure on this workload, not proof that context optimization cannot work.

The 432 actual answers and all reported provider invocations were measured at Luna low reasoning through subscription access only. The runtime exposed 828,400 usable context tokens; peak baseline input was 814,596. The baseline is a pinned source-derived adapter, not the native OpenCode executable.

The custom historical-dependency questions are concentrated before the first baseline summary. These results do not establish post-compaction answer-quality equivalence. A zero Compache authored-source miss count checks context availability only. The next experiment should spread old-evidence questions throughout the later turns and add [public RAG and memory benchmarks](public-benchmarks.md).

[Compact machine-readable results](../examples/rag_long/results/summary.json) and [per-invocation timeline](../examples/rag_long/results/timeline.tsv) preserve the paired totals. Source/policy hashes, timing assumptions and protocol/tool-isolation changes are recorded beside them. Full raw prompts, answers, events and failed-attempt evidence were retained in the local experiment artifact; they are not in this repository.

The zero-recovery scenario is identical here: no package recovery call was made. Internal Codex extra invocations are included; discarded external-tool attempts are separately itemized below. Credit equivalents are conditional calculations, not subscription invoices. Rates: [Codex pricing](https://learn.chatgpt.com/docs/pricing#token-rates).


Completed six runs. Credit equivalents use standard rates for reported usage; they are not a subscription billing statement.

| Arm | Input | Physical cached input | Output | Reported usage credit equivalent |
|---|---:|---:|---:|---:|
| opencode | 78,982,362 | 2,654,464 | 15,693 | 191.6795 |
| compache | 105,727,845 | 4,529,664 | 15,140 | 254.3171 |

Additional Codex invocations within successful logical calls: 9 (included in measured totals and modeled scenarios). Their correction tails are not visible; modeled cache reuse assumes the same snapshot blocks.

Discarded transport attempts are retry overhead, separate from package recovery. Combined totals include successful calls and retries; setup and controller activity are incomplete.

| Arm | Retry model calls | Retry input | Retry cached | Retry output | Retry credits | Combined credits |
|---|---:|---:|---:|---:|---:|---:|
| opencode | 2 | 1,286,566 | 652,416 | 107 | 1.7498 | 193.4293 |
| compache | 3 | 1,768,517 | 1,187,200 | 194 | 1.7525 | 256.0696 |
| Both | 5 | 3,055,083 | 1,839,616 | 301 | 3.5023 | 449.4990 |

Setup probe counts are partial; see setup-overhead.json. Controller usage is unavailable.

Modeled prefix-cache savings are counterfactual; physical cache behavior differs because each snapshot uses a fresh ephemeral thread.

| TTL | OpenCode credits | Compache credits | Savings | Savings % |
|---|---:|---:|---:|---:|
| 300 seconds | 36.5457 | 100.2590 | -63.7133 | -174.3% |
| 300s, replicate 1 | 12.1098 | 33.2035 | -21.0937 | -174.2% |
| 300s, replicate 2 | 12.3208 | 33.8447 | -21.5238 | -174.7% |
| 300s, replicate 3 | 12.1151 | 33.2108 | -21.0958 | -174.1% |
| 1800 seconds | 32.2125 | 100.2590 | -68.0465 | -211.2% |
| 1800s, replicate 1 | 10.6632 | 33.2035 | -22.5403 | -211.4% |
| 1800s, replicate 2 | 10.8739 | 33.8447 | -22.9708 | -211.2% |
| 1800s, replicate 3 | 10.6753 | 33.2108 | -22.5355 | -211.1% |

| Run | Answer / summary / ping / recovery calls | Max input | Wall seconds | Missing requirements | Rewrites | Plain text / unknown format |
|---|---:|---:|---:|---:|---:|---:|
| opencode-1 | 72/1/0/0 | 799,519 | 512.7 | not assessed | 0 | 14 / 0 |
| opencode-2 | 72/1/0/0 | 799,405 | 493.0 | not assessed | 0 | 12 / 8 |
| opencode-3 | 72/1/0/0 | 814,596 | 457.7 | not assessed | 0 | 10 / 0 |
| compache-1 | 72/0/1/0 | 617,570 | 536.5 | 0 | 17 | 1 / 16 |
| compache-2 | 72/0/1/0 | 614,748 | 654.1 | 0 | 17 | 4 / 17 |
| compache-3 | 72/0/1/0 | 617,692 | 547.4 | 0 | 17 | 4 / 8 |

The run continued after an initial protocol pause. External-tool isolation changed for affected replicas after discarded attempts; see continuation-note.json for details.

Recovery is included in the primary totals. The separate recovery exclusion removes direct calls only and does not model later carryover.

- Physical cached counters are observed for fresh ephemeral snapshots; modeled prefix reuse is not observed cache behavior.
- Runtime input overhead is included in usage but cannot be split exactly.
- Additional provider invocations use the same modeled snapshot blocks and logical time; their native correction tails are not observable.
- The direct recovery exclusion does not remove downstream carryover.
- OpenCode is a source-derived adapter, not the native executable.
