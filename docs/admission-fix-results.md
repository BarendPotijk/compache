# Compache 0.4.1: admission repair and repeated evaluation

The repair completes the expanded workload while preserving the known source requirements. It does not establish a universal cost winner: on the same recorded trajectories, Compache is cheaper with the five-minute cache assumption and more expensive with thirty minutes.

## What changed

- Citation protection is relevant to the current question and resolves only to occurrences available when the cited answer was written. Distinctive identifiers and uncommon term pairs replace broad single-word matches; ambiguous relevant citations remain protected.
- If a needed source was already compacted, the gate appends a canonical-source copy to the current request. It retains the original placeholder and cached prefix, reuses an existing full copy, and can compact a recalled copy later. Admission and costs include all recalled text. This is a deterministic local index operation, with no model decision call; the manual reopen/retrieve tools remain available and their actual model calls are recorded.
- An optional causal token ruler uses preceding successful provider input, a fixed runtime allowance and an explicit safety margin. The live plan uses 16,000 overhead tokens and a 10% margin over the maximum observed payload ratio, starting at one. It recalculates after every cut and records before/after states. This is an empirical safeguard, not native tokenization or a guaranteed bound. Synthetic replay has no provider counters and therefore uses the original byte ruler.

The selected thresholds remain unchanged: 20% target, 22% cold trigger, 50% warm trigger and a 32,000-token reserve within the configured 828,400-token usable context. Live thresholds use the calibrated estimate, including its margin. No policy or profile was refitted. The corpus now serves as a development regression benchmark because its failures informed these fixes; its historical split names do not imply an untouched test set.

## Completion and dependency availability

| Evaluation | Before repair | After repair |
|---|---:|---:|
| Offline completed conversations | 145/300 | 300/300 |
| Offline admitted turns | 15,707/23,760 | 23,760/23,760 |
| Known historical requirements on admitted offline turns | 7,598/12,498 planned | 12,498/12,498 |
| Fresh Compache live answers, three cases | 19, 18, 16 of 96 | 96, 96, 96 of 96 |
| Live historical requirements available before model recovery | 12/12 reached; 143 later unexecuted | 155/155 |

All 160 training, 40 validation and 100 heldout-split conversations completed. Their known historical denominators were 5,313, 2,868 and 4,317 respectively. The 647 unknown requirements remain unassessed. The offline replay made 6,610 rewrites: 5,599 cold and 1,011 warm; 6,607 reached the target, and there were no emergency rewrites. It assumes zero model recovery cost but includes deterministic recall input tokens. Its cache-only cost comparator can exceed the context window and is not an executable OpenCode baseline; do not use that comparison as measured savings.

## Measured calls and conditional cost

Three fresh subscription Luna-low Compache cases were compared with the three unchanged 96-turn OpenCode recordings. Immediate calls used the full configured usable window; there were no artificial expiry sleeps and no API calls. All 936 reused baseline files match their original hashes.

| Assumed warm window | OpenCode credits | Compache credits | Compache difference |
|---|---:|---:|---:|
| Five minutes | 125.3116 | 80.5920 | **35.69% lower** |
| Thirty minutes | 72.3185 | 79.8041 | **10.35% higher** |

These are modeled credit equivalents for reported successful invocations, using the frozen return cadence and intended prefix reuse. They are not subscription invoices or provider-cache proof. Both TTLs reprice the same five-minute-policy trajectory; the policy is not independently optimized for thirty minutes. Three historical OpenCode network timeouts have no reported usage, so total attempt cost remains unknown. Controller, construction and audit-agent usage is also unavailable rather than zero.

Fresh Compache produced 288 answers, 11 recovery calls and 13 keep-warm calls: 312 logical calls and 365 reported provider invocations, including **53 internal Codex invocations**. It appended 169 deterministic recall passages. Reported fresh usage totals **75,966,047 input and 24,881 output tokens**. The OpenCode comparator contains 306 logical calls and 328 provider invocations, including 22 internal calls, 14 summaries and four recoveries. Runtime scaffolding and all reported internal invocations are included in the table.

The thirty-minute result is sensitive to the observed internal Codex work: it contributes 11.9412 credits for Compache versus 3.0704 for OpenCode. Excluding that role would reverse the sign, but the reported result includes it. More frequent rewrites also mean Compache gains little when the assumed cache window grows, while OpenCode gains substantially. This accounting does not prove that the policy caused the difference in stochastic internal invocations.

The causal audit reconstructs all 312 new logical-call estimates from prior usage. None underestimates the final reported invocation of its call; the smallest observed cushion is 8,265 tokens. Internal invocations are counted separately in costs and are outside that estimate-accuracy claim. The highest reported Compache input across all invocations is 709,978 tokens. A future payload or framing change can exceed the empirical estimate.

## Answer quality remains separate

We froze a small selection of one late-turn case per available question family for each company before those answers arrived: 32 paired cases, or 64 answers. Independent blinded Luna and Terra audits were compared with exact candidate quotes. The initial Luna audit contained mapping errors and is preserved. Three Terra findings based on missing holiday details in abridged references were corrected after the actual visible retrieved text was checked.

After adjudication, **29/32 Compache answers and 14/32 OpenCode answers were semantically correct** in this diagnostic. These are not full-run accuracy estimates or production rates. Three Compache answers still failed: one omitted a requested exact prior-output quote, one missed an earlier allocation, and one answered the previous turn. Sources were available, showing why dependency retention cannot stand in for answer quality. Citation findings are recorded separately.

The next economic check should use a policy planned for the actual cache TTL and representative new traces. The current five-minute policy must not be presented as a demonstrated thirty-minute cost improvement.

## Evidence and reproduction

See [the committed result bundle](../examples/rag_scaled/admission-fix-results/README.md), [workload generation](scaled-rag-workload.md) and [offline evaluation](scaled-rag-evaluation.md). Large raw traces, prompts, answers and provider events are delivered separately. The complete evaluation is reproducible from the same generator, frozen policy/profile, source and plan.

Final source passes 131 tests on Python 3.11.13 and 3.14.3 plus Ruff E9/F; all 600 corpus shard hashes and 936 reused baseline file hashes were rechecked.
