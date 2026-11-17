# Interactive cost explorer

The README links to a browser-local simulator because GitHub sanitizes executable JavaScript from rendered Markdown. Source lives in `site/`, with no dependencies, keys, telemetry, provider requests or build step. This is an aggregate deterministic scenario model, **not a JavaScript port or execution of Compache**, and not an OpenClaw runtime benchmark. The existing subscription experiments compare OpenCode; this new explorer explicitly models a separate OpenClaw client-summary configuration.

## Run and reproduce

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
# Open http://127.0.0.1:8765/
node --test site/engine.test.mjs
```

Use the scenario link to preserve inputs in the URL fragment, or export the JSON ledger to inspect every request. The export includes configuration, seed, generated workload, selected-TTL per-call ledgers, per-role totals, both TTL comparisons, decision diagnostics and limitations. Decision previews keep the latest 16 blocks and report omitted counts/tokens; totals by kind cover the entire view. No content is sent to a service. Serve over HTTP; opening `index.html` directly as a file cannot load its module worker reliably. The same files can be served by GitHub Pages or any static host; GitHub Pages is not enabled by this PR.

## Workload assumptions

- One seed creates one workload shared by both arms. Policy, TTL and pricing changes leave that workload unchanged. Workload inputs such as dependency intensity can change random draws; this is not a common-random-number causal experiment between different workloads.
- The calculator models one company with one parameter set: `companies=1` and `companyVariation=0`. It samples 100 threads by default; sample count and seed are advanced settings. No company profile or real customer data is used.
- Retrieved tokens per run follow a lognormal distribution specified by median and p95, with no company-specific multiplier. User input is 200 tokens, system context 2,048 tokens and output length is the configured billable total, including reasoning. Retrieval tokens are bounded at one million in the generator. One answer request per run is modeled, after retrieval. Actual multi-call tools and generation duration are not modeled.
- Planned length is fixed, geometric (parameter is its mean), or lognormal (median/p95). It is capped at 160. Before each subsequent turn, a continuation Bernoulli can end the thread. Planned length and abandonment are distinct causes of thread termination. The survival chart shows both planned and realized lengths; the p95 length control only affects the lognormal distribution.
- Conditional return gaps are lognormal, specified by median/p95 in minutes. Never-return mass comes from termination, not an arbitrary long gap. Request duration is zero. TTL is measured from request start; a snapshot expires exactly at its TTL.
- On a dependent run, one to three distinct previous input/retrieval/output kinds are selected, each with a past turn chosen within the lookback. A retrieval block represents a whole bundle. This simple graph is not a benchmark-derived distribution, calibrated company behavior, or a semantic AND/OR proof.

The top-level cost per run is total cost of all answer and maintenance calls divided by the same number of workload runs. It is unavailable when either arm is incomplete. Cache hit rate is cached answer-input tokens divided by all answer-input tokens; maintenance/ping traffic is excluded from this rate. The expandable decision view displays the actual simulated gate and planner decisions for the selected thread/run. It is not a static illustration of guaranteed behavior.

## Token and cost accounting

Ordered blocks have immutable identities and token counts. Cache hits use the longest common prefix with any unexpired earlier request of the same thread, rounded down to 128 tokens, with a 1,024-token minimum. Generated outputs do not enter the cache until included in a later request. Summaries and flushes use separate system identities. Each request refreshes its own prefix; compacting the middle invalidates the subsequent prefix. Cross-thread reuse is excluded. Prefixes are assumed cacheable and retained until expiry; real provider caching is not guaranteed.

For each successful request, with rates per million tokens:

```
fresh = input - cached
cost = fresh × input_rate × write_multiplier × input_tier / 1,000,000
     + cached × cached_rate × input_tier / 1,000,000
     + output × output_rate × output_tier / 1,000,000
```

The long tier is selected from total input, not fresh input, and applies to the whole request. Zero disables the threshold. Maintenance requests are priced independently. Negative savings remain negative. Zero-cost denominators are unavailable, not an infinite saving. Any admission failure suppresses the full-workload savings percentage; displayed partial totals must not be compared as completed costs.

Presets were checked on **2026-11-17** against the official model pages:

| Preset | Context / max input | Input / cached / output $ per 1M | Cache write | Long request tier |
|---|---:|---:|---:|---|
| [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) | 1,050,000 / 922,000 | 0.10 / 0.01 / 0.50 | 1.25× fresh input | Above 272k: 2× input/cache, 1.5× output |
| [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) | 1,050,000 / 922,000 | 2.00 / 0.20 / 12.00 | 1.25× fresh input | Above 272k: 2× input/cache, 1.5× output |
| [GPT-4.1, legacy/deprecated](https://developers.openai.com/api/docs/models/gpt-4.1) | 1,047,576 / 1,014,808 | 2.00 / 0.50 / 8.00 | 1× | None modeled |

GPT-4.1 max input here conservatively reserves its documented 32,768 maximum output. Prices are editable Standard API assumptions, excluding tax, regional premiums, batch/flex discounts and infrastructure. These are not subscription credits or invoice measurements. GPT-6 Luna's Chat Completions function-calling path requires reasoning `none`; no model is called by this explorer. Model behavior and rates can change after the source date.

## Compache projection

The current user input, retrieval and previously generated answer blocks enter the sequence. Prior inputs and outputs remain verbatim. At a cold return, a view above target + two percentage points triggers retrieval compaction; a warm return waits for 50% or 60%. Targets are 10/15/20/25%. Oldest eligible retrieval blocks are removed toward the target. Current-query dependencies are protected; missing retrieval dependencies are appended as recalled source blocks, preserving the earlier prefix. Their input cost is charged. Extra recovery model calls are zero.

This assumes perfect current-dependency identification, excludes lexical ambiguity and unknown dependencies, and uses the selected target directly. It does not run the package's training/held-out optimizer or establish 90–95% semantic retention. Budget pressure can leave the view above target. Unlike the Python gate, this simplified projection removes blocks without pricing handle-placeholder text or applying the exact warm rollback, token calibration and serialized tool-pairing rules. It is useful for sensitivity, not a package-equivalence claim.

Keep-warm planning compares configured return probabilities and delay CDF against cold-compacted versus retained context, with priced ping requests, output and one subsequent carrying-cost allowance. It never reads the realized next gap, planned thread end or future dependency graph. It assumes no future query in its cold preview. Pings stop on actual arrival; all planned pings are charged after abandonment. The planner is an expected-value approximation, not the package's fitted survival estimator. Consequently pings can increase realized cost. Turning them off isolates passive cache reuse and compaction. The age-specific return probability is the continuation probability multiplied by `P(planned length >= next run) / P(planned length >= current run)`. It uses the configured distribution, never the realized future thread length. The displayed survival curve is this theoretical survival; the sampled workload may differ.

Admission uses the model's max input and `input + max(output, 32,000 reserve) <= context` for Compache. The optional custom context can make admission fail; failures are reported rather than dropping expensive turns from a savings denominator.

## OpenClaw baseline scope and source pin

Pinned revision: [`c49b553b85b82bc700465d4607949574f892366c`](https://github.com/openclaw/openclaw/commit/c49b553b85b82bc700465d4607949574f892366c), checked 2026-11-17. Model the **client-side Chat Completions summarization path**, not every OpenClaw deployment and not its default OpenAI Responses/native compaction route. New OpenClaw configs use safeguard mode. Source defaults inform the schedule and budgets, while generated text and token counts are necessarily assumptions.

| Behavior | Explorer representation |
|---|---|
| Client maintenance threshold | `min(75% × context, context - min(20k, 25% × context))` before the next answer |
| Recent tail | Keep complete newest synthetic turns until the 20k token target is reached; actual OpenClaw cuts at valid messages/tool pairs and can split turns |
| Summary input | Older head only; retrieval blocks projected to at most 500 tokens; capped at 40k conversation tokens plus 2,048 system tokens |
| Summary output | Configurable, default 4,000 tokens; proxy for a 16,000-character ceiling, not observed output |
| Deterministic record | 1,500 tokens appended after summary; no generation cost, subsequent input cost charged |
| Memory flush | One full-view call per compaction, default 256 output tokens, separate cache identity; prompt and exact flush timing are simplified |
| Tool pruning | Off for this OpenAI scenario; Anthropic-specific seeded TTL pruning is not applied |
| Recovery / retries | Zero extra model recovery, summary-quality retry and failure calls assumed |
| Answer admission | Input cap plus output/reserve; reserve is `min(20k, 25% × context)` |

The actual summarizer bounds conversation input to 160,000 characters, trims long tool results and samples over-budget histories. The explorer approximates that with token counts and an oldest-first cap. It does not reproduce sampling content, CJK density, recent-turn excerpts, split-turn summary calls, semantic audits, compressed reasoning, response latency or memory-tool I/O. Actual memory flushes also have a 2 MB transcript trigger and may be skipped by runtime conditions; only the per-compaction cost is represented here. Native Responses checkpoint compaction, after-delivery maintenance and tool-round pressure scheduling require separate future adapters. This should not be described as an exact simulation of all OpenClaw defaults.

Primary sources:

- [Compaction behavior, thresholds and reserves](https://github.com/openclaw/openclaw/blob/c49b553b85b82bc700465d4607949574f892366c/docs/reference/session-management-compaction/compaction.md)
- [Bounded summarizer input, safeguard mode and summary record](https://github.com/openclaw/openclaw/blob/c49b553b85b82bc700465d4607949574f892366c/docs/concepts/compaction.md)
- [Recent-tail cut point](https://github.com/openclaw/openclaw/blob/c49b553b85b82bc700465d4607949574f892366c/packages/agent-core/src/harness/compaction/compaction-cut-point.ts)
- [Memory-flush plan](https://github.com/openclaw/openclaw/blob/c49b553b85b82bc700465d4607949574f892366c/src/auto-reply/reply/memory-flush-plan.ts)
- [Session pruning](https://github.com/openclaw/openclaw/blob/c49b553b85b82bc700465d4607949574f892366c/docs/concepts/session-pruning.md)
- [GitHub Markdown sanitization](https://github.com/github/markup#github-markup)

Raw dependency availability is deliberately separate from answer accuracy. A missing original block can still have its useful facts represented by the OpenClaw summary. The explorer does not inspect or generate that summary, so it cannot score semantic preservation or claim a quality advantage.

## Validation and next step

Engine tests cover deterministic causal graphs, company heterogeneity, shared-prefix/TTL accounting, generated-output cache timing, long-tier rates, protected dependencies and recall, summary/flush costs and bounds, deterministic-record replay cost, abandonment pings, negative savings, invalid configuration, admission and zero denominators. Browser checks cover model and TTL changes, reset, share links, JSON export, validation and responsive layout.

The next evidence step is an adapter using actual documents and generated answers, compared against the real package under the same settings. See [the benchmark source plan](benchmark-trace-plan.md): QASPER is the proposed reusable source, MuSiQue checks controlled multi-hop dependencies, and LongMemEval is a separate memory diagnostic. No public benchmark dataset has yet been downloaded or used in these calculator results.
