# Interactive cost explorer

The README links to a browser-local simulator because GitHub sanitizes executable JavaScript from rendered Markdown. Source lives in `site/`, with no dependencies, keys, telemetry, provider requests or build step. This is an aggregate deterministic scenario model, **not a JavaScript port or execution of Compache**, and not an OpenClaw runtime benchmark. The existing subscription experiments compare OpenCode; this new explorer explicitly models a separate OpenClaw client-summary configuration.

## Run and reproduce

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
# Open http://127.0.0.1:8765/
node --test site/*.test.mjs
```

Use the scenario link to preserve inputs in the URL fragment, or export the JSON ledger to inspect every request. The export includes configuration, seed, generated workload, model-default cache settings, per-call ledgers, per-role totals, every searched candidate, selected policy, expected/observed retention, decision diagnostics and limitations. Decision previews keep the latest 16 blocks and report omitted counts/tokens; totals by kind cover the entire view. No content is sent to a service. Serve over HTTP; opening `index.html` directly as a file cannot load its module worker reliably. The same files can be served by GitHub Pages or any static host; GitHub Pages is not enabled by this PR.

## Workload assumptions

- One seed creates one workload shared by both arms. Policy, TTL and pricing changes leave that workload unchanged. Workload inputs such as dependency intensity can change random draws; this is not a common-random-number causal experiment between different workloads.
- The calculator models one company with one parameter set: `companies=1` and `companyVariation=0`. It samples 100 threads by default; sample count and seed are advanced settings. No company profile or real customer data is used.
- Retrieved tokens per run follow a lognormal distribution specified by median (0–500,000) and p95 (up to 1,000,000), with no company-specific multiplier. A zero median is a dedicated no-retrieval mode: every retrieval is zero tokens and retrieval dependency edges/requirements are absent. The p95 and retrieval dependency controls are disabled in that mode; a zero-median lognormal is not evaluated. User input is configurable (default 200 tokens), system context 2,048 tokens and output length is the configured billable total, including reasoning. Retrieval tokens are bounded at one million in the generator. One answer request per run is modeled, after retrieval. Actual multi-call tools and generation duration are not modeled.
- Planned length is fixed, geometric (parameter is its mean), or lognormal (median/p95). It is capped at 160. Before each subsequent turn, a continuation Bernoulli can end the thread. Planned length and abandonment are distinct causes of thread termination. The survival chart shows theoretical survival after both length limits and abandonment; the p95 length control only affects the lognormal distribution.
- Conditional return gaps are lognormal, specified by median/p95 in minutes. Never-return mass comes from termination, not an arbitrary long gap. Request duration is zero. TTL is measured from request start; a snapshot expires exactly at its TTL.
- Input, retrieval and output dependencies are sampled independently from their own per-run probabilities. Each source type has a separate lookback in runs; if selected, one earlier source is chosen uniformly within that lookback (or all available prior runs when the thread is younger). A retrieval block represents a whole bundle. This simple graph is not a benchmark-derived distribution, calibrated company behavior, or a semantic AND/OR proof.

The top-level cost per run is total cost of all answer and maintenance calls divided by the same number of workload runs. It is unavailable when either arm is incomplete or no searched Compache policy meets the retention requirement. Cache hit rate is cached answer-input tokens divided by all answer-input tokens; maintenance/ping traffic is excluded from this rate. Context defaults to the mean input at each run across threads admitted by both arms at that run. Ended or failed threads are not padded with zero; a dedicated thread-count subchart with an aligned run axis and point tooltips shows the surviving cohort. The subchart shows counts and shares of the initial workload, with the selected run highlighted; it is hidden for individual-thread views. Select a thread and a run to inspect its context curve and ping plan. The redundant expandable run-context panel is removed. The Assumptions disclosure contains only fixed modeling assumptions; pricing sources remain with model settings and full sources are in the methodology. Incomplete comparisons retain diagnostic charts but suppress comparable cost metrics. The default planned median is 8 runs and p95 is 20, editable assumptions rather than measured chatbot norms. The survival chart follows the configured length distribution through approximately 1.15 times its p95 (capped at 160), with a median/mean/fixed-length marker. Changing a median scales its p95 proportionally; p95 remains independently editable.

## Token and cost accounting

Ordered blocks have immutable identities and token counts. Cache hits use the longest common prefix with any unexpired earlier request of the same thread, at an exact eligible block boundary, with the selected model's minimum (1,024 visible tokens for the OpenAI presets; 512 for the current Claude presets). Generated outputs do not enter the cache until included in a later request. Summaries and flushes use separate system identities. Each request refreshes its own prefix; compacting the middle invalidates the subsequent prefix. Cross-thread reuse is excluded. The calculator assumes caching is enabled through the final eligible input block, a matching breakpoint remains discoverable, and entries persist until a hard cutoff at the documented default lifetime. Routing, hidden provider tokens, explicit breakpoint placement and provider lookup limits are not simulated. Providers may retain entries longer; an unexpired prefix is not a guarantee of a real hit. Below the minimum cacheable request size, fresh tokens pay base input rather than the cache-write rate.

For each successful request, with rates per million tokens:

```
fresh = input - cached
cost = fresh × input_rate × write_multiplier × input_tier / 1,000,000
     + cached × cached_rate × input_tier / 1,000,000
     + output × output_rate × output_tier / 1,000,000
```

The long tier is selected from total input, not fresh input, and applies to the whole request. Zero disables the threshold. Maintenance requests are priced independently. Negative savings remain negative. Zero-cost denominators are unavailable, not an infinite saving. Any admission failure suppresses the full-workload savings percentage; displayed partial totals must not be compared as completed costs.

Presets were checked on **2026-11-17** against official pricing, model and caching documentation. Rates are Standard API USD per million tokens; writes replace the base input charge, rather than adding to it.

| Preset | Input / cached / output | Cache write | Default cache | Long request tier |
|---|---:|---:|---:|---|
| [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) | 0.10 / 0.01 / 0.50 | 0.125 | 30 min | Above 272k: 2× input/cache/write, 1.5× output |
| [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) | 10 / 1 / 50 | 12.50 | 30 min | Above 272k: 2× input/cache/write, 1.5× output |
| [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) | 2 / 0.20 / 12 | 2.50 | 30 min | Above 272k: 2× input/cache/write, 1.5× output |
| Claude Opus 5.5 | 4 / 0.20 / 20 | 5 | 5 min | No long-context premium |
| Claude Sonnet 5.5 | 2 / 0.10 / 10 | 2.50 | 5 min | No long-context premium |
| Claude Haiku 5.5 | 0.10 / 0.01 / 0.50 | 0.125 | 5 min | Above 100k: 5× all token rates |
| Claude Fable 5.2 | 8 / 0.20 / 40 | 10 | 5 min | No long-context premium |

Claude prices come from [Anthropic's developer pricing page](https://platform.claude.com/docs/en/about-claude/pricing), with [model context limits](https://platform.claude.com/docs/en/models/overview) and [caching defaults/minimums](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). The Claude presets use 1,000,000 context/input limit with the engine's output/reserve admission check. OpenAI presets use 1,050,000 context and 922,000 maximum input. [OpenAI caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching) specifies the 30-minute default and exact eligible-boundary accounting for GPT-5.6 and later. GPT-4.1 is removed.

Selecting a model sets pricing, context, cache minimum and lifetime together. There is no cache-window selector or 5/30-minute comparison table. Editing an advanced model price switches to custom pricing and retains the last preset's cache policy. Prices exclude tax, regional premiums, batch/flex discounts and infrastructure. No model is called; these are neither subscription credits nor invoice measurements. Rates may change after the source date.

## Retention-constrained policy search

A retention slider runs from **75% to 100%**, in one-percentage-point steps, defaulting to **95%**. This is a requirement on modeled historical dependency availability before current-query recovery, not the fraction of all input tokens and not measured semantic answer quality.

For each reached run and source type, expected required dependencies equal that type's configured probability. Expected retained dependencies equal this probability multiplied by the fraction of possible source turns still present within its uniform lookback. Sum retained and required across runs and types, then divide. Thus a type with no reuse probability contributes no weight. Report both the expected score used for optimization and availability for the independently sampled dependency edges. A zero denominator is unavailable and imposes no constraint. Earlier inputs and outputs remain verbatim in this projection; retrieval is the compactable source type. The per-type table makes this weighting visible.

The deterministic search evaluates targets **10/15/20/25/35/45%**, warm triggers **50/55/60%**, and protected retrieval horizons of **0/75/90/95/100%** of the retrieval lookback, rounded up to whole runs and deduplicated. The horizon is a minimum set of recent retrieval sources to retain, so actual context can remain above the target. The default 24-run lookback yields 90 candidates. No target or warm-trigger input is shown to the user.

Candidates must complete all workload runs and meet the expected retention floor. Among feasible candidates, select the smallest total answer, summary, flush, ping and recalled-token cost. Ties are stable by target, trigger and horizon. All candidates share the same seeded workload; tightening the floor filters the same candidate set. A diagnostic candidate can still be inspected if none is feasible, but it is not a recommendation and savings are suppressed.

This is a **finite-grid optimum fitted to the synthetic scenario**, not a continuous global optimum, held-out evaluation or learned production policy. The search uses whole-scenario outcomes to choose fixed policy parameters; execution of those parameters uses only current/past context and configured return distributions. Current-query recovery cannot inflate the retention metric. The retention requirement applies to Compache; the OpenClaw summary's semantic retention is not evaluated, so the comparison is not quality-matched.

## Compache projection

The current user input, retrieval and previously generated answer blocks enter the sequence. Prior inputs and outputs remain verbatim. At a cold return, a view above target + two percentage points triggers retrieval compaction; a warm return waits for the selected 50%, 55% or 60% trigger. Oldest eligible retrieval blocks are removed toward the target. The selected recent retrieval horizon and current-query dependencies are protected; missing retrieval dependencies are appended as recalled source blocks, preserving the earlier prefix. Their input cost is charged. Extra recovery model calls are zero.

This assumes perfect current-dependency identification, excludes lexical ambiguity and unknown dependencies, and uses the finite-grid selected target, trigger and horizon. It does not run the package's Python training/held-out optimizer or establish semantic retention. Budget pressure can leave the view above target. Unlike the Python gate, this simplified projection removes blocks without pricing handle-placeholder text or applying the exact warm rollback, token calibration and serialized tool-pairing rules. It is useful for sensitivity, not a package-equivalence claim.

Keep-warm planning compares configured return probabilities and delay CDF against cold-compacted versus retained context, with priced ping requests, output and one subsequent carrying-cost allowance. It never reads the realized next gap, planned thread end or future dependency graph. It assumes no future query in its cold preview. Pings stop on actual arrival; all planned pings are charged after abandonment. The planner is an expected-value approximation, not the package's fitted survival estimator. Consequently pings can increase realized cost. Turning them off isolates passive cache reuse and compaction. The age-specific return probability is the continuation probability multiplied by `P(planned length >= next run) / P(planned length >= current run)`. It uses the configured distribution, never the realized future thread length. The displayed survival curve is this theoretical survival; the sampled workload may differ.

Admission uses the model's max input and `input + max(output, 32,000 reserve) <= context` for Compache. The optional custom context can make admission fail; failures are reported rather than dropping expensive turns from a savings denominator.

## Ping payoff and cache diagnostics

The ping graph replaces the four-step decision walkthrough. After a selected run it plots **expected avoided cost** and **expected ping cost** as lines, with **net saving** as bars. Zero pings is an explicit option. The planner now evaluates up to 12 pings by default; the advanced cap can be lowered. The winning positive value is highlighted, together with the ping interval and actual planned/sent counts. Sending stops when a user returns; waiting costs include abandonment. The lifetime is still bound to the chosen model.

In the average context view, the graph averages the same N-ping alternatives across eligible thread prefixes at that run. It shows eligible and total counts; gated prefixes are excluded, and the result is labeled “best average plan.” It is not the average of individual optimal values and does not replace each thread's own selected plan. Individual thread selection shows that thread's actual planner rows. If all prefixes are gated, no positive ping curve is invented. The average planned/sent numbers include all admitted run ends at the chosen position.

The token-mix bars reconcile exactly to answer-input tokens, with mutually exclusive categories:

- **Cached:** actual cache credit in the simulator.
- **First request:** all fresh input in the first successful request of a thread.
- **Expired prefix:** an exact cache-aligned match available in the last 16 expired snapshots, beyond the live match.
- **New context:** fresh tail blocks whose identity and length have never appeared in a previous successful request.
- **Unmatched history:** previously seen fresh tail blocks outside those matching prefixes, including rewrites and lookup/alignment limits.

Snapshots and seen-block identities include successful maintenance calls; aggregate percentages include answer calls only. Diagnostic expiry history is bounded and does not change actual cache credits or prices. Unmatched history is **not** a proven compaction effect, and these are modeled accounting categories, not provider counters. A large retrieval payload or a first request can keep the token hit rate low even if every reusable prefix is warm. A ping cannot cache future new content. Long gaps beyond the selected ping horizon can still expire.

The layout and payoff/cost chart use the supplied reference as design guidance only. Its private production figures and fitted return curves are not included.

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
| Tool pruning | Off for this controlled client-summary configuration across all models; Anthropic-specific seeded TTL pruning is not applied |
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

Engine tests cover deterministic causal graphs, company heterogeneity, shared-prefix/TTL accounting, generated-output cache timing, long-tier rates, protected dependencies and recall, summary/flush costs and bounds, deterministic-record replay cost, abandonment pings, negative savings, invalid configuration, admission and zero denominators. Tests also cover pre-recovery retention, per-type dependencies, constrained minimum cost, causal execution, model-bound cache defaults and Haiku whole-request pricing tiers. Tests cover matched-cohort averages, adaptive survival axes, zero retrieval, cache-mix reconciliation and ping aggregation. Browser checks cover slider boundaries, model changes, zero/large retrieval, short-thread scaling, average and individual views, reset, validation and responsive layout.

The next evidence step is an adapter using actual documents and generated answers, compared against the real package under the same settings. See [the benchmark source plan](benchmark-trace-plan.md): QASPER is the proposed reusable source, MuSiQue checks controlled multi-hop dependencies, and LongMemEval is a separate memory diagnostic. No public benchmark dataset has yet been downloaded or used in these calculator results.
