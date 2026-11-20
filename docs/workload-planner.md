# Workload planner

Compache 0.8 includes a trace-fitted planner for a serial RAG thread. It selects the request view and keep-warm schedule using expected cumulative cost over the workload's distribution of returns, stops, token growth and foreground model calls. A held-out replay compares **summed charges across the same arrivals**, including pings sent after the last answer.

From the checkout root, use `PYTHONPATH=src python3 -m compache planner --help` with Python 3.11 or newer or install with `python -m pip install .`. The Python package has no required runtime dependencies. Selecting a tiktoken encoding in the CLI requires a separate tiktoken installation. [Try the two frozen synthetic-RAG fits](../examples/fitted_policy/README.md).

The [input-accounting walkthrough](input-accounting.md) covers measured request anchors, edit-only token counts, error allowances and a runnable compaction example.

## Count the shared prefix

Give the host's actual system prompt and ordered default tool schemas to the planner. A tool's description, parameter schema and serialized punctuation all contribute. These fields can make a substantial part of a first request eligible for a cache read. Each process shares a catalog among sessions using the same WorkloadPolicy. The catalog isolates installation, provider, model and cache key. A new conversation can benefit only from a matching, eligible prefix of a prior successful request within the TTL. The provider must route these requests into a shared cache namespace. Set `share_headers=False` when the route isolates threads.

```sh
compache planner headers --model-spec model.json --header header.json \
  --encoding o200k_base --additional-input-tokens 2048 --gap-seconds 30
```

| Input or result | Meaning |
|---|---|
| `system_prompt` | System instructions including any visible wrapper the host sends. |
| `default_tools` | Ordered array of default tool definitions or a single serialized fragment. The report gives the group total and per-tool counts in array order. For opaque fragments the report identifies the unallocated tool total. |
| `component_order` | Optional array containing `system_prompt` and `default_tools` in provider wire order. A change in the first component can invalidate both. |
| Token ruler | A host callback returning token IDs measures the supplied representation. A count-only callback estimates common lengths at component boundaries. The default estimates tokens from UTF-8 bytes. The optional CLI encoding uses locally installed tiktoken. An unknown model mapping requires an explicit encoding or the byte estimate. |
| Cache overlap | The longer of the thread's matching prefix and a matching catalog entry, rounded down to full cache blocks and subject to the model minimum. Shared header and thread reuse may cover the same tokens. |
| Dynamic tools | The Python host may supply a thread-specific trailing tool segment. It consumes context and input budget. Cross-thread credit stops at the end of the static prefix. |

For an example with a 374-token system prompt and 755 tokens of tools, a matching catalog entry can supply 1,024 of the 1,129 header tokens under a 1,024-token minimum and 128-token blocks. With a read multiplier of 0.1 and a write multiplier of 1, this header costs 207.4 input equivalents on a hit and 1,129 on a cold request. Both policies get the discount when they send the same prefix.

Provide exact wire fragments when an SDK changes serialization. Token IDs are exact for the bytes supplied and selected encoding. Provider-controlled framing, routing, writes and eviction still affect the actual counters. Read/write multipliers, cache minimum, block size and TTL come from the model spec. Ordinary input costs one equivalent, eligible uncached input costs the write multiplier, a matching token costs the read multiplier and output costs the output multiplier. [tiktoken](https://github.com/openai/tiktoken)

## Fit and compare

```sh
compache planner fit --threads train.jsonl --labels train-labels.jsonl \
  --model-spec model.json --header header.json --observation-end 2026-01-31T00:00:00Z \
  --retention-floor 0.95 --output policy.json

compache planner replay --threads heldout.jsonl --labels heldout-labels.jsonl \
  --policy policy.json --header header.json --observation-end 2026-01-31T00:00:00Z \
  --output replay.json
```

Choose an observation end through which returns and timers could actually be observed. Keep train and evaluation threads disjoint. If using a specific encoding, pass it to both commands.

- The demand model learns return probability, idle gaps, new input and retrieval sizes, response length and duration by turn position. It distinguishes fully observed idle horizons from censored ones. The default Brownian sampler models increments in log incoming tokens. The empirical sampler reuses observed transitions. Return timing and termination remain empirical in both.
- A cut recipe defines separate lookbacks for user inputs, outputs and complete retrieval groups. A full-text alternative must be present before recovery for a requirement to count as retained. The default retention floor is 95%.
- Without `--recipes`, fitting scores a grid: every pairing of a dialogue window with a retrieval window, at 1, 2, 5, 10, 20, 50 and so on below the longest training thread, each with and without query-match protection. It admits one recipe, the cheapest in projected training tokens among those that pass the floor, save tokens and met at least one assessed requirement. Each row of `recipes` records its `admission` outcome. The admitted recipe sits at the floor by construction, so check its retention on held-out threads with `planner replay` before relying on it.
- With `--recipes`, fitting admits the supplied recipes in their order, including combinations of irreversible cuts, as before. `--select` overrides either default.
- The floor applies to assessed requirements (`coverage`). A requirement nobody could assess no longer vetoes a cut. Each row also reports `coverage_lower_bound`, which counts every unknown requirement on a labelled turn the recipe changed as lost, and `--unknowns worst_case` applies the floor to that bound instead. A turn without a label is treated as unsampled: it is reported as `unlabeled_changed_runs` and does not affect admission. `--max-unknown-change-rate 0` restores the earlier rule, which rejects a recipe that changes any unassessable or unlabelled turn.
- At a request, the planner compares eligible checkpoints with keeping the current view. At completion it compares lapse with a cancellable ping schedule. A bounded search integrates expected charges over future returns and exits on common workload draws. The default looks ahead three runs with three sampling strata. The continuation value at the horizon is zero. Admission limits still apply.
- Supplying a common header at fit time measures the fraction of eligible static tokens another thread could have warmed at otherwise cold returns in the training calendar. First runs have their own denominator. Forecasts use the matching installation and header. A different header or unseen scope receives zero forecast credit until calibrated. On an actual arrival the catalog can recognize successful recent traffic immediately. Supply the deployment provider name and cache key when fitting and replaying. The Python and CLI options allow an explicit probability for scenario analysis.
- Replay charges a preserve-all/no-ping reference and Compache under identical cache economics, header traffic, timestamps and recorded answers. `--no-share-headers` isolates threads in both arms. To use a deployed host policy as the reference, export a full call ledger on the matched workload and use the comparison below. Recovery calls and new answer generation are outside this workload replay.

## Price a keep-warm schedule

At foreground completion, Compache compares zero or more timers on the fitted returns and thread endings for this run position. A timer is billed only if it fires. A user arriving exactly at a timer cancels it. Reuse expires exactly at the configured TTL. For each training return, the planner computes the incremental cache reads available to its first foreground request with and without the schedule. Later same-run calls already follow that foreground request and pay their own read charges.

The planner averages the extra foreground savings and the maintenance charges across the full timing cohort. It checks the resulting net payoff alongside the recursive compaction alternatives. A planned rewrite can shorten the reusable prefix. A maintenance-only suffix can help an identical repeat ping, while a returning user's question can reuse the shared foreground prefix. The idle decision's `ping_economics` records the probability of dispatch, the probability of a beneficial return, the expected maintenance charge, the upper bound on foreground savings and the estimated net savings by schedule. Prices use standard fresh-input-token equivalents.

To fit the layout of your traffic, pass `--next-endpoint-reuse-fraction fractions.json` to `compache planner fit`, or use the `next_endpoint_reuse_fraction` keyword of `fit_workload`. Each key is an originating training run ID with a subsequent foreground request. A value of 0 means the next request cannot read the measured thread prefix, 1 means it can address the entire prefix, and intermediate values represent a confirmed partial overlap. Omit unknowns. The compatibility assumption for missing observations is an append-only continuation with full structural overlap. TTL, compaction and per-request eligibility still apply.

For a host with one known foreground publication, `prior.published_endpoint_reachable_in(following)` compares two `CacheRequest` records using their partition and eligible ending identities. It returns a boolean within one partition and `None` across partitions. Use this binary result only where the prior endpoint accounts for the entire incremental thread prefix. If another earlier write can serve part of the prefix, supply its verified fractional overlap or omit the observation. The fit can use content-free layout measurements from completed training runs. The next evaluation question is unavailable when its timer is scheduled.

A `compache-model-spec-v2` can also set `long_input_threshold`, `long_input_factor` and `long_output_factor`. The tier is chosen independently for each request using its total input before splitting fresh tokens, writes and reads. For the Luna benchmark, a request over 272,000 input tokens costs twice the standard input rate and 1.5 times the standard output rate. The planner prices a ping at its own size and its potential benefit at the returning request's size.

## Count foreground calls

Pass `--calls-per-run train-calls.json` to `planner fit` and a separate held-out file to `planner replay`. The Python keyword for both is `calls_per_run`. A value of 4 counts the initial request and three subsequent same-model calls that reuse the initial view within the run. Count each successful physical foreground model response once within the same model, route and cache key. One model response can request several tools.

```json
{"run-a": 1, "run-b": 4, "run-c": null}
```

The CLI also accepts a JSON object with `complete_run_ids` and `events`. Each event has a stable physical `call_id`, a `run_id`, `role` and `status`. The Python helper `count_foreground_calls(events, complete_run_ids=...)` deduplicates repeated deliveries. It counts successful `answer` and `foreground_step` events. Use `ping`, `compaction`, `recovery` or `support` for other activity. Supply only fully observed runs with one successful final answer in the completeness list. Foreground events can declare model, provider and cache key to validate membership. Ambiguous meter increments alone cannot identify multiple distinct physical calls.

For each run position the planner estimates E[N] from the observed counts. A position with no observations uses the pooled mean. Without any evidence `E[N] = 1`, including when loading older policy artifacts. The forecast adds `(E[N] − 1) × charge(I, I, 0)` for a candidate with initial input I. A repeat can read eligible complete cache blocks from that initial input even if the first request was cold. The unaligned remainder takes the write price. Sub-minimum or cache-disabled requests take the ordinary input price. The estimate excludes subsequent tool payload growth, cache expiry within a run and extra output that the trace did not record. Recorded run output is charged once.

Held-out counts are consulted only after the action is chosen. Both arms use the same held-out counts. A missing held-out count assumes one and is excluded from the supplied-count denominator. The answer ledger gives first-request occupancy. Separate `foreground_repeat` rows aggregate the repeated initial input with its read/write/fresh token split and price. The receipt and call-count functionality uses existing host counters.

## Compare the installed host policy

Call `replay_workload(..., include_host_contract=True)` to export the matched workload, pricing and observation window. Use `compare_host_ledger(replay, export)` from `compache.optimizer` with the same contract, a policy name, the source meter's total call count, an explicit completeness attestation and the host's events. Each event identifies the call, thread, run, timestamp, role, status and token counts. Fresh, write and read counts must sum to total input. Include charged retries, pings, recovery and summary calls. When foreground counts are supplied, identify intermediate model calls as foreground_step and the final call as answer. The verifier checks their count, timing and the frozen answer's output. Extra intermediate output outside the modeled output budget withholds the matched savings result. Incomplete coverage withholds the savings result. The signed difference between total host cost and candidate cost can be negative.

## Connect the host

```python
from compache.optimizer import CacheBreakChecker, CacheScope, WorkloadPolicy, WorkloadSession

policy = WorkloadPolicy(fitted_artifact)  # share this instance across sessions
checker = CacheBreakChecker("compache-cache.sqlite3")
scope = CacheScope(installation_id, provider_name, model_id, thread_id, routing_key)
session = WorkloadSession(policy, scope, checker, tokenize=encode_wire_bytes)

plan = session.prepare(request, started_at=request_start, system_prompt=wire_system,
                       tools=ordered_default_tools, dynamic_tools=thread_only_tools)
# Send the same header and plan.view.parts through the normal model call.
result = session.complete(plan, output_source, ended_at=request_end,
                          response=completed_provider_response)
if session.next_ping_at is not None:
    schedule_timer(session.next_ping_at, request.run_id)
```

Use one sequential session per thread. Preserve the canonical archive and expose the package's `reopen` recovery handler to the model whenever history can be replaced by a handle. A new arrival revokes the preceding run's timers. At dispatch check `session.ping_view(run_id=..., at=...)` immediately before sending. Send only when it returns a view and pass the resulting receipt to `session.acknowledge_ping`. The host owns the scheduler, provider adapter and retry transport. Store provider counters separately from expected charges. The in-memory catalog bounds sharing to sessions on that worker; the SQLite checker supports persistence and multiple workers using the same database.

Pass the ping’s own completed response to `session.acknowledge_ping(run_id=..., at=ping_started_at, completed_at=receipt_time, response=ping_response)`. The optional completion time defaults to the dispatch time. A true result confirms the refresh. Always book the dispatched call’s charge. Then read `session.next_ping_at` and schedule that deadline if present. Recheck the view before sending each ping, including any timers your host queued earlier.

### Reconcile the provider's published prefix

For GPT-6 Luna's default Responses API caching, configure the planner with `cache_ttl_seconds=1800`, `cache_min_tokens=1024` and `cache_block_tokens=1`. The planner uses the fitted model specification to schedule timers. OpenAI's default 30-minute lifetime applies to cache-read pricing and can extend beyond the minimum. [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching#cache-lifetime-gpt-5-6-and-later)

For automatic caching with one contiguous prefix, pass the completed Responses object to both `complete` and `acknowledge_ping`. When both cache counters are present, Compache uses the sum of observed reads and writes, rounded down to the configured cache block and subject to the minimum. Total input still anchors occupancy and billing. For example, a foreground request with 8,873 total input tokens, 0 reads and 8,798 writes can offer 8,798 tokens to a subsequent ping when the block size is one. Its ordinary 75 input tokens stay outside that boundary. Output tokens, including reasoning, do not extend the input cache. If a counter is unavailable, the existing estimate from block-rounded total input applies. Zero is an observed count.

With explicit breakpoints, the adapter must establish the contiguous prefix covered for the configured lifetime from the actual request layout. Set `reusable_input_tokens` on each completion and ping receipt to that verified bound, or to zero if coverage is unknown. When a receipt reports positive 30-minute and 12-hour writes together, Compache requires this bound before using the aggregate for a refresh. The host supplies it for other multi-breakpoint layouts whose shape is absent from usage counters. A breakpoint on a user input can leave later assistant, reasoning and tool items outside the covered prefix.

For native Responses maintenance that appends a maintenance-only user message, keep the original input items and provider settings, append the actual native response items and the maintenance request, and identify the ping as a separate branch. When `acknowledge_ping` receives an OpenAI Responses object reporting the fitted `gpt-6-luna` model and the scope provider is `openai` (case-insensitive), it defaults to this treatment. On any provider, pass `ping_has_private_suffix=True` when the actual maintenance input branches from foreground history. Pass `False` from the first ping when the host has verified the whole ping input is the foreground continuation. Other providers, explicit counters without a response object and modeled replay keep their existing default. The branch choice persists across an identical repeat. The next real user turn branches from the foreground conversation and omits the ping and its output. An identical repeat can reuse the extra blocks written for the maintenance request. The foreground branch retains its last established input prefix and occupancy anchor. A zero-output prewarm of only the original input may cover that original prefix. Record its actual receipt against the request it sent. Extra assistant or tool coverage requires separate evidence.

### Reprice an expensive refresh

When a completed ping read the eligible prior prefix and returned usable counters, its actual dispatch sets the new cache clock. Extra billed output can increase the expected cost of the next ping. The first ping’s newly appended visible answer can legitimately be a write. A later ping can read that answer along with the prior prefix.

After a confirmed ping, the planner compares stopping now with the remaining offered timer schedules using its measured input mix and output. For a complete receipt the incurred charge in standard fresh-input equivalents is `input_factor × (fresh + write_multiplier × writes + read_multiplier × reads) + output_rate × output`. The request’s total input determines both rates. The completed ping’s charge enters the ledger once. All choices share that sunk cost.

The next ping sends the same prompt. Compache prices its newly reusable blocks at the cache-read rate and uses the observed output length for this chain. When the receipt distinguishes fresh inputs and cache writes, the projected price per remaining uncached token is `(fresh + write_multiplier × writes) / (fresh + writes)`. If the receipt has no uncached tokens or omits the write count, the planner uses the model’s configured rate for that tail. The receipt records the pricing basis. The next foreground run starts with its configured ping forecast.

For each schedule, the decision conditions on the thread still being idle when the receipt arrived. It calculates expected charges for additional pings actually sent, plus the cost of future foreground requests at the resulting cache age. A return at the same instant as a timer wins. The schedule with the lowest expected future charge wins, with fewer pings preferred on a tie. With no supported future return or no feasible remaining deadline, the chain ends and the verified refresh still retains its original expiry. The remaining candidates are bounded by the deadlines offered at foreground completion.

`session.last_ping_receipt` gives the billed token counters, cost basis, output-overrun flag and continuation decision. Missing or inadequate evidence stops the chain. An observed on-time zero-read or verified static-only history miss can also disable future pings for that scope. Offline replay marks synthetic receipts as modeled. Set `--modeled-ping-output-tokens 9` on `compache planner replay` to exercise a different realized ping output while keeping the model spec’s initial forecast fixed.

## Replay warm and cold boundaries

`VirtualCache` replays completed calls on a logical clock in dispatch order. Give each trajectory its own cache for each lifetime. At the exact expiry time a lookup is cold. The five-minute scenario is hypothetical for Luna. The 30-minute scenario models expiry at the provider’s minimum lifetime. [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching#cache-lifetime-gpt-5-6-and-later)

```python
from compache.optimizer.usage import response_usage
from compache.optimizer.virtual_cache import DEFAULT_TTLS, CacheRequest, VirtualCache

caches = [VirtualCache(seconds) for seconds in DEFAULT_TTLS]
for logical_seconds, payload, completed_response in recorded_calls:
    request = CacheRequest.from_payload(payload, scope="benchmark-arm/replicate-1")
    receipt = response_usage(completed_response)
    for cache in caches:
        event = cache.apply(request, at=logical_seconds, usage=receipt)
        save_event(event)
```

Only an earlier write to a matching eligible endpoint can produce a modeled read. The first request pays to write the cacheable input even if its physical receipt reports a hit. A ping’s private message ending can serve an identical repeat. A sibling foreground return can read through the previously published common ending. The model assumes that a successful read also refreshes eligible, live ancestor entries covered by that read. [Native lookup boundaries](https://developers.openai.com/api/docs/guides/prompt-caching#how-prefix-matching-works)

Pass `state="warm"` to allow an expired, matching write at one boundary, or `state="cold"` to evict the matching lookup entries before that call. `observed_usage` retains the measured counters. In `modeled_usage`, reads plus writes equal the receipt’s cacheable input, while ordinary input and generated output retain their measured counts. To replay an estimated archive, supply its token assumptions to `cache.project(request, at=..., input_tokens=..., cacheable_input_tokens=..., output_tokens=...)`. Those events have `observed_usage=None`. A host that records complete prefix identities can build `CacheRequest(partition, boundaries)` directly, with ascending `(item_count, prefix_identity)` pairs for its eligible lookups and the published endpoint last. The native payload adapter supports implicit caching with one write endpoint and fully materialized history.

An output-capped generation can carry a billable input receipt. Pass `response_usage(response)` to replay its reported reads, writes and output, then account for the next attempt as a separate request. An incomplete answer stays outside the foreground conversation history.

## Diagnose lost cache

An eligible on-time zero read disables pings on that thread and route and records a warning. The warning identifies which system/tool component changed when one is observable. A changed header alone can be flagged as unconfirmed. Expiry, deliberate changes and below-minimum requests have distinct outcomes. When a provider reports one contiguous cached prefix, a host can pass a verified static overlap with `verified_shared_static_tokens`. Hosts with verified native tokenization, stable provider framing and contiguous prefix counters can instead enable `verified_contiguous_prefix_reads=True` on the session to check this automatically. This identifies a partial read that ends inside the shared header before reaching the previous thread history. A host-established upper bound on the exact common prefix can also establish invalidation.

Inspect `result["cache_event"]`, application warnings from `compache.optimizer.cache_breaks` or `checker.events(scope)`. Warning records contain hashes, changed component names, timestamps and token counters. The latch survives restarted workers. Call `checker.reset(scope)` after the host corrects the cause. If both stable and necessary system/tool changes are expected, mark planned changes explicitly with `intentional_change=True`.
