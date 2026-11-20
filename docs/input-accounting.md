# Verify the input count in Compache 0.6

The workload planner anchors occupancy to the unrounded input meter for the request view the host actually sent. Counting happens locally. The ordinary model response supplies the next measurement. Compache makes zero extra requests to a generation or token-count endpoint.

## 1. Record the matched request

```python
from compache.optimizer import CacheBreakChecker, CacheScope, WorkloadPolicy, WorkloadSession

policy = WorkloadPolicy(fitted_artifact)
scope = CacheScope(installation, provider, model_id, thread_id, routing_key)
checker = CacheBreakChecker("cache.sqlite3")
session = WorkloadSession(policy, scope, checker, count_part_tokens=host_count_block)

plan = session.prepare(request, started_at=start, system_prompt=system, tools=default_tools)
# Send the plan through the host's existing provider call using the same serializer.
completed = session.complete(plan, visible_answer_source, ended_at=end, response=provider_response)
```

Pass a completed Responses API dictionary, SDK response object or terminal stream event. The host associates it with the pending plan. Other transports can pass `input_tokens`, `cached_input_tokens` and optionally `response_id` directly. A reported total input of 1,423 remains 1,423 even if 128 tokens were cache reads. Rounding down to a cache block happens separately for reuse. A rejected or superseded completion leaves its meter uncommitted. The session rejects a known response ID supplied again for a different request.

For multiple calls within a tool loop, keep one view and matching meter per call. The serial workload session models a foreground request and its pings. Other renderers and intermediate tool calls can use `PartMeter`, `InputAnchor` and `estimate_input` directly on their actual rendered views. For the experimental Codex snapshot runner, `view_usage` is the first metered increment paired with the submitted snapshot, `last_model_usage` describes the final metered increment, and `usage` accounts for the whole turn. An aggregate without an identifiable per-view meter cannot establish an occupancy anchor.

## 2. Count the edits locally

```text
next input = prior reported input
           − local size of removed or replaced body blocks
           + local size of added or replacement body blocks
           + (new header/framing size − previous header/framing size)
```

A block comprises a user message, rendered answer, retrieval result or the handle that replaces a compacted item. A changed block with the same ID contributes one removal and one addition. Reordering the same blocks preserves their combined size under the additive ruler, while their order matters to cache-prefix matching. A raw billed output can include reasoning and formatting; use the items actually replayed in the next request. Pass `answer_history_tokens` when the host already knows that rendered block's size.

The default body ruler uses the configured UTF-8 bytes-per-token and per-part framing estimate. A host can supply `count_part_tokens(source)` to measure the block its renderer sends using a local tokenizer. The header has separate `tokenize` and `count_tokens` callbacks. A stable header reuses its measurement. A changed header is measured as one ordered unit to preserve token merges across component boundaries. Historical source IDs, order and content still need comparison, but stable sent blocks reuse their saved token counts. Compache requires no tokenizer dependency. Local counts assume additive block boundaries and can differ from an exact full-request encoding.

## 3. Decide how far to compact

Evaluate keeping the current view and the cut recipes approved by the fitted dependency constraint (95% by default). Each candidate reports its own estimate in `plan.decision["candidate_input_accounting"]`. The planner uses the point estimate for price. Admission uses the estimate plus the observed undercount allowance for that type of edit. It checks the model's input cap (context window minus the configured reserve), wire-byte limit and modeled output headroom. It then chooses the feasible action with the lowest expected workload cost. Some workloads supply only a keep candidate. If none is feasible, `WorkloadAdmissionError` exposes the candidate arithmetic, cap and forecast status.

A warm timestamp affects pricing. Cache TTL, route and matched prefix remain separate from occupancy. The provider's aggregate input meter carries no information about where unobserved framing lies, so an unexplained residual receives no automatic shared-prefix credit. A thread's input anchor remains usable across cache expiry.

## 4. Learn from the actual response

`completed["input_reconciliation"]` shows the forecast, reported input, request identifier when available and signed difference (actual minus estimate). For a matched anchored forecast the session records `max(0, difference)` in the appropriate history for an append, rewrite, unchanged view or header change. Each history holds the latest 64 observations. The admission allowance is their nearest-rank 95th percentile, initially zero. An early positive observation can therefore supply the entire allowance until there is more evidence. The allowance is empirical and applies to the current request's admission, independently of the configured reserve and modeled future growth. It adds no tokens to price.

A completed foreground call or ping with a matching input meter becomes the next occupancy anchor. A ping also needs a confirming receipt before the planner credits its cache refresh. When a completed call lacks a matching input meter, the last compatible measured anchor remains available. The first request uses the local ruler, and a subtraction inconsistent with its anchor falls back to the local size with a reason. Call `invalidate_input_anchor()` before preparing another request after unobserved changes to context or serialization. Use a new scope/session for a new route or model. Historical normalized traces contain insufficient wire identity to assign their logged input count to a counterfactual compacted view.

## 5. Verify a decision without a provider

Run `python examples/rag_workload/verify_input_accounting.py` from the source package with Compache installed. The example uses a small local fixture, a two-turn fitted policy and a deliberately simple block ruler.

| Calculation | Tokens |
|---|---:|
| First local estimate (header 10, question 30, retrieval 1,000) | 1,040 |
| Input meter supplied by the first response | 1,423 |
| Keep on the next turn (1,423 + replayed answer 40 + question 20 + retrieval 80) | 1,563 |
| Input cap (1,600-token window less a 100-token reserve) | 1,500 |
| Cut (1,563 − old retrieval 1,000 + its handle 12) | **575** |

The response specifies 900 billed output tokens; the actual assistant record going back into history occupies 40 in this fixture. At this point there are zero earlier anchored-error observations. The keep view exceeds the cap, and the approved cut fits. The same calculations are available as structured fields on the decision. [Runnable example](../examples/rag_workload/verify_input_accounting.py)

## Captured verification

On two saved 96-turn low-effort Luna replays, a renderer adapter fed the package the actual serialized records measured with a locally cached `o200k_base` encoding. All 196 anchored transitions matched the first recorded input counter, including 37 rewrites. One recorded compaction gives **175,006 − 165,543 + 283 + 41,056 + 68 + 48 = 50,918**: prior input, removed retrievals, handles, new retrieval, replayed answer, and new question respectively. The 6,144 reported cache reads are included in that 50,918.

A separate end-to-end session replay used the prepared messages and visible answers from 45 persistent Luna-medium conversations. Across 162 continuations it measured zero unchanged historical blocks again. The median undercount was 46 tokens, p95 was 82 and the maximum was 94. These checks used saved subscription telemetry, with outbound sockets blocked. They validate sizing for the captured requests. They do not measure new answer quality, savings or provider billing.
