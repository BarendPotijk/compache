# Implemented RAG optimizer, version 0.2

Compache learns a policy from normalized threads and frozen dependency labels, evaluates it with the same gate and planner exposed by the package, and saves a reviewable recommendation. This is a local, deterministic implementation: no network or model calls happen during fitting, optimization, preparation or replay.

## Supported contract

Inputs use the [trace contract](trace-contracts.md): sequential runs, one generation per run and full canonical history. Each request contains previous inputs, exposed retrievals and outputs followed by the current input and exposed retrievals. One supplied `ModelSpec` must match every generation's model ID. Mixed models are rejected for fitting/replay rather than silently priced with the wrong assumptions.

Only historical retrieved bodies can be replaced with placeholders. User inputs, assistant answers and current retrievals stay verbatim. This version does not implement arbitrary tool transcripts, preserved thinking, multi-call running turns, general turn compaction, provider wire serialization or framework wrappers.

## Fit, select and validate

1. Split whole threads by a stable hash of thread ID, or reserve one company with `--heldout-company`. Default holdout fraction: 25%, rounded up. No thread appears on both sides.
2. Fit training-only return curves using Kaplan–Meier survival estimates. An idle gap runs from the previous run's end to the next run's start. Last runs are right-censored at the explicit observation cutoff, limited to a seven-day horizon. Residual survival is not a measured probability of never returning. Run indices with fewer than five observations use a pooled curve.
3. Summarize historical dependency occurrences by source kind and run distance. These are counts in sufficient evidence alternatives, not token-weighted reuse or causal recall. The profile stores aggregates and hashes, not questions, retrieved text or answers.
4. Search a fixed grid. Grace distances come from training labels plus 0 and 1; targets are 10%/15%/20%/25%, cold triggers two percentage points above target, warm triggers 50%/60%, and maximum ping counts 0/1/3/6, bounded by `--max-pings`. A preserve-all candidate is always included. Other controls are explicit engineering constants. Ties preserve more history, then use fewer pings, a larger target and a higher warm threshold.
5. Reject training candidates with admission failures, skipped runs, too many known misses or too many changed unknown runs. The historical-requirement coverage floor defaults to 95%, configurable with `--min-dependency-coverage` (for example 0.90). The changed-unknown-run ceiling remains zero. The optional `--max-miss-rate` is an additional run-level guard, defaulting to 1.0 so the historical coverage rule determines eligibility. Missing labels never become zero-risk negative examples.
6. Evaluate the training winner once on held-out threads and compare preserve-all/no-cache, cache-only, compaction-only and combined costs. Validation may veto promotion; it never reselects a candidate. Lower held-out modeled cost, satisfied risk/admission constraints and complete visible-text cost coverage are all required. Otherwise recommend preserve-all/no-pings if admissible.

The supplied development corpus has already informed the implementation; thread splitting prevents direct fitting leakage, but does not make this an untouched external test set. The tool does not calculate confidence intervals or certify semantic answer quality. A production release needs new representative traces and answer replays under the selected views.

## Gate and source recovery

`prepare` reads a canonical request, policy, explicit thread state and clock. It has no access to labels or the target/future answer. Its byte-stable projection reuses existing placeholders while warm and appends new content. It considers a warm replan only above the configured 50% or 60% pressure threshold (standalone default 60%). A warm rewrite must reach the compaction target; when protected context makes that impossible, it keeps the cached projection (`deferred_warm`) until a cold return. If the unchanged view would breach the hard token/byte admission limit, it may take a partial safety cut (`replaced_warm_for_admission`); admission still fails if protected/current context cannot fit. Cold compaction can take partial savings even when the target is unreachable. Telemetry records these decisions. New policies latch at cold boundaries. Missing, edited or reordered canonical prefix parts invalidate the old recipe and recoveries.

Oldest eligible retrievals are replaced until reaching the target, or until no eligible bodies remain. Grace is the number of preceding runs whose retrievals remain protected. A conservative lexical guard also protects distinctive current-question overlaps and document IDs cited in prior outputs. Identical fresh retrievals can make an older occurrence eligible. These rules are heuristics, not semantic classifiers. Callers can additionally pass `protected_ids`.

Cuts remain sticky while canonical history extends, including after cache expiry. A later query that needs a removed source must expand it. Each placeholder contains a stable handle scoped to the thread, part identity and body fingerprint. `expand` returns at most 48,000 characters per page; the caller must implement the model-visible tool and persist recovered content in its integration. Replay includes a simulated recovery mechanism, but no live tool integration ships here.

Estimated token and byte admission limits protect known-model requests; failures do not commit thread state. Unknown models and masked inputs are observe-only, make no new cuts and explicitly report that admission is unverified. The pass-through is not a guarantee the provider will accept the request. Do not send null/masked bodies as if they were complete production prompts.

## Planner and cache model

`end_run` returns `keep_warm` with due timestamps only when the fitted return curve and supplied economics imply positive value; otherwise it returns `lapse`. It includes the first ping's new answer write, recurring prefix reads, output cost and survival-weighted probability of sending each ping. It accounts for request duration, cache minimum, TTL, ping lead and the warm ceiling. The [retention test vectors](retention.md) are executable tests. No curve means no pings.

The replay cache holds one latest prefix per thread. It matches whole serialized parts, rounds shared tokens down to the configured block size, expires at TTL from request start, and refreshes on cacheable requests. It does not model provider cache eviction, routing, multiple surviving cache branches, cache-write latency, or queue delays. Ping plans stop when a new run starts, the collection cutoff arrives, or the simulated ping misses/produces billed output. The implementation sends no actual pings. An integration must recheck idleness, lease ownership, deadline and provider capability before executing a decision.

The planner's expected-value formula uses continuous token sizes; the replay ledger applies the explicit minimum/block rounding. It prices a hypothetical cold view using only the just-completed history, without seeing the next user question. The one-generation RAG contract has zero later calls per run. Persistent stores, `observe`/bill calibration, cache breakpoints and provider-native requests are not implemented.

## Cost, risk and evidence

The token ruler estimates `ceil(UTF-8 bytes / bytes_per_token)` for each body plus configured framing overhead per part. It does not tokenize provider requests or calibrate from Codex's large system/tool prompt. Ledger costs are base-input-token equivalents: fresh input + read multiplier × cached input + write multiplier × cache writes + output multiplier × output tokens. They are not currency amounts, actual usage, or subscription charges. Unavailable bodies have unknown cost; totals cover visible projection text and affected runs are flagged.

Task requirements use OR between alternatives and AND within each alternative and between requirements. A known miss means no sufficient alternative survives in full before recovery. Output-support claims do not substitute for task requirements. A partial, unknown or absent label makes the run unassessable; changing its context counts against the separate unknown-change ceiling. Neither source labels nor zero measured misses establish answer quality.

Historical coverage is measured before recovery: retained historical requirements divided by assessed historical requirements. A requirement counts as retained when every source in at least one sufficient alternative is available. Direct lookup requirements do not dilute this denominator. Unknown requirements remain unassessed and the separate unknown-change guard applies. No historical requirements yields an explicit null coverage with denominator zero. A five-requirement sample cannot distinguish 90% from 95%: either threshold needs all five.

The target is a requested post-compaction fraction; current-turn content and protected evidence can prevent attaining it. Report achieved sizes alongside configured targets. Policy selection must not be described as having reached a target based solely on its configured value. Constraints artifacts use v2; v1 artifacts retain their original run-level semantics when loaded.

For a miss, replay uses the frozen requirement sets as an explicitly labeled recovery oracle. It chooses a cheapest incremental missing alternative for each requirement, charges a small recovery-request output and a second model call, appends restored sources, and carries them forward. This is not a globally optimal set solver or a model's demonstrated ability to notice omissions. It models one extra call at the original request timestamp and does not estimate recovery latency. Known misses remain in the miss rate even after successful modeled recovery.

## CLI artifacts

```sh
compache fit --threads threads.jsonl --labels labels.jsonl --model-spec model.json \
  --observation-end 2026-06-21T11:21:27+00:00 --output profile.json
compache solve --threads threads.jsonl --labels labels.jsonl --model-spec model.json \
  --observation-end 2026-06-21T11:21:27+00:00 --output policy-run
compache replay --threads threads.jsonl --labels labels.jsonl \
  --policy policy-run/policy.json --profile policy-run/profile.json \
  --observation-end 2026-06-21T11:21:27+00:00 --output replay.json
compache prepare --threads threads.jsonl --policy policy-run/policy.json \
  --thread-id rag-001 --run-id rag-001-run-03 --output prepared.json
```

Choose a collection cutoff and IDs from your data. `fit` alone uses all supplied threads; `solve` performs its own training split and fit. `prepare` reconstructs state sequentially through the chosen run, without hypothetical pings or label-driven recovery. `replay` on the entire training+held-out corpus is diagnostic, not a held-out score.

`solve` writes:

| Artifact | Meaning |
|---|---|
| `profile.json` | Training-only aggregates, timing provenance, source hashes and ruler assumptions |
| `candidate-policy.json` | Lowest-cost feasible training candidate, even if validation rejects it |
| `policy.json` | Recommended candidate or preserve-all/no-pings fallback; null if no admissible fallback |
| `report.json` | Every training candidate, winner, held-out rows, baselines, promotion checks and costs |
| `heldout-ledger.jsonl` | Per-request, expansion and ping costs for the candidate, not the fallback |

All serialized policies require their versioned fields. The planner/replay verify the fitted profile hash. JSON artifacts and per-run ledgers are deterministic for identical inputs. Keep source traces and labels private; profiles omit their text, while reports still include thread/run/company identifiers and source IDs.

## Python API

```python
import json
from pathlib import Path
from compache.optimizer import ModelSpec, Request, ThreadState, optimize, prepare
from compache.optimizer.contracts import seconds
from compache.traces import load_threads

data = Path("examples/rag_workload")
threads = load_threads(data / "threads.jsonl")
labels = [json.loads(s) for s in (data / "labels.jsonl").read_text().splitlines()]
spec = ModelSpec.from_dict(json.loads((data / "pressure-model.json").read_text()))
result = optimize(threads, labels, spec,
                  observation_end="2026-06-21T11:21:27+00:00")

from compache.optimizer import Policy
if result["recommended_policy"] is not None:
    policy = Policy.from_dict(result["recommended_policy"])
    thread = threads[0]
    state = ThreadState(thread.thread_id)  # explicit, caller-owned, per-thread
    for run in thread.runs:
        request = Request.from_thread(thread, run.run_id)
        view = prepare(request, policy, state, seconds(run.generation.started_at))
        if not view.admitted:
            raise RuntimeError(view.reason)
        # Provider adapters must translate view.parts and supply expansion tools.
        print(view.telemetry())
```

The fixture model ID describes the source answers only. Its context window, TTL, cache eligibility, multipliers and ruler are illustrative inputs, not verified specifications of that model. Nothing in this example sends requests or executes retention plans.
