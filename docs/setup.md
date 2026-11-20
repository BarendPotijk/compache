# Fit Compache to one workflow

Start with one serial workflow on one model and cache route. A legal-research assistant and a support assistant can have different return times and context growth even in the same installation. Use separate exports and policies when those routes serve different traffic. Choosing a workflow name changes the example and guidance. The private fit learns its decisions from the supplied observations.

```sh
compache setup
compache setup --workflow support --demo
```

The default reads no customer files, looks up no credentials, and makes no model calls. Setup permissions apply to that invocation. There is no background collector, upload, live configuration change or paid timer. The demonstrations run after installation using built-in synthetic counts.

## 1. Estimate cache costs from metadata, if you choose

Export one JSONL row per completed user turn using counters you already collect. Include an opaque thread key, a route alias, model ID, 1-based run index, timezone-aware start and end timestamps, newly added input tokens and billed output tokens. Optional fields measure tokens exposed by retrieval, visible assistant output and foreground model calls. Supplying an export explicitly opts in to processing it locally. Keep customer names, messages, document contents, system prompts and tool definitions out of this file. Extra fields are rejected. Pseudonyms and activity patterns can still be sensitive.

```json
{"thread_id":"t-a291","route_id":"support","model_id":"example-model","run_index":1,"started_at":"2026-01-02T10:00:00Z","ended_at":"2026-01-02T10:00:04Z","added_tokens":2140,"retrieved_tokens":240,"output_tokens":65,"visible_output_tokens":42,"calls_per_run":2}
```

- `added_tokens` counts the new input and exposed retrieval for this run, including framing. Include the static header once at the start of the thread. Repeated cached history is excluded from added tokens. `retrieved_tokens` is a subset of that total. A provider’s entire request input count is usually much larger than the addition.
- `output_tokens` is billed output. `visible_output_tokens` measures the portion appended to the next prompt, in tokens including framing as applicable. If omitted, replay assumes it equals billed output. A missing call count assumes one. Treat optional measurements as unknown when unavailable.
- Use the initial model-request dispatch timestamp for `started_at` and user-visible completion for `ended_at`. Cache lifetimes and timer offsets start at dispatch. The gap from completion to the next dispatch can include host processing as well as time spent waiting for a user. Start the next turn after the preceding one ends. The cutoff is the actual end of observation, including idle time. The standard fit needs some complete 24-hour observation horizons to learn the return distribution. Near-cutoff runs are censored for this estimate.
- For a host integration, `compache.onboarding.make_metadata_row` accepts only the listed fields and returns a dictionary. It performs no I/O. Optionally use `pseudonymize_thread_id(raw_id, key=host_secret_bytes)` to join turns via a keyed digest. Keep the key with the host. Reuse the same aliases when exporting normalized thread content for the combined mode.

Use the actual economics for your model and route. The bundled model specs are examples. From an implementation checkout, see the versioned fields in the [illustrative model spec](../examples/rag_workload/roomy-model.json).

```sh
compache setup --workflow support --metadata counts.jsonl \
  --model-spec model.json --observation-end 2026-02-01T00:00:00Z \
  --policy-out candidate.json
```

By default this command prints a report and saves nothing. The optional output path must be a new file in an existing directory. It contains fitted numerical timing and token samples, model economics and training-thread digests. Treat this artifact as private. Add `--json` for aggregate results, censoring, data coverage and assumptions.

Setup reserves about one quarter of threads for evaluation, selected by a stable hash. At least one thread stays in training. A single thread can supply a fit, but cannot support a held-out comparison. For each evaluation turn, the planner sees only what was available at the decision. Both fitted policies face the same unseen arrivals, model prices and call counts. Modeled savings equal one minus the total workload-fitted cost divided by the reference cost. Both positive and negative results are reported. The references are the host preserving an append-only prefix without maintenance and a fixed, mixed-workload synthetic fit. Production host savings need a matched host ledger. Cross-thread header reuse, mid-thread host compaction, provider-specific eviction and generated recovery are outside this count-only comparison. If the context limit is exceeded, it withholds the savings figure.

## 2. Assess dependencies only if you authorize content access

Metadata cannot establish that a later answer needed a specific earlier clause, customer correction or assistant promise. The optional dependency reviewer sees real, model-visible text to establish which earlier input, retrieved passage or output was required. A deterministic local program handles timing and token counts. There is no metadata-reviewing model in this flow.

Choose a local, prefiltered and preferably redacted export of normalized threads. The command scans its lines locally, decodes up to three whole threads by default, and creates one private packet per selected run. Providing a preselected export bounds which records the local process can read at all. Sampling is repeatable for the same ordered file. Packets contain the current question and observed answer, the model-visible historical text and source identifiers. Unexposed retrievals, future turns, timestamps and usage receipts stay outside the packets.

```sh
compache setup --content-sample selected-threads.jsonl --packets-dir private-review
```

Inspect `private-review/packets` before choosing the reviewer. A person or a fully local model can label the packets locally. A hosted coding agent can transmit files it opens, even when it runs inside a local editor. Authorize the provider, data scope and selected packets separately before giving a hosted agent access. An available API key gives setup no authority to send data. The skill produces one evidence label per run, including partial or unassessable cases and short source quotes. Treat the labels as content. [Dependency-labeling skill](../skills/compache-label-dependencies/SKILL.md) · [Normalized trace contract](trace-contracts.md)

```sh
compache setup --content-sample selected-threads.jsonl --labels reviewed.jsonl \
  --model-spec model.json --retention-floor 0.95 --json
```

This path validates the labels against the selected source text and reports which candidate cuts preserved their required history. It fits no return-time distribution. Use the same export and `--max-threads` during preparation and assessment. The 95% floor covers assessed historical requirements. Setup scores two fixed candidate cuts, rejects a cut that changes any unresolved run, and withholds recipes that never exercised a cut. The [workload fit](workload-planner.md) searches a grid of cuts and bounds unresolved requirements instead. A few sampled threads can identify a missed dependency, but they supply weak evidence for admitting an irreversible cut. Your private packet directory and label files stay where you chose to save them until you delete them.

For a host that already decides when to compact, the JSON report includes the exact input, retrieval and output lookback of each candidate in `dependencies.recipe_results`. After held-out validation, the host can use an eligible `RetentionRecipe` at its own checkpoints and expose the reopen tool. This use needs no timing export or keep-warm scheduler. [Rendering a recipe](../src/compache/optimizer/retention.py)

## 3. Combine the two choices and test the candidate

```sh
compache setup --workflow research --metadata counts.jsonl \
  --observation-end 2026-02-01T00:00:00Z --model-spec model.json \
  --content-sample selected-threads.jsonl --labels reviewed.jsonl \
  --policy-out research-candidate.json
```

Use the same thread pseudonyms for both exports. The selected, labelled threads must also belong to the metadata cohort for the same model and route. Setup assigns them to training and excludes them from the cache evaluation. If none remain for evaluation, savings are unavailable. The policy contains the fitted cache distribution and any eligible recipes. The setup cost comparison continues to price caching alone. To price compaction or validate retention on unseen content, separately select and label independent validation threads, then run the existing local replay. This replay also reads a local header file containing the actual system prompt and ordered tool descriptions. Use only header data approved for the evaluation.

```sh
compache planner replay --threads validation-threads.jsonl --labels validation-labels.jsonl \
  --policy research-candidate.json --header header.json --observation-end 2026-03-01T00:00:00Z \
  --output validation-report.json
```

Host activation is explicit. For count-only timer advice load a candidate saved from the metadata-only invocation with `WorkloadPolicy` and call `compache.setup_metadata.plan_metadata_idle` at run end. Supply the verified reusable prefix from the receipt. Unknown capacity or an active cache-break latch should suppress pings. The host checks that the thread has not moved on, sends any approved ping, accounts for its real receipt, and cancels remaining timers on user return. Use `WorkloadSession` for a combined candidate with eligible recipes. It performs receipt-based replanning and supports compaction with source recovery. Disabling your export hook stops future collection. Reverting the host to its existing policy stops use of the fitted candidate. [Workload planner and matched host comparison](workload-planner.md) · [Reopen tool](integration.md#give-the-model-a-reopen-tool)

```python
import json
from pathlib import Path
from compache.optimizer import WorkloadPolicy
from compache.setup_metadata import plan_metadata_idle

policy = WorkloadPolicy(json.loads(Path("candidate.json").read_text()))

def on_run_finished(run_index, input_count, added_count, visible_output_count,
                    duration_seconds, verified_prefix_count, cache_break_detected):
    decision = plan_metadata_idle(
        policy, run_index=run_index, input_tokens=input_count, added_tokens=added_count,
        visible_output_tokens=visible_output_count, duration_seconds=duration_seconds,
        reusable_input_tokens=verified_prefix_count, ping_allowed=not cache_break_detected,
    )
    return decision.get("due_after_start_seconds", [])
```

These offsets are measured from request start. A first shadow integration can record the proposed times without sending any pings.
