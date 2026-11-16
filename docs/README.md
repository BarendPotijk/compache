# compache

A harness-agnostic optimizer for long agent conversations. It keeps a conversation inside the model's context window without rewriting the provider's cached prefix by accident, and decides at every run end whether keeping that prefix warm is worth paying for.

**Status (0.4.0):** [RAG workload fitting, constrained search, cost replay, a deterministic retrieval gate and retention decisions](optimizer.md) are implemented, alongside the trace contracts and coding-agent labeling skill. The broader architecture below is a roadmap. Experimental subscription execution and an OpenCode policy baseline are available; production adapters, bill calibration, durable state and live idle-time ping scheduling remain unimplemented. See the implementation guide for the supported subset and assumptions.

## The one idea

A cached prefix is cheap to read and expensive to write. So the view of the conversation sent to the model must stay byte-stable while the cache is warm, and may only be rewritten when the cache is already gone. Everything else follows from that.

## Two mechanisms

| | Recipe gate | Retention planner |
|---|---|---|
| Runs | Before every model request | Once, when a run finishes |
| Decides | What view of the history the model sees | Ping to keep warm, lapse, or force the next start cold |
| Reads | The previous request's bill, the stored recipe, the cache window | Token counts, the gate's own cold plan, a return-time curve |

They cooperate: the planner prices a cold return at the gate's own cold plan, and the gate reads a ping as a restarted TTL.

## Principles

- **Canonical history is never changed.** The model sees a view: history with a recipe applied.
- **Planning is pure and deterministic.** The same history, policy and measure give the same bytes.
- **Nothing is summarized by a model.** Reductions are placeholders and deterministic index notes.
- **Everything removed is recoverable** by handle through one tool.
- **Decisions use the bill**, never a count.
- **Value comes in order:** prefix stability by a wide margin, then compaction thresholds, then the planner.

## Docs

| Doc | Covers |
|---|---|
| [Implemented RAG optimizer](optimizer.md) | Runnable fit/solve/replay, promotion checks, pure gate/planner, economics and coverage limits |
| [Live comparison](live-comparison.md) | Subscription runner, measured usage, cache scenarios and limitations |
| [OpenCode baseline](opencode-baseline.md) | Pinned default compaction behavior and adapter boundaries |
| [Public benchmarks](public-benchmarks.md) | RAG, memory and evidence-dependency evaluation options |
| [Offline trace contracts](trace-contracts.md) | Implemented JSONL APIs, scoped Langfuse conversion, evidence packets, label validation |
| [Foundations](foundations.md) | Provider assumptions, request layout, prefix stability |
| [Compaction](compaction.md) | The gate: window, sizing, tiers, last resort, admission, projection, recovery |
| [Retention](retention.md) | The planner: decision rule, test vectors, return curve, pings |
| [Integration](integration.md) | Harness contract, providers, build order, evaluation |
| [Design](design.md) | Architecture, fitting the policy from traces, integrations, build plan |
| [Reference](reference.md) | Defaults, invariants, design decisions, open issues |

## Terms

| Term | Meaning |
|---|---|
| **run** | One user message and every model call made to answer it |
| **turn** | A user prompt through the next completed answer (text and no tool call). The last turn is *running* until it has one. A clarifying-question resume stays inside its turn |
| **canonical history** | The stored messages |
| **view** | What the model is sent: canonical history with a recipe applied |
| **recipe** | The reductions decided over the first covered messages. Later messages are appended unchanged |
| **cold / warm** | Whether the provider still holds the previous prefix. Cold: a replan is free. Warm: the view may only extend |
| **the bill** | The previous request's billed input plus its response's output |
| **request-scoped context** | Rebuilt every request and never saved: plan state, memory, an open document, notices. Rides after the history's cache point |
| **W** | The model's context window. Thresholds are ratios of it |
| **shared part** | Eager tools plus static instructions: what every conversation of the model has cached |
| **handle** | What a placeholder names to fetch the original: `tool:<id>`, `document:<uri>`, `source:<key>`, `answer:<n>`, `turn:<n>` |
| **ping** | A replay of the run's last request with the answer appended, a cache point at its end and `max_tokens: 0`. It reads the prefix, caches the answer, and restarts the TTL |
