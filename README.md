# compache

Fit context-compaction and prompt-cache policies to your RAG workload.

**[Try the calculator ↗](https://barendpotijk.github.io/compache/)** · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Comparisons](docs/comparisons.md) · [Implementation](https://github.com/BarendPotijk/compache/pull/2)

Compache uses historical threads, dependency labels and return timing to choose what context to retain and whether keeping the cache warm is worth paying for. The retention requirement defaults to 95% of assessed historical dependencies. It is not an answer-accuracy guarantee.

**Experimental.** The calculator is live. The Python implementation is in [PR #2](https://github.com/BarendPotijk/compache/pull/2), based on [PR #1](https://github.com/BarendPotijk/compache/pull/1). Production provider adapters and a live ping scheduler are not shipped.

## Quickstart

Python 3.11+. Use the implementation branch while the PRs are open:

```sh
git clone --branch codex/rag-optimizer https://github.com/BarendPotijk/compache.git
cd compache
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
compache traces validate examples/rag_workload/threads.jsonl
python scripts/run_rag_optimizer.py --output ./work/evaluation
```

Validation reports **45 threads / 207 runs**. Evaluation fits on training threads and writes a policy, workload profile and held-out cost ledger. No API key or model call is needed. The fixture combines fabricated questions, sources and return timing with captured agent answers; its token prices are illustrative.

For your own traces: [normalize → label dependencies → fit and evaluate → integrate](https://github.com/BarendPotijk/compache/blob/codex/rag-optimizer/docs/optimizer.md). A [coding-agent skill](https://github.com/BarendPotijk/compache/blob/codex/rag-optimizer/skills/compache-label-dependencies/SKILL.md) prepares and labels dependency evidence.

## How it works

![Threads and labels feed offline fitting; a frozen policy drives the runtime gate and planner](docs/assets/workflow.svg)

- **Fit offline.** Estimate return timing and dependency lookback, select a policy under the retention requirement, then evaluate separate threads.
- **Prepare each request.** Preserve warm prefixes, compact eligible historical retrievals and keep recovery handles. Inputs, answers and current retrievals remain protected by the supported RAG contract.
- **Plan at run end.** Compare expected keep-warm benefit with ping and cold-return cost. The application owns execution and cancellation.

Policies are versioned artifacts. Runtime decisions do not consult evaluation labels or future answers. [Application structure](docs/application-structure.md) · [Runtime contract](https://github.com/BarendPotijk/compache/blob/codex/rag-optimizer/docs/optimizer.md#gate-and-source-recovery)

## Compare approaches

| Baseline | Available comparison | Evidence |
|---|---|---|
| OpenClaw | Interactive client-summary scenario | Deterministic cost estimate |
| OpenCode | Pinned compaction policy and subscription experiments | Captured usage, repriced cache assumptions |
| Headroom | Native lossless-library replay; append-only and cold-refresh variants | Actual compression, modeled cost; full proxy not evaluated |

[Scopes, results and reproduction](docs/comparisons.md). Headroom also has cache-aware cost gating and trace-based learning components; those capabilities are not unique to Compache.

The initial Headroom complete-text diagnostic covers **11 threads / 49 runs** and finds **0% modeled cost difference**. It exercises short prose traces in a roomy window with a conservative Compache policy. One incomplete held-out thread is reported separately. This does not establish long-thread performance or answer quality.

## Explore your workload

[Open the calculator](https://barendpotijk.github.io/compache/) to set token growth, dependencies, thread survival and return timing. It shows cost per run, context growth, cache hit rate and keep-warm payoff against OpenClaw. Headroom comparisons use actual payloads through the separate replay command.

Costs are scenario estimates, not provider billing savings. The calculator and initial Headroom replay assume zero extra recovery-call overhead. Source availability and answer quality are separate; incomplete comparisons do not produce a savings percentage.

[Docs](docs/README.md) · [Methodology](https://barendpotijk.github.io/compache/methodology.html) · [Benchmark source plan](https://barendpotijk.github.io/compache/benchmarks.html) · [Issues](https://github.com/BarendPotijk/compache/issues) · [Earlier calculator README](docs/design/calculator-readme.md) · [Original design README](docs/design/README.md)
