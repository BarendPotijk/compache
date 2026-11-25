# Documentation

| Page | Covers |
|---|---|
| [Guide](guide.md) | Thread and label formats, recording without past traces, `compache fit`, `replay` and `profile`, applying a policy |
| [Agents](agents.md) | `compache wrap` for coding agents, what it does in a loop of tool calls, how it learns, what it saves |
| [Integrations](integrations.md) | Every way into the request path: SDK clients, LangChain, the proxy for agent harnesses. And every source of traces |
| [How it works](how-it-works.md) | What is read from the threads, how candidates are priced, what the runtime decides |
| [Calculator](calculator.md) | The browser cost model, what it compares, and opening it on your own workload |
| [Other tools compared](comparisons.md) | caching.ai, OptScale, Headroom, OpenClaw and provider-side compaction, and what was measured |
| [Benchmarks](../benchmarks/README.md) | DocChat, BEAM and LongDS: what the fit chose and what it saved. [Live runs](../benchmarks/live/README.md): what the provider billed through every way in. [Agents](../benchmarks/agents/README.md): Claude Code and Codex, live and modeled |

Figures live in `assets/` and are drawn by `scripts/figures.py` from the data files in `assets/data/`.
