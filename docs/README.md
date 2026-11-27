# Compache documentation

Compache chooses what conversation history to send to a model and when to keep its prompt cache warm. To integrate a chatbot, fit a policy on representative conversations, evaluate it on other threads, then apply it where your application sends model requests.

[Run the quickstart](guide.md) · [Choose an integration](integrations.md) · [Explore the calculator](https://barendpotijk.github.io/compache/)

## Start with your application

| I want to… | Start here | What I can verify |
|---|---|---|
| Try Compache locally | [Runnable examples](../examples/README.md) | A sample policy fits, a held-out replay reports cost and retention, and a chatbot can apply the policy |
| Add it to an existing model client or framework | [Integration guide](integrations.md) | My application's model requests pass through Compache |
| Keep control of model calls and scheduled jobs | [Application-managed sessions](sessions.md) | Each turn, tool round, and ping has an owner and a usage receipt |
| Run it with Claude Code, Codex, or another agent | [Coding agents](agents.md) | The agent goes through the local proxy and its session appears in the record |

## Fit to your own traffic

1. [Record or import traces](traces.md). Choose metadata-only recording, provider trace exports, or labeled conversations. Check which inputs contain text before sharing them.
2. [Fit and evaluate](fitting.md). Keep entire threads together when splitting the data. Read the held-out cost and the share of observed dependencies in view.
3. [Apply the saved policy](guide.md#5-apply). Check that calls reach the integration and compare the resulting receipts with your application's billing or usage logs.

## Reference and evidence

- [CLI and Python reference](reference.md) covers the public entry points, files and report units.
- [Troubleshooting](troubleshooting.md) covers an empty record, a policy that preserves history, unexpected cache reads and extra upstream calls.
- [How the planner works](how-it-works.md) explains which candidates the fit compares and which decisions happen at request time.
- [Calculator guide](calculator.md) explains editable assumptions and how to open a workload profile.
- [Benchmarks](../benchmarks/README.md) and [comparisons](comparisons.md) give the datasets, baselines, answer scores and cost measurements.

**Version.** These pages describe the source in this checkout. Run `python -m pip install -e .` from the repository root to use its examples and flags. The 0.15.1 release predates `--min-retention` and the separate training and held-out retention fields. [Documentation shipped with 0.15.1](https://github.com/BarendPotijk/compache/tree/v0.15.1/docs)

Figures are generated from the [checked-in measurements](assets/data/) by [the figure script](../scripts/figures.py). The logo uses [Onest](https://github.com/googlefonts/onest) under the SIL Open Font License.
