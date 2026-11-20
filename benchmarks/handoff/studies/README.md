# Previous benchmark runs

| Study | Fitting data | Evaluation | How to use the result |
|---|---|---|---|
| [Luna long conversations](luna/comparison.json) | The `generic` arm used 200 synthetic RAG conversations. The `tuned` arm used eight other threads from each respective public suite. | Two long GoodAI conversations and two MemoryArena searches. | The portable [verifier](../README.md) reproduces the cost and retention totals for the externally fitted arm and the two controls. Both suites had informed development. |
| [Terra and Sol, short](terra-sol/short/report.md) | Fits from the same benchmark families. | Two short MemoryArena threads, 14 user turns. | Full conversation cost, including maintenance on the virtual clock. Every policy and model answered one of the two terminal questions correctly. |
| [Terra and Sol, long checkpoints](terra-sol/checkpoints/report.md) | Luna's previously fitted contexts. | 24 first answers on frozen layouts. | First-answer cost and quality. Earlier summary and ping calls lie outside the measurement. |

Four of the six short GoodAI probes reused an entire training script and its target. We excluded short GoodAI from the matched stronger-model report. The [split audit](split-audit.json) records question occurrences, script and target reuse, and document overlap. Its source counts describe the original archives. The portable data allow independent cost aggregation and inspection of recorded grades. Rejudging model outputs against licensed source texts requires the source material.

The stronger-model experiment settled at $42.91110 across the short and long calls. Sol spending has ended. The cost of these studies is separate from the historical Luna charges. Reviewing this archive requires no provider access.

## Regenerate the Luna figures

The [cost and retention plotting script](luna/plot_figures.py) reads the committed [aggregate metrics](luna/comparison.json). Run `python3 benchmarks/handoff/studies/luna/plot_figures.py` with Matplotlib installed to create the desktop and mobile figures in `luna/figures/`. The standard-library verifier prints the same cost and retention cells without any plotting dependency.
