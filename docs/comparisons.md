# Comparisons

Compare named configurations on the same workload. Compression ratio, modeled cost and answer accuracy are different measurements.

| Baseline | Available scope | Not established |
|---|---|---|
| OpenClaw | Browser model of pinned client-summary behavior | Actual answers or billing |
| OpenCode | Pinned policy and recorded subscription experiments | General performance across workloads |
| Headroom 0.44.0 | Native verified-lossless library compression, with frozen-prefix and cold-refresh variants | Default proxy/CCR/Kompress, native cold hook, net-cost gate or answer quality |

OpenClaw and OpenCode are distinct baselines: [calculator methodology](https://barendpotijk.github.io/compache/methodology.html), [OpenCode protocol](https://github.com/BarendPotijk/compache/blob/codex/rag-optimizer/docs/opencode-baseline.md).

## OpenClaw

`node scripts/openclaw_comparison.mjs` prints the README table: modeled cost per 1,000 turns for seven model presets at three thread lengths, on the calculator's default workload. `--requirement` sets the least share of what later turns need that Compache must keep in view (default 0.8). `--svg` redraws the chart.

| Models | 8-turn threads | 30-turn threads | 80-turn threads |
|---|---|---|---|
| Claude (four presets) | 44% to 60% cheaper | 47% to 62% cheaper | 30% to 55% cheaper |
| OpenAI (three presets) | 28% cheaper | 61% cheaper | 62% cheaper |

- Neither tool is executed. Compache's policy is the cheapest of 90 candidates on one set of threads and is priced on a second set.
- OpenClaw follows its documented defaults at a pinned revision and sends no keep-warm requests. With a Claude API key it clears old tool results on the server above 30% of the window, keeping the newest three. With OpenAI it is priced on its client-side summary at 75% of the window. Its default there calls OpenAI's own compaction at the same point, which is not modeled.
- A needed source that was archived or cleared is fetched back with one extra call, on both sides. With Claude both then have every needed source in view. With OpenAI, OpenClaw's summaries stand in for 11% of what 30-turn threads need and 27% of what 80-turn threads need.
- Still in view before fetching back: Compache 81% to 97%, OpenClaw 73% to 100%.

## Headroom overlap

Headroom has a TTL learner, optional cold-prefix recompaction, a cache-aware mutation cost gate and observation-based compression recommendations. Its gate defaults to ten expected future reads and can derive cache-survival weight from idle time. At the reviewed commit its old request-time recommendation API returns `None`; the Rust loader says its dispatcher does not consume published recommendations yet. Do not present every documented learning feature as enabled runtime behavior.

Compache's intended distinction is fitting whole-thread decisions to dependencies and return timing under an explicit dependency-retention requirement. This is a scope distinction, not a superiority claim.

Reviewed source: [`18067b1`](https://github.com/headroomlabs-ai/headroom/tree/18067b1d19f6606bae3b4d6b50efeeaa48dc477e), 2026-11-17. Executed release: `headroom-ai==0.44.0`. Reports fingerprint the installed module separately; a release wheel is not assumed identical to that Git commit.

## Headroom replay

An earlier offline replay ran Headroom's lossless `compress(mode="agent")` and a fitted Compache policy over the same recorded threads, with one token ruler, cache ledger and price list. On that small prose fixture neither changed anything: 25,864 base-input-token equivalents in every arm over 11 complete-text held-out threads. It checked the replay path, not behaviour under long-context pressure. The harness left the package with the first-generation optimiser. [Frozen reports, policy, profile and environment](https://github.com/BarendPotijk/compache/tree/codex/rag-optimizer/examples/headroom/results).

## Full proxy comparison: not run

Evaluate default cache/CCR mode, optional cold recompaction and optional net-cost policy separately on the same long RAG payloads. Record recovery, retrieval, summary and ping calls; cache usage; rejected requests; source availability; and independent answer scores. Tune only outside the final evaluation split. The calculator does not assign Headroom an invented percentage reduction.

Sources: [cold hook](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/transforms/cold_prefix.py), [TTL learner](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/cache/ttl_estimator.py), [cost gate](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/transforms/content_router.py#L6244-L6349), [recommendation status](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/crates/headroom-core/src/transforms/recommendations.rs#L1-L22).
