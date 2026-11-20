# Comparisons

Compare named configurations on the same workload. Compression ratio, modeled cost and answer accuracy are different measurements.

| Baseline | Available scope | Not established |
|---|---|---|
| OpenClaw | Browser model of pinned client-summary behavior | Actual answers or billing |
| OpenCode | Pinned policy and recorded subscription experiments | General performance across workloads |
| Headroom 0.44.0 | Native verified-lossless library compression, with frozen-prefix and cold-refresh variants | Default proxy/CCR/Kompress, native cold hook, net-cost gate or answer quality |

OpenClaw and OpenCode are distinct baselines: [calculator methodology](https://barendpotijk.github.io/compache/methodology.html), [OpenCode protocol](https://github.com/BarendPotijk/compache/blob/codex/rag-optimizer/docs/opencode-baseline.md).

## Headroom overlap

Headroom has a TTL learner, optional cold-prefix recompaction, a cache-aware mutation cost gate and observation-based compression recommendations. Its gate defaults to ten expected future reads and can derive cache-survival weight from idle time. At the reviewed commit its old request-time recommendation API returns `None`; the Rust loader says its dispatcher does not consume published recommendations yet. Do not present every documented learning feature as enabled runtime behavior.

Compache's intended distinction is fitting whole-thread decisions to dependencies and return timing under an explicit dependency-retention requirement. This is a scope distinction, not a superiority claim.

Reviewed source: [`18067b1`](https://github.com/headroomlabs-ai/headroom/tree/18067b1d19f6606bae3b4d6b50efeeaa48dc477e), 2026-11-17. Executed release: `headroom-ai==0.44.0`. Reports fingerprint the installed module separately; a release wheel is not assumed identical to that Git commit.

## Reproduce

Use the implementation branch from the README and a separate environment:

```sh
python3 -m venv .venv-headroom
. .venv-headroom/bin/activate
python -m pip install . headroom-ai==0.44.0
export HEADROOM_OFFLINE=1 HEADROOM_STATELESS=1 HEADROOM_TELEMETRY=off HEADROOM_TOIN_BACKEND=none
python scripts/compare_headroom.py --output ./work/headroom-comparison
```

The script checks fixture hashes, fits on training threads, freezes the recommendation and evaluates held-out threads. It reports both the entire held-out cohort and a complete-text diagnostic that excludes whole threads solely for missing content.

For an existing frozen policy:

```sh
compache compare --threads threads.jsonl --labels labels.jsonl \
  --policy policy.json --profile profile.json \
  --observation-end 2026-11-17T00:00:00+00:00 --output comparison.json
```

Replace the example cutoff with your collection cutoff. Labels must match the exact traces. Offline mode is required. No provider API or answer-generation call is made.

## Arms and accounting

- **headroom-lossless-cache:** native `compress(mode="agent")`, with prior projected messages frozen and reused.
- **headroom-lossless-cold:** the same mode, with a caller-managed refresh after the scenario TTL plus 60 seconds. This is an adapter variant, not Headroom's native cold-prefix hook.
- **Compache:** the frozen recommended policy, including modeled keep-warm decisions. Extra recovery-call overhead is zero; recalled input remains charged.
- All arms use the same recorded answers, times, byte-based token ruler, cache ledger, reserve and prices. Headroom's native tokenizer counters are separate diagnostics; its `gpt-4o` tokenizer selection does not invoke that model.
- Each canonical part becomes one user/assistant/tool message for compression. This is not a complete provider tool-call transcript. Runtime scaffolding is zero unless explicitly supplied.
- Source availability in the Headroom arm relies on its lossless verification. It is not independent semantic or answer-quality validation. Missing text or failed admission suppress aggregate savings.
- Local compression time is measured, including lazy initialization. Sequential arms share process caches, so these timings are diagnostics rather than a comparative latency benchmark. CPU time is not priced as provider tokens.

## Initial development result

The small prose fixture and roomy illustrative context produce no compression in either Headroom arm. Compache's validated recommendation also makes no cuts or pings.

| Cohort | Compache | Headroom cache | Headroom cold refresh | Interpretation |
|---|---:|---:|---:|---|
| 12 held-out threads / 53 runs | Incomplete text | 52 completed | 52 completed | Savings unavailable |
| 11 complete-text threads / 49 runs | 25,864 | 25,864 | 25,864 | 0% modeled difference |

Costs are base-input-token equivalents, not dollars or bills. `rag-028` is the excluded incomplete thread. The corpus already informed development. A separate 200-record JSON control reduced 8,622 native tokens to 2,260 with Headroom-verified losslessness; it is excluded from every workload score. This checks the replay path, not performance under long-context pressure. [Frozen reports, policy, profile and environment](https://github.com/BarendPotijk/compache/tree/codex/rag-optimizer/examples/headroom/results).

## Full proxy comparison: not run

Evaluate default cache/CCR mode, optional cold recompaction and optional net-cost policy separately on the same long RAG payloads. Record recovery, retrieval, summary and ping calls; cache usage; rejected requests; source availability; and independent answer scores. Tune only outside the final evaluation split. The calculator does not assign Headroom an invented percentage reduction.

Sources: [cold hook](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/transforms/cold_prefix.py), [TTL learner](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/cache/ttl_estimator.py), [cost gate](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/transforms/content_router.py#L6244-L6349), [recommendation status](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/crates/headroom-core/src/transforms/recommendations.rs#L1-L22).
