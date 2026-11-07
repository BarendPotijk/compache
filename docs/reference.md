# Reference

## Defaults

Starting points, not measurements of your product.

| Knob | Default | Why |
|---|---|---|
| Cold tier-2 trigger | 0.17 W | Near the cost-retention knee |
| Tier-2 target | 0.15 W | Room for several turns whole on a cold start |
| Warm trigger | 0.65 W | High, because a warm replan rewrites everything |
| Running-turn allowance | 0.25 W | A long turn needs room to work |
| Last-resort trigger / target | 0.60 / 0.30 W | Settle far under the warm trigger |
| Next-turn reserve | 0.05 W | Admission allows 0.95 W |
| Tier-1 grace turns | 1 | A follow-up builds on the turn just answered |
| Note ask / cited documents | 240 chars / 8 | Note size |
| Cache TTL | 300 s | 1 hour charges a higher write on all in-run growth |
| Expiry and freshness margins | 20 s | Equal on purpose; must exceed ping lateness |
| Ping lead | 45 s | Shorter saves more, until lateness eats the margin |
| Max pings | 12 | Value flattens; cost grows linearly |
| Byte bound | 16 MB | Transport limit |
| Calibration pairs | 8 | Older pairs describe another prefix |
| Ruler ridge | 0.05 | One lopsided pair must not swing the fit |
| Priors, tokenizer unit | Prose 1.5, structured 2.0, bounds 0.5–4.0 | Claude against cl100k |
| Priors, byte unit | Prose 0.40, structured 0.46, bounds 0.15–1.0 | Claude against UTF-8 bytes |
| Expansion window | 48,000 chars | Bounded recall |
| Captured requests, active plans | 64, 64 | Memory bound per worker |
| Return curve, `last_call_seconds`, `later_calls` | Fit them | They describe your users |

Keep these relationships: target < cold trigger; last-resort target < warm trigger; keep-warm ceiling = warm trigger − reserve; freshness margin < ping lead.

## Invariants

**Compaction**

1. Identical history, policy and measure give an identical recipe and projection.
2. Every tool call in the view has its result, and vice versa.
3. Tiers 1 and 2 never touch the running turn. The latest response's thinking always stays, and only admission reaches the latest request's unread results.
4. Prompts and answers stay verbatim, apart from citation markers of removed sources.
5. A citation key travels only with its text.
6. Tier 2's reductions include tier 1's.
7. A document a kept turn cites stays whole.
8. A placeholder or note the model has seen survives a warm replacement.
9. Earlier cuts never come back at a cold start.
10. Framework-owned returns and user-authored results are never reduced.
11. After a last resort settles, the next tool result extends the prefix.
12. Thinking leaves only from the front of the run.
13. The store never fails a request.

**Planner**

1. A prefix above the keep-warm ceiling is never kept warm.
2. The static prefix earns no ping.
3. The first ping pays the write of the answer; later pings read it.
4. Cheaper reads never make keeping warm less attractive.
5. A TTL no longer than the ping lead is never kept warm.
6. Nothing of the conversation is persisted for a ping.
7. The first billed output token stops the plan.
8. Pings stop once another run starts, and a newer request retires the plan.
9. A run ending in a tool call or thinking alone plans nothing.
10. Planning and pinging never block or fail a run.

## Design decisions

Don't reopen these without new evidence.

| Rejected | Why |
|---|---|
| Model-written summaries as memory | They rewrite the prefix, can't be audited, and lose the prompts and answers later turns need most |
| Replanning every request to the smallest view | Every request pays to rewrite the prefix |
| Unloading capabilities while warm | Saved declarations cost more than repeated prefix writes |
| A fixed ping count, or simulating the gate inside the planner | Hides policy in a second mechanism. Price the gate's own cold plan |
| Persisting conversation content for pings | Reuse the last request in memory, and let a restart drop the plan |
| Recipe-bound recall | Resolve handles from history, independent of the recipe |
| Counting views before deciding | Costs latency and money, and the bill is exact |

## Open issues

**Design**

- **Block position:** choose one request-scoped position that holds from the first request on.
- **Preserved thinking:** replacing a result an earlier response read may be rejected by an enforcing provider. Mitigation needs designing.
- **Unbounded tool results:** a turn reading large documents on a small window can loop, with the model expanding what the shed replaced. Results need bounds at the source and the shed needs a guard.
- **Fork carry:** what a cold plan builds on when the stored digest no longer matches. The safe reading is nothing.
- **Unknown window:** whether a model missing from the table falls back to a default W or fails.

**Project**

- Language and first harness.
- Which providers first.
- Licence.
- Whether citations are core or optional, and whether protected evidence stays.
- Whether to ship a neutral return curve or require fitting.
- Which threads to build an evaluation dataset from.
