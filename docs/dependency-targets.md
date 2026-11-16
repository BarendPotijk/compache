# Dependency-based compaction targets

The revised optimizer searches targets of **10%, 15%, 20% and 25%**, with warm-pressure triggers of **50% or 60%**. It chooses the lowest modeled training cost satisfying the requested historical-dependency coverage floor, default **95%**. A 90% floor is evaluated as a separate sensitivity. Cold returns are the normal compaction opportunity.

The fitted candidate selected **20% target, 22% cold trigger, 50% warm trigger, zero-turn blanket retrieval grace**. Query/evidence guards still apply. A standalone policy defaults to a 60% warm trigger until fitted.

| Diagnostic | Previous policy | Revised policy |
|---|---:|---:|
| Training historical requirements retained | 30/30 | 30/30 |
| Previously used test thread retained | 5/5 | 5/5 |
| Cold rewrites on that thread | 2 | 2 |
| Warm rewrites on that thread | 15 | 3 |
| Total rewrites | 17 | 5 |
| Modeled cost, input-token equivalents | 14,770,217.0 | 4,098,621.2 |

The 72.3% modeled reduction is relative to the previous Compache policy in the same offline replay. It is **not a new live savings result or a comparison with OpenCode**. No model calls were made. Inputs, outputs, return timings, labels and price ratios are frozen; costs use byte-based token estimates. The previously used test conversation is a diagnostic, not a fresh unseen holdout.

Coverage counts historical requirements with one complete sufficient source alternative retained, before recovery: AND within alternatives, OR between them. Direct lookups do not dilute the denominator. Unknown-context changes remain disallowed. Both the 90% and 95% settings selected this policy; with only five historical requirements in the diagnostic, missing one drops coverage to 80%, so both floors need 5/5. This small synthetic sample does not establish a population retention rate or answer quality.

The target is not forced by discarding protected evidence. Four of five diagnostic rewrites reached 20%; one cold rewrite retained 37.2% because the target was unreachable. Across the six training threads, 22/29 rewrites reached the target and post-rewrite sizes ranged from 19.53% to 37.20%. Warm partial rewrites are deferred when they cannot reach the target, unless the unchanged view would violate a hard admission limit. Cold compaction can take partial savings immediately.

All fractions use the configured context window and the current estimated token ruler. A [subsequent live rerun](live-comparison-0.4.0.md) records provider token fractions and costs separately: three updated Compache repeats saved 22.3% against the historical OpenCode baseline at 5 minutes and 11.8% at 30 minutes under the same counterfactual cache model. Package version 0.4.0 updates default/search settings and constraints v2; the earlier measured 0.3.0 experiment is preserved unchanged.

[Selected policy](../examples/rag_long/cold-compaction-results/policy.json), [full candidate grid and selections](../examples/rag_long/cold-compaction-results/new.json), [previous-policy replay](../examples/rag_long/cold-compaction-results/old.json), [input manifest](../examples/rag_long/cold-compaction-results/manifest.json).
