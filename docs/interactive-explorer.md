# Cost calculator

The [calculator](https://barendpotijk.github.io/compache/) prices one chat workload twice: with a fitted Compache policy and with OpenClaw's default rules. It runs in the browser from `site/`, with no dependencies, keys or provider requests. The full method is on its [methodology page](https://barendpotijk.github.io/compache/methodology.html) (`site/methodology.html`).

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site   # then open http://127.0.0.1:8765/
node --test site/*.test.mjs
node scripts/openclaw_comparison.mjs                             # the README table
```

## What it models

| Part | Rule |
|---|---|
| Workload | Seeded threads from the page's settings: retrieved tokens per turn, what later turns need, thread length and return delay |
| Prices | List prices and cache rules of seven presets, checked against the provider pages on 2026-11-20 |
| Compache | Archives old retrieved text when the cache is cold, or at a higher mark while it is warm, and sends keep-warm pings when a return is likely. The cheapest of 90 candidate policies that keeps the chosen share in view is fitted on one set of threads |
| OpenClaw | Its documented defaults at revision [`c49b553`](https://github.com/openclaw/openclaw/commit/c49b553b85b82bc700465d4607949574f892366c), with no keep-warm requests. Claude presets: tool results cleared on the server above 30% of the window, keeping the newest three. OpenAI presets: a client-side summary at 75% of the window |
| Fetching back | A needed source that was archived or cleared costs one extra call, on both sides. A source OpenClaw summarized is not fetched again |
| Reported cost | Both sides on a second set of threads, drawn with the next seed |

## What it does not show

- No model is called. Answer quality is not measured, and both sides are assumed to know which earlier sources a turn needs.
- For OpenAI models OpenClaw's own default calls OpenAI's compaction at the same 75% mark. That is not modeled.
- The Compache side is a simplified projection, not the package's fitted policy. [DocChat](../benchmarks/docchat/README.md) runs the package itself.
