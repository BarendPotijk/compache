# Other tools compared

Every tool below ships fixed rules. compache fits the rule to recorded traffic and decides compaction and cache keep-warm together. That is the difference, and the table says what was and was not measured.

| Tool | What it does by default | Fitted to traffic | Keeps the cache warm | Compared how |
|---|---|---|---|---|
| OpenClaw | Clears old tool results or summarizes at fixed shares of the context window | No | No | Modeled in the [calculator](calculator.md) |
| Headroom | Compresses the content of new tool output, keeps every message, one of four fixed profiles | No | No | Not run head to head |
| Provider compaction | Compacts on the server at a context-size threshold | No | No | Not modeled |

## OpenClaw

Defaults as documented at revision [`c49b553`](https://github.com/openclaw/openclaw/commit/c49b553b85b82bc700465d4607949574f892366c):

| With | Default |
|---|---|
| A Claude API key | Server-side tool-result clearing once the request reaches the larger of 50,000 tokens and 30% of the window. The three newest results stay, and at least the larger of 12,500 tokens and 5% of the window must come free. Server-side compaction follows at 70% and a client summary at 75% |
| An OpenAI API key | OpenAI's own compaction endpoint at 75% of the window |
| Both | A five-minute cache and no keep-warm requests |

![Cost per 1,000 turns on Claude Sonnet 5.5 at three thread lengths: OpenClaw's defaults against compache](assets/openclaw-cost.svg)

`node scripts/openclaw_comparison.mjs` prices both sides with the calculator's model for seven model presets and three thread lengths. Neither tool is executed. For OpenAI the calculator prices OpenClaw's client-side summary, because the compaction endpoint's output cannot be priced from documentation.

| Models | 8-turn threads | 30-turn threads | 80-turn threads |
|---|---|---|---|
| Claude Fable 5.2, Opus 5.5, Sonnet 5.5 | 58% cheaper | 61% to 63% cheaper | 60% to 63% cheaper |
| Claude Haiku 5.5 | 90% cheaper | 91% cheaper | 88% cheaper |
| GPT-6 Astra, GPT-5.6 Terra, GPT-6 Luna | 59% cheaper | 80% cheaper | 82% cheaper |

- The workload is the calculator's default: 25k retrieved tokens per turn, users back after a median of four minutes, and what later turns need as measured on DocChat. compache's policy is fitted on one set of threads and priced on another.
- compache leaves 81% to 93% of what turns need in the request and fetches the rest back, on 3% to 9% of turns. OpenClaw leaves 92% to 99% in the request. With OpenAI its summaries stand in for up to 8% of what 80-turn threads need.
- Haiku charges five times as much for a request above 100k tokens. OpenClaw's thresholds let requests grow past that, and the fitted policy stays under it.

## Headroom

Read first-hand on 2026-11-21 from the README, documentation and source of [headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom), release 0.44.0.

- **Adoption.** `compress(messages, model=...)` in process, a proxy behind a base-URL change, or `headroom wrap <agent>`. It needs no recorded traffic.
- **Rules.** Four profiles (`coding`, `balanced`, `agent-90`, `general`) with fixed thresholds. It records statistics about tool output, and its documentation says those do not change request-time decisions.
- **History.** "Headroom never drops messages from the conversation history." It shrinks what a tool returned: 21% to 57% fewer input tokens on its four agent scenarios, over 90% on repeated JSON and log lines, little on prose.
- **Cache.** The default mode compresses only the newest part of each turn and forwards earlier turns byte for byte, so it does not break the cached prefix. It sends no keep-warm request. Recompacting a cold prefix and estimating the cache lifetime from traffic exist as options that are off by default.
- **Recovery.** Compressed content can be fetched back through a retrieve tool, in the proxy.

The two are not alternatives. Headroom makes a single tool result smaller. compache decides which whole parts stay in the request and when the cache is kept warm, from how your users return and what your turns need. No head-to-head run exists, so no number is claimed here.

## Provider-side compaction

Anthropic's tool-result clearing and compaction, and OpenAI's compaction endpoint, act at a context-size threshold. They do not know how your users return or which earlier parts later turns need, and they send no keep-warm request. The calculator models Anthropic's clearing as OpenClaw configures it.
