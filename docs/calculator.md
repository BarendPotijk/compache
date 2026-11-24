# Calculator

The [calculator](https://barendpotijk.github.io/compache/) prices one chat workload under a fitted compache policy, your own rules and the defaults of other tools. It runs in the browser from `site/`, with no keys and no provider requests.

## What it needs, and the easiest way to give it

| It needs | Least effort first |
|---|---|
| What a turn adds, how long threads run, how long users are away, what later turns need | A starting point · drop a chat log, with labels if you have them · drop a `policy.json` · the link `compache profile` prints · draw the distributions · type a median and a 95th percentile |
| What you do today | Send everything, unless you say otherwise · pick a tool's defaults · edit the rules under *Your app today* · enter what you pay per 1,000 turns |
| Model and prices | A preset, or your own prices |

A file is read in the browser and sent nowhere. A chat log is measured as `compache profile` measures it, with the observed distributions instead of a fitted curve.

```sh
compache profile threads.jsonl --labels labels.jsonl --model claude-sonnet-5-5 --policy policy.json
```

prints a link with the workload and the policy filled in. `compache fit` also writes the measured workload into `policy.json`, so that file alone is enough.

## Share it

The address bar always holds the scenario on screen, and **Share** copies the result with its link. A link names only what it changes, so it can be written by hand:

```text
https://barendpotijk.github.io/compache/#v=7&model=haiku&start=documents&gapMedianMinutes=30&app.keepTurns=10&compare=app
```

## What it shows

| Part | Shows |
|---|---|
| Cost per 1,000 turns | Every side at once, with the share of needed text each one no longer has. Click one to compare with it |
| The policy | What stays of questions, answers and retrieved text, over where later turns need it. The one fitted here, the one in your `policy.json`, or your own: drag a window to price it. Every candidate, by cost and needed text in view |
| One conversation | What each turn sends, when the cache expired, what was trimmed, fetched back and pinged. Click a turn for the expected cost of both choices and of each number of pings |

## How close it is

The policy has the package's form and decides the way it does: before each request and after each answer it compares the expected cost three turns ahead. On the quickstart's 26-turn chat both cut on an expired cache and never on a warm one, and where both saw the same cache the requests agree to within 4% (`site/policy.test.mjs`).

| DocChat, saving against sending everything | 30 seconds | 4 minutes | 30 minutes |
|---|---|---|---|
| Calculator, on the training threads' profile, mean of six seeds | 32% | 84% | 87% |
| Package replay, held-out threads | 43% | 83% | 85% |
| Package, training threads | 32% | 84% | 85% |

Treat its figure as an estimate and `compache fit` with `compache replay` as the number for your threads.

## What it models

| Part | Rule |
|---|---|
| Workload | Seeded threads drawn from the inputs. Lognormal with the given median and 95th percentile, or drawn bin by bin |
| Prices | List prices and cache rules of seven models, checked on 2026-11-20. The same sheet as the package: `src/compache/models.json` |
| compache | Windows for questions, answers and retrieved text, and 0, 1 or 3 pings. The cheapest of up to 21 candidates on one set of threads that keeps the chosen share in view, priced on a second set |
| Comparisons | One vocabulary of rules: cache lifetime, turns kept, retrieved text dropped, shrunk or cleared, summaries, pings. [Where each tool's come from](comparisons.md) |
| Fetching back | A part compache archived costs one extra call for the turn that needs it. A comparison pays when its tool would fetch again, and otherwise the text is missing |
| The range | The 5th to 95th percentile of the saving when the sampled threads are drawn again |

## What it leaves out

- No model is called. Answer quality is not measured, and every side is assumed to know which earlier parts a turn needs.
- A policy that keeps retrieved text matching the question, or cited sources, is priced without that.
- compache uses the standard cache lifetime. The long one is priced for the comparisons only.

## Run it locally

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site   # then open http://127.0.0.1:8765/
node --test site/*.test.mjs
node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json && python scripts/figures.py openclaw
```

The full method is on the [methodology page](https://barendpotijk.github.io/compache/methodology.html).
