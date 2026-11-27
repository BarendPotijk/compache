# Calculator

The [calculator](https://barendpotijk.github.io/compache/) estimates cost per 1,000 user turns under a compache policy, your own rules and selected configurations of other tools. It runs in the browser from `site/`, with no keys and no provider requests.

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
| Cost per 1,000 turns | Estimated spend by configuration, with the share of sampled dependencies still missing after any modeled recovery. Click one to compare with it |
| The policy | What stays of questions, answers and retrieved text, over where later turns need it. The one fitted here, the one in your `policy.json`, or your own: drag a window to price it. Every candidate, by cost and needed text in view |
| One conversation | What each turn sends, when the cache expired, what was trimmed, fetched back and pinged. Click a turn for the expected cost of both choices and of each number of pings |

## How close it is

The policy has the package's form and decides the way it does: before each request and after each answer it compares the expected cost three turns ahead. On the quickstart's 26-turn chat both cut on an expired cache and never on a warm one, and where both saw the same cache the requests agree to within 4% (`site/policy.test.mjs`).

| DocChat, saving against sending everything | 30 seconds | 4 minutes | 30 minutes |
|---|---|---|---|
| Calculator, on the training threads' profile, mean of six seeds | 32% | 84% | 90% |
| Package replay, held-out threads | 43% | 83% | 89% |
| Package, training threads | 32% | 84% | 90% |

The browser draws synthetic threads from a workload profile. Package replay uses the held-out threads you supply and still estimates cost. The table compares both estimates against sending the full history with the default cache. A provider bill requires live traffic.

## What it models

| Part | Rule |
|---|---|
| Workload | Seeded threads drawn from the inputs. Lognormal with the given median and 95th percentile, or drawn bin by bin |
| Prices | List prices and cache rules of seven models, checked on 2026-11-20. The same sheet as the package: `src/compache/models.json` |
| compache | Windows for questions, answers and retrieved text, 0, 1 or 3 pings, and on Claude the five-minute or the one-hour cache. The cheapest of up to 42 candidates on one set of threads that keeps the chosen share in view, priced on a second set |
| Comparisons | One vocabulary of rules: cache lifetime, turns kept, retrieved text dropped, shrunk or cleared, summaries, pings. [Where each tool's come from](comparisons.md) |
| Fetching back | A turn needing one or more archived parts incurs one extra recovery call. A comparison pays when its tool would fetch again, and otherwise the text is missing |
| The range | The 5th to 95th percentile of the saving when the sampled threads are drawn again |

## What it leaves out

- Browser retention estimates the share of earlier-part dependencies available before recovery. Package chat replay can evaluate a labelled need with multiple acceptable sets of sources.
- No model is called. Answer quality is not measured, and every side is assumed to know which earlier parts a turn needs.
- A policy that keeps retrieved text matching the question, or cited sources, is priced without that.

## Run it locally

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site   # then open http://127.0.0.1:8765/
node --test site/*.test.mjs
node scripts/openclaw_comparison.mjs --json docs/assets/data/openclaw.json && python scripts/figures.py openclaw
```

The full method is on the [methodology page](https://barendpotijk.github.io/compache/methodology.html).
