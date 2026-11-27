# CLI and Python reference

Use `compache <command> --help` for the full option list in your installation. The Python API and CLI use the same fitter and replay engine. Local demos, fits and replays work without a model key.

## Commands by task

| Task | Command | Output |
|---|---|---|
| Try the package | `compache demo -o policy.json` | A fitted policy for the bundled four-chat sample. Omit `-o` to print the fit without saving. |
| Start before you have traces | `compache default --model MODEL -o policy.json` | A starter policy. Add `--agent` for tool loops. |
| Find model names | `compache models` | Models with a built-in price sheet. |
| Import conversations | `compache pull SOURCE -o threads.jsonl.gz` | A local threads file. Sources, credentials and export formats are in the [trace guide](traces.md). |
| Prepare labels | `compache packets threads.jsonl --sample 200 -o packets.jsonl` | Labeling inputs for up to 200 runs. Packets can contain conversation text. |
| Validate a dataset | `compache check threads.jsonl --labels labels.jsonl` | Thread and label counts plus validation errors, if any. |
| Fit and save | `compache fit train.jsonl --labels train-labels.jsonl --model MODEL -o policy.json` | The selected policy and training estimates. `--min-retention` accepts a fraction from 0 to 1 in this checkout. |
| Evaluate a policy | `compache replay test.jsonl --labels test-labels.jsonl --policy policy.json -o replay.json` | A comparison on unseen threads. `-o` includes detailed replay data. |
| Open a scenario in the calculator | `compache profile threads.jsonl --model MODEL --policy policy.json` | Measured workload parameters and a shareable calculator URL. |
| Send traffic through a proxy | `compache proxy --policy policy.json` | Local HTTP proxy on port 8787 by default. [Route a client to it](integrations.md#the-proxy). |
| Run a coding agent | `compache wrap claude` | Starts the agent and a local proxy. Use `compache wrap --watch claude` to record first. |
| Refit a growing agent record | `compache learn claude` | Updates eligible policies for that record when enough calls have accumulated. |
| Inspect runtime usage | `compache stats compache-record --json` | Usage-based costs and an estimated full-history comparison for recorded calls. |
| Diagnose a local installation | `compache doctor` | A local synthetic request-path check and detected integrations. |

Replace `MODEL` with a name from `compache models`. For a custom model, pass a JSON file of [model spec fields](guide.md#beyond-the-top-level-api). Explicit specs supply relative token prices and cache rules. Known model names also supply a dollar price sheet.

## Python entry points

| Entry point | When to use it |
|---|---|
| `compache.fit(threads, labels, model=..., min_retention=...)` | Train on your own conversations. Returns a `Policy`. Chat fitting accepts an unlabeled record for caching decisions. [Fit inputs and options](fitting.md) |
| `policy.describe()`, `policy.summary` | Read the decision as text or a dictionary. The summary reports costs on the training workload. |
| `policy.save(path)`, `compache.Policy.load(path)` | Persist and reload the chosen policy. A policy may include the system prompt and tool definitions supplied to the fit. |
| `compache.replay(policy, threads, labels)` | Evaluate a loaded policy or artifact dictionary on other threads. If you have a filename, call `Policy.load` first. |
| `compache.wrap(client, policy, record=...)` | Apply a policy to a supported SDK client and optionally record usage. Reuse the returned client. |
| `compache.thread(thread_id)` | Bind outgoing model requests in the block to an application conversation ID. Works with sync and async calls. |
| `policy.session(thread_id)` | Start a session when your app assembles and sends the provider requests. Retain the session across turns. [Session lifecycle](sessions.md) |
| `compache.Live(...)` | Share one policy runtime across supported adapters or inspect its call counters. [Integration options](integrations.md) |

### Read a chat replay in Python

With a saved policy and two held-out JSONL files in the current directory:

```python
import compache

policy = compache.Policy.load("policy.json")
report = compache.replay(policy, "test.jsonl", "test-labels.jsonl")
print(report["cost_per_1000_turns"])  # includes the unit and both policies' costs
print(report["min_retention"], report["training_retention"], report["heldout_retention"])
```

Agent replays report `cost_per_1000_calls`. For either planner, read the normalized cost object's `unit` before formatting a number as currency. Raw `policy_cost` and `keep_everything_cost` use fresh-input-token equivalents. Retention is a fraction from 0 to 1, or `None` if unmeasured. [Interpret replay reports](fitting.md#replay-and-interpret-the-report)

### Counts and costs from a running application

| Signal | Meaning |
|---|---|
| `client.compache.stats["calls"]` | Model calls seen at the application boundary of the wrapped client. |
| `client.compache.stats["physical_calls"]` | Dispatches made through the Compache transport, including refusals, retries, reopen calls, labeling calls and pings. |
| `client.compache.tokens` | Aggregated input, cache-read, cache-write and output counts from receipts, plus an estimate of tokens left out. Cache reads and writes are portions of input. |
| `compache stats RECORD` | Costs from recorded usage and known prices, using provider-reported cost when available for a priced model. Its full-history comparison is an estimate. |

Per-process counters restart with the process. An application-owned session sends through your own client, so capture its transport receipts there. [Investigate unexpected counts or costs](troubleshooting.md)
