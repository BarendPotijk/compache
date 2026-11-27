# Guide

From a repository checkout, use Python 3.11 or later to install this version and fit a policy on the four conversations bundled with the package. This example runs locally without credentials.

```sh
python -m pip install -e .
compache demo -o policy.json
```

```text
Fitted on 4 sample threads, 71 turns (71 labelled) for claude-sonnet-5-5.

Keep questions from the last 4 turns, answers from the last 9 turns and retrieved text from the last turn.
Send up to 1 keep-warm ping while a user is away.
Minimum training retention: 80.00%.
On the training threads, 81.82% of what the labelled turns need from earlier turns stays in view. The rest is fetched back, and that call is in the cost.
$16.10 per 1,000 turns against $50.25 keeping everything: 68% less.
Caching alone, at its best, would cost $18.66.
```

These figures come from the demo’s training conversations. To estimate savings on your application, fit on your own data and replay the saved policy on other complete threads.

## 1. Threads

Choose the [input that matches your application](traces.md). Compache accepts timestamped chat logs, structured turns and supported trace exports. Keep a stable ID for each conversation so you can reserve whole threads for evaluation.

### From Langfuse

[Pull sessions by environment, tag or filter, or read an existing export](traces.md#from-langfuse).

### From MLflow or an OpenTelemetry export

[Import MLflow traces or OTLP JSON spans](traces.md#from-mlflow-or-an-opentelemetry-export). For local agent sessions, [import Claude Code or Codex transcripts](traces.md#from-agent-transcripts).

## 2. Labels

For chat, label which earlier sources a later answer required. Without labels the planner keeps the full history and fits caching. [Build packets, encode alternative sources and check the labels](fitting.md#label-historical-dependencies). For tool-loop workloads, the [agent planner](agents.md) can infer dependencies from reuse of earlier results.

## Without past traces

[Record sizes, usage, timing and hashed conversation IDs](traces.md#without-past-traces) through the wrapper or proxy. Recording defaults to metadata only. Opting into sampled labeling can send additional requests containing conversation text to the provider.

## 3. Fit

Fit on training threads and save the policy. The [fit walkthrough](fitting.md) links to a runnable split of the packaged sample and documents the retention floor, model prices and observation cutoff.

## 4. Replay on unseen threads

[Replay the saved policy](fitting.md#replay-and-interpret-the-report) against keeping everything. Compare held-out retention and modeled cost on the same arrivals. Keep training and evaluation IDs disjoint.

## 5. Apply

Wrap a supported client or run the proxy. Your application continues sending its conversation history. Use the [integration guide](integrations.md) to choose a supported request format and pass a stable conversation ID. This fragment assumes your app already has an Anthropic client (`client`), model name (`MODEL`), system prompt (`SYSTEM`), and message history (`messages`). For a complete, offline SDK call, run the [wrapped chatbot](../examples/README.md#put-a-policy-in-a-chatbot-request).

```python
import compache

client = compache.wrap(client, "policy.json")
with compache.thread("conversation-17"):
    response = client.messages.create(
        model=MODEL, system=SYSTEM, messages=messages, max_tokens=1024
    )
```

Pings need a running process. For a chat application without a timer, fit in Python with `ping_counts=(0,)`. If retrieved text is pasted into a user message, the request path can only keep or archive them together. Fit from a forward recording to reflect that grouping. To keep collecting evidence while applying a policy, supply `record=` and optionally `sample=` to the wrapper.

### Your app manages requests

Use an [application-owned session](sessions.md) if you manage prompt assembly, billing, retries or timers. A [runnable example](../examples/quickstart/app_managed.py) exercises the full tool-loop lifecycle with an in-memory provider.

## 6. The calculator on your workload

```sh
compache profile threads.jsonl --labels labels.jsonl --model claude-sonnet-5-5
```

It measures tokens per turn, thread length and time away by how far a thread has come, calls and answer time per turn, what the cache returned and what later turns needed, and prints a link that opens the [cost calculator](https://barendpotijk.github.io/compache/) with those numbers filled in. `-o profile.json` writes the same as a file for the calculator's Import button. Details in [calculator.md](calculator.md).

## Beyond the top-level API

`compache.optimizer` holds the engine: `solve_workload`, `replay_workload`, `WorkloadPolicy` and `WorkloadSession`. Use it for a tokenizer callback instead of the byte estimate, a cache-break checker stored in SQLite, shared system prompts across threads, several model calls per turn, or a comparison against your host's own call ledger. Each is a keyword argument documented on the function.

| Model spec field | Meaning |
|---|---|
| `model_id` | Name written on the policy |
| `context_tokens` | Largest input the model accepts |
| `cache_ttl_seconds`, `cache_min_tokens`, `cache_block_tokens` | The provider's cache lifetime, smallest cacheable prefix and block size |
| `read_multiplier`, `write_multiplier`, `output_multiplier` | Price of a cached, a newly cached and an output token, relative to an input token |
| `long_input_threshold`, `long_input_factor`, `long_output_factor` | Surcharge above a request size, with `"schema_version": "compache-model-spec-v2"` |
| `ping_output_tokens`, `expansion_output_tokens` | Output a ping and a fetch-back request are expected to produce |
| `bytes_per_token`, `part_overhead_tokens`, `byte_limit` | The size estimate and the request size limit |
