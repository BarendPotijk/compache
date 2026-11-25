# Guide

Record threads, fit a policy, apply it. Python 3.11 or later, no dependencies.

```sh
pip install compache
```

| Step | Command | Python |
|---|---|---|
| See it on sample chats | `compache demo` | |
| Read traces | `compache pull langfuse -o threads.jsonl.gz`, also `mlflow`, `otel`, `claude` and `codex` | `compache.langfuse.pull()` |
| Or record going forward | `compache proxy --record compache-record --sample 0.1` | `compache.wrap(client, record=..., sample=0.1)` |
| Check the data | `compache check threads.jsonl --labels labels.jsonl` | `compache.load_threads(path)` |
| Fit | `compache fit threads.jsonl --labels labels.jsonl --model claude-sonnet-5-5` | `compache.fit(threads, labels, model=...)` |
| Price on unseen threads | `compache replay heldout.jsonl --policy policy.json` | `compache.replay(policy, threads, labels)` |
| Apply | `compache proxy --policy policy.json` | `compache.wrap(client, policy)` |
| Open the calculator on your workload | `compache profile threads.jsonl --model claude-sonnet-5-5` | `compache.profile(threads, labels, model=...)` |

## 1. Threads

One JSON object per line, one conversation each. A file, a gzipped file or a folder of either. The simplest shape is the chat log you already store:

```json
{"thread_id": "t-17", "messages": [
  {"role": "system", "content": "Answer from the handbook."},
  {"role": "user", "content": "How many leave days do I get?", "at": "2026-11-02T09:14:03Z"},
  {"role": "retrieval", "content": "Full-time staff accrue 25 days a year.", "id": "handbook#leave-3"},
  {"role": "assistant", "content": "25 days a year.", "at": "2026-11-02T09:14:09Z"}
]}
```

| Field | What it is for |
|---|---|
| `role` | `user`, `assistant` or `system`. Retrieved or tool text is `retrieval`, `tool`, `context`, `document` or `function`. A `tool_result` block inside a user message also counts as retrieved text. Every assistant message counts as one model call of its turn |
| `content` | Text, or a list of text blocks. Sizes are estimated from it at four bytes a token |
| `at` | When the message was sent: ISO time or seconds since the epoch. Needed on every user message, used on assistant messages when present. `created_at`, `timestamp` and `time` are read too. A time without a zone is taken as UTC |
| `id` | Optional name of a retrieved document |
| `usage` | Optional, on an assistant message: the provider's usage as returned. Its output count replaces the estimate, and its cache read says how much of the last request the turn found cached |
| `model` | Optional, on an assistant message or on the thread: the model that answered. `fit` prices the threads as the model most of them record |
| `tool_calls` | Optional, on an assistant message, or `tool_use` blocks in its content: their arguments say what an agent's reply relied on, see [Agents](agents.md) |

A user message opens a turn. Everything retrieved and everything the assistant says before the next user message belongs to it. User messages sent before any answer merge into one question, and a last message that was never answered is dropped. The first system message becomes the system prompt of the fit.

If you keep turns instead of messages, write them directly:

```json
{"thread_id": "t-17", "runs": [
  {"started_at": "2026-11-02T09:14:03Z", "ended_at": "2026-11-02T09:14:09Z",
   "input": "How many leave days do I get?",
   "retrievals": [{"text": "Full-time staff accrue 25 days a year.", "id": "handbook#leave-3"}],
   "output": "25 days a year."}
]}
```

A turn that made several model calls can say so with `"calls": 3`. Parts are numbered `<thread>:<turn>:input`, `<thread>:<turn>:retrieval:<k>` and `<thread>:<turn>:output`. Labels refer to these IDs. A thread that spells out every identity itself, as the benchmark builders write them, is taken as is.

If your application records which sources an answer cited, say so: give the retrieved message a `"part_id"` that is unique in the thread, such as `"lookup-17"`, and the assistant message `"cited_part_ids": ["lookup-17"]`. With turns, put the list on the run. The fit can then keep cited sources longer than the rest of what was retrieved, for as long as your threads show it pays.

A thread in either shape may also carry `"tools"`, the tool definitions sent on every request, and `"observed_until"`, when the record was read.

### From Langfuse

```sh
export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-...
compache pull langfuse -o threads.jsonl.gz --days 30 --environment production
```

```text
Read 1,542 observations: 514 model calls in 80 sessions.
Wrote 80 threads, 257 turns to threads.jsonl.gz.
514 model calls were recorded twice and are counted once.
Answered by claude-opus-5-5 (514 calls).
```

A session becomes a thread. The file carries the model, the system prompt, the tools and the time of the read, so `compache fit threads.jsonl.gz` needs nothing else.

| In the traces | How it is read |
|---|---|
| One trace per turn, or one per model call | Turns follow the user's messages, rebuilt from what each model call was sent |
| A tool loop | Every call of the turn is counted. Tool results are retrieved text |
| Retrieved text pasted into the prompt | Taken out of the prompt where a retriever or tool observation of the trace returned it |
| A model call recorded by two instrumentations | Counted once |
| Helper calls, such as a query rewriter | Left out |
| Calls that do not carry the conversation | The thread is read trace by trace: each root's input and output is a turn |

- **It needs** a session ID on the traces, with prompts and answers recorded. Sessions that began before the first row read are left out, and the summary counts everything it leaves out.
- **Choose one app** in a project with `--environment`, `--tag` or `--filter`, which takes Langfuse's own filter JSON. `--rows export.json` reads exported observations instead of the API.
- **Host.** Langfuse Cloud by default, `LANGFUSE_BASE_URL` for another region or your own Langfuse 4.
- **Checked** on Langfuse Cloud with the OpenAI drop-in, the Anthropic SDK's own spans, the OpenTelemetry Anthropic instrumentation and `@observe` functions.

### From MLflow or an OpenTelemetry export

```sh
compache pull mlflow --tracking-uri http://localhost:5000 --experiment chat -o threads.jsonl.gz
compache pull otel spans.jsonl -o threads.jsonl.gz
```

Both are read the way Langfuse is: a session or conversation ID makes a thread, and the same rules rebuild its turns.

| Source | What is read |
|---|---|
| MLflow 3 | Traces over the tracking server's REST API, in the shapes its OpenAI, Anthropic and LangChain autologging record. The session is `mlflow.trace.session`. `MLFLOW_TRACKING_URI` and `MLFLOW_TRACKING_TOKEN` are used when set |
| OpenTelemetry | An OTLP JSON export, one document or one per line, in the GenAI conventions (`gen_ai.*`) or OpenInference (`llm.*`), as Phoenix and others write |

Checked against an MLflow 3.18 server with the three autologgers, whole and streamed, and with spans sent to it over OTLP.

## 2. Labels

Without labels the fit tunes the keep-warm pings and keeps the whole history. Labels say what later turns needed from earlier ones, which is what lets it drop the rest. A sample is enough: an unlabelled turn counts as not sampled.

```json
{"thread_id": "t-17", "run": 7, "needs": [["t-17:3:output"], ["t-17:2:retrieval:1"]]}
{"thread_id": "t-17", "run": 8, "needs": []}
{"thread_id": "t-17", "run": 9, "needs": null}
```

| `needs` | Meaning |
|---|---|
| A list of alternatives | Any one alternative, with all of its parts in view, is enough to answer the turn |
| `[]` | The turn needs nothing from earlier turns |
| `null` | Nobody could tell |

A part is its ID, or `{"part_id": ..., "quote": ...}`. A quote is checked to appear verbatim in that part, which keeps a model that writes labels honest.

To have a coding agent write them:

```sh
compache packets threads.jsonl --sample 200 -o packets.jsonl   # what a labeller may see, one run per line
# the agent labels each packet with the skill in skills/compache-label-dependencies
compache check threads.jsonl --labels labels.jsonl             # validates and summarizes
```

## Without past traces

Traces can be protected, incomplete or not kept at all. The same fit runs on a record made going forward by compache in the request path. With no policy it changes nothing in your requests.

```python
client = compache.wrap(client, record="compache-record", sample=0.1)
```

```sh
compache fit compache-record --model claude-sonnet-5-5    # threads and labels both come from the folder
```

| The record holds | |
|---|---|
| For every turn | When it started and ended, the size in bytes of its question, of each retrieved text and of its answer, the model, its calls and the provider's usage |
| For a sampled turn | The labels of the earlier parts its reply named, such as `R3.1` or `A2` |
| Never | Any text of a conversation. A conversation's ID is kept as a hash |

`sample` is the share of turns that are asked. Once a sampled reply is in, compache sends one short request of its own: the conversation, the reply, and the conversation's parts by label and first words, with the request to name the parts the reply needed in one line, `<!-- refs: {R3.1} {A2} -->`. Your application's request and its reply are untouched, whatever their form. On Claude Haiku 5.5 a note cost about half of what a turn does, so one turn in ten adds some 5%.

`notes="inline"` asks in the reply itself and saves that request. The reply then answers a request with one more paragraph in it, and that can change the answer: in the [live runs](../benchmarks/live/README.md#a-note-after-the-reply-or-in-it) a fact was wrong in 3 of 8 conversations on Gemini 3.5 Flash Lite and in 2 of 10 on Claude Haiku 5.5 through the Messages API, against none of them when the note was asked afterwards or not at all.

| With `notes="inline"`, a reply that is | Carries the note as |
|---|---|
| Text | A last line, taken out before your application sees the reply |
| JSON under a schema | One more property, cut out of the JSON text. The schema is widened on every turn, so the provider caches one schema |
| Streamed | Nothing: what has been shown cannot be taken back, so it is asked after the reply all the same |

| If, with the note in the reply, | Then |
|---|---|
| The provider refuses the request | It is sent again without. After three refusals notes are switched off |
| The reply names a label where its reader would see it, is cut off by the output limit, or holds nothing but the note | The request is sent again without the note, and that reply is the one returned |
| The property cannot be cut out of the JSON | The same, with the application's own schema |

Either way, a reply that ignores the request leaves its turn unlabelled, and a part the model fetched back while answering counts as one the turn needed.

For a policy from the first day, `compache default --model claude-sonnet-5-5` writes the one fitted to the default workload of the calculator. Refit on your record as it grows.

For an agent, no notes are needed: [it learns from its loops](agents.md#how-it-learns).

## 3. Fit

```sh
compache fit threads.jsonl --labels labels.jsonl --model claude-sonnet-5-5 -o policy.json
```

```text
Fitted on 4 threads, 71 turns (71 labelled) for claude-sonnet-5-5.

Keep questions from the last 4 turns, answers from the last 9 turns and retrieved text from the last turn.
Send up to 1 keep-warm ping while a user is away.
82% of what the labelled turns need from earlier turns stays in view. The rest is fetched back, and that call is in the cost.
$16.10 per 1,000 turns against $50.25 keeping everything: 68% less.
Caching alone, at its best, would cost $18.66.
```

| Option | Use |
|---|---|
| `--model` | A model with a built-in price sheet (`compache models` lists them), or a JSON file of model spec fields for any other. Fields left out take the defaults of `ModelSpec`. It can be left out when the threads record their model, as a file from `compache pull` does |
| `--system-prompt`, `--tools` | The text and tool definitions sent on every request, when the threads do not carry them |
| `--observed-until` | When the record was cut, if conversations were still going and the threads do not say. By default every thread is taken to be over |
| `--json` | Print the summary as JSON |

In Python, `compache.fit` returns a `Policy`: `policy.describe()` is the text above, `policy.summary` the numbers, `policy.save(path)` the file. The file holds the decision, the fitted model of returns, the priced candidates and the system prompt it was fitted with.

## 4. Replay on unseen threads

```sh
compache replay heldout.jsonl --labels heldout-labels.jsonl --policy policy.json
```

It prices the policy and keeping everything on the same arrivals, including every ping. With labels it also reports how much of what those threads needed stays in view and adds the cost of fetching the rest back. A fit sits near its floor by construction, so this is the number to trust. `--json` adds the same per thread. If a request outgrows the model's context window, the costs are compared on the turns both sides answered.

## 5. Apply

Put compache where your model calls leave. Your application keeps its own message list and sends it whole, as before.

```python
import compache

client = compache.wrap(client, "policy.json")          # an OpenAI, Anthropic or Google GenAI client
with compache.thread("conversation-17"):
    response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, max_tokens=1024)
```

```sh
compache proxy --policy policy.json                    # for anything that lets its base URL be set
```

On the way out, the parts the policy archives are replaced by a notice, the end of the conversation is marked cacheable, and the model is given a `reopen` tool whose calls are answered before your application sees the reply. A background thread sends the keep-warm pings. [Integrations](integrations.md) lists every way in: frameworks, LiteLLM, Hermes and agent harnesses included.

- **Record and apply together.** `record=` and `sample=` work with a policy too, so the next fit has fresher threads.
- **No long-lived process.** Pings need one. Where there is none, fit with `ping_counts=(0,)`. Trimming needs no state: a process that meets a conversation in the middle takes the cache to have expired.
- **Retrieved text pasted into the question.** The request path sees one message, so it can only keep or archive the two together. Fit such an application on a record made going forward, which sees it the same way.

### By hand

When you assemble prompts yourself, a session gives you the request for each turn:

```python
policy = compache.Policy.load("policy.json")
session = policy.session("thread-17")          # one per conversation, kept while it is active

turn = session.prepare(question, retrieved)     # retrieved: the texts fetched for this question
response = client.messages.create(model=MODEL, system=SYSTEM, messages=turn.messages, ...)
session.complete(turn, answer_text, usage=response.usage)

if session.next_ping_at:                         # epoch seconds, or None
    schedule(session.next_ping_at)
```

| Call | What it does |
|---|---|
| `session.prepare(question, retrieved)` | Returns the request for this turn. `turn.parts` are its parts in order, `turn.messages` the same as chat messages, `turn.archived_ids` the parts replaced by a short notice |
| `session.complete(turn, answer, usage=...)` | Records the answer and returns when a ping is due. `usage` is the provider's usage object or dict. With it the session tracks real sizes and stops pinging a cache that no longer hits |
| `session.ping()` | When the timer fires: the request to send with the smallest output the provider allows, or `None` if the ping is no longer worth sending |
| `session.ping_sent(at=..., usage=...)` | Records the ping. More pings follow only when its usage shows the cache was read |
| `session.reopen(reference)` | Answers the model's `reopen` tool call with the archived text |

- **Cited sources.** For a policy fitted with them, pass `session.complete(turn, answer, cited_part_ids=[...])`. The IDs are the `part_id` of items given to `prepare`, or the generated ones in `turn.parts`, and must be in view in that request. They add nothing to what the model is sent.
- **Cache marks.** Mark the end of a Claude request with `policy.cache_control`. It names the lifetime where the policy was fitted for another than the default, and then every mark of the request has to.
- **Archived parts.** A cut replaces a part with a notice that carries a handle. Give the model `compache.REOPEN_TOOL` and `compache.REOPEN_INSTRUCTIONS`, and pass its tool calls to `session.reopen`.
- **Placement.** `turn.messages` puts answers under `assistant` and everything else under `user`. If your application sends retrieved text another way, build the request from `turn.parts` and keep that mapping fixed.
- **Tool loops.** Call `prepare` once per user message, append the calls of that turn yourself, and hand what the tools returned to `session.extend(turn, results)` before `complete`. With several calls, pass the first call's usage as `usage` and the last call's as `last_usage`.
- **Determinism.** No model is called. The same history, times and usage give the same request, so a session can be rebuilt after a restart by feeding the stored turns through `prepare` and `complete` again.
- **Time.** Both calls take `at=` in epoch seconds and default to now.

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
