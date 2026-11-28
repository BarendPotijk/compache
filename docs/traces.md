# Get conversation traces

Choose the source that has both the user’s messages and their arrival times. Each command below writes a local file. Imported threads can contain prompt text, tool definitions, retrieved documents and answers. Label packets made from these threads can contain the same material. Use a [forward record](#without-past-traces) when you need to keep conversation text out of the files.

| What you have | Where to start |
|---|---|
| Chat logs or structured turns | [Write one conversation per JSONL row](#write-your-own-threads) |
| Langfuse observations | [Pull sessions or read an export](#from-langfuse) |
| MLflow traces or OTLP JSON | [Read model spans](#from-mlflow-or-an-opentelemetry-export) |
| Local Claude Code or Codex sessions | [Import agent transcripts](#from-agent-transcripts) |
| An application you can instrument | [Record future turns](#without-past-traces) |

## Write your own threads

One JSON object per line, one conversation each. A file, a gzipped file or a folder of either. The examples below show individual rows expanded for readability. Serialize each object onto a single line in the file.

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

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to the credentials for the project you want to export. For a self-hosted or regional instance, set `LANGFUSE_BASE_URL` to its origin.

```sh
compache pull langfuse -o threads.jsonl.gz --days 30 --environment production
```

```text
Read 1,542 observations: 514 model calls in 80 sessions.
Wrote 80 threads, 257 turns to threads.jsonl.gz.
514 model calls were recorded twice and are counted once.
Answered by claude-opus-5-5 (514 calls).
```

A session becomes a thread. The export can carry the model, system prompt, tools and observation cutoff. With a recognized model recorded, `compache fit threads.jsonl.gz` can fit caching from the export. To fit chat compaction, also supply [dependency labels](fitting.md#label-historical-dependencies).

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

## From agent transcripts

Run the command for the harness that wrote your sessions.

```sh
compache pull claude -o claude-threads.jsonl.gz
compache pull codex -o codex-threads.jsonl.gz
```

An optional path after the source selects the session directory. The [agent guide](agents.md) describes which call and tool-result dependencies can be recovered.

## Without past traces

Traces can be protected, incomplete or not kept at all. The same fit runs on a record made going forward by compache in the request path. With no policy and the default sampling rate, the recorder collects sizes and usage as requests pass.

```python
client = compache.wrap(client, record="compache-record")
```

```sh
compache fit compache-record --model claude-sonnet-5-5    # threads and labels both come from the folder
```

| The record holds | |
|---|---|
| For every turn | When it started and ended, the size in bytes of its question, of each retrieved text and of its answer, the model, its calls and the provider's usage |
| For a sampled turn | The labels of the earlier parts its reply named, such as `R3.1` or `A2` |
| Never | Any text of a conversation. A conversation's ID is kept as a hash |

The default `sample=0` records metadata. With `sample=0.1`, approximately one turn in ten can trigger a billable labeling request to the model provider.

`sample` is the share of turns that are asked. Once a sampled reply is in, compache sends one short request of its own: the conversation, the reply, and the conversation's parts by label and first words, with the request to name the parts the reply needed in one line, `<!-- refs: {R3.1} {A2} -->`. Your application's request and its reply are untouched, whatever their form. The extra cost depends on the provider, prompt length and sampling rate.

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

For an exported threads file, run `compache check threads.jsonl.gz` and compare the reported thread and turn counts with the source. For a recording, pass the record directory itself. Continue with [labeling, fitting and replay](fitting.md).
