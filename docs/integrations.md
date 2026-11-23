# Integrations

compache goes in at two places: where your model calls leave, and where your traces are kept.

## Where the model calls leave

| Your stack | compache goes in with |
|---|---|
| OpenAI SDK (Chat Completions, Responses), Anthropic SDK, Google GenAI SDK, sync or async | `client = compache.wrap(client, "policy.json")` |
| LangChain and LangGraph on `ChatOpenAI` | `ChatOpenAI(..., **compache.langchain.clients(live))` and `middleware=[compache.langchain.middleware()]` |
| Anything with a base URL: OpenClaw, Hermes, Claude Code, Codex, OpenCode, Cline, Aider, Goose, CrewAI, LlamaIndex, Pydantic AI, the Vercel AI SDK, LiteLLM, `ChatAnthropic` | `compache proxy --policy policy.json`, and the base URL set to it |
| Your own HTTP client | `compache.Live(...).exchange(path, body, send)` |
| Your own prompt assembly | `policy.session(thread_id)`, see the [guide](guide.md#5-apply) |

All of them run the same code on the JSON that goes over the wire, so a request leaves the same whichever way compache went in.

| On every model call | |
|---|---|
| Policy | Archived parts go out as a notice with a handle. The model gets a `reopen` tool for them, and its calls are answered before the application sees the reply |
| Cache | The end of the conversation is marked cacheable where the provider has to be told and the application has not done so. Keep-warm pings go out when the policy says one pays |
| Record | Sizes, times and usage of every turn go to a folder, without any text. See [the guide](guide.md#without-past-traces) |
| Notes | For a sampled share of turns the model is asked, once its reply is in, which earlier parts the reply needed. That request is compache's own, and the application never sees it |
| Anything else | A request shape compache does not know passes through untouched. A request the provider refuses after it was changed is sent again as the application wrote it |

### SDK clients

```python
import compache
from anthropic import Anthropic

client = compache.wrap(Anthropic(), "policy.json", record="compache-record", sample=0.1)
with compache.thread("conversation-17"):
    client.messages.create(model=MODEL, system=SYSTEM, messages=messages, max_tokens=1024)
```

`create`, `parse`, `stream`, tools, structured output and async clients work as before: the client's own HTTP client is wrapped, nothing above it. `client.compache.stats` counts what was done.

### LangChain and LangGraph

```python
live = compache.Live("policy.json", record="compache-record", sample=0.1)
model = ChatOpenAI(model="gpt-6-luna", **compache.langchain.clients(live))
agent = create_agent(model, tools, middleware=[compache.langchain.middleware()], checkpointer=saver)
agent.invoke({"messages": [...]}, {"configurable": {"thread_id": "conversation-17"}})
```

The middleware names each model call's conversation after the graph's thread ID and changes nothing else.

### The proxy

```sh
compache proxy --policy policy.json --record compache-record --sample 0.1 --port 8787
```

It serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages` and Gemini's `:generateContent` on the provider's own paths, forwards the caller's credentials, and passes everything else through. `--upstream URL` sends all of it to one gateway.

| Harness | Setting |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` |
| Codex, Aider, Goose, CrewAI and other OpenAI-compatible clients | `OPENAI_BASE_URL=http://127.0.0.1:8787/v1` |
| OpenClaw | In `~/.openclaw/openclaw.json`, a provider under `models.providers` with `baseUrl: "http://127.0.0.1:8787/v1"` and `api: "openai-completions"`, or `baseUrl: "http://127.0.0.1:8787"` and `api: "anthropic-messages"`. Add `compat: { sendSessionAffinityHeaders: true }` to its models so that sessions are named |
| Hermes | In `~/.hermes/config.yaml`, an entry under `providers` with `api: http://127.0.0.1:8787/v1` and `session_affinity_header: x-compache-thread`, selected by `model.provider: custom:<name>` |
| `ChatAnthropic` | `anthropic_api_url="http://127.0.0.1:8787"` |

### Naming a conversation

In order: `compache.thread(...)` around the call, the header `x-compache-thread`, a session header a harness already sends (`x-session-affinity`, `session_id`, `x-session-id`, `x-opencode-session-id`, `agent-session-id`), the body's `prompt_cache_key`. Without any of them a conversation is recognized by what it has said so far, from its second turn on.

## Where the traces are kept

| Traces in | Read them with |
|---|---|
| Langfuse | `compache pull langfuse -o threads.jsonl.gz` |
| MLflow | `compache pull mlflow -o threads.jsonl.gz` |
| An OpenTelemetry export in the GenAI or OpenInference conventions, as Phoenix and others write | `compache pull otel spans.jsonl -o threads.jsonl.gz` |
| Your own chat log | Nothing: `compache fit` reads it, see the [guide](guide.md#1-threads) |
| Nowhere you can read | A record made going forward: `compache fit compache-record` |

## What was run

| | Checked against |
|---|---|
| The four APIs through the OpenAI, Anthropic and Google GenAI SDKs, sync and async, whole and streamed, and a LangChain agent | The SDKs themselves, on every test run, against a local stand-in for the provider |
| The proxy | Standard-library callers and both SDKs, against the same stand-in |
| Live providers | Chat Completions whole and streamed, Responses and Messages through the SDKs, the proxy and a LangChain agent, on Claude Haiku 5.5, GPT-6 Luna and Gemini 3.5 Flash Lite through OpenRouter. See [the live runs](../benchmarks/live/README.md) |
| Gemini's own API | The stand-in only: OpenRouter does not serve it |
| The harness settings above | Taken from each harness's documentation. Not run here |
