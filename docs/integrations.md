# Integrations

compache goes in at two places: where your model calls leave, and where your traces are kept.

## Where the model calls leave

| Your stack | compache goes in with |
|---|---|
| Claude Code, Codex, OpenCode, Gemini CLI, Qwen Code, Aider, Goose, or any other agent on the command line | `compache wrap claude`, see [Agents](agents.md) |
| OpenAI SDK (Chat Completions, Responses), Anthropic SDK, Google GenAI SDK, sync or async | `client = compache.wrap(client, "policy.json")` |
| A Python framework on one of these SDKs: OpenAI Agents SDK, Pydantic AI, LlamaIndex, Agno, AutoGen, Strands, Haystack, Instructor, Langfuse's OpenAI drop-in | The wrapped client, or an HTTP client: [one line each](#python-frameworks) |
| LangChain and LangGraph, any chat model | `create_agent(..., middleware=[compache.langchain.middleware(live)])` |
| LiteLLM, and what is built on it | `compache.litellm.install(live)`, or a [callback](#litellm) that records in your LiteLLM proxy |
| Hermes Agent | Its [context engine](#hermes): `context.engine: compache` |
| A LiteLLM proxy or another gateway | `compache proxy --policy policy.json --upstream http://localhost:4000` in front of it |
| Anything else with a base URL: the Vercel AI SDK, OpenClaw, CrewAI, Cline | `compache proxy --policy policy.json`, and the base URL set to it |
| Your own HTTP client | `compache.Live(...).exchange(path, body, send)` |
| Your own prompt assembly | `policy.session(thread_id)`, see the [guide](guide.md#5-apply) |

All of them run the same code on the JSON that goes over the wire, so a request leaves the same whichever way compache went in. Wherever a policy file can be given, `learn="compache-record"` (`--learn`) can be given instead: the policy then [follows from your traffic](agents.md#how-it-learns).

| On every model call | |
|---|---|
| Policy | Archived parts go out as a notice with a handle. The model gets a `reopen` tool for them, and its calls are answered before the application sees the reply |
| Cache | The end of the conversation is marked cacheable where the provider has to be told and the application has not done so. Where the policy was fitted for one-hour entries, every mark of the request asks for them, yours included. Where it counts on five minutes, marks on the conversation ask for that. Keep-warm pings go out when the policy says one pays |
| Record | Sizes, times and usage of every turn go to a folder, without any text. See [the guide](guide.md#without-past-traces) |
| Notes | For a sampled share of turns the model is asked, once its reply is in, which earlier parts the reply needed. That request is compache's own, and the application never sees it |
| A restart | A process that meets a conversation in the middle, as after a deploy, in a function or on the command line, takes the cache to have expired and trims at once. No state has to be kept. Behind a load balancer, keep a conversation on one worker, or each that takes it up trims it again |
| Anything else | A request shape compache does not know passes through untouched. A request the provider refuses after it was changed is sent again as the application wrote it. Where it refuses one-hour entries, the request goes out under its default and everything after is planned on that |

### SDK clients

```python
import compache
from anthropic import Anthropic

client = compache.wrap(Anthropic(), "policy.json", record="compache-record", sample=0.1)
with compache.thread("conversation-17"):
    client.messages.create(model=MODEL, system=SYSTEM, messages=messages, max_tokens=1024)
```

`create`, `parse`, `stream`, tools, structured output and async clients work as before: the client's own HTTP client is wrapped, nothing above it. `client.compache.stats` counts what was done.

### Python frameworks

| Framework | One line |
|---|---|
| OpenAI Agents SDK | `set_default_openai_client(compache.wrap(AsyncOpenAI(), "policy.json"))` |
| Pydantic AI | `OpenAIProvider(openai_client=compache.wrap(AsyncOpenAI(), "policy.json"))`, and `AnthropicProvider(anthropic_client=...)` the same way |
| LlamaIndex | `OpenAI(model=..., http_client=compache.http_client(live), async_http_client=compache.async_http_client(live))` |
| Agno | `OpenAIChat(id=..., client=compache.wrap(OpenAI(), "policy.json"))` |
| AutoGen | `OpenAIChatCompletionClient(model=..., http_client=compache.async_http_client(live))` |
| Strands | `OpenAIModel(client=compache.wrap(AsyncOpenAI(), "policy.json"), model_id=...)`, in one event loop |
| Haystack | `generator.warm_up()`, then `generator.client = compache.wrap(generator.client, "policy.json")` |
| Instructor | `instructor.from_openai(compache.wrap(OpenAI(), "policy.json"))` |
| Langfuse's OpenAI drop-in | `compache.wrap(langfuse.openai.OpenAI(), "policy.json")`. Langfuse goes on tracing what your application wrote, in full, so the next fit has everything |

`live` is `compache.Live("policy.json")`. Put `with compache.thread("conversation-17"):` around the run to name its conversation.

### LangChain and LangGraph

```python
live = compache.Live(learn="compache-record")          # or a policy file
agent = create_agent(model, tools, middleware=[compache.langchain.middleware(live)], checkpointer=saver)
agent.invoke({"messages": [...]}, {"configurable": {"thread_id": "conversation-17"}})
```

Any chat model. The graph keeps its messages whole, archived results reach the model as a notice, and the agent gets a `reopen` tool that runs like any other. The thread ID names the conversation. compache makes no request of its own here, so it sends no keep-warm pings and asks no notes.

With `ChatOpenAI` it can sit in the HTTP client instead, with pings and notes and fetching back unseen by the graph: `ChatOpenAI(..., **compache.langchain.clients(live))`, and `middleware()` without arguments to name the conversations.

### LiteLLM

```python
live = compache.Live("policy.json")
compache.litellm.install(live)           # every OpenAI-compatible call LiteLLM makes from now on
with compache.thread("conversation-17"):
    litellm.completion(model="openrouter/anthropic/claude-haiku-5.5", messages=messages)
```

Providers that LiteLLM calls over handlers of its own, such as Anthropic's Messages API, go through `compache proxy` with `api_base="http://127.0.0.1:8787"`. A LiteLLM proxy takes compache in front: `compache proxy --policy policy.json --upstream http://localhost:4000`, whatever providers it routes to.

To learn first, record in a LiteLLM proxy you already run, whatever the provider:

```yaml
litellm_settings:
  callbacks: ["compache.litellm.callback"]     # COMPACHE_RECORD=/data/compache-record COMPACHE_SAMPLE=0.1
```

The conversation is named by `metadata: {"session_id": ...}`. In process: `litellm.callbacks = [compache.litellm.Callback(record="compache-record", sample=0.1)]`.

### Hermes

```sh
hermes plugins install BarendPotijk/compache/integrations/hermes
compache default --model claude-sonnet-5-5 -o ~/.hermes/compache.json        # or a policy fitted to your own sessions
echo "COMPACHE_POLICY=$HOME/.hermes/compache.json" >> ~/.hermes/.env
```

```yaml
# ~/.hermes/config.yaml
context:
  engine: compache
prompt_caching:
  cache_ttl: 5m        # or 1h, where the policy was fitted for one-hour entries
```

The engine chooses what each request holds and leaves the conversation Hermes keeps as it is. The model fetches a trimmed part back with a `reopen` tool. When a conversation outgrows the model, Hermes' own compressor does what it always did. Needs Hermes 0.23 or later.

Hermes sends the requests, so the engine sends no keep-warm pings and asks no notes. For those, and to learn a policy from your own sessions, run Hermes through the proxy below. `COMPACHE_RECORD` keeps a record of what the engine sees, without labels.

### The proxy

```sh
compache proxy --policy policy.json --record compache-record --sample 0.1 --port 8787
```

As a container, it learns in a volume: `docker build -t compache . && docker run -p 8787:8787 -v compache:/data compache`. Arguments of its own replace that, such as `--policy`.

It serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages` and Gemini's `:generateContent` on the provider's own paths, forwards the caller's credentials, and passes everything else through. `--upstream URL` sends all of it to one gateway.

| Harness | Setting |
|---|---|
| Claude Code, Codex, OpenCode, Gemini CLI, Qwen Code, Aider, Goose | `compache wrap <agent>` starts the proxy and sets this for one run. `compache wrap --print <agent>` shows it, to set it yourself |
| CrewAI | `LLM(model="openai/...", base_url="http://127.0.0.1:8787/v1")`, or `OPENAI_BASE_URL` |
| OpenClaw | In `~/.openclaw/openclaw.json`, a provider under `models.providers` with `baseUrl: "http://127.0.0.1:8787/v1"` and `api: "openai-completions"`, or `baseUrl: "http://127.0.0.1:8787"` and `api: "anthropic-messages"`. Add `compat: { sendSessionAffinityHeaders: true }` to its models so that sessions are named |
| Hermes | In `~/.hermes/config.yaml`, an entry under `providers` with `api: http://127.0.0.1:8787/v1` and `session_affinity_header: x-compache-thread`, selected by `model.provider: custom:<name>` |
| Vercel AI SDK | `createAnthropic({baseURL: "http://127.0.0.1:8787/v1"})`, `createOpenAI(...)` the same. The conversation's name goes in `headers: {"x-compache-thread": id}`. OpenAI's Responses API needs `store: false`: with it on, the SDK sends earlier answers as references, which nothing can trim |
| `ChatAnthropic` | `anthropic_api_url="http://127.0.0.1:8787"` |

### Naming a conversation

In order: `compache.thread(...)` around the call, the header `x-compache-thread`, a session header a harness already sends (`x-session-affinity`, `session_id`, `session-id`, `x-session-id`, `x-claude-code-session-id`, `x-opencode-session-id`, `agent-session-id`), the body's `prompt_cache_key`. Without any of them a conversation is recognized by its tool calls, and by what it has said so far from its second turn on. Agents that share a name, such as a main agent and the ones it starts, are kept apart.

## Where the traces are kept

| Traces in | Read them with |
|---|---|
| Langfuse | `compache pull langfuse -o threads.jsonl.gz` |
| MLflow | `compache pull mlflow -o threads.jsonl.gz` |
| An OpenTelemetry export in the GenAI or OpenInference conventions, as Phoenix and others write | `compache pull otel spans.jsonl -o threads.jsonl.gz` |
| The sessions that Claude Code or Codex keeps on this machine | `compache pull claude -o threads.jsonl.gz`, or straight into a record: `compache learn claude --from claude` |
| Your own chat log | Nothing: `compache fit` reads it, see the [guide](guide.md#1-threads) |
| Nowhere you can read | A record made going forward: `compache fit compache-record` |

## What was run

| | Checked against |
|---|---|
| The four APIs through the OpenAI, Anthropic and Google GenAI SDKs, sync and async, whole and streamed, a LangChain agent, and LiteLLM | The SDKs themselves, on every test run, against a local stand-in for the provider |
| The proxy | Standard-library callers and both SDKs, against the same stand-in |
| OpenAI Agents SDK 0.26, Pydantic AI 2.73, LlamaIndex (OpenAI 0.9), Agno 3.2, AutoGen 0.7, Strands 1.66, Haystack 3.5, Langfuse 5.2, Instructor 1.17 | A conversation of four turns with a search tool on 2026-11-24, against the stand-in: earlier results were archived from the third turn on. Instructor's has no tool results, so it was recorded only |
| Vercel AI SDK 7.0 | Anthropic, Chat Completions and Responses through the proxy, whole and streamed, with the proxy restarted in the middle |
| LiteLLM 1.110 | In process on every test run. Its proxy with the callback, and its Anthropic path through `compache proxy`, by hand |
| Hermes 0.23.3 | The released agent with a local model. The engine, installed with Hermes' own installer: a conversation trimmed once its cache had expired, and a passage fetched back. The proxy, with the settings above |
| Live providers | Chat Completions whole and streamed, Responses and Messages through the SDKs, the proxy and a LangChain agent, on Claude Haiku 5.5, GPT-6 Luna and Gemini 3.5 Flash Lite through OpenRouter. See [the live runs](../benchmarks/live/README.md) |
| Gemini's own API | The stand-in only: OpenRouter does not serve it |
| Claude Code 2.1, Codex 0.172, OpenCode 1.19 | Started with `compache wrap`, on Claude Haiku 5.5 and GPT-6 Luna: a small repository repaired by each. [Sessions of real issues](../benchmarks/agents/README.md) with Claude Code and Codex |
| OpenClaw 2026.11.7, Aider 0.86, CrewAI 1.15 | The released harness through the proxy with a local model, with the settings above. OpenClaw names its sessions itself |
| LangChain 1.5 | An agent on a scripted chat model with the middleware, on every test run: results trimmed, one fetched back, sync and async |
| Gemini CLI, Qwen Code, Goose, Cline | Taken from each one's documentation. Not run here |
