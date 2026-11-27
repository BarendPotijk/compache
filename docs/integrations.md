# Integrations

Choose the place where your application sends its **full conversation history**. Keep the original messages in your application so later requests can be prepared again and archived material can be recovered.

## Where the model calls leave

| If you control… | Use |
|---|---|
| An OpenAI, Anthropic or Google GenAI Python client | [Wrap the SDK client](#sdk-clients) once at startup. |
| A framework that accepts a provider client or an HTTP client | [Inject that client](#python-frameworks) into the framework. |
| A LangChain `create_agent` agent | [Install middleware](#langchain-and-langgraph) and pass the graph thread ID. |
| A custom LangGraph graph | [Wrap the model in its node](#custom-langgraph-node). |
| A LiteLLM application or gateway | [Install its adapter, configure its recording callback or front the gateway](#litellm). |
| A CLI or editor agent | [Launch the harness](agents.md) or [configure an always-on proxy](#the-proxy). Hermes also offers a [context engine](#hermes). |
| A service in another language, or a configurable base URL | [Run the HTTP proxy](#the-proxy). |
| Your own HTTP transport | Call `live.exchange(path, body, send, thread=thread_id)`. Your `send(body)` performs the request and returns a `compache.live.Result`. |
| Your own prompt assembly | Create one `policy.session(thread_id)` per conversation. See the [session lifecycle](sessions.md). |

Install `compache` alongside your application's existing provider or framework packages. Use a fixed policy to control rollouts, a record to observe traffic, or a learning directory to update the policy from traffic. SDK examples below assume a configured provider account, a model available to that account and a policy file for that model. For example:

```sh
python -m pip install compache
compache default --model claude-sonnet-5-5 -o policy.json
```

### SDK clients

Wrap the client at construction and reuse it across requests. Pass your application's stable conversation ID around each turn. For example, in an Anthropic application whose `messages` contain the conversation:

```python
import compache
from anthropic import Anthropic

client = compache.wrap(Anthropic(), "policy.json")
with compache.thread("conversation-17"):
    response = client.messages.create(
        model="claude-sonnet-5-5", messages=messages, max_tokens=1024
    )
print(client.compache.stats["calls"])
```

The same wrapper accepts provider-configured clients. Each row below is a fragment using the application's existing model, history and imports.

| API | Construct once | Call inside the thread block |
|---|---|---|
| OpenAI Chat Completions | `client = compache.wrap(OpenAI(), "policy.json")` | `client.chat.completions.create(model=model, messages=messages)` |
| OpenAI Responses | Same OpenAI client | `client.responses.create(model=model, input=history, store=False)`; send the full input items on each turn. |
| Anthropic Messages | `client = compache.wrap(Anthropic(), "policy.json")` | `client.messages.create(model=model, messages=messages, max_tokens=1024)` |
| Google GenAI | `client = compache.wrap(genai.Client(), "policy.json")` | `client.models.generate_content(model=model, contents=contents)` |

Use `AsyncOpenAI()` or `AsyncAnthropic()` with `await` inside the same ordinary `with compache.thread(...)` block. With Google use the wrapped client's `client.aio.models`. Streaming, parsing and tool definitions stay on the original SDK methods. Existing Azure OpenAI, Anthropic Vertex AI and Anthropic Foundry clients keep their provider configuration.

To **observe first**, use `compache.wrap(client, record="compache-record")`. To **learn and apply**, use `compache.wrap(client, learn="compache-record")`. An explicit `sample=0.1` enables extra model requests for notes on a sampled share of turns. Keep-warm requests can also incur provider charges when a policy enables them. Use `pings=False` on the wrapper to disable background pings. After a completed model request, check `client.compache.stats["calls"]` and `client.compache.tokens`. The `reopened` counter counts tool invocations, including searches that found no match.

Follow the [offline chatbot walkthrough](../examples/README.md#put-a-policy-in-a-chatbot-request) to see the archived fact, the extra provider request and the answer returned to the application.

### Python frameworks

These are constructor fragments for an existing application with its framework and model configured. Create the client once, then pass it where the framework constructs its model.

| Framework | Integration point |
|---|---|
| OpenAI Agents SDK | `set_default_openai_client(compache.wrap(AsyncOpenAI(), "policy.json"))` |
| Claude Agent SDK | `ClaudeAgentOptions(env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787", "ENABLE_TOOL_SEARCH": "true"})`, with `compache proxy` running |
| Google ADK | `LlmAgent(model=Gemini(model=..., base_url="http://127.0.0.1:8787"))`, with `compache proxy` running |
| Pydantic AI | `OpenAIProvider(openai_client=compache.wrap(AsyncOpenAI(), "policy.json"))`, and `AnthropicProvider(anthropic_client=...)` the same way |
| LlamaIndex | `OpenAI(model=..., http_client=compache.http_client(live), async_http_client=compache.async_http_client(live))` |
| Microsoft Agent Framework | `OpenAIChatClient(async_client=compache.wrap(AsyncOpenAI(), "policy.json"), model=...)`, and `default_options={"store": False}` on the agent: otherwise OpenAI keeps the history and there is nothing to trim |
| Semantic Kernel | `OpenAIChatCompletion(ai_model_id=..., async_client=compache.wrap(AsyncOpenAI(), "policy.json"))` |
| Agno | `OpenAIChat(id=..., client=compache.wrap(OpenAI(), "policy.json"))` |
| AutoGen | `OpenAIChatCompletionClient(model=..., http_client=compache.async_http_client(live))` |
| Strands | `OpenAIModel(client=compache.wrap(AsyncOpenAI(), "policy.json"), model_id=...)`, in one event loop |
| Haystack | `generator.warm_up()`, then `generator.client = compache.wrap(generator.client, "policy.json")` |
| Instructor | `instructor.from_openai(compache.wrap(OpenAI(), "policy.json"))` |
| Langfuse's OpenAI drop-in | `compache.wrap(langfuse.openai.OpenAI(), "policy.json")`. Langfuse goes on tracing what your application wrote, in full, so the next fit has everything |

`live` is `compache.Live("policy.json")`. Put `with compache.thread("conversation-17"):` around the run to name its conversation.

### LangChain and LangGraph

For a LangChain agent created with `create_agent`, middleware can prepare the messages for any compatible tool-calling chat model. The checkpointer keeps the original graph messages. The agent can call the middleware's `reopen` tool to recover an archived result. This route makes no independent requests for notes or keep-warm pings.

In your existing application, with its `model`, `tools` and `saver` already configured:

```python
import compache
from langchain.agents import create_agent

live = compache.Live("policy.json")
agent = create_agent(
    model, tools, middleware=[compache.langchain.middleware(live)], checkpointer=saver
)
config = {"configurable": {"thread_id": "conversation-17"}}
agent.invoke({"messages": [{"role": "user", "content": "Find the return window."}]}, config)
```

Reuse the same ID on subsequent invocations. Without it, the middleware forwards the model call unchanged. To put compache in a `ChatOpenAI` transport instead, construct `ChatOpenAI(model=model_name, **compache.langchain.clients(live))` and use `middleware=[compache.langchain.middleware()]` solely to propagate the thread ID. That transport supports provider-level cache controls, keep-warm pings and sampled notes. Use one policy application point per request. [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)

#### Custom LangGraph node

For a hand-built `StateGraph`, configure the model called by the node. The following node fragment uses the application's existing `model_name` and policy, and assumes the graph uses `MessagesState`:

```python
import compache
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

live = compache.Live("policy.json")
model = ChatOpenAI(model=model_name, **compache.langchain.clients(live))

def call_model(state: MessagesState, config: RunnableConfig):
    with compache.thread(str(config["configurable"]["thread_id"])):
        reply = model.invoke(state["messages"], config=config)
    return {"messages": [reply]}
```

Register `call_model` as the graph's model node. Invoke your compiled graph with the same `configurable.thread_id` as above and pass the complete message history to the model inside the node. After invoking, inspect `live.stats["calls"]`. [LangGraph nodes and configuration](https://docs.langchain.com/oss/python/langgraph/graph-api)

### LiteLLM

For an existing LiteLLM application with a model route, credentials and `messages` configured:

```python
import compache
import litellm

live = compache.Live("policy.json")
compache.litellm.install(live)           # every OpenAI-compatible call LiteLLM makes from now on
with compache.thread("conversation-17"):
    litellm.completion(model="openrouter/anthropic/claude-haiku-5.5", messages=messages)
```

Providers that LiteLLM calls over handlers of its own, such as Anthropic's Messages API, go through `compache proxy` with `api_base="http://127.0.0.1:8787"`. A LiteLLM proxy takes compache in front: `compache proxy --policy policy.json --upstream http://localhost:4000`, whatever providers it routes to.

To record traffic in an existing LiteLLM proxy, set the record directory in the environment of the LiteLLM process, then register the callback in its configuration:

```sh
export COMPACHE_RECORD=./compache-record
```

```yaml
litellm_settings:
  callbacks: ["compache.litellm.callback"]
```

Supply the conversation ID as `metadata={"session_id": conversation_id}`. For an in-process callback, use `compache.litellm.Callback(record="compache-record")` alongside your existing callbacks. These configurations record metadata. Enabling `COMPACHE_SAMPLE=0.1` on the proxy opts into model-assisted labels for sampled turns and can send conversation text to the provider in an additional paid request.

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
compache proxy --policy policy.json --record compache-record --port 8787
```

Point the worker or SDK at the proxy, keeping its normal credentials. OpenAI-compatible SDKs typically expect `http://127.0.0.1:8787/v1`; Anthropic's Python client expects `http://127.0.0.1:8787`. Verify the listener before sending traffic, then inspect counters after one application call:

```sh
curl -fsS http://127.0.0.1:8787/compache/health
curl -fsS http://127.0.0.1:8787/compache/stats
```

Health returns `"status": "ok"`. A supported model call increments `calls`; a health probe alone leaves it unchanged. For multiple workers, route each conversation to the same worker. A new process that first sees a conversation midstream assumes the cache expired and may trim immediately. Store records and learned policy files on a persistent volume. Per-process counters start again after restart.

As a container that is always on, it learns in a volume: `docker build -t compache . && docker run -d --restart unless-stopped -p 127.0.0.1:8787:8787 -v compache:/data compache`. Arguments of its own replace that, such as `--policy`.

**Set the upstream before sending a credentialed request.** By default, the standalone proxy forwards to the public OpenAI, Anthropic or Google endpoint selected by request format. For a gateway, Azure, Vertex AI or Foundry deployment, supply its intended origin with `--upstream URL`, or set `--openai-url`, `--anthropic-url` or `--gemini-url` per provider. The proxy appends the incoming request path to the configured base URL. Include each path prefix once. It forwards the client's provider credentials. The optional `--careful` flag checks for unfamiliar key formats before sending them to public endpoints.

Compache recognizes Chat Completions, Responses, Messages and Gemini generation requests on supported paths. Other HTTP requests pass through. Bedrock signs the request body, which prevents rewriting it in transit. Keep the listener on its default loopback address or restrict access at your service boundary. The proxy exposes its health, stats and Prometheus counters (`/compache/metrics`) without authentication.

| Harness | Setting |
|---|---|
| The agents that `wrap` knows, in an editor or always on | `compache wrap --print <agent>` shows what to set. Claude Code: `"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787", "ENABLE_TOOL_SEARCH": "true"}` in `~/.claude/settings.json`, which its editor extensions read too |
| Cline | `cline auth -p openai -b http://127.0.0.1:8787/v1 -m <model>`, or the base URL of an OpenAI Compatible or Anthropic provider in its settings |
| CrewAI | `LLM(model="openai/...", base_url="http://127.0.0.1:8787/v1")`, or `OPENAI_BASE_URL` |
| OpenClaw | In `~/.openclaw/openclaw.json`, a provider under `models.providers` with `baseUrl: "http://127.0.0.1:8787/v1"` and `api: "openai-completions"`, or `baseUrl: "http://127.0.0.1:8787"` and `api: "anthropic-messages"`. Add `compat: { sendSessionAffinityHeaders: true }` to its models so that sessions are named |
| Hermes | In `~/.hermes/config.yaml`, an entry under `providers` with `api: http://127.0.0.1:8787/v1` and `session_affinity_header: x-compache-thread`, selected by `model.provider: custom:<name>` |
| Vercel AI SDK | `createAnthropic({baseURL: "http://127.0.0.1:8787/v1"})`, `createOpenAI(...)` the same. The conversation's name goes in `headers: {"x-compache-thread": id}`. OpenAI's Responses API needs `store: false`: with it on, the SDK sends earlier answers as references, which nothing can trim |
| `ChatAnthropic` | `anthropic_api_url="http://127.0.0.1:8787"` |

The HTTP adapter can answer an internal `reopen` before the application receives the final reply. If the provider rejects a modified request, compache can retry the original body. Use `retried` and `policy_refused` to investigate these cases. [Troubleshooting](troubleshooting.md)

### Naming a conversation

For the SDK wrapper, set `with compache.thread(conversation_id):` around the outgoing call or send the `x-compache-thread` header. The block takes precedence. The proxy first checks `x-compache-thread` and the supported [session headers](../src/compache/proxy.py), then the request body's `prompt_cache_key` or `session_id`. Without an explicit ID the runtime uses conversation history and tool calls to recognize continuations. Use a stable application conversation ID when available.

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

The following table summarizes the project’s published integration checks. Optional SDK tests require their corresponding packages. The scripted coverage for [SDK wrappers](../tests/test_wrap.py), [LangChain middleware](../tests/test_langchain.py), [the launcher](../tests/test_launch.py) and [Hermes](../tests/test_hermes.py) runs locally.

| | Checked against |
|---|---|
| The four APIs through the OpenAI, Anthropic and Google GenAI SDKs, sync and async, whole and streamed, a LangChain agent, and LiteLLM | The SDKs themselves, on every test run, against a local stand-in for the provider |
| The proxy | Standard-library callers and both SDKs, against the same stand-in |
| OpenAI Agents SDK 0.26, Pydantic AI 2.74, LlamaIndex 0.14, Agno 3.2, AutoGen 0.7, Strands 1.66, Haystack 3.5, Langfuse 5.2, Instructor 1.17 | A conversation of four turns with a search tool on 2026-11-25, against the stand-in: earlier results were archived from the third turn on. Instructor's has no tool results, so it was recorded only |
| Claude Agent SDK 0.2, Google ADK 2.15, Microsoft Agent Framework 1.26, Semantic Kernel 1.36, CrewAI 1.15 | An agent in a loop of searches on 2026-11-25, against the stand-in: older results were archived |
| Vercel AI SDK 7.0 | Anthropic, Chat Completions and Responses through the proxy, whole and streamed, with the proxy restarted in the middle |
| LiteLLM 1.111 | In process on every test run. Its proxy with the callback, and its Anthropic path through `compache proxy`, by hand |
| Hermes 0.23.4 | The released agent with a local model. The engine, installed with Hermes' own installer: a conversation trimmed once its cache had expired, and a passage fetched back. The proxy, with the settings above |
| Live providers | Chat Completions whole and streamed, Responses and Messages through the SDKs, the proxy and a LangChain agent, on Claude Haiku 5.5, GPT-6 Luna and Gemini 3.5 Flash Lite through OpenRouter. See [the live runs](../benchmarks/live/README.md) |
| Gemini's own API | The stand-in only: OpenRouter does not serve it |
| Claude Code 2.1, Codex 0.172 | Started with `compache wrap`, on Claude Haiku 5.5 and GPT-6 Luna: [sessions of real issues](../benchmarks/agents/README.md). Claude Code and the Anthropic SDK on Vertex AI and Foundry, against the stand-in |
| OpenClaw 2026.11.10, Aider 0.86 | The released harness through the proxy with a local model, with the settings above. OpenClaw names its sessions itself |
| LangChain 1.5 | An agent on a scripted chat model with the middleware, on every test run: results trimmed, one fetched back, sync and async. Live on Claude Haiku 5.5 with reasoning, in both forms LangChain keeps it, and on GPT-6 Luna |
| Qwen Code 0.28, Goose 1.61, Crush 0.102, Copilot CLI 1.0, Kilo 7.8, OpenCode 1.19, pi 0.73, OpenHands 1.16, Cline 3.0 | Live on 2026-11-25: each repaired a small package under `compache wrap`, on Claude Haiku 5.5 or GPT-6 Luna, Goose and Copilot CLI on both, with a policy that trims after three rounds. Every request was taken, no reasoning was dropped, every module came out right. Gemini CLI 0.71: a loop of reads, against the stand-in |
