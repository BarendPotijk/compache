---
name: compache-setup
description: "Put compache into a codebase or around a coding agent, so that long conversations and agent loops cost less: find where model calls leave, choose the way in, check that it works and say what it saves. Use when asked to lower LLM cost, trim context, keep a prompt cache warm, or add compache."
---

# Put compache into an application

compache goes in where model calls leave the process. It sends less of a long conversation, at the moments when that is cheap, and keeps the provider's cache warm while that pays. It learns both from the traffic itself. There is nothing to configure, so do not add thresholds or settings.

## 1. Find where the model calls leave

Look for SDK clients (`OpenAI(`, `Anthropic(`, `genai.Client(`, their async forms), chat models of a framework, `litellm.completion`, and base URLs in settings. One client may be made in one place and used in many: put compache in where it is made.

## 2. Choose the way in

| What the code has | What to change |
|---|---|
| A client of the OpenAI, Anthropic or Google GenAI SDK | `client = compache.wrap(client, learn="compache-record")` where it is made |
| A LangChain or LangGraph agent | `middleware=[compache.langchain.middleware(compache.Live(learn="compache-record"))]` in `create_agent`, and a checkpointer with a `thread_id` |
| OpenAI Agents SDK, Pydantic AI, LlamaIndex, Agno, AutoGen, Strands, Haystack, Instructor | Hand the framework the wrapped client. One line each in `docs/integrations.md` |
| LiteLLM in process | `compache.litellm.install(compache.Live(learn="compache-record"))` |
| A LiteLLM proxy or another gateway | `compache proxy --learn compache-record --upstream <the gateway>` in front of it |
| Anything else with a base URL, in any language | `compache proxy --learn compache-record`, and the base URL set to `http://127.0.0.1:8787/v1` (Anthropic: without `/v1`) |
| A coding agent on the command line | `compache wrap claude`, `codex`, `opencode`, `gemini`, `qwen`, `copilot`, `aider`, `goose`, `crush`, `kilo`, `pi`, or `compache wrap -- <command>` |

Then:

- Add `compache` to the dependencies (`pip install compache`, Python 3.11 or later, no dependencies of its own).
- Add the record folder to `.gitignore`. It holds sizes, times and usage, and no text.
- Name each conversation where the application has a name for it: `with compache.thread(conversation_id):` around the call, or the header `x-compache-thread`. Without a name compache recognizes a conversation by what it has said and by its tool calls.
- Keep sending the whole conversation. If the application cuts its own history (a sliding window, dropped tool results, a summary at a threshold), say so to the user before removing it: both at once defeat each other.

## 3. Check it

```sh
compache doctor                       # a request goes through, and what is installed here
python your_app.py                    # a conversation of several turns or tool calls
compache stats compache-record        # what it cost, and what it would have
```

Short conversations save nothing, and that is right: a first cut comes when old results outweigh what follows them, or when a user returns to an expired cache.

## 4. Chats with retrieved text

An agent's policy needs nothing but the traffic. A chat that retrieves text once per message is fitted on labels, which the model can give on a sample of turns:

```python
client = compache.wrap(client, record="compache-record", sample=0.1)   # some 600 messages
```

```sh
compache fit compache-record -o policy.json
```

```python
client = compache.wrap(client, "policy.json")
```

Past traces in Langfuse, MLflow or an OpenTelemetry export do as well: `compache pull langfuse -o threads.jsonl.gz`, then `compache fit threads.jsonl.gz`. The skill `compache-label-dependencies` writes labels for those.

## 5. Report

Tell the user which call sites changed, what `compache stats` or `compache fit` says in dollars, and what is not covered: calls that go around the changed client, and conversations that another part of the application trims.
