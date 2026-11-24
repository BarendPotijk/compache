# Agents

```sh
pip install compache
compache wrap claude
```

An agent makes dozens of model calls per message, and every call sends every tool result so far. compache goes between the agent and the model, keeps in view what its next steps rely on, and keeps the cache warm while a tool runs. The agent itself is left as it is.

![Cost per 1,000 model calls on Claude Sonnet 5.5 for one task at a stretch, for a few steps with a 4-minute break, and for three tasks with 4-minute breaks: everything, cached at its best, and compache](assets/agents-cost.svg)

| | |
|---|---|
| `compache wrap claude` | Claude Code. Also `codex`, `opencode`, `gemini`, `qwen`, `aider`, `goose` |
| `compache wrap codex -- exec "fix the failing test"` | With arguments of its own |
| `compache wrap -- my-agent` | Any other command. It is told by the variables that SDKs read |
| `compache wrap --watch claude` | Record only, and change nothing |
| `compache wrap --print claude` | Show the proxy to start and what to set, to do it yourself |
| `compache stats` | What was saved, per agent and model |
| `compache doctor` | Whether it works here, and what it can be used with |

`wrap` starts a proxy on a free local port, starts the agent with its base URL set to it, and says at the end what the session cost and what it would have cost. No file of the agent's is changed. A base URL that is set already, such as a gateway's, is where requests go on to. Credentials pass through. On a subscription nothing is billed by the token: what is saved there is room in the window and in the plan's limits.

## What it does

| Moment | What happens |
|---|---|
| A tool returns | Its result goes out in full, and stays for as long as it is one of the newest |
| Older results outweigh what follows them, or the request has reached a size that is billed dearer | They become a notice. Rewriting what follows pays for itself in the next calls |
| The cache has expired | They become a notice at once: everything has to be paid for again anyway |
| What the model reasoned in turns that are over | Goes with the results of its time. It is sent again with every call and takes as much room as they do |
| The model needs one after all | It calls `reopen`. The agent never sees the call, and what it fetched stays in view until it is old |
| A tool takes longer than the cache lives | The same request goes out once more with next to no output, which keeps the cache |
| Somebody is back after a break | The instructions and tools that open every request are read from an entry that lives an hour |

How many rounds of results stay, when a cut pays, how many pings and which lifetimes are learned. A cut leaves a cache mark where the next one will begin, so the provider still reads what comes ahead of it.

## How it learns

- **It starts from public runs.** The policy for the first session was fitted on 174 coding-agent runs: the newest 8 rounds in full on models that read their cache at a twentieth of the price, 12 where they read at a tenth.
- **It goes on from yours.** After some 200 calls, and again whenever that has grown by half, it is fitted on what is on record in `~/.compache/<agent>`, per model, in the seconds after a session.
- **No labels.** A reply that repeats a line which only a tool had returned shows which result it relied on. Those lines are held as hashes while the conversation runs, and the record keeps which result it was, never the line.
- **It stays honest.** A policy that hides results hides that they were needed. So one conversation in ten goes out as the agent wrote it. Those say what replies rely on, and what the others save.
- **No text.** The record holds sizes, times and usage. Conversations go by a hash.

At least 85% of the calls that rely on an earlier result find it in view. The others are priced as fetching it back.

## In your own agent

```python
client = compache.wrap(Anthropic(), learn="compache-record")        # or OpenAI(), genai.Client(), async clients
```

```python
live = compache.Live(learn="compache-record")
agent = create_agent(model, tools, middleware=[compache.langchain.middleware(live)], checkpointer=saver)
```

```sh
compache proxy --learn compache-record                              # anything else with a base URL
```

Every way in of the [integrations](integrations.md) takes `learn`. To decide yourself when a policy changes, record with `record="compache-record"`, fit with `compache fit compache-record -o policy.json` and give the file: an agent's traffic is recognized by its loops, and needs no labels. `compache default --agent --model claude-sonnet-5-5` writes the policy that learning starts from.

With your own prompt assembly, a session decides call by call:

```python
session = policy.session("task-17")
turn = session.prepare(question)
...
session.replied(turn, reply_text, tool_arguments, usage=response.usage)
turn = session.extend(turn, tool_results)          # one round of tool calls. turn.archived_ids went out as a notice
...
session.complete(turn, answer, usage=response.usage)
```

## What it saves

Modeled on the public runs, fitted on half of them and priced on the other half, against sending everything on the provider's default cache.

| | Claude Sonnet 5.5 | Claude Haiku 5.5 | GPT-6 Luna |
|:--|:-:|:-:|:-:|
| One task at a stretch, 64 calls | 5% | 15% | 9% |
| Prompts of eight calls, back after 30 seconds | 8% | 16% | 10% |
| Prompts of eight calls, back after 4 minutes | 30% | 31% | 11% |
| Prompts of eight calls, back after 15 minutes | 45% | 42% | 19% |
| Three tasks in a session, back after 4 minutes | 27% | 80% | 31% |
| Three tasks in a session, back after 30 minutes | 32% | 81% | 33% |

- **Breaks are where most of it is.** Part of that is caching alone: pings and the right lifetimes save 26% of the 30% in the third row on Sonnet. Trimming is the other part, and grows with the length of a session.
- **Long sessions on Haiku** pass 100,000 tokens, where its list price is five times higher. A cut keeps them below.
- **Short sessions save nothing.** A task that is done in ten calls has nothing to trim, and compache's tool costs it 1%.
- **About 1 call in 80** fetches something back.

[Agent benchmarks](../benchmarks/agents/README.md): these numbers, and real sessions with Claude Code and Codex next to Headroom and Kompressor.
