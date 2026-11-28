# Coding agents

Use `compache wrap` to run an installed coding agent through a local proxy. The agent keeps its conversation history. On each model call, compache uses that history and the policy to decide which earlier tool results to send in full.

## Start a session

Use Python 3.11 or later and an agent that already works with your provider or gateway. To observe a Claude Code session first:

```sh
python -m pip install compache
compache doctor
compache wrap --watch claude
```

Complete a task in the launched agent and exit it. Then check that model calls were recorded:

```sh
compache stats claude --json
```

Look for your model and a positive `calls` count. The report counts calls with provider usage. It prices those calls and estimates what the same traffic would have cost with the full history under the provider’s default cache lifetime. If the launcher reports that no model calls came through, check the agent's effective endpoint with `compache wrap --print claude` and specify `--upstream` for a custom gateway. For the next session, run `compache wrap claude` to apply the learning policy. To use a fixed policy instead, run `compache wrap --policy policy.json claude`. The agent's own credentials and billing arrangement apply. [Troubleshooting](troubleshooting.md)

![Cost per 1,000 model calls on Claude Sonnet 5.5 for one task at a stretch, for a few steps with a 4-minute break, and for three tasks with 4-minute breaks: everything, cached at its best, and compache](assets/agents-cost.svg)

If you want a coding agent to configure compache in an application, install the [setup skill](../skills/compache-setup/SKILL.md) with `npx skills add BarendPotijk/compache`. In Claude Code, use `/plugin install compache --marketplace BarendPotijk/compache`.

## Launch options

| Command | When to use it |
|---|---|
| `compache wrap claude` | Claude Code. Also `codex`, `opencode`, `gemini`, `qwen`, `aider`, `goose`, `crush`, `copilot`, `kilo`, `pi`, `openhands` |
| `compache wrap codex -- exec "fix the failing test"` | With arguments of its own |
| `alias claude='compache wrap claude --'` | Every session, from your shell's profile |
| `compache wrap -- my-agent` | Any other command. It is told by the variables that SDKs read |
| `compache wrap --watch claude` | Record only, and change nothing |
| `compache wrap --print claude` | Show the proxy to start and what to set, for an agent in an editor or one that is always on |
| `compache learn claude --from claude` | Start from the sessions Claude Code keeps on this machine. Also `codex` |
| `compache stats` | What was saved, per agent and model. For Claude Code's status line: `"statusLine": {"type": "command", "command": "compache stats claude --line"}` in its settings |
| `compache doctor` | Whether it works here, and what it can be used with |

`wrap` starts a proxy on a free local port, starts the agent with its base URL set to it, and says at the end what the session cost and what it would have cost. No file of the agent's is changed, and a session that began with compache goes on without it, or the other way round. compache's own messages go to `~/.compache/<agent>/compache.log` while the agent has the terminal. Where the agent's own settings or command line name a server, such as a gateway, requests go on to that. Where a project's file names another one, or a key is none of the provider's own, nothing is sent until `--upstream URL` says where. Gemini CLI takes a base URL once its settings say how it signs in (`/auth`). On a subscription nothing is billed by the token: what is saved there is room in the window and in the plan's limits.

## What it does

| Moment | What happens |
|---|---|
| A tool returns | Its result goes out in full, and stays for as long as it is one of the newest |
| Older results outweigh what follows them, or the request has reached a size that is billed dearer | They become a notice, which says that they were read in full and how to see them again. Rewriting what follows pays for itself in the next calls |
| The cache has expired | They become a notice at once: everything has to be paid for again anyway |
| What the model reasoned before | Goes with the results of its time. It is sent again with every call and takes as much room as they do. Claude only takes it back behind what it was made behind, so a cut takes all that stands behind it |
| The model needs one after all | It calls `reopen`. The agent never sees the call, and what it fetched stays in view until it is old. Once it fetches the same again, or goes round with tools of its own for what it no longer sees, results stay twice as long in this conversation |
| A tool takes longer than the cache lives | If the policy schedules a ping, a short-output request attempts to refresh the cached prefix |
| Somebody is back after a break | Shared instructions and tools can still hit a longer-lived cache entry when the provider supports it and the policy selected that lifetime |

How many rounds of results stay, when a cut pays, how many pings and which lifetimes are learned. A cut leaves a cache mark where the next one will begin, so the provider still reads what comes ahead of it.

## How it learns

- **It starts from public runs.** The policy for the first session was fitted on 174 coding-agent runs and keeps the newest 8 rounds in full.
- **It goes on from yours.** After some 200 calls, and again whenever that has grown by half, it is fitted on what is on record in `~/.compache/<agent>`, per model, in the seconds after a session. `compache learn claude --from claude` adds the sessions Claude Code or Codex kept before, so that it need not wait.
- **No labels.** A reply that repeats a line which only a tool had returned shows which result it relied on. Those lines are held as hashes while the conversation runs, and the record keeps which result it was, never the line.
- **Control conversations.** A policy that hides results hides that they were needed. So one conversation in ten goes out as the agent wrote it. Those say what replies rely on, and what the others save.
- **No text.** The record holds sizes, times and usage. Conversations go by a hash.

The default fit requires 80% of training calls known to rely on earlier results to find them in view. Set `--min-retention` on `compache fit` to choose another minimum. Each observed miss is priced as fetching the result back.

## In your own agent

Use the same integration as the application that owns the model call:

| Your agent | Integration |
|---|---|
| Uses an OpenAI, Anthropic or Google GenAI Python client | `compache.wrap(client, learn="compache-record")`. Reuse the returned client. [SDK examples](integrations.md#sdk-clients) |
| Uses LangChain's `create_agent` or a custom LangGraph node | Configure the [agent middleware or node's HTTP client](integrations.md#langchain-and-langgraph) and propagate the graph thread ID. |
| Accepts a base URL | Start `compache proxy --learn compache-record` and [configure the harness](integrations.md#the-proxy). |
| Builds prompts itself | Create a session per task, use `session.prepare` to start a turn, `session.replied` and `session.extend` for tool rounds, and `session.complete` for the final answer. [Session lifecycle](sessions.md) |

To control when policies change, record with `record="compache-record"`, run `compache fit compache-record -o policy.json`, and give the resulting file to your integration. Tool loops can supply evidence of which results subsequent replies used. To inspect the starting agent policy, run `compache default --agent --model claude-sonnet-5-5`.

## What it saves

![Long Claude Code sessions on Claude Haiku 5.5, the bill after the last prompt. One task in 13 prompts: alone $0.33, Headroom $0.40, Kompressor $0.28, compache $0.13. 16 issues in one conversation: alone $0.98, compache $0.15. Alone and compache are averages of two runs](assets/agents-live.svg)

The published live runs measured 51% to 85% less on long sessions and 16% less on sessions of some fifty calls, with as many solved. [Agent benchmarks](../benchmarks/agents/README.md): real sessions with Claude Code and Codex next to Headroom and Kompressor, and the numbers below.

The table below reports modeled savings on the public runs, fitted on half of them and priced on the other half, against sending everything on the provider's default cache:

| | Claude Sonnet 5.5 | Claude Haiku 5.5 | GPT-6 Luna |
|:--|:-:|:-:|:-:|
| One task at a stretch, 64 calls | 5% | 17% | 8% |
| Prompts of eight calls, back after 30 seconds | 7% | 20% | 9% |
| Prompts of eight calls, back after 4 minutes | 29% | 32% | 10% |
| Prompts of eight calls, back after 15 minutes | 43% | 44% | 18% |
| Three tasks in a session, back after 4 minutes | 23% | 80% | 28% |
| Three tasks in a session, back after 30 minutes | 30% | 81% | 30% |

- **Breaks are where most of it is.** In the eight-call-per-prompt Sonnet scenario with four-minute breaks, optimizing caching alone saves about 25% against the default cache. Adding compaction brings the total saving to 29% against the same baseline.
- **Long sessions on Haiku** pass 100,000 tokens, where its list price is five times higher. A cut keeps them below.
- In the ten-call benchmark, the recovery tool added about 1% to cost and the session ended before compaction helped.
- In the public-run replay, about one call in 85 needed an archived result.
