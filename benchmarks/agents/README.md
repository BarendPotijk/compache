# Agents

What an agent's calls cost with and without compache: live with Claude Code and Codex next to Headroom and OptScale Kompressor, and modeled on public runs.

![Long Claude Code sessions on Claude Haiku 5.5, the bill after the last prompt. One task in 13 prompts: alone $0.33, Headroom $0.40, Kompressor $0.28, compache $0.13. 16 issues in one conversation: alone $0.98, compache $0.15. Alone and compache are averages of two runs](../../docs/assets/agents-live.svg)

## Live

| | Alone | Headroom | Kompressor | compache |
|:--|:-:|:-:|:-:|:-:|
| **One task in 13 prompts**, Claude Code | $0.41, $0.25 | $0.40 | $0.28 | **$0.13, $0.12** |
| Hidden tests passed | 14 of 14 | 14 of 14 | 14 of 14 | 14 of 14 |
| Largest request | 143k, 118k | 142k | 138k | 100k, 88k |
| **17 issues in one conversation**, Claude Code | | | | |
| After 8 prompts | $0.35, $0.08 | $0.14 | $0.40 | **$0.08, $0.05** |
| Right, of 8 | 7, 5 | 7 | 8 | 8, 6 |
| After 12 prompts | $0.76, $0.47 | | $0.95 | **$0.12, $0.09** |
| Right, of 12 | 9, 6 | | 10 | 9, 7 |
| After 16 prompts | $1.20, $0.76 | | | **$0.19, $0.12** |
| Right, of 16 | 11, 7 | | | 11, 8 |
| **Sessions of five or six issues**, Claude Code | $0.059 | $0.056 | $0.055 | **$0.050** |
| Right, of 34 | 26 | 27 | 22 | 24 |
| **The same**, Codex | $0.033 | $0.029 | $0.038 | $0.029 |
| Right, of 34 | 20 | 20 | 19 | 21 |
| **One task in 13 prompts**, Codex | $0.058 | | | $0.063 |
| Calls, hidden tests passed | 56, 14 of 14 | | | 66, 14 of 14 |

- **Long sessions are where it pays.** 67% and 51% less on the task, and 85% and 84% less on sixteen prompts of the day. Requests stay near 100,000 tokens, above which Haiku costs five times as much. Alone they grew to between 118,000 and 184,000. How soon they get there is the agent's own choice, so the same work alone costs $0.25 or $0.41, $0.76 or $1.20.
- **Short sessions: 16% less with Claude Code.** 19% once the policy had learned from sessions on other issues ($0.048, 24 right). With Codex 10% less, as with Headroom, and within chance: an agent makes 24 to 105 calls for the same session, and that moves the bill more than any tool does.
- **With Codex there is less to gain.** GPT-6 Luna reads 97% of this input from its cache, at a tenth of the price, keeps it for half an hour, and charges more for a request only from 272,000 tokens on. On the long task compache cut nothing, and the two runs differ by the calls the agent chose to make. The best cut in hindsight would have saved 14%.
- **The same calls, priced.** What Claude Code did alone would have cost 26% less under compache, and what Codex did 13%: their own calls, one by one, with the policy that `wrap` starts with.
- **Answers hold up.** No difference in right answers is beyond chance (p = 0.43 to 1).

## Modeled

174 runs of [SWE-rebench with OpenHands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (CC-BY-4.0), 64 calls each. The policy is fitted on every second run and priced on the others, by the provider's list prices and cache rules, fetching back included. The runs carry no times: a call takes eight seconds, a tool three, and a user is back after a while that is drawn around the median.

![Cost per 1,000 model calls on Claude Sonnet 5.5 for three ways of working](../../docs/assets/agents-cost.svg)

| Less than everything on the default cache | Claude Sonnet 5.5 | Claude Haiku 5.5 | GPT-6 Luna |
|:--|:-:|:-:|:-:|
| One task at a stretch | 5% | 17% | 8% |
| Three tasks, back after 30 seconds | 20% | 79% | 28% |
| Three tasks, back after 4 minutes | 23% | 80% | 28% |
| Three tasks, back after 30 minutes | 30% | 81% | 30% |
| Eight calls a prompt, back after 30 seconds | 7% | 20% | 9% |
| Eight calls a prompt, back after 4 minutes | 29% | 32% | 10% |
| Eight calls a prompt, back after 15 minutes | 43% | 44% | 18% |

85% to 96% of the calls that rely on an earlier result find it in view, and 0.4% to 1.5% of all calls fetch one back. Haiku's list price is five times higher for a request above 100,000 tokens, which three tasks in a row reach.

## How the live runs are made

| | |
|---|---|
| Issues | [SWE-bench Lite](https://www.swebench.com)'s sympy issues that run without a container: kept where the fix's tests fail before and pass after on this machine, graded by those tests |
| A session | Five or six issues in one checkout, one prompt each, and between them the user is away for 60, 20, 420, 120 and 780 seconds |
| A long session | A generated package (`tasks.py`): fourteen modules to correct, then to test, review, document and check against random inputs, in thirteen prompts. Graded by hidden tests |
| A day | All seventeen issues in one conversation, with breaks of 20 to 420 seconds. Runs that end early had spent their money: the first alone after 16 prompts, Kompressor's after 12 and Headroom's after 8 |
| Models | Claude Haiku 5.5 in Claude Code 2.1.331, GPT-6 Luna in Codex 0.172.0, through OpenRouter, billed on 2026-11-24 and 2026-11-25 |
| Runs | Two of each session, and of the task and the day alone and with compache in Claude Code. One of the others. A prompt may take 15 minutes |
| compache | What `compache wrap` starts, learning from nothing. Once learned: after `compache learn`, on the sessions of the first column on other tasks |
| Headroom 0.45.0, Kompressor 1.6.1 | Their proxies, with their defaults and with the settings the package ships |

Agents and tools hold no key: the last server on the way puts it in and reads the bill. They run in a sandbox that reaches this machine only.

```sh
python benchmarks/agents/public.py fetch runs && python benchmarks/agents/public.py price runs --output docs/assets/data/agents.json
python scripts/figures.py agents

python benchmarks/agents/swe.py swe                              # needs swe/venv: Python 3.9 with mpmath 1.3 and pytest
python benchmarks/agents/run.py play --agent claude --swe swe --session --repeats 2 --env .env --headroom headroom --kompressor kompressor
python benchmarks/agents/run.py play --agent claude --long --tasks 2 --env .env
python benchmarks/agents/run.py play --agent claude --swe swe --day --tasks day --pause 60 20 420 45 330 90 --env .env
python benchmarks/agents/run.py report
python benchmarks/agents/run.py priced --agent claude            # the first arm's sessions, priced under compache
```
