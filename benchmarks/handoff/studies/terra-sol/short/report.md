# Terra and Sol on short public conversations

Mean total cost per user run in USD. Each policy answered the same questions with low reasoning effort. Compache used eight training conversations from the same benchmark to fit each policy. 0 of 14 MemoryArena Progressive Search evaluation questions also appeared in those training conversations.


| Model | Benchmark | Cache | Compache (fitted) | OpenCode | Headroom* |
|---|---|---:|---:|---:|---:|
| GPT-6.1 Terra | MemoryArena Progressive Search | 5 min | $0.02111 | $0.02122 | $0.02149 |
| GPT-6.1 Terra | MemoryArena Progressive Search | 30 min | $0.00827 | $0.01624 | $0.01648 |
| GPT-6.2 Sol | MemoryArena Progressive Search | 5 min | $0.04129 | $0.04144 | $0.04143 |
| GPT-6.2 Sol | MemoryArena Progressive Search | 30 min | $0.01646 | $0.03180 | $0.03180 |

| Model | Benchmark | Cache | Scored questions | Compache | OpenCode | Headroom* |
|---|---|---:|---|---:|---:|---:|
| GPT-6.1 Terra | MemoryArena Progressive Search | 5 min | Original final target | 1/2 | 1/2 | 1/2 |
| GPT-6.1 Terra | MemoryArena Progressive Search | 5 min | Direct questions (all gradable turns) | 9/14 | 9/14 | 8/14 |
| GPT-6.1 Terra | MemoryArena Progressive Search | 30 min | Original final target | 1/2 | 1/2 | 1/2 |
| GPT-6.1 Terra | MemoryArena Progressive Search | 30 min | Direct questions (all gradable turns) | 9/14 | 9/14 | 8/14 |
| GPT-6.2 Sol | MemoryArena Progressive Search | 5 min | Original final target | 1/2 | 1/2 | 1/2 |
| GPT-6.2 Sol | MemoryArena Progressive Search | 5 min | Direct questions (all gradable turns) | 9/13 (1 ungraded) | 9/14 | 9/13 (1 ungraded) |
| GPT-6.2 Sol | MemoryArena Progressive Search | 30 min | Original final target | 1/2 | 1/2 | 1/2 |
| GPT-6.2 Sol | MemoryArena Progressive Search | 30 min | Direct questions (all gradable turns) | 9/13 (1 ungraded) | 9/14 | 9/13 (1 ungraded) |

The matched sample covers MemoryArena Progressive Search (2 conversations, 14 user runs). Costs include retries, summaries, tool follow-ups and keep-warm pings. We recorded **231 distinct API responses**, priced at **$1.29824** in total at standard API rates.

Across physical trajectories the host made 0 overflow summaries, 88 proactive pings and 31 tool rounds. 0 foreground runs had an archived source. These short cases have no independently labeled set of cross-run dependencies.

User arrivals and foreground durations on the replay clock are synthetic draws shared by the policies. The API ran back to back. Compache ran a separate trajectory for each expiry. OpenCode and Headroom each ran once per model and case, and both expiry estimates reuse that trajectory. The five-minute window is a counterfactual for these models.

Earlier eight-thread workload fit per dataset, transferred with each stronger model's own context and prices. On MemoryArena Progressive Search, 0 of 14 evaluation questions exactly matched a training question after text normalization (arena-006 0/5, arena-029 0/9).

MemoryArena uses blind rubric-based grading, with its released final target scored separately. The direct-question score includes gradable terminal turns.

The two blind reviewers agreed on 104 of 107 distinct answers before adjudication. The frozen rubric excluded 3 unverified attributions to García Márquez from direct accuracy. The table identifies ungraded answers by policy.

The short GoodAI comparison was stopped after repeated questions were found in the fit. The experimental ledger records $42.91110 across all Terra and Sol requests. This includes $1.29824 for the matched short MemoryArena comparison, $2.38575 for excluded short GoodAI trajectories and $39.22711 for the separate long checkpoints.

An offline replay reproduced all 231 API request fingerprints using the public prompts, source packets, prior model replies and fitted policy parameters. Replacing evaluator-only answers and scoring fields with a sentinel left those requests unchanged.

\* Headroom 0.44.0 used its Responses cache handler with compression disabled. The host used OpenCode v1.19.5 overflow handling for both controls.

[Numeric results](summary.json) · [GoodAI benchmark](https://github.com/GoodAI/goodai-ltm-benchmark) · [MemoryArena release](https://huggingface.co/datasets/ZexueHe/memoryarena) · [Terra API rates](https://developers.openai.com/api/docs/models/gpt-6.1-terra) · [Sol API rates](https://developers.openai.com/api/docs/models/gpt-6.2-sol)
