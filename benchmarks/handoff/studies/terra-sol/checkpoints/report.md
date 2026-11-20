# Terra and Sol checkpoints · same-benchmark development comparison

24/24 first answers recorded. Physical API cost for these checkpoints is **$39.22711**.

Development comparison. Compache was fitted to other threads from these same benchmarks, before the held-out conversations were replayed. Each model received one previously compacted prompt per policy and held-out conversation. GoodAI uses its first native name-recall probe. MemoryArena uses the final search question. Each benchmark supplies two conversations. The price is for one new answer at low reasoning effort with a 1,536-token output cap.

## GoodAI LTM

| Model | Policy | Calls | USD / answer | Mean seconds | Cache reads | Cache writes | Correct / graded | Dependencies kept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Terra | Compache | 2/2 | $1.90667 | 13.7 | 0 | 1,525,028 | 2/2 | 10/10 |
| Terra | OpenCode | 2/2 | $0.01663 | 3.3 | 0 | 26,181 | 0/2 | 0/10 |
| Terra | Headroom | 2/2 | $0.01628 | 2.8 | 0 | 25,795 | 0/2 | 0/10 |
| Sol | Compache | 2/2 | $3.81321 | 6.6 | 0 | 1,525,031 | 2/2 | 10/10 |
| Sol | OpenCode | 2/2 | $0.03287 | 1.9 | 0 | 26,175 | 0/2 | 0/10 |
| Sol | Headroom | 2/2 | $0.03232 | 2.0 | 0 | 25,793 | 0/2 | 0/10 |

## MemoryArena Progressive Search

| Model | Policy | Calls | USD / answer | Mean seconds | Cache reads | Cache writes | Correct / graded | Dependencies kept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Terra | Compache | 2/2 | $1.47642 | 19.2 | 0 | 1,179,081 | 0/2 | 1/3 |
| Terra | OpenCode | 2/2 | $1.47414 | 12.3 | 0 | 1,178,425 | 1/2 | 1/3 |
| Terra | Headroom | 2/2 | $1.64720 | 12.3 | 0 | 1,317,025 | 0/2 | 1/3 |
| Sol | Compache | 2/2 | $2.95421 | 15.9 | 0 | 1,179,076 | 1/2 | 1/3 |
| Sol | OpenCode | 2/2 | $2.94896 | 10.0 | 0 | 1,178,427 | 1/2 | 1/3 |
| Sol | Headroom | 2/2 | $3.29464 | 10.4 | 0 | 1,317,027 | 1/2 | 1/3 |

Read and write columns are actual provider token totals over the listed calls. Elapsed time covers the single foreground request. Physical cost includes output and default cache writes; each request uses a separate routing namespace.

The original Luna runs supplied history, summaries, prior answers, and retained sources at their 30-minute policy decisions. Terra and Sol answered those fixed prompts with recovery tools disabled. The dependency count describes source availability in the original layout before recovery. These prices exclude its earlier model calls, compactions and pings. There is no simulated five-minute or 30-minute bill in this test.

GoodAI uses the frozen native grader. MemoryArena uses the original terminal answer and a policy-blind semantic grade. Unresolved grades stay pending. Each quality estimate covers at most two questions.

[Aggregate numeric results](summary.json)
