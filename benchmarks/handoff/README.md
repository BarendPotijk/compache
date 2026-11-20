# Long-chat evidence for the Compache handover

[Synthetic corpus and exact policy fit](fit/README.md) · [Worked fixture](example/README.md) · [Earlier studies](studies/README.md) · [Public source adapters](../public_chat/README.md) · [Runnable planner PR](https://github.com/BarendPotijk/compache/pull/3)

[aggregate.json](aggregate.json) contains grouped measurements for two long GoodAI LTM conversations and two MemoryArena Progressive Search conversations. All were development workloads. The displayed Compache arm fitted its policy on 200 synthetic RAG conversations from 20 companies. This comparison uses historical GPT-6 Luna runs at low reasoning effort. Each control ran once per case. Compache ran a separate trajectory at each cache window.

From a fresh checkout, run the verifier with Python 3.10 or later. It uses the standard library, runs offline, and requires no credentials.

```sh
python3 benchmarks/handoff/verify.py
```

It prints the twelve comparison cells and the two MemoryArena case diagnoses. It independently reprices each published token group with the study tariff, reconciles the costs to the archived aggregate results, and checks turn counts, original-source retention, grades, compaction and tool rounds.

## Contents and scope

| Field in the aggregate | What it contains |
|---|---|
| `physical` | 38 billing groups over 459 distinct settled responses. The controls' receipts appear once. Both paid Compache trajectories count. Total physical usage valued at the study's list prices was **$21.423110690**. |
| `modeled` | 73 billing groups over 685 response-window allocations. For each case, policy and assumed five- or 30-minute lifetime, the original study replayed eligible reads and writes on its simulated arrival clock, including the tool-choice correction. The controls' different windows reuse the same physical trajectory. |
| `trajectories` | 16 physical trajectories, with one row of outcome counts per user turn. Compache uses one window per trajectory; controls apply to both. The 51 distinct questions correspond to 204 physical turn observations and 306 window-specific turn observations. |
| `reference` | Totals from the original study's aggregate score tables for the twelve suite/policy/window combinations. Each includes the number of turns, modeled costs and tokens, pre-recovery retention, host summaries, discretionary cuts, tool rounds and graded answer counts. |
| `definitive_rejections` | One definitively rejected provider request in this slice. Its response yielded no billable token receipt. |

Financial groups combine calls by source case alias, policy, tariff band and call role, and by expiry window for modeled charges. Retries contribute to the role of the original request. Any role with fewer than three calls in a stratum is combined with the largest role into `combined`. Every exported billing group includes at least three calls. The publication contains grouped token and cost totals, public case aliases, and task-level outcomes. It contains no individual response records, provider or document identifiers, raw fingerprints, latency traces, prompt text, answer text or tool arguments.

The recorded study tariff in USD per million tokens is $0.10 for fresh input, $0.01 for cache reads, $0.125 for writes, and $0.50 for output. The long tier applies if the original request exceeded 272,000 input tokens. For this tier input rates double and output rates multiply by 1.5. Within each preassigned tariff band the price is linear, so grouped counts suffice to recompute money. The public result can check that arithmetic and each published benchmark total. Determining individual request tier assignment, simulating a new expiry window, or independently reconstructing the original cache correction requires the original research workbench.

## Why MemoryArena cost more at 30 minutes

| Source case | Policy | Tool rounds | Rounds before first host summary | Invalid retrieve calls asking for unseen IDs | Source requirements kept / assessed |
|---|---|---:|---:|---:|---:|
| arena-080 | Compache | 45 | 41 | 42 | 4 / 10 |
| arena-080 | Headroom | 20 | 17 | 18 | 6 / 10 |
| arena-090 | Compache | 25 | 14 | 9 | 15 / 32 |
| arena-090 | Headroom | 29 | 27 | 26 | 15 / 32 |

Each count of unseen IDs compares a retrieval request with documents introduced by the current user turn. Tool outcomes are grouped by case, policy and phase relative to the first host summary. In arena-080, Compache's 41 earlier rounds comprised 38 invalid lookups and three exhausted tool budgets. There were zero proactive pings on Compache's two 30-minute MemoryArena trajectories and no discretionary cuts. On arena-090 it used four fewer rounds than Headroom. One realization per condition measures the observed difference; it leaves uncertainty about the cause of the model's extra work.

`kept` counts a requirement whose complete original source set or an accepted alternative was still visible before recovery. The two cases contributed 10 and 32 assessable historical requirements and 11 unresolved entries each. In GoodAI every arm retained 2/19. GoodAI scores native delayed probes and supplemental probes separately. MemoryArena scores verified intermediate clues and the original final targets separately. An ungraded turn carries a null grade. Headroom 0.44.0 used Responses cache mode with compression disabled. OpenCode's overflow summary ran in the shared host.

## Provenance

[export.py](export.py) builds only this grouped release. It reads the archived Luna study's two settled receipt ledgers, clock scenarios, corrections, held-out fixtures, saved grades and suite comparisons. If that independent local research archive is available, the following command regenerates the aggregate to a new folder:

```sh
python3 benchmarks/handoff/export.py /path/to/outputs/tuned-chat /tmp/compache-benchmark-aggregate
```

The bundled verifier and aggregate suffice for a fresh-checkout accounting check. The [synthetic example](example/README.md) has separate fit, model-visible and evaluator inputs for testing an adapter. Its fictional events contribute no measurements to this evidence. The benchmark sources are [GoodAI LTM](https://github.com/GoodAI/goodai-ltm-benchmark) and [MemoryArena](https://github.com/ZexueHe/MemoryArena).
