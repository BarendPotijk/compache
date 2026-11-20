# Research and support walkthrough

The same five-minute cache produces different decisions on these two invented workloads. Run either demo with an installed copy of Compache. Each demo generates its own counts locally. Your trace store and model credentials are unnecessary.

```sh
compache setup --workflow research --demo
compache setup --workflow support --demo
```

## Which cache policy costs less?

The examples fit each profile on 24 synthetic threads. Twelve separate synthetic threads per workflow supply the evaluation. Both profiles use the same model economics. A third fit draws its training threads from a mixture of research and support. These distributions illustrate decisions a team can check using its own observations. They are invented for the walkthrough.

| Training statistic | Document research | Customer support |
|---|---:|---:|
| Runs | 200 | 85 |
| Median thread length | 8 turns | 3 turns |
| Median new input per turn | 28,884 tokens | 497 tokens |
| Median retrieved portion of new input | 25,950 tokens | 70 tokens |
| Median model run duration | 69 s | 4 s |
| Median gap after a run when the user returned | 377 s | 37 s |
| Runs followed by another turn within one observed day | 88% | 72% |

| Held-out workload | Turns | No-ping cost | Cost using mixed fit | Cost using own fit | Own-fit pings | Own-fit savings vs no pings |
|---|---:|---:|---:|---:|---:|---:|
| Document research | 78 | 11,599,613 | 6,997,358 | 6,997,358 | 69 | 39.68% |
| Customer support | 51 | 83,934 | 88,798 | 83,934 | 0 | 0.00% |

Costs are modeled in ordinary input-token equivalents and rounded to the nearest unit. In research, either fit recognized enough likely returns just past expiry to justify paying for a refresh. In support, the workflow-specific fit cancelled or skipped every maintenance call. The mixed fit sent 13 and increased cost by 5.79% over doing nothing. The support-specific fit cost 5.48% less than the mixed fit.

The replay assumes an append-only, reusable prefix, one cache namespace per thread, and a known five-minute lifetime. It includes terminal pings, repeated foreground reads and generated output. It excludes cross-thread sharing and host rewrites. A cache-break check and the host's actual usage receipts are needed to test those assumptions in a deployment. [Replay assumptions](../../src/compache/setup_metadata.py)

## Which earlier sources can be dropped?

Consider the same candidate cut in two four-turn threads. It keeps historical user inputs and assistant outputs, and retains only the latest historical retrieval turn in full. These are hand-labelled, fictional questions. A handle for reopening a source counts as a miss when measuring what was retained before recovery.

| Earlier evidence needed for the fourth question | Document research | Customer support |
|---|---|---|
| User input in turn 1 | Restrict review to the 2022 agreements | Send the replacement to the Utrecht pickup point |
| Assistant output in turn 2 | Renewal clause flagged for review | Replacement case R41 opened |
| Retrieved text | Turn 1 signed addendum specifies delivery by courier | Turn 3 dispatch estimate is three working days |
| Sources visible after the cut | 2 of 3 | 3 of 3 |

The research question loses the signed notice provision. The support question keeps the latest dispatch estimate and the earlier user correction. At a 95% requirement, 2/3 falls below the floor and 3/3 clears it for this individual example. The production fitter evaluates coverage across the labelled sample and also checks whether unresolved turns would change. Three labels alone offer little evidence for deploying a policy. A team can use the count-only setup to evaluate caching and separately opt in to examining source text when it wants to fit cuts. [Retention rules](../../docs/workload-planner.md#fit-and-compare)

From the repository root, regenerate both cost comparisons and both source checks with `python examples/workflows/compare.py`. [Example code](compare.py) · [Synthetic workload definitions](../../src/compache/workflow_presets.py)
