# A six-turn adapter smoke test

Two authored fictional conversations show how to represent prior inputs, assistant outputs and retrieved passages with stable IDs. The two-turn bicycle workshop is development data. The four-turn harbor dispatch is an adapter test with a separate scorecard. Both are tiny software fixtures. A passing smoke test establishes the input and scoring contract. Context pressure, production quality, cache hits and savings require separate measurements.

From a fresh checkout, using Python 3.8 or later, run

```sh
cd benchmarks/handoff/example
python3 -B read_fixture.py check
python3 -B read_fixture.py fit-input
python3 -B read_fixture.py serve --turn 4
python3 -B read_fixture.py score --answer 'Cedar | 18 crates | 17:40' \
  --visible harbor/1/in harbor/1/out harbor/2/r3 harbor/2/r4
```

The last command returns an exact answer match and **3/3 requirements kept**. Drop `harbor/2/r3` and the same answer still passes while retention drops to **2/3**. This distinction matters when a model guesses or recovers information from an archive. Use the actual IDs visible immediately before recovery to score a planner’s cut.

| Input | Consumer | Visibility |
|---|---|---|
| [fit.json](fit.json) | Fitter | Historical workshop turns, simulated times and training labels. The `fit-input` command reads only this file. |
| [serve.json](serve.json) | Fixed-work replay host | Complete harbor recording. The host emits only the requested user turn, current retrieval and earlier dialogue via `serve --turn N`. Each assistant output becomes visible on the **next** turn. Pass only the emitted parts to a model. |
| [grading.json](grading.json) | Offline evaluator | Final target and sufficient source sets. The serving path never opens this file. |

For the last turn the name comes from an earlier assistant output, the crate count from an earlier user input, and the time from the current notice previously retrieved on turn two. Two partial passages jointly supply an alternative derivation of the time. A superseded notice and the unrelated badge exchange are distractors. The scorer accepts a complete alternative for each requirement. Scores assume any reported visible ID still represents its original verbatim passage.

The three idle gaps are **112, 689 and 2,163 seconds**, measured from the preceding answer's end to the next user's arrival. These span both simulated cache windows. Compute cache eligibility separately from returning in time, using the actual matching prefix and the model tariff.

The reader checks SHA-256 fingerprints and writes nothing. Its verification perturbs future availability and the current recorded answer in memory to check causal visibility. The normalized messages are transport independent. A live adapter would record each arm’s generated answer in its own history in place of the authored output.

The [RAG trace-contract PR](https://github.com/BarendPotijk/compache/pull/1) has the larger native fixture with generation-level visibility. The [optimizer implementation PR](https://github.com/BarendPotijk/compache/pull/2) contains the package integration.
