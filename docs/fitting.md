# Fit and evaluate a policy

Start with a [threads export or a recording](traces.md). A fit chooses a history-retention and caching policy for a model. Replay estimates what a saved policy would have cost on other conversations at their recorded arrival times. Both commands run locally.

## Split whole conversations

Set aside complete threads before choosing a policy. Preserve each original `thread_id` and keep all its turns together. Both planners reject replay IDs used for fitting. Renaming the same conversation defeats this check. If you label before splitting, partition the labels by the same IDs.

To try the workflow without your own traces, install this checkout with `python -m pip install -e .` and run the [offline Python example](../examples/quickstart/fit_replay.py) from the repository root. It fits on three complete sample threads and reports retention and modeled costs on the fourth.

```sh
python examples/quickstart/fit_replay.py
```

On the bundled sample, the example fits on 46 turns and replays 25. The held-out thread retains 5 of 8 historical requirements (62.5%), below the 80% training floor. The estimated costs are $14.54 per 1,000 turns with the policy and $50.17 keeping the full history. All 25 held-out turns fit in both arms.

For your own data, the CLI examples below use `train-threads.jsonl` and `heldout-threads.jsonl` for the two exports, with matching `train-labels.jsonl` and `heldout-labels.jsonl` for their labels.

## Label historical dependencies

For chat threads, without labels the fit tunes the keep-warm pings and keeps the whole history. Labels say what later turns needed from earlier ones, which is what lets it drop the rest. A sample is enough: an unlabelled turn counts as not sampled.

```json
{"thread_id": "t-17", "run": 7, "needs": [["t-17:3:output"], ["t-17:2:retrieval:1"]]}
{"thread_id": "t-17", "run": 8, "needs": []}
{"thread_id": "t-17", "run": 9, "needs": null}
```

| `needs` | Meaning |
|---|---|
| A list of alternatives | Any one alternative, with all of its parts in view, is enough to answer the turn |
| `[]` | The turn needs nothing from earlier turns |
| `null` | Nobody could tell |

Specify a part by its ID or by an object with `part_id` and `quote`. If supplied, the quote must appear verbatim in that part. In the example, either the answer from turn 3 or the first retrieval from turn 2 provides the evidence for turn 7.

To have a coding agent write them:

```sh
compache packets threads.jsonl --sample 200 -o packets.jsonl   # what a labeller may see, one run per line
# Write one label per assessed turn to labels.jsonl.
compache check threads.jsonl --labels labels.jsonl             # validates and summarizes
```

The `packets` and `check` examples above assume an export named `threads.jsonl`. The offline example reads the supplied labels. For your own export, generate packets separately for each split. A packet shows the target question, the actual answer and available parts of that request. It can expose conversation text to whoever labels it. Give a labeller only the packets they are authorized to see.

Use the [dependency-labeling instructions](../skills/compache-label-dependencies/SKILL.md) to distinguish a source necessary for the answer from one that merely mentions the same subject. Where the evidence is insufficient, use `null` or omit the label. Use `[]` only when the current turn is sufficient. Validation checks IDs, visibility and supplied quotes. A valid label can still misidentify what an answer depended on.

## Fit and save

```sh
compache check train-threads.jsonl --labels train-labels.jsonl
compache fit train-threads.jsonl --labels train-labels.jsonl --model claude-sonnet-5-5 --min-retention 0.80 -o policy.json
```

| Option | Use |
|---|---|
| `--min-retention` | Minimum observed training retention before reopening, between 0 and 1 (default 0.80). Python uses `fit(..., min_retention=0.95)`. Chat fits count labeled historical requirements. Agent fits count calls known to rely on earlier results. A cohort with no observed dependencies has no measured percentage |
| `--model` | A model with a built-in price sheet (`compache models` lists them), or a JSON file of model spec fields for any other. Fields left out take the defaults of `ModelSpec`. It can be left out when the threads record their model, as a file from `compache pull` does |
| `--system-prompt`, `--tools` | The text and tool definitions sent on every request, when the threads do not carry them |
| `--observed-until` | When the record was cut, if conversations were still going and the threads do not say. By default every thread is taken to be over |
| `--json` | Print the summary as JSON |

In Python, `compache.fit` returns a `Policy`: `policy.describe()` renders a human-readable summary, `policy.summary` the numbers, `policy.save(path)` the file. The file holds the decision, the fitted model of returns, the priced candidates and the system prompt it was fitted with.

## Replay and interpret the report

```sh
compache check heldout-threads.jsonl --labels heldout-labels.jsonl
compache replay heldout-threads.jsonl --labels heldout-labels.jsonl --policy policy.json -o replay.json
```

| Read this result | What it tells you |
|---|---|
| Training minimum and observed training retention | The floor used to select the policy and the fraction of known dependencies it actually kept in view before recovery. |
| Held-out retention | The fraction retained on unseen conversations. With no observed dependencies the percentage is unmeasured. For a chat policy inspect `coverage_by_thread` in the JSON report for variation. |
| Policy and keep-everything costs | Modeled costs for the same workload, including policy pings and recovery supported by dependency evidence. Use `cost_per_1000_turns` for chat or `cost_per_1000_calls` for agents and read its `unit` field (USD for priced models, input-token equivalents otherwise). Raw `policy_cost` and `keep_everything_cost` are input-token equivalents. |
| Incomplete replay | When context limits stop a chat arm early, compare the costs on the turns both answered under `common_prefix` and check each arm’s turn count. |

Replay does not generate new answers. For answer accuracy or billing impact, evaluate responses and provider usage in your application. When ready, [apply a policy](guide.md#5-apply).
