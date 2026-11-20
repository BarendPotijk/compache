# Pick up the Compache benchmark study

Start with the [illustrated handover](benchmark-handover.html). To read it as laid out, open the HTML in a browser from this checkout. It works offline. Once the PR is merged into main and Pages deploys, the [GitHub Pages copy](https://barendpotijk.github.io/compache/docs/benchmark-handover.html) will render directly in a browser. The brief covers why the present suites are development evidence, how to screen a fresh workload, and how to freeze the next comparison before scoring it.

## Start from GitHub while the PRs are open

The handover branch includes the [trace collection and exact fits from PR #4](https://github.com/BarendPotijk/compache/pull/4). The [runnable planner is in PR #3](https://github.com/BarendPotijk/compache/pull/3) on the existing optimizer branch. Use two worktrees to inspect both versions.

```sh
git clone --branch codex/benchmark-agent-handover https://github.com/BarendPotijk/compache.git
cd compache
git fetch origin refs/pull/3/head
git worktree add --detach ../compache-planner FETCH_HEAD
(cd ../compache-planner && PYTHONPATH=src python3 -B examples/fitted_policy/run.py)
```

After the handover merges, use the default branch for the first checkout. The [pinned runtime example](https://github.com/BarendPotijk/compache/tree/a813d8d72d7d39f44b447aecf1aaaaad42dd1573/examples/fitted_policy) identifies the exact code used in the handover check.

## First ten minutes

Run from the repository root with Python 3.11 or later. These checks read committed data and require no API key.

```sh
python3 benchmarks/handoff/verify.py
python3 benchmarks/handoff/fit/verify_fit.py
python3 benchmarks/handoff/example/read_fixture.py check
```

The first command reconciles grouped usage covering 459 Luna responses across 16 trajectories and prints 12 cost and retention rows. The second checks both synthetic training artifacts and their 200 fitted conversation IDs. The last exercises a two-thread trace fixture with separate model-visible inputs and evaluator labels.

| To continue | Open |
|---|---|
| Inspect a tool loop, a compaction or the cost of a turn | [Grouped benchmark cost and tool-use evidence](../benchmarks/handoff/README.md) |
| Regenerate the 300 fictional RAG conversations, with timestamps and dependency labels | [Generator, fit and lineage](../benchmarks/handoff/fit/README.md) |
| Implement a new trace adapter without sending model requests | [Worked example](../benchmarks/handoff/example/README.md) |
| Rebuild the earlier short public fixtures from their upstream data | [GoodAI and MemoryArena adapters](../benchmarks/public_chat/README.md) |
| Inspect split overlap or reuse completed Terra and Sol runs | [Research archive](../benchmarks/handoff/studies/README.md) |
| Exercise the actual planner and original fitted policies | [Runnable planner example](https://github.com/BarendPotijk/compache/tree/a813d8d72d7d39f44b447aecf1aaaaad42dd1573/examples/fitted_policy) |
| Find the earlier package contributions | [Trace contract and dependency review](https://github.com/BarendPotijk/compache/pull/1) · [Optimizer and baseline implementation](https://github.com/BarendPotijk/compache/pull/2) |

## The next assignment

Produce an offline feasibility report for a fresh external benchmark family. First demonstrate on independent development traces when the 95% retention gate admits a cut, when the planner can reuse a prefix, and when keeping the existing host policy is cheaper. Record refusals and failed retrievals. Freeze the fitted policies, clocks, source-access rules, scorecards and cohort selection before examining performance on the external family. The [HTML brief](benchmark-handover.html#handoff) defines the deliverables. Sol spending has ended. The prior experiment budget has been accounted for.
