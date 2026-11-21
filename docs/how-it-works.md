# How it works

compache answers two questions from your recorded threads: what must stay in the request, and when a keep-warm ping pays. It answers them together, because each changes what the other is worth. A cut rewrites the cached prefix, and a ping costs in proportion to the prefix it keeps warm.

![Record threads, fit a policy, apply it on every request](assets/workflow.svg)

## What is read from the threads

| From | The fit learns |
|---|---|
| Timestamps | For each turn position: whether the user came back, after how long, and how long the answer took. A wait that the record cuts short is treated as unfinished, not as a user who left |
| Text sizes | How much each turn adds: its question, its retrieved text, its answer |
| Labels | For each kind of part, how far back the evidence a turn needed lies |

## The candidates

A cut is three windows: how many turns of questions, of answers and of retrieved text stay in view. Two switches refine it: earlier retrieved text that matches the new question may stay, and parts under a size may always stay.

- Windows follow 1, 2, 5, 10 and so on, plus the look-backs that cover 80% to 100% of the labelled needs of each kind.
- Every cut is scored without a replay: an applied cut archives a part once it leaves its window, so how much of what turns needed is still in view follows from how far back that evidence lies.
- The cuts worth pricing are those on the frontier: each keeps at least one point more in view than every smaller one, down to 80%. Up to six of them, plus keeping everything, are each paired with 0, 1 and 3 pings.

## The price of a candidate

Token counts cannot rank policies, because what a cut or a ping is worth depends on how often the cache is still warm when the user returns.

- Each candidate runs through the runtime session against the provider's cache rules: lifetime, smallest cacheable prefix, write and read prices and any long-request surcharge. Requests and pings fall at the threads' own timestamps.
- A thread is priced with a model of returns fitted on the other threads, in three folds.
- A cut pays for what it loses. Every labelled need that is no longer in view costs one more request, which reads the cached view, writes the reopened parts and emits a tool call. That rate carries from the labelled turns to all turns.
- The cheapest candidate wins. Ties go to more in view, then to fewer pings.

The winner is the cheapest policy the fit priced on your threads, and keeping everything with and without pings is always among those. `compache replay` prices it on threads it has not seen.

## At runtime

| Moment | Decision |
|---|---|
| A new user message | Send the history as it is, or apply the cut. The session compares expected cost over the fitted returns, three turns ahead. A cut usually waits for a return that finds the cache cold, when rewriting the prefix costs nothing extra |
| An answer is recorded | Let the cache lapse, or schedule up to the fitted number of pings before its lifetime ends. Pings are charged for users who never return. A returning user cancels the timer |
| A ping comes back | Its usage shows whether the cache was read. If so, the next ping is priced again. If not, pinging stops for that wait |
| The model asks for an archived part | The reopen tool returns the original text from the session's history |

A request that should have hit the cache and read nothing suspends pings for that conversation and records which part of the prefix changed.

## Assumptions

- One answer call per user message, unless the engine is given call counts.
- A cache entry lives exactly the lifetime in the model spec.
- Tokens are estimated at four bytes each unless the engine is given a tokenizer.
- A need counts as in view only when a whole alternative is in the request before anything is fetched back.
- Labels are judgements about need. [DocChat](../benchmarks/docchat/README.md) measures them instead, by answering each turn again with parts withheld.
