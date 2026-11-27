# Troubleshooting

Start with one completed conversation. Identify the integration that forwarded it, the policy file or record folder it used, and the usage the provider returned.

## My app runs, but the record is empty

- **Wrapped SDK.** Send a normal application turn with the client returned by `compache.wrap`. Inspect `client.compache.stats["calls"]` on that same client. Creating a wrapper around a second, unused client leaves the count at zero.
- **Proxy.** Request `http://127.0.0.1:8787/compache/health` and check the call counter before and after a normal model request. A health response confirms the process is reachable. A rising call count confirms application traffic passed through it. For another port, change the URL. [Proxy configuration](integrations.md#the-proxy)
- **Recording.** Supply `record="compache-record"` or `--record compache-record` when starting the integration. Inspect that directory in the process's working directory. A policy file alone enables application of the policy. Finish the turn and consume the response if streaming before inspecting the record.

## The fitter keeps the entire chat history

Check the training summary for labeled turns and historical requirements. Sizes and timing let a chat fit compare caching policies. To learn which questions, answers and tool results may be archived, it needs labels. An unknown label and a turn explicitly labeled independent have different meanings. [Label historical dependencies](fitting.md#label-historical-dependencies)

With labels, full history can still be the cheapest eligible policy. Check how often users return within the cache lifetime, the amount of historical text and the requested retention minimum. If retrieved documents are embedded inside a single user message, that message is one unit in the live request. Separate tool results make documents independently addressable. Fit on the same message format your app sends.

## Replay rejects a thread or reports no retention

`held-out evaluation includes a training thread` means a thread ID in the evaluation set also contributed to the fit. Split whole conversations before fitting. Keep their original IDs and put all labels for each thread in the same split. [Runnable held-out example](../examples/quickstart/fit_replay.py)

A missing held-out percentage means the cohort had no observed historical dependencies to score. A measured value below the training target is possible. Inspect the number of labeled historical requirements and, for chat, `coverage_by_thread` before interpreting a difference.

## A warm return has few cache reads

Check the usage on the **first physical call of the returning turn**. Compare it with the prior turn's final receipt. A cache lifetime is a modeling assumption. Actual reuse depends on timing, the provider's minimum cacheable prefix, and whether the next request starts with the same model inputs. Compare the system prompt, tool definitions and order, and the history on both sends.

A ping is a separate model request with its own output, cost and receipt. For an application-managed session, acknowledge the ping with `session.ping_sent(..., usage=...)` and read `session.next_ping_at` again. A paid ping with little cache reuse can change the next decision. [Manage a ping timer](sessions.md#schedule-and-account-for-sends)

## I see more provider requests than user messages

One user turn can include several model calls. Tool continuations, retries after a refusal, reopens, opt-in labeling and pings can each add a dispatch. Check `physical_calls` on the wrapped runtime. If recording, inspect the `kind` and `usage` fields in `calls.jsonl`. A refused dispatch may have no usage receipt. [Counter definitions](reference.md#counts-and-costs-from-a-running-application)

## My measured bill differs from replay

Replay models both policies at the trace timestamps. Check the model, currency unit, cache prices, number of calls per turn, and share of requests with recorded usage. Compare equivalent denominators: a chat turn may contain several agent calls. In a live record, `compache stats` can use billed cost for a priced model when the provider supplies it. Its comparison with full history still uses a counterfactual. [Interpret costs](fitting.md#replay-and-interpret-the-report)

## The CLI rejects an option from these docs

Run `compache --version` and `compache fit --help`. Use `python -m compache fit --help` to check the Python environment running your app. For example, the published 0.15.1 CLI predates `--min-retention`. Install this checkout with `python -m pip install -e .` from the repository root to follow the examples for this revision.
