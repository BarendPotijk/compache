# Manage sessions in your application

When you own prompt assembly, billing, retries or timers, keep one session per active conversation. Run `compache demo -o policy.json` followed by `python examples/quickstart/app_managed.py` from the repository root to exercise the [application-managed example](../examples/quickstart/app_managed.py) without a provider key.

The example uses an in-memory provider. It prepares a question, records a tool call, appends its result, records the final answer with usage, and checks for a ping between turns.

## Handle a conversation

Load the policy once and keep the object returned by `policy.session(thread_id)` for the conversation. Calling that method again creates a new session. Serialize turn and timer events for each session so a new user turn supersedes its old timer.

```python
import compache

policy = compache.Policy.load("policy.json")
session = policy.session("order-17")
turn = session.prepare("When does it ship?", ["Order 17 ships Friday."])
# In an application, send turn.messages or assemble the request from turn.parts.
# Substitute the actual answer and its provider usage here.
due = session.complete(turn, "Friday.", usage={"prompt_tokens": 42, "completion_tokens": 2})
```

| Call | What it does |
|---|---|
| `session.prepare(question, retrieved)` | Returns the request for this turn. `turn.parts` are its parts in order, `turn.messages` the same as chat messages, `turn.archived_ids` the parts replaced by a short notice |
| `session.complete(turn, answer, usage=...)` | Records the answer and returns when a ping is due. `usage` is the provider's usage object or dict. With it the session tracks real sizes and stops pinging a cache that no longer hits |
| `session.ping()` | When the timer fires: the request to send with the smallest output the provider allows, or `None` if the ping is no longer worth sending |
| `session.ping_sent(at=..., usage=...)` | Records the ping. More pings follow only when its usage shows the cache was read |
| `session.reopen(reference)` | Answers the model's `reopen` tool call with the archived text |

- **Cited sources.** For a policy fitted with them, pass `session.complete(turn, answer, cited_part_ids=[...])`. The IDs are the `part_id` of items given to `prepare`, or the generated ones in `turn.parts`, and must be in view in that request. They add nothing to what the model is sent.
- **Cache marks.** Mark the end of a Claude request with `policy.cache_control`. It names the lifetime where the policy was fitted for another than the default, and then every mark of the request has to.
- **Archived parts.** A cut replaces a part with a notice that carries a handle. Give the model `compache.REOPEN_TOOL` and `compache.REOPEN_INSTRUCTIONS`, and pass its tool calls to `session.reopen`.
- **Placement.** `turn.messages` puts answers under `assistant` and everything else under `user`. If your application sends retrieved text another way, build the request from `turn.parts` and keep that mapping fixed.
- **Tool loops.** Call `prepare` once per user message. Report the intermediate reply and tool calls with `session.replied(turn, text, calls=..., usage=...)`. Assign `turn = session.extend(turn, results)` after the tools return, assemble the next provider request, and pass the latest turn to `complete`. With several calls, pass the first call's usage as `usage` and the last call's as `last_usage`.
- **Restart.** The session lives in memory. To reconstruct it, replay the original ordered inputs, tool events, receipts, ping acknowledgements and timestamps from your application. A sizes-only recording lacks the text needed to rebuild a conversation.
- **Time.** Turn preparation and completion take `at=` in epoch seconds and default to now.

## Schedule and account for sends

After a completed answer, use the epoch time returned by `complete` or read `session.next_ping_at`. A value of `None` means there is nothing to schedule. When a timer fires, call `ping(at=...)` again to check whether a request is still due. If it returns a turn, send that view with a minimal output limit and acknowledge the send with its own usage and completion time. Read `next_ping_at` again after acknowledgement. Missing usage cannot confirm a refresh.

A session constructs views and consumes receipts. Your host performs and accounts for every provider request, including retries, tool continuations, reopens and pings. Keep the provider receipts alongside the send type if you need to reconcile charges. See [integrations](integrations.md) for modes that manage the request path for you.
