---
name: compache-setup
description: Integrate Compache into a chatbot or coding agent, fit a policy on its conversations, and verify request routing and usage.
---

# Integrate Compache into an application

Locate the code that constructs the model client, the conversation ID and the history passed on each call. Identify any existing gateway, compaction, retry loop and scheduler. Use one integration point per request path and retain the application's original conversation history for recovery.

## Choose the integration

- **Existing provider SDK.** Reuse the application’s configured OpenAI, Anthropic or Google GenAI client. Wrap it where it is created and use the returned client throughout the app. Bind the application conversation ID with `compache.thread(id)`.
- **Framework.** For LangChain `create_agent`, use the middleware with a stable graph thread ID. For a custom LangGraph node, inject the wrapped transport into the node that calls the model. For other frameworks, follow the [integration guide](https://github.com/BarendPotijk/compache/blob/main/docs/integrations.md) for their actual extension point.
- **Gateway or configurable base URL.** Point the application at `compache proxy`. Preserve the existing final destination by explicitly setting the intended upstream when a gateway or cloud deployment is in use.
- **Application-owned prompt and sends.** Create one `policy.session(thread_id)` per active conversation. The host supplies usage receipts and handles the reopen tool and scheduled pings. [Session lifecycle](https://github.com/BarendPotijk/compache/blob/main/docs/sessions.md)
- **Coding agent CLI.** Use `compache wrap --watch <agent>` to observe, or `compache wrap <agent>` for the learning integration. `compache wrap --print <agent>` shows the harness-specific routing settings.

## Select the data and policy

Apply an existing policy with `compache.wrap(client, policy)`. If starting with observation, explicitly enable local recording with `compache.wrap(client, record="compache-record")`. Keep that directory out of version control. The default record holds sizes, timing, usage and identifiers for observed dependencies. Exported traces and generated labeling packets may contain full text.

Chat fitting on metadata alone tunes caching and preserves history. To fit chat compaction, use existing dependency labels, or sampled labeling when the user has opted into sending conversation text to their configured provider in additional billable requests. For tool-loop agents, quote-based dependency evidence can be inferred during recording. Keep training and evaluation conversations disjoint and preserve their original IDs.

## Verify the application's path

Use the project’s existing tests or an in-memory provider. Confirm a completed turn appears in the record or increases the live call counter. For multi-turn changes, inspect the actual outgoing request and verify the original application history remains available. An application that sends only server-side references or an already-truncated history gives Compache less context to manage.

Report the integration point, policy source, and what was verified. State whether a cost is a modeled replay estimate, a usage-based estimate or a provider charge. For evaluated workloads, include cost units, the baseline and observed held-out retention. [Offline examples](https://github.com/BarendPotijk/compache/blob/main/examples/README.md) · [Diagnostics](https://github.com/BarendPotijk/compache/blob/main/docs/troubleshooting.md)
