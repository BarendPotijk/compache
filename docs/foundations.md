# Foundations

## Provider assumptions

Written for Anthropic-style prompt caching. Verify each against current provider documentation before relying on it, `max_tokens: 0` first, since pings depend on it.

- A request reads the longest cached prefix that matches byte for byte up to a breakpoint. At most four breakpoints per request.
- Lifetime is 5 minutes, or 1 hour at a higher write price. Every read restarts the clock, which counts from the request's start. Assume no grace.
- Cache write costs about 1.25× base input, cache read 0.05–0.10× depending on the model. So re-reading a prefix while a user thinks can beat rewriting it when they return.
- A breakpoint cannot sit on a thinking block.
- Keep one lifetime per request.
- The cache lives per deployment. Pin one deployment per model, or pings write instead of read.
- Token-count endpoints can be unreliable, so the gate reads the bill instead.

## Request layout

The gate's whole job is to keep everything before the history-end marker byte-identical between consecutive warm requests. Breakpoints are marked ●.

```text
tools (eager definitions only)                     ●
static instructions                                ●
dynamic instructions (per conversation, not request)
history view                                       ● history-end marker
request-scoped context
user's latest message / tool results
```

**Deferred tools** travel outside the billed prefix until a capability load reveals them, so a large catalog should be deferred.

**The moving block rule.** Request-scoped context changes every request, so a breakpoint on or after it writes content that is never read back.

1. Never put a breakpoint on it. If the framework adds a trailing breakpoint, withhold it whenever the block is present.
2. Keep its position stable from the first request on, or the second call re-writes the first turn.
3. Keep it small. Its cost recurs on every call.

**Placing the marker.** The user's request stays last, and a tool call's results must follow it directly, so context goes after them. Only the first context in a row places the marker. Never place one with nothing before it, or behind a block that isn't text or a tool call: the provider rejects both.

## Prefix stability

> While the cache is warm, eager tools, static instructions and the history view are byte-stable, and the view only extends at its end. Anything request-specific rides after the history-end marker.

**Detecting a break.** Record fingerprints of eager tools, static instructions and all instructions, plus the view's identity. Classify every request against the previous one: `first`, `extended`, `tools_changed`, `instructions_changed` or `view_rewritten`. Anything but `extended` while warm is a defect.

**View identity.** Serialize the view as the provider's prefix sees it: kinds, instructions and part content. Exclude clocks, bills, provider ids and names, which differ between a live response and its reloaded copy. Use a JSON list and hash it without the closing bracket, so a longer view is a byte extension of a shorter one: hash the same length of the new view and compare.

**The failure class** is anything that varies by request and renders before the marker: a resume that rebuilds the system prompt from only some inputs, an open file's title in the dynamic instructions, a notice a capability adds only while loaded. The fix is always to move it behind the marker, or carry it unchanged.
