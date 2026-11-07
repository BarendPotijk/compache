# Integration

## Harness contract

**Required:**

1. Messages with typed parts: user prompt, instructions marked static or dynamic, text, thinking (with producing model), tool call and return (stable id), and a request-scoped flag.
2. Usage on every response: input (inclusive), cache read, cache write, output.
3. When each request started.
4. Hooks: before each request, able to replace messages and wrap the model; after each response; at run end; and a release on every other exit.
5. A recipe store, one row per conversation version, holding recipe JSON, covered count, prefix digest and calibration JSON. It holds identities and counts, never content.
6. A per-model table: window, TTL, read and write multiples, retention enabled.
7. A local token counter and its unit.
8. A model factory for pings with the same settings, and a way to send `max_tokens: 0`.
9. A "conversation moved on" check and a stable run id.
10. A way to recognise framework-owned tool returns.

**Optional:** source extraction (tool return to documents and keys, answer text to cited keys), protected evidence, and deferred tools. Source extraction is the one domain-specific part.

## Providers

| Type | Notes |
|---|---|
| Explicit prefix cache | The design target. Check that gateways honour `max_tokens: 0`, that routing is pinned, and whether preserved-thinking is enforced |
| Automatic prefix cache | Set `write = 1.0` and the provider's read discount. Verify the idle lifetime. A ping uses the minimum output allowed |
| Context cache billed per hour | The planner needs a storage term in place of the per-ping read. The byte-stable rule still applies |

The planner uses price ratios only, so a read-and-write provider needs new multiples, not new logic.

## Build order

Log the prefix transition from step 1 onward: it's how you know step 1 worked.

1. **Prefix stability:** request layout, fingerprints, view digest. Nothing else is worth tuning while warm requests rewrite.
2. **Cold/warm classification.**
3. **Core compaction:** bill sizing and ruler, tiers 1 and 2, recipe store, projection with placeholders and notes, the expansion tool.
4. **Warm replan, last resort, admission shed.**
5. **The planner:** capture, the rule, a fitted curve, the ping loop, gate read-back.

Pin the [invariants](reference.md#invariants) as tests, and use the [test vectors](retention.md#test-vectors) for the planner.

**Test techniques:** a one-token window forces the deepest reduction. Use a real embedded store with a fake model. Use a clock whose sleeps advance it. Grow the running turn after a last resort and compare serialized prefixes.

## Telemetry

Content-free counts only.

- **Per request:** window, action, tier, prefix transition, before and after tokens, trigger tokens, cache age.
- **Per run end:** action, pings, held, cold view, expected saving, run index.
- **Per ping:** outcome and deployment id. `prefix_missed` plus that id reveals routing problems.
- **Continuity:** expected cache read = previous read + previous write. This catches partial losses a hit-or-miss status can't.

## Evaluation

Defaults are starting points. Measure before changing one.

- **Thresholds:** replay real threads with compaction off, and have blind judges mark which earlier prompts, answers and sources each turn requires. Then replay any recipe and ask whether each required piece is verbatim, a placeholder or gone. Sweep cold trigger and target for cost against retention, and look for the knee: above it, cost rises for little retention; below it, retention collapses for little saving.
- **Answer quality:** seed those turn starts into fresh conversations and have judges grade them, with full context run twice for a noise floor. At low thresholds, watch for the model searching again for what's gone.
- **The planner:** replay real runs with their real gaps and a frozen gate simulator, keeping a checkable token ledger. Report a waterfall of caching, compaction and planner. An oracle that knows each return time bounds what a better curve could earn.
- **Data needed:** per run, the conversation, index, times, call count, first and last input, and the gap to the next run. A start-to-start gap histogram per lineage covers TTL simulation.
