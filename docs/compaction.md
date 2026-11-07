# Compaction: the recipe gate

The gate wraps every model request, innermost, so it sees the fully assembled request.

1. Load the conversation's recipe row.
2. Classify the window, cold or warm.
3. Size the previous request from its bill.
4. Keep, replace or plan the recipe.
5. Admit the request, replanning or shedding if needed.
6. Store the recipe, project the view, send.
7. After the response, record the calibration pair and the view identity.

A store failure degrades to planning from history alone. It never fails a request.

## Cold or warm

Checked in order, without calling the provider:

| Condition | Result |
|---|---|
| No prior response, a model switch, or a planned compaction | Cold |
| Age of last use > TTL − margin | Cold |
| The last response reports cache reads or writes | Warm |
| Otherwise | Unknown, treated as warm |

- **Last use** is the later of the last request's start and the latest ping that read the prefix. Do not age from when the response finished: a long response would hide an expiry.
- **The margin** is subtracted, never added, and equals the ping freshness margin. A much larger one plans cold starts the provider could still have served.
- **Unknown follows warm**, because a wrong cold plan rewrites a prefix that was still there.

## Sizing

Decide on the bill. It is exact, free and immediate. Two corrections apply:

- **Subtract the request-scoped share.** No recipe can shrink it.
- **Scale planned views by the bill.** The local ruler misses in proportion to size, so multiply by bill ÷ ruler(view as sent), but only when the rebuilt view's identity matches the billed one. After a fork or edit, plan on the ruler alone.

```text
billed = input + output
scale  = billed / (ruler(view) + scoped)    if identity verified, else 1
tokens = max(billed − round(scoped × scale), 0)
pressure = max(tokens, ceil(bytes × W / byte_bound))
```

Thresholds compare against pressure, so a history the byte bound would refuse is reduced like one the window would.

**The ruler** estimates between bills:

```text
estimate = prose_ratio × prose + structured_ratio × structured + exact + offset
```

Text and thinking are prose. Tool calls, tool results and the standing part (tools and instructions) are structured. Exact is replayed thinking, taken from the producing response's output usage; thinking from another model is excluded. The offset is what the provider bills but local counting can't see.

Keep each conversation's last eight (prose, structured, exact, billed) pairs and fit by ridged least squares toward the priors. Fit the offset only with three or more points, and only if it comes out positive. Priors deliberately overcount, so an uncalibrated conversation is never refused admission.

## Keep, replace or plan

A stored recipe is **usable** if it covers no more messages than the history has and its prefix digest still matches. The digest is content-free: kinds, call ids, tool names and hashes of prompt and answer text. It is keyed by content, not lineage, so a fork simply misses and recomputes.

| Action | When | Effect on the prefix |
|---|---|---|
| `applied_warm` | Warm, usable, under the warm trigger | Extends |
| `replaced_warm` | Warm, above the warm trigger, or forced by admission | Rewritten once, to a small view |
| `planned_cold` | Cold; builds on the stored recipe so cuts never come back | Lost anyway |
| `planned_missing`, `planned_prefix_changed` | Warm with no usable recipe | The one allowed warm rewrite |

## The tiers

| Step | Fires when | Does | Stops at |
|---|---|---|---|
| Tier 1 | Every cold start | Documents no answer cited become placeholders, except in the newest finished turn | n/a |
| Tier 2, cold | Bill > 0.17 W | Keeps the newest finished turn, then older ones newest-first while the view ≤ target. Reduces the rest | 0.15 W |
| Tier 2, warm replan | Bill > 0.65 W | The same, carries what the model has seen, then notes the oldest and replaces results | 0.15 W, plus the running-turn allowance |
| Last resort | View after tier 2 > 0.60 W | Oldest reduced turns become their note alone, then tool results become placeholders | 0.30 W |
| Admission shed | Request fails admission | Running-turn results, largest first, unread ones included | 0.30 W and admitted |
| Admission | Always, once calibrated | Refuses above 0.95 W or the byte bound | Fails before dispatch |

**Tier 1.** Citing one passage keeps the whole document whole everywhere. Without source extraction, tier 1 is a no-op.

**Tier 2** contains tier 1 and drops replayed thinking from every finished turn, since the provider bills it again. A reduced turn also loses its intermediate narration, its source-free results, and documents no kept turn cites. Prompts and final answers stay verbatim.

**Never replaced:** framework-owned tool returns such as capability loads, and tools carrying the user's own words.

**A cold plan** never brings back an earlier cut, including results and front-of-run thinking a warm trim left out of the running turn: the later reducers can't remove thinking, so a request that fit without it may not fit with it.

**A warm replan** rewrites the whole prefix, so it must pay for itself. It compacts to the target rather than to just under the trigger. A placeholder or note the model has seen stays, or a returning body would rewrite the prefix again. A low warm trigger fails twice over: replans remove little while rewriting everything.

## The running turn

It keeps up to 0.25 W whole. Above that it loses its oldest rounds first, thinking and then results, with these limits:

- The latest response keeps its thinking, because its tool calls await results.
- The latest request keeps its unread results. Only admission can touch them.
- An expansion stays: the model asked for it.
- Thinking leaves only from the front of the run. Removing a middle block invalidates the later ones.

## Last resort and the settle point

Tier 2's floor can still be too large. The last resort notes the oldest reduced turns, then replaces tool results: the running turn's older rounds first, since it has already acted on them, then the kept turns', whose results back the answer a follow-up builds on.

It settles at 0.30 W, far under the warm trigger, so the next result extends the prefix. A view stopped just under a trigger crosses it again on every request, and each crossing rewrites the prefix.

## Admission

The request as sent, context included, must fit 0.95 W and the byte bound. Before refusing:

1. A warm request pushed over by request-scoped context force-replans the finished turns.
2. `shed_running` replaces running-turn results, largest first. This covers one result too large for the window, which planning on the last bill never sees. Without it, the run fails, and so does every later message, because a failed turn has no answer and the next prompt continues it.

A request that still can't fit fails before any provider call, with a typed reason.

## What the model sees

- A reduced turn's note rides as an extra part at the head of the next shown request, never as a second consecutive user message, and borrows that request's timestamp so it projects identically.
- A noted turn's messages leave, except capability load exchanges and protected evidence. Its tool calls leave with their results, wherever those sit.
- Messages at or after `covered` are unchanged.
- A reduced document's passages collapse into one placeholder, in place.

**Placeholders** name a handle, never a citation key:

```text
[Removed from context by compaction: result of <tool>. Call expand_compacted with handle 'tool:<id>' if its exact content is needed.]
```

**Notes** are one deterministic line built from canonical history: state and handles, then capabilities, tools with counts, up to eight cited documents as title and handle, a count of uncited reads, and for a noted turn the first 240 characters of the ask and the answer's length.

A static instruction teaches this to the model: content still exists, a placeholder is never evidence of absence, and it must expand before quoting, citing, relying on or contradicting removed content.

## Citations and recovery

**A key reaches the model only with its passage text.** Covered answers lose the markers of removed sources, and the prose stays. Only covered answers change, so a warm request still extends its predecessor, and a source restored later never rewrites an earlier answer. The answer validator accepts only keys whose text reached the model in this run.

**`expand_compacted(handle, offset)`** resolves handles from the run's canonical history, whatever the current recipe says.

- `answer:<n>` and `turn:<n>` return text; `turn:` puts the prompt first.
- Bodies over 48,000 characters page in windows.
- It peels up to three trailing characters, because the model copies quotes and brackets along with handles.

## Capability unloading

The load, its instructions and the tool reveal are history entries, so loading leaves the prefix untouched. A capability that no kept or running turn calls is retired at a cold run start, never while warm: the load exchange is stripped, recipe positions are remapped, and the turn's note says so. A response losing its load call also loses its thinking. A pending or newer call keeps the load. Capabilities with no tools are never retired, because loading them is using them.
