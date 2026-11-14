# compache

compache fits a context-compaction and cache keep-warm policy to your own traces, then applies it deterministically on every model request.

**Status:** design only, no code yet. The full design is in [docs](docs/README.md).

![Architecture. Offline, two fitters turn traces into a profile and a solver turns that into a policy. Online, the gate and the planner read the policy.](docs/assets/architecture.svg)

- **Offline:** two fitters describe your workload, and the solver combines that with the provider's prices into one policy file.
- **Online:** the gate decides how and when to compact, and the planner decides when to ping. Neither calls a model, so the same inputs always give the same request.

## One request through the layers

![Sequence diagram. Before every request the integration calls prepare, the gate returns a recipe and a view, and the rewritten request is sent. After the response it calls observe. At run end the planner decides whether to ping.](docs/assets/request-sequence.svg)

- **Prepare:** the gate decides the view before the request leaves.
- **Observe:** the bill comes back, and the runtime calibrates and records it.
- **End of run:** the planner decides, and pings repeat while the thread is idle.

The blue arrows are the two decisions compache outputs. The order the layers get built in is in the [build plan](docs/design.md#build-plan).

## Open design choices

Already settled: [statistical fitters, no shipped return curve, Python 3.11 and up](docs/design.md#decided).

**Gate and planner**

1. **Prompts and answers.** Do they always stay verbatim, or may a fitted policy reduce old ones? Proposed: verbatim in the first version, with their reuse reported.
2. **Reference defaults.** Ship the current thresholds as they are, or derive them again from public or synthetic threads first?
3. **The `compact` action.** Keep it, or let the planner only ping or lapse? Its useful band is narrow.
4. **Request-scoped block.** Which single position holds from the first request on?
5. **Preserved thinking.** Replacing a result an earlier response read may be rejected on models that bind thinking blocks to unedited history. How does a rewritten view avoid that?
6. **Unbounded tool results.** A turn reading large documents on a small window can loop, with the model expanding what the shed just replaced. Where are results bounded, and what guards the shed?
7. **Fork carry.** What does a cold plan build on when the stored digest no longer matches? The safe answer is nothing.
8. **Unknown model.** Observe only, assume a default window, or fail? Proposed: observe only.
9. **Citations and protected evidence.** Core, or optional?
10. **Three compaction rules to confirm.**
    - Tier 2 stops at the target and never keeps one more turn past it.
    - Tiers 1 and 2 never touch the running turn. Only the allowance, the last resort and admission do.
    - A capability can be unloaded on a cold start where only tier 1 fires.

**Fitting and solving**

11. **Miss ceiling.** What share of turns may need an expansion before the solver rejects a policy? To be set from the first real traces.
12. **Profile scope.** One profile per application, or one per agent or segment inside it? Proposed: per application first.
13. **Evaluation dataset.** Which threads do we judge the defaults and the solver against?

**Integration**

14. **First targets.** Proposed: Anthropic then OpenAI, Langfuse then MLflow, LangChain then Hermes then opencode.
15. **Hosts that aren't Python.** Proposed: a local service with a thin plugin, a proxy as the fallback, and no TypeScript port.
16. **Built-ins that can't be switched off.** When a host places its own cache breakpoints, does compache stand down to audit, or run beside them?
17. **State and pings.** Which stores ship first, and do pings run in-process or go to the host's own queue? Proposed: in-memory and SQLite, in-process by default.
18. **Own telemetry.** Is the runtime's content-free log on by default? Proposed: yes, kept local.

**Project**

19. **Licence.** Needed before the first release.
20. **Development interpreter.** `.python-version` pins 3.15.0 while the floor is 3.11. Proposed: pin 3.11.
