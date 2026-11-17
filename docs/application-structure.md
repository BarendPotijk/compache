# Application structure

Separate learning, runtime decisions, external adapters and evaluation. Observed traffic must not silently change an already-cached prompt halfway through a thread.

| Layer | Implementation | Responsibility |
|---|---|---|
| Intake and evidence | `traces/`, `dependencies/`, labeling skill | Validate normalized RAG traces and labels; preserve unknowns |
| Offline fitting | `optimizer/fit.py`, `solve.py` | Training-only statistics, constrained search, held-out evaluation |
| Frozen artifacts | `optimizer/contracts.py` | Versioned assumptions, profile hash, policy and retention constraints |
| Runtime | `optimizer/gate.py`, `planner.py` | Project context and plan retention using current history and explicit clock |
| Integration | `optimizer/integration.py`, experimental runners | Recovery protocol and execution hooks; production adapters remain future work |
| Evaluation | `optimizer/replay.py`, `comparison.py` | Shared cost ledger; scope, incomplete arms and per-run evidence |
| Optional baselines | `baselines/headroom.py`, existing `optimizer/opencode.py` | Pinned external behavior, isolated from runtime imports |
| Presentation | `site/` | Browser-local calculator and comparison scopes; no private trace upload |

These paths describe [PR #2](https://github.com/BarendPotijk/compache/pull/2), not a released production runtime. OpenCode retains its existing import path for compatibility. New external adapters live under `baselines/`; the core package still has no runtime dependencies.

## Lessons applied from Headroom

1. **Separate observation from activation.** Fit artifacts offline and latch a policy at a cold boundary. Headroom retired mutable request-time recommendations because changed compressed bytes broke caching. Compache retains its existing profile-hash and cold-boundary contracts.
2. **Put integrations at the edge.** Only the explicit comparison command imports Headroom. Fitting, the gate and planner do not require it.
3. **Include recovery in the integration contract.** Removed content needs stable handles and a caller-exposed tool. A lossless comparison does not validate lossy recovery.
4. **Attach provenance to results.** Record package version, installed-code fingerprint, configuration, trace/label hashes and ledger. Measured compression time is separate from modeled provider cost.
5. **Make evidence failures visible.** Unsupported output, frozen-prefix violations, missing text and failed admission cannot silently become savings. Unknown labels stay unknown.

The comparison runner accepts a frozen policy and never refits it. The fixture reproduction script fits on its training partition first. This development corpus is not an untouched external benchmark.

## Presentation

The README now leads with purpose, status and a runnable quickstart, followed by a compact workflow. The calculator remains the main interactive demo. Comparisons distinguish simulated scenarios, native library execution and recorded provider usage. Older presentations remain under `docs/design/`.

Sources: [Headroom observation-only contract](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/telemetry/toin.py#L1-L55), [README](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/README.md), [lossless library mode](https://github.com/headroomlabs-ai/headroom/blob/18067b1d19f6606bae3b4d6b50efeeaa48dc477e/headroom/compress.py#L166-L182).
