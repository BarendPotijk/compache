# Replayed RAG workload

This fixture uses the 45 previously generated conversations and 207 actual Codex subagent answers. Questions and retrieval passages were fabricated. Run/generation timestamps are restored from the original fabricated customer cadence, not the native agent harness's scheduling times. Usage is null: the native Codex system/tool overhead cannot calibrate this standalone RAG request projection.

`labels.jsonl` contains the 207 frozen skill-route observational dependency labels from the earlier independent comparison. There are 198 assessable runs and 9 unknown runs under the optimizer's task-requirement rule. Three threads (`rag-013`, `rag-028`, `rag-043`) have a masked source at their final run. No source body is filled in from the benchmark corpus. Labels are evidence for deterministic evaluation, not experimentally tested causal necessity.

`manifest.json` records exact input hashes, provenance and a synthetic collection cutoff seven days after the final fabricated run. Every terminal run is right-censored. `pressure-model.json` (2,048-token window) and `roomy-model.json` (131,072-token window) are deliberately illustrative scenarios. They use a 300-second TTL, zero cache minimum, 0.1× reads, 1.25× writes and 4× outputs. Those are not verified provider settings; the zero cache minimum exercises retention on these short fixture requests.

The evaluation script that used this fixture belonged to the earlier gate-based optimiser and was removed with it. The fixture still backs the trace, label and input-accounting checks.
