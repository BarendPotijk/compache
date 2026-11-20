# Replayed RAG workload

This fixture uses the 45 previously generated conversations and 207 actual Codex subagent answers. Questions and retrieval passages were fabricated. Run/generation timestamps are restored from the original fabricated customer cadence, not the native agent harness's scheduling times. Usage is null: the native Codex system/tool overhead cannot calibrate this standalone RAG request projection.

`labels.jsonl` contains the 207 frozen skill-route observational dependency labels from the earlier independent comparison. There are 198 assessable runs and 9 unknown runs under the optimizer's task-requirement rule. Three threads (`rag-013`, `rag-028`, `rag-043`) have a masked source at their final run. No source body is filled in from the benchmark corpus. Labels are evidence for deterministic evaluation, not experimentally tested causal necessity.

`manifest.json` records exact input hashes, provenance and a synthetic collection cutoff seven days after the final fabricated run. Every terminal run is right-censored. `pressure-model.json` (2,048-token window) and `roomy-model.json` (131,072-token window) are deliberately illustrative scenarios. They use a 300-second TTL, zero cache minimum, 0.1× reads, 1.25× writes and 4× outputs. Those are not verified provider settings; the zero cache minimum exercises retention on these short fixture requests.

After installing the package, run:

```sh
python3 scripts/run_rag_optimizer.py --output /tmp/compache-evaluation
```

The script checks input hashes before evaluating four declared scenarios:

| Scenario | Purpose |
|---|---|
| Company-held-out, pressure | Fit Alder/Briar; test all 15 Cinder threads, whose return timing differs |
| Thread-held-out, pressure | Stable whole-thread split across all 45 threads; exercise compaction and retention |
| Thread-held-out, roomy | Same split with room for all short requests; isolate retention |
| Complete-text threads, pressure | Secondary diagnostic excluding the three masked threads solely for cost coverage, then split the remaining 42 |

All 45 threads stay in the primary audit. The complete-text diagnostic is separately named and cannot justify a claim about excluded/unknown bodies. The script writes every candidate, the held-out request/ping ledger and `summary.json`. Held-out labels never choose thresholds or ping limits. The data is a development benchmark already inspected while building the package, not a pristine external validation set.

The release results show the intended conservative behavior: an unseen slow company rejects the training keep-warm policy; known-company thread splits show modeled reductions; incomplete costs block promotion. Exact numbers and policies live in the generated reports, rather than being hard-coded in the algorithm.
