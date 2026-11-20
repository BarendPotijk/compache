# Long RAG workload

This deterministic fixture extends the source `rag_workload` question families and company profiles into a 72-turn onboarding and policy-review conversation. Six training threads cover Alder & Finch, Briar Works, and Cinder & Co.; the single held-out thread uses the already-seen Alder & Finch profile with distinct seeded case-register values. The split, seed, selected source references, dependency-lag counts, and synthetic cadence are frozen in `manifest.json` and `question_design.jsonl`.

`handbook.md` is the standalone fictional retrieval manual. It contains company-specific policies, related operating procedures, and seeded case registers. The trace retrieves three to five topic-matched sections per request. It includes early-to-late dependencies for a carried instruction, prior output quote, original retrieval, version comparison, and fresh duplicate evidence. `labels.jsonl` contains authored dependency expectations for every run and is for offline grading only.

The trace `output` values are scaffolding for offline validation. A native execution harness must ignore those outputs, reconstruct each request from only its current input and selected retrievals plus actual earlier native answers, then store each actual answer and measured provider token usage. Never send the fixture's future questions, template outputs, dependency labels, or manifest/design metadata to the live model. Keep OpenCode and Compache runs paired on the same thread and question order. Run all native calls immediately; replay the 2–3 second, 420-second, and 2,400-second timestamps as separate offline warm, medium-return, and long-return scenarios.

The final prompt size is a planning estimate from stored source text, not actual model usage. Native token counts and any automatic compaction must be recorded by the execution harness. Do not claim a context-window threshold from this estimate alone.

Rebuild the fixture with:

```sh
python3 scripts/build_long_rag_workload.py
```

Validate this fixture with:

```sh
python3 -m unittest discover -s tests -p test_long_workload.py -v
```
