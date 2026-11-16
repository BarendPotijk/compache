# Expanded RAG results

The frozen 300-conversation corpus contains 23,760 authored turns across 30 fictional companies. See [the result and its limits](../../docs/scaled-rag-results.md) before interpreting costs or coverage: Compache stopped in 155 offline conversations and all three measured cases.

`manifest.json` is an index of the final corpus. Per-run gold, case-ID lists, per-run pattern/lag lists and repeated company profiles are omitted; aggregate counts and the company table remain. Its `full_manifest_sha256` identifies the complete generated manifest; it is not the hash of this smaller index. Regenerate the complete dataset with `python3 scripts/build_scaled_rag_workload.py --output ./work/scaled-rag-dataset`. The separately delivered dataset ZIP and its SHA-256 are described in `results/dataset-archive.json`.

`results/` contains corpus counts, frozen policy/profile, offline aggregate denominators, actual subscription usage summaries, five-/thirty-minute modeled costs, the per-invocation timeline, validation, independent semantic audit and admission reconstruction. File-byte hashes in the offline manifest and canonical-object hashes in the live plan use different encodings of the same policy/profile. The source-hash manifests preserve execution and final-source provenance.

Raw traces, label shards, prompts, answers, provider events, failed transport records, per-run replay records and source snapshots remain in the separate local evidence bundles. They are excluded from Git because of size. Full-thread savings are unavailable; truncated trajectories must not be compared as equivalent completed work. Controller usage and three failed network attempts have no reported usage and are not priced as zero.
