# Benchmark review record

Authoring used `gpt-6-luna`; package construction and review used separate `gpt-6.1-terra` agents. The review agent inspected visible trace content before opening the authored expectations. This is model review, not human certification or an intervention experiment.

| Stage | Record | Scope |
|---|---|---|
| Initial independent assessment | `benchmark-independent-review.jsonl` and `benchmark-review-findings.txt` | All 45 final runs, judgments frozen before comparing truth |
| Post-fix assessment | `benchmark-postfix-review.jsonl` and `benchmark-postfix-findings.txt` | All 207 runs, final cases in depth; input and judgment hashes recorded |
| Correction verification | `benchmark-final-disposition.json` | Current input hashes and resolution of reported issues, including all affected companies |

The first two judgment files are preserved unchanged. They describe earlier versions and must not be treated as current benchmark labels. The initial input snapshot was not retained as a separately hashed release; that first record is a development audit trail. The post-fix findings record their source hashes. The final disposition identifies the current dataset and is correction verification, not another independent blind assessment.

Corrections covered arithmetic comparisons, effective dates, persistent instructions, missing return conditions, ambiguous references, sufficient historical/current alternatives and delivery qualifications. The old-retrieval case now asks for a packaging detail absent from the prior answer, so it tests retained retrieval content. Prior assistant answers count as conversational evidence where applicable, without being treated as independent documentary verification.

No concrete concern from the bounded correction review remains open. This does not establish exhaustive source equivalence, calibrated annotation accuracy or causal necessity. Previously unseen reasonable alternatives need adjudication. The 162 intermediate runs have model review records but no authored dependency truth rows; the 45 final runs carry the authored expectations.

Deterministic tests separately verify exact quotes, visible IDs, full-history ordering, expected families, arithmetic/date regressions, simulation disclosures, manifest hashes and byte-identical regeneration. Passing those checks does not prove semantic completeness.
