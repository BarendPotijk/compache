# Admission repair evaluation

See [the result and its limits](../../../docs/admission-fix-results.md). This bundle records Compache 0.4.1 on the same 300-conversation corpus, unchanged selected thresholds/profile, three fresh Luna-low cases and unchanged OpenCode recordings. It is a development regression result, not an untouched production test.

The repaired gate admits all 23,760 offline turns and retains all 12,498 known historical requirements. The fresh sample completes all three 96-turn cases with all 155 historical requirements available. Modeled known-call costs are 35.69% lower at five minutes and 10.35% higher at thirty minutes, including internal Codex overhead. Total attempt cost remains unavailable because three old baseline timeouts have no reported usage. Source retention is distinct from the small semantic answer audit.

The files contain aggregate denominators, frozen plan/policy/profile, per-invocation timing and token counts, cost by role, audit adjudication and validation. The raw corpus and execution evidence are separate large local artifacts. The two recorded-output diagnostics use pre-format source hashes; their byte provenance is preserved rather than relabeled as the final source.
