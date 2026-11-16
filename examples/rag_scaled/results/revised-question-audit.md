# Revised scaled RAG pilot audit

## Disposition

No material answer/label mismatch was found in the sampled revised pilot. It is suitable to freeze as a **synthetic stress fixture for bounded live calls**, with the scope limits below stated. This is a spot audit of the three 96-run gzip shards (`scaled-24-07`, `scaled-26-07`, `scaled-29-07`) and `scripts/build_scaled_rag_workload.py`, not a full generator validation or live answer-quality assessment. Samples covered every implemented family, question/retrieval/label joins, and late targets beyond turns 50 and 80.

The revised fixture has **11 families**: `direct`, `scoped_negative`, `mixed`, `fresh`, `version`, `correction`, `arithmetic`, `input`, `history`, `prior_output`, and `abstention`. `history` is an earlier-case retrieval family; there is no distinct topic-return family.

## Concrete checks

- **Separated correction events:** `scaled-24-07-run-019` asks to compare the allocation for procurement case `C9D61C8882D`. The original allocation is in `scaled-24-07-run-015`'s input, and a separate correction is in run 016's input. Label requirement `req-2` is an AND-set requiring both exact source occurrences. The target question does not state either value.
- **Version update:** `scaled-24-07-run-089` asks to compare earlier case `CA1D876005C` with its revised case record. The old fact is supported by either `scaled-24-07-run-006` retrieval or output, and the current target retrieval supplies a revised fact for the same case. The label correctly requires old AND revised evidence. This exercises supersession of a synthetic case record, not effective-date interpretation of a real policy revision.
- **Fresh duplicate alternative:** `scaled-24-07-run-012` asks for prior case `CE9EFE446EC` “using the current case extract”; the current retrieval contains an exact duplicate of the historical fact. Its alternatives are that current occurrence OR the earlier retrieval OR the earlier output, and `historical_context_needed` is `no`. `scaled-29-07-run-031` shows the same pattern. This is coherent as an authored fresh-evidence alternative, but the current retrieval is an explicitly inserted duplicate rather than an independently authored contemporary document.
- **Prior output:** `scaled-24-07-run-030` asks to quote the first sentence of the answer in turn 20, with the source label pointing to run 020's output. The frozen answer is generator scaffolding; live evaluation must use the actual earlier answer. Do not count a template match as live quote accuracy.
- **Source placement and domain specificity:** current target questions and inserted facts use domain-specific fields (for example, delivery duration plus destination; hosting location plus backup retention; pricing plus seat allowance). The exact target fact is inserted at a varying position within its domain retrieval, rather than always at the beginning. Opaque case IDs are not a direct encoding of run numbers. Labels' exact quoted strings were found in the cited source occurrences.
- **Abstention:** the unlogged pilot loss is labeled unknown with no supporting alternative; the current domain fact remains a separate supported claim. The response still needs to be checked for unsupported inference from nearby synthetic record fields.
- **Long dependencies:** sampled long targets beyond turns 50 and 80 point to earlier input, retrieval, or output occurrences. AND/OR groupings inspected were coherent with the requested multi-part claims.

## Limits to retain in reporting

The companies, case records, values, and handbook are fabricated. Case-specific terms are scenario records and can differ from general handbook guidance; describe them as such, not as ordinary company policy. Registers have more varied fields than the first pilot but remain generated tabular rows, not empirical operational evidence. `history` does not establish topic-return behavior across a genuinely distinct intervening topic narrative. The fresh-duplicate route is a same-value insertion into the current retrieval. Prior-output cases remain answer-scaffold dependent until runtime rebinding.

For runtime rebinding, checking whether a literal authored fact quote appears in the actual prior answer is a conservative exact-string screen. It may exclude semantically valid paraphrases. Treat counts from that screen as a **lower bound on supported output alternatives**, report dropped alternatives and diagnostics, and do not present the screen as semantic answer grading. This does not imply a change to the optimizer policy.

No claim is made about public-benchmark status, population representativeness, causal source use, real-world company behavior, or production retrieval quality. The earlier pilot audit remains available separately as `question-audit.md`.
