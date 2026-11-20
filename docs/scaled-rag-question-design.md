# Scaled synthetic RAG question design

This is the design inventory. The frozen [implemented workload](scaled-rag-workload.md) uses 11 of these patterns; the full inventory is not a coverage claim.

## Scope and interpretation

Build a new deterministic synthetic standard-RAG corpus with 300 conversations: 30 fictional companies, 10 threads per company. Every company has a sufficiently detailed, internally consistent document bundle (policies, procedures, dated revisions, tables/case records, and near-duplicate distractors); threads retrieve relevant and irrelevant excerpts in ordinary-sized chunks. This is a designed conformance/stress corpus, not a public benchmark, prevalence sample, natural-traffic estimate, or evidence of causal source use. Do not describe the fictional material as externally validated.

Use 12–16 families below, crossed and balanced across companies and thread positions. Each thread should include multiple families and ordinary direct questions; rotate family order and values by seeded generation. User turns must not repeat the historical answer or key fact. They may identify an earlier turn, case ID, document, or version, provided that identifier does not encode the answer. Retrieval at the target turn may provide competing current evidence, but must not simply restate the old fact when the family is intended to test history.

## Question families

Examples use fictional company Northstar Loom and are templates; generate actual company facts and gold records from the document bundle.

| Family | Self-contained target question example | Intended evidence / gold form |
|---|---|---|
| 1. Direct current lookup | “What is the current standard dispatch window, and what condition limits it?” | Current authoritative policy occurrence; structured fields for value and condition. Include exclusions nearby. |
| 2. Multi-document AND synthesis | “Compare standard and priority delivery, including the shared calendar caveat and the priority region restriction.” | Both distinct service documents jointly required (AND); separately encode each claim. |
| 3. Historical exact fact | “In turn 18 I asked about the original warranty period. What duration did the then-current document state?” | Earlier retrieval occurrence, exact value and version; target must not repeat duration. |
| 4. Long-lag topic return | “Return to case NS-4821 from the supplier-risk review. Which two controls did we record?” | Earlier user case details and/or retrieved record; cover lags 51+ and unrelated intervening topics. |
| 5. Exact quote | “Repeat the approved phrase from the brand review at turn 27, preserving punctuation and capitalization.” | Earlier visible source quote and/or assistant answer; exact string match. Include close decoy tagline. |
| 6. Prior-output edit | “Revise the second bullet from your turn-34 draft into one sentence, keeping its factual scope and changing only tone to neutral.” | Specific earlier assistant output occurrence; evaluator validates requested edit against actual generated prior answer, not a canned expected sentence. |
| 7. Version supersession | “For case REV-06, give the rule that applied on the approval date and say whether the older limit still governs.” | Dated versions and approval date; select effective version, explicitly identify superseded value. |
| 8. Correction propagation | “I corrected the account tier in turn 42 after your draft. Recalculate the estimate using the corrected tier and the rate card.” | Earlier user correction plus rate-card retrieval; correction supersedes earlier input. Avoid stating corrected tier in question. |
| 9. Constraint carryover | “Apply my formatting instruction from turn 9 to this answer: summarize the renewal steps.” | Earlier user instruction plus current renewal source. Score factual content and instruction compliance separately. |
| 10. AND/OR alternative evidence | “What is the approved escalation route recorded in the incident log?” | Gold alternatives: (A AND B) OR (C), where A+B jointly suffice and C is a signed later summary. Keep alternatives semantically equivalent and visible in separate retrieval events. |
| 11. Explicit abstention / unsupported | “What was the exact number of units lost in the unlogged pilot?” | No authoritative evidence; answer must state unavailable/unknown and not infer from adjacent counts. Gold has no support occurrence. |
| 12. Near-duplicate disambiguation | “For the West depot, what cutoff applies to refrigerated returns?” | One exact document applies; distractors differ by depot, product class, draft/current status, or one digit. Require all disambiguating attributes. |
| 13. Fresh duplicate alternative | “What is the documented service window?” | Current target retrieval directly supplies fact; older occurrence is an alternative, not required. Tests correct attribution without falsely demanding history. |
| 14. Explicit retrieval/input/output composite | “Using the allowance I gave in turn 21 and the unit rate in the current schedule, calculate the revised ceiling.” | Earlier user input AND current retrieval; output calculation deterministic. Optionally another authored sufficient source route may be OR alternative. |
| 15. Topic return with correction and version | “For project OR-19, use the amendment we accepted after the first review. What limit should the renewal note carry?” | Earlier retrieval plus later correction/amendment; distinguish acceptance date from stale draft date. |
| 16. Bounded negative / scope | “Does the contract promise a delivery date, or only an estimate? Answer only for the in-stock service.” | Source plus scoped negative; answer must not generalize from a near-duplicate service or imply an unsupported guarantee. |

Represent each question's authored sufficient evidence as alternatives of AND-sets. A set can include user-input occurrences, retrieval occurrences, and earlier assistant-output occurrences. OR across alternatives means any whole alternative suffices; no claim is made that listed evidence is causally necessary. Use explicit unknown/abstention labels for unsupported facts, never fabricated evidence. Include distractors in retrieval candidates and label model-visible evidence precisely; a selected-but-not-exposed passage is not evidence.

## Dependency distance and thread schedule

Define dependency lag as `target_user_turn_index - source_occurrence_turn_index`, in user/assistant conversational turns using a single documented zero/one-based convention. For multi-source alternatives, report each source lag and the alternative's maximum lag. Do not collapse current retrieval into a historical lag. Stratify target dependencies into 1–5, 6–20, 21–50, and 51+ turns; ensure every family intended to test history appears in each applicable stratum across the corpus. Reserve at least one long-dependency target after turn 50 and another after turn 80 in every thread long enough to support it.

Recommended deterministic thread lengths (10 threads per company; 300 total):

| Threads per company | Length |
|---:|---:|
| 1 | 24 |
| 1 | 36 |
| 1 | 48 |
| 1 | 60 |
| 1 | 72 |
| 1 | 84 |
| 1 | 96 |
| 1 | 108 |
| 1 | 120 |
| 1 | 144 |

Thus each company contributes 10 threads with the full length ladder. Place long dependencies after turns 50 and 80 where possible; in 24/36/48-turn threads, use the longest feasible lag and preserve the 51+ stratum in longer threads. Do not make all long questions end-of-thread: distribute targets across the post-threshold region. Keep several ordinary intervening tasks between source and target, with no repeated answer paraphrase or accidental summary that leaks it.

## Deterministic factual checking

Store immutable source occurrences, canonical fact records, version/effective-date metadata, and per-claim evidence alternatives. Normalize only harmless presentation differences for semantic fields (whitespace, Unicode normalization, case where case is not meaningful, and declared unit aliases); do not normalize away digits, signs, dates, scope, entity, version, negation, or qualifiers. Grade structured factual claims field-by-field against canonical records; exact-quote families require literal quote equality after only newline normalization. Arithmetic gold uses a fixed decimal/rounding rule declared in the case record. Unsupported questions pass only when the response explicitly signals unknown/unavailable and makes no unsupported substantive claim.

For prior-output edits, compare the edit request against the actual preceding assistant output captured in that run: verify source bullet identity, preserved factual propositions, requested transformation, and no newly added facts. Keep the deterministic authored template as an intent scaffold only; never treat a prewritten synthetic answer as the actual prior output. Where actual output diverges from the template, validate the requested operation on actual text and report answer correctness separately from template match. For generated prose not reducible to structured fields, use blinded human adjudication or a separately reported model judge; do not silently convert that score into deterministic fact accuracy.

## Leakage, shortcuts, and validity checks

- Prevent answer leakage in question wording, case IDs, filenames, retrieval ordering, turn summaries, and gold/template fields sent to live models. Keep gold and authored labels in evaluator-only sidecars.
- Avoid fixed family positions, repeated company-specific phrasing, answer-bearing IDs, invariant numeric values, and a rule that “latest snippet always wins”; balance them with genuine effective-date logic.
- Include plausible near-duplicates and decoys, but make the authoritative choice decidable from version, scope, entity, date, and status metadata. Avoid contradictory gold routes.
- Separate model-visible retrievals from unselected candidates and preserve exact occurrence provenance. Ensure every cited/evaluable quote exists literally in an eligible visible occurrence.
- Keep assistant summaries and later user restatements from inadvertently carrying the answer across the intended long lag. Where summaries are part of the real trace, label them as output evidence and disclose the route.
- Ensure company and family values are independently seeded; split train/evaluation by thread and include held-out combinations of known document patterns. Do not leak company-specific answer mappings through thread IDs.
- Report designed denominators by family, lag, company, source type, and thread length. Separate dependency-label coverage, factual answer accuracy, abstention quality, quote fidelity, and actual source-use inference. This corpus alone establishes none of population representativeness, causal necessity, or public benchmark standing.
