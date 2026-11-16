# Label contract v1

The same contract applies to a user's coding agent and the proposed local OpenAI worker. The canonical schema is `label.schema.json` in this directory. Compache implements trace validation, packet preparation and label validation. An OpenAI runner is not included in this milestone.

## Units and origin

A thread is a conversation, a run is one user request through its final answer, and a generation is one model request. The starter fixtures have one generation per run. Multi-call production traces require an extension that labels each call against its own visible input before aggregating to runs.

Every source occurrence has `part_id`, `origin_run_id`, `origin_run_index`, `kind`, `text`, and `content_status`. Kinds are input (user-authored), retrieval (inserted document text), and output (assistant final answer). Instructions, tool invocations and intermediate working content are retained separately by the full normalized trace contract; they do not become retrieved evidence just because they occur in a tool span. A re-retrieved document keeps its document/version/chunk identity but gets a new occurrence ID. Hashes may establish equal bytes, not equal meaning or causal use.

`request_parts` is the exact ordered context presented to the target generation. A part outside that list is not usable evidence. Text `null` means unavailable, even if the original model is known to have seen it. `target_output` is the observed answer, not an input source. Preserve origin identities when history is repeated in later requests.

## Two distinct assessments

`task_requirements` captures the current user's requested information, constraints and references, including requests the answer missed. Each entry quotes the current question, describes the requirement, records fulfillment and whether historical context is needed, and lists sufficient source alternatives when assessable. A historical language preference may affect an answer even when the new question does not restate it; quote the current question that the preference governs.

`historical_context_needed` assesses the supplied request's evidence, not the wording of the question. Use `no` when a sufficient current-run source set satisfies the requirement and resolves its references. Use `yes` when every established sufficient set needs an earlier-run part and the evidence supports that inference. Use `unknown` when visibility, source completeness or reference resolution is insufficient. These are evidence-based judgments, not experimentally proven necessity. An explicit reference to an old policy can still be `no` if a current retrieval fully supplies and identifies that policy.

`fulfillment` is `fulfilled` when the answer demonstrably satisfies the requirement, `unfulfilled` when it omits a required item or contradicts the available evidence, and `uncertain` when the available material cannot establish correctness. Unsupported is not automatically false: an invented-looking revenue assertion without any revenue evidence remains unsupported with uncertain fulfillment; a 90-day answer against an applicable 30-day policy is unfulfilled.

`claims` evaluates assertions or transformations in the observed answer. Each claim quotes the answer and records supported, unsupported or uncertain. Its requirements describe the evidence needed to support that claim. Unsupported output must not be treated as a correct answer simply because a similar number appeared somewhere earlier.

Each requirement uses `alternatives`: OR across alternatives, AND across the sources inside one alternative. All requirements for a claim/task are AND. Example: either the old pricing chunk OR its freshly retrieved identical copy supports a fee. A comparison may instead need BOTH a standard-delivery chunk and an express-delivery chunk inside one alternative. Empty alternatives mean no supported source set is established, not a free satisfied condition. A supported claim requires at least one requirement with nonempty alternatives; abstention/meta-claims can be uncertain when they cannot be supported by a positive source span.

`source_kind` is the single kind of all sources across the alternatives, or `mixed` if kinds differ; use `unknown` only when no sources can be established. Quotes are exact nonempty substrings of their source text. Retrieval provenance needs document/version/chunk metadata on the source part. Current input/retrieval may support a claim without creating a historical edge.

## Historical edges and certainty

`historical_dependencies` points only to visible earlier-run parts. Link an edge to target claims, target task requirements, or both. Relations are:

- `used`: visible structural or textual evidence supports reuse; this is still observational.
- `possibly_used`: plausible reliance remains ambiguous.
- `not_needed_given_visible_alternative`: another visible source supplies the required information; this says nothing definitive about the path actually used.
- `contradicted_or_superseded`: an earlier value is corrected or replaced; it is not authoritative support for the new factual claim.

Repeated retrieval alone cannot justify `used`. A source may be part of the provenance of an erroneous extrapolation: keep the answer unsupported and describe only the observable link. Prior output can be authoritative for “rewrite your option B” while being unreliable for a factual question about the world.

Necessity values are inferred, tested, unknown. Authored synthetic task truth is stored outside this label format. The current validator rejects `tested`; future intervention records must identify the removal/replacement, alternative evidence availability, replay settings, repeated full-context controls, and outcome assessor. A model assertion or exact citation is not an intervention.

## Coverage and provenance

`complete` means the supplied, visible evidence was assessed, not that causal necessity or exhaustive latent requirements were proved. Missing visible text requires partial/unassessable plus coverage notes. If the target answer is absent, task requirements can still be partial; never fabricate claims. Unknown visibility or truncated candidate coverage must be reported, and those cases must not become negative reuse examples.

Host envelope (outside model output): input SHA256; normalizer/selector, prompt and schema versions; code revision; backend and model identifier; reasoning/settings; response/request IDs; timestamp; attempt number; status; token usage; priced/unpriced cost coverage; human review or disagreement if any. Cache by every input/configuration component that can change labels. Freeze accepted labels for reproducible downstream fits.

Do not compute token-weighted dependency recall from counts of whole documents or coarse edges. It requires a pinned tokenizer, exact supported spans, overlap unions and a stated denominator. The starter supports part/requirement-level checks only. Preserve unknowns and report both macro company/thread results and volume-weighted totals when population evaluation is added.
