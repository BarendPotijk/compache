# OpenAI execution mode: implementation plan

Status: specification, not a working API client in this milestone. The package input builder, schema and validator are the common components for both execution modes. A coding agent uses its existing model session; the API worker uses the user's OpenAI API key. These are separate authentication and billing paths.

## Local worker first

Accept a normalized trace export, explicit target scope, model identifier, maximum calls/tokens/cost, and `OPENAI_API_KEY` from environment or a user-chosen .env file. Do not require hosting raw traces or credentials in a Compache service. Run normalization and candidate extraction locally; send the selected prefix/evidence packet to OpenAI. A configured key does not itself expand the approved dataset scope. Existing approval to process a named dataset with OpenAI is sufficient; do not repeatedly ask.

Use the Responses API with `text.format` type `json_schema`, `strict:true`, and the shared schema. Supply the instructions from SKILL.md and label-contract.md as the policy; place the evidence packet in a distinct user-data message. Do not allow tools or arbitrary URLs during annotation. Set `store:false` for application-state storage; this is not a promise of zero provider retention. Pin SDK/model versions supported by the customer's account and verify their capabilities before a paid run.

The worker owns normalization, byte hashes, prompt compilation, retries, budgets, validation, provenance and resumable output. The model owns semantic judgments only. A narrow worker is sufficient; no autonomous browsing/coding agent is needed for the API path.

## Request/attempt lifecycle

1. Validate the input packet; compute its immutable hash and full configuration cache key.
2. Reuse an accepted cached label for that exact key, or reserve a bounded request budget.
3. Submit one run packet. For very long threads, use audited candidate selection and coverage reporting; never silently truncate. Selection recall must be measured separately from labeling recall.
4. Accept only a completed, non-refusal response with valid structured content. Schema adherence does not check evidence IDs, quotes, temporal order, source kinds or semantic correctness; run the shared validator and sample semantic review.
5. Record timeout, rate limit, refusal, incomplete, parse failure and semantic-validation failure separately. Avoid hidden duplicate retry layers: the OpenAI Python SDK retries some failures by default. Choose one bounded retry owner and retain every attempt's IDs and known usage. A timeout can have unknown billing, so do not report its cost as zero.
6. Preserve failed/unassessable cases in planned denominators. Do not turn them into “no dependency.” Stop at the user-configured budget, preserving a resumable cursor and input/config hashes.
7. Freeze accepted labels, then run the deterministic profile fitting step. Keep labels and reports separate from original Langfuse observations. Publishing scores or synthetic data is a distinct requested operation.

## Product surface

The package already provides local-file `traces normalize`, per-run `dependencies prepare`, and `dependencies validate`. See the repository's `docs/trace-contracts.md` for their exact arguments. Normalization currently accepts only the declared Compache RAG envelope.

Still to implement: native Langfuse export adapters and bounded fetching, batch packet selection, `dependencies label --backend openai` with resumable attempts and budgets, and deterministic fitting over accepted labels. The API labeling and fitting commands do not exist yet.

The coding-agent route uses the same prepared packets and import/validation contract. Install the skill in that agent's supported skill directory; its local model writes the label JSONL. Contract parity does not imply identical model judgments. Compare both modes on held-out fixtures and preserve disagreements.

## Sources checked 2026-11-16

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): strict schema, refusals, incomplete responses and semantic validation.
- [OpenAI Python SDK](https://github.com/openai/openai-python): retry configuration and request IDs.
- [Langfuse sessions](https://langfuse.com/docs/observability/features/sessions): session grouping; actual generation content remains essential.
- [Compache design](https://github.com/BarendPotijk/compache/blob/a8e6adf329127a6e102bd63f4c7bc70bcd7ce1f2/docs/design.md): offline statistical fitting; the optional judge feeds frozen evidence, not online decisions.
