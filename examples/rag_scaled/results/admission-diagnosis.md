# Compache admission diagnosis

All three stopped because the frozen gate estimated the next request above its **796,400-token admission limit**, with every earlier retrieval protected and therefore no eligible retrieval available to compact. The blocked turn was not sent to the model. This reproduces each recorded stop using the frozen execution source and policy, the actual earlier answer texts, fixture timestamps, and the runner's 16,000-token planning overhead. It is an analysis of these three synthetic conversations, not a production prevalence estimate or a proof that any particular protected source is safe to remove.

| Arm / stopped run | Earlier retrievals protected | Earlier protected retrieval tokens | Current retrieval tokens | Estimated request | Above limit | Window |
|---|---:|---:|---:|---:|---:|---|
| `compache-1` / `scaled-26-07-run-020` | 19 / 19 | 740,074 | 38,973 | 797,556 | 1,156 | warm |
| `compache-2` / `scaled-24-07-run-019` | 18 / 18 | 745,841 | 41,414 | 805,284 | 8,884 | warm |
| `compache-3` / `scaled-29-07-run-017` | 16 / 16 | 743,096 | 46,464 | 807,650 | 11,250 | cold |

Counts and token estimates use the frozen policy's `ceil(UTF-8 bytes / 4) + 6` per part. The requests also contain input and actual-output parts plus the 16,000-token planning allowance. The context setting is 828,400 tokens with a 32,000-token reserve. Serialized request sizes were 3.14–3.18 MB, below the separate 16 MB byte limit: the estimated-token limit, not the byte limit, caused these decisions. No retrieval had previously been removed, and the stopped decisions removed zero parts. The current retrieval is not a compaction candidate; the prior ones all fail the protection check. [Stops: arm 1](live/runs/compache-1/stopped.json), [arm 2](live/runs/compache-2/stopped.json), [arm 3](live/runs/compache-3/stopped.json).

## Why the old retrievals were protected

The frozen policy has `protect_query_matches: true` and `retrieval_grace_runs: 0`. Its guard protects an earlier retrieval if **either** a prior output contains its `document_id` string **or** a non-common stemmed query term appears in its text or document ID, unless the current retrieval is an exact duplicate under its four-field comparison. Here, every old retrieval matched the citation condition. Lexical matching also protected 19 of 19, 16 of 18 and 15 of 16 respectively; the remaining two and one were citation-only. No old chunk was an exact duplicate of the current chunk under that comparison.

The citation rule operates on the document ID alone, not the cited chunk or occurrence. Multiple generated retrieval occurrences share an ID such as `company-24-pricing-register`, and the live harness asks for citations. For example, the actual turn-2 answer in `compache-2` cited `[company-24-pricing-register v2]`; at the blocked turn the guard treats that as a citation of the **turn-8** retrieval with the same document ID, even though the turn-2 answer predates that occurrence. The turn-8 chunk also matched lexical terms, so this example illustrates the rule's breadth rather than showing that citation matching alone caused that chunk's retention. The cited text is in the [actual answer ledger](live/runs/compache-2/answers.jsonl); all per-occurrence reasons and output indices are in the [machine-readable reconstruction](admission-diagnosis.json).

The lexical guard removes question words appearing in more than half of all historical and current retrieval vocabularies, but remaining generic words still match long documents. Examples are:

- `compache-1`: `requir` matched 9 old chunks, `procurement` 10, and `calculate`, `plann`, `total`, and `unit` 5 each. All 19 were also citation-protected.
- `compache-2`: `cover` matched 9, `original` 8, and `compare` and `procurement` 6 each. Two people-register chunks were protected by citation without a remaining lexical match.
- `compache-3`: `allowance`, `annual`, `procurement`, and `requir` matched 6 each; one warranty-register chunk was protected by citation without a remaining lexical match.

These terms are stemmed forms from the actual blocked questions. The corresponding wide matches arise from long procedural and register passages; they are lexical associations, not demonstrated dependencies. The profile describes return and retention behavior; it is not consulted by `protected_sources` to choose protected chunks. The frozen policy and code implement the actual decisions: [policy](live/policy.json), [guard](live/source/compache-0.4.0-scaled-harness/src/compache/optimizer/gate.py#L194), [compaction and admission](live/source/compache-0.4.0-scaled-harness/src/compache/optimizer/gate.py#L296), and [runner's stop handling](live/source/compache-0.4.0-scaled-harness/scripts/run_live_comparison.py#L535).

## Planning estimates versus observed inputs

The preceding completed call is the closest measured comparison. The stopped requests have no provider usage because they were not sent.

| Arm / last completed turn | Gate estimate, including 16,000 planning overhead | Provider-reported input tokens | Estimate minus measured |
|---|---:|---:|---:|
| `compache-1` / 19 | 758,402 | 798,187 | −39,785 (−5.0%) |
| `compache-2` / 18 | 763,736 | 805,291 | −41,555 (−5.2%) |
| `compache-3` / 16 | 761,037 | 804,285 | −43,248 (−5.4%) |

This is a material difference on these inputs, but the two counters have different construction: the planner uses bytes, per-part overhead and a fixed allowance, while the provider reports the actual fresh ephemeral-thread request including native runtime framing. The first completed-call estimates exceeded reported inputs by 3,618–4,815 tokens; the gap changed sign and grew as these passages accumulated. This comparison does not isolate tokenizer error from framing or prove what the unsent blocked request would have consumed. Actual usage comes from the [arm 1 ledger](live/runs/compache-1/ledger.jsonl), [arm 2 ledger](live/runs/compache-2/ledger.jsonl), and [arm 3 ledger](live/runs/compache-3/ledger.jsonl).

The mechanism is an interaction between an algorithmic guard that protects whole occurrences based on broad document-ID and lexical matches, and this workload's very large repeated-domain bundles, reused document IDs and citation-oriented answer prompt. The evidence demonstrates over-broad *matching* at document-versus-occurrence granularity; it does not establish that these guards should be removed or that any particular compaction preserves answer quality. Since all three arms stopped early, these observations do not establish full-thread savings or completion behavior.
