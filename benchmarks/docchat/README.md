# DocChat

Long chats over public documents, for measuring what a context policy costs and what it loses. One model writes the conversation, a second one grades, and which turns need earlier context is measured instead of assumed.

## The idea

A fabricated benchmark is only useful if the parts a policy is judged on are not made up. DocChat separates three kinds of input.

| Part | Where it comes from |
|---|---|
| Document text | Public files, pinned by revision and content hash |
| What is retrieved each turn | BM25 over the document, run on the user's own words |
| User messages and reference answers | Written by Claude Haiku 5.5 from one planned excerpt and the earlier turns it builds on |
| **Which turns need history, and which earlier parts** | **Measured.** Haiku answers each turn again with parts of the history withheld, and Claude Sonnet 5.5 grades every answer |
| Who asks, the mix of turn types, thread length, timing | Declared in [`domains.json`](domains.json) and at the top of [`docchat.py`](docchat.py) |

The plan only proposes that a turn builds on an earlier one. If the retriever fetches the old excerpt again, or the question gives the answer away, the turn is labelled as needing no history.

## Domains

Each domain is one common chat experience over one kind of public document. Ten documents per domain.

| Domain | Experience | Source | Licence |
|---|---|---|---|
| `finance` | Working through a company's annual report | Form 10-K filings, [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) | Apache-2.0 |
| `legal` | Reviewing a commercial contract | [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad) | CC BY 4.0 |
| `healthcare` | Asking about a medicine from its official label | [openFDA drug labelling](https://open.fda.gov/apis/drug/label/) | Public domain, CC0 |
| `support` | Getting help from product documentation | [Kubernetes documentation](https://github.com/kubernetes/website) | CC BY 4.0 |
| `hr` | Asking an internal handbook about policies | [TTS Handbook](https://github.com/GSA-TTS/handbook), US General Services Administration | Public domain, CC0 |

Finance, legal and support are three of the five industry domains in RAGBench. The medicine label stands in for its biomedical domain, and the handbook adds the internal helpdesk. [`sources.json`](sources.json) lists every file. Nothing is bundled: the builder downloads the files. The openFDA label is the one source that can change upstream, and a build stops when a hash no longer matches.

## How a thread is built

1. **Plan.** A seeded outline picks a persona, a thread length and, for every turn, a type and an excerpt. The excerpts follow a reader's path through the document: mostly nearby, sometimes a jump.
2. **Write.** Haiku writes the user message and a reference answer from the planned excerpt and the earlier turns. A turn the material cannot carry becomes a plain lookup.
3. **Retrieve.** BM25 returns the six best chunks for the user message. The planned excerpt is always among them.
4. **Verify.** Haiku answers the turn from the current retrievals plus one set of earlier parts at a time. Sonnet checks that the reference answer is supported, writes one to four key facts, and grades each answer against them.
5. **Assemble.** Threads in the [trace format](../../docs/trace-contracts.md), labels, grading notes, a split and a manifest.

| Turn type | What the user does | What it may depend on |
|---|---|---|
| `lookup` | Asks something new | Nothing earlier |
| `follow_up` | Asks on from the previous turn | The previous excerpt |
| `recall` | Returns to a topic from a while ago | An old excerpt |
| `compare` | Sets an earlier answer against a related excerpt | An old answer or excerpt |
| `user_context` | Asks "in my case", after stating a fact about themselves earlier | An old user message |
| `reuse_output` | Asks for an earlier answer as a table, a note or a recalculation | An old answer |
| `synthesis` | Asks for a summary of several earlier turns | Several old answers |

## Labels

Every turn is answered with these sets of earlier parts shown, one at a time: none, the previous turn, and for the turns it builds on their user messages, their answers, their excerpts, and all three together.

- **No history needed:** the answer with nothing shown was graded correct.
- **History needed:** that answer was wrong and at least one set made it right. Each such set is an alternative in the label. The combined set counts only when no single kind was enough.
- **Unknown:** no set gave a correct answer, or the judge found the reference answer unsupported.

`grading/` keeps the key facts and every verdict, so a live run can grade its own answers with the same judge.

## Build

```sh
python benchmarks/docchat/docchat.py build --cache dl --work work --output docchat --dry-run   # plan only, no model calls
PYTHONPATH=src python benchmarks/docchat/docchat.py build --cache dl --work work --output docchat
```

- Model calls go through the `claude` command line (`claude -p`) under your own login. Every answer is stored in `work/calls`, so a stopped build resumes and a second build with another `--clock` asks nothing new.
- `--docs`, `--short`, `--medium` and `--long` set the size: documents per domain and threads of 8 to 12, 25 to 35 and 70 to 90 turns per document.
- `--clock` is `fast`, `mixed` or `slow` (median idle 30 s, 4 min, 30 min), or `tracelab`, which follows quantiles published for coding-agent sessions.
- A third of each domain's documents is the test set, chosen by hash of the document ID. Threads follow their document.

Output: `traces/`, `labels/`, `grading/` and `manifest.json`, one file per thread.

## Run

```sh
export PYTHONPATH=src:benchmarks/docchat
python benchmarks/docchat/run.py cost --data docchat-fast docchat-mixed docchat-slow --output cost.json
python benchmarks/docchat/run.py answers --data docchat-mixed --recipe '<recipe from cost.json>' --work work --output answers.json
```

- `cost` fits a policy on each build's training threads with `planner solve` and prices it on the test threads. Prices come from [`sonnet-5-5.json`](sonnet-5-5.json), Claude Sonnet 5.5 at list price.
- `answers` has a model answer held-out turns twice: with the full history, and with the policy's view, in which archived parts show the package's compaction notice and the model may call the package's reopen tool. The judge grades each answer against the turn's key facts.

## First build

Built on 2026-11-20 from two documents per domain with one thread of each length: 30 threads, 1,227 turns, 15 threads for training and 15 for testing. Long threads hold 240k to 350k tokens.

| Domain | Turns | Need history | Do not | Unknown |
|---|---|---|---|---|
| `finance` | 244 | 31% | 51% | 18% |
| `legal` | 242 | 38% | 55% | 8% |
| `healthcare` | 254 | 28% | 58% | 14% |
| `support` | 254 | 22% | 60% | 18% |
| `hr` | 233 | 27% | 64% | 9% |

- **How far back.** Of the 354 turns that need history, the nearest sufficient evidence is one turn back for 53%, two to five for 22%, six to twenty for 18% and further for 7%.
- **Which kind.** Earlier retrieved text alone is enough for 45% of them, earlier answers alone for 38% and earlier user messages alone for 21%.
- **Plan against measurement.** Half of the planned follow-ups and two thirds of the planned recalls turned out to need no history. In 96% of those cases the retriever had fetched the passage again. One lookup in ten did need the previous turn.

### Cost with Claude Sonnet 5.5

`planner solve` chose the same cut for all three clocks: keep the questions of the last 49 turns and the answers of the last 59, and keep retrieved text for one turn. Only the pings differ.

| Clock | Pings | Keep everything | Fitted policy | Saving |
|---|---|---|---|---|
| `fast` | None | $34 | $19 | 43% |
| `mixed` | Up to 3 | $143 | $24 | 83% |
| `slow` | Up to 3 | $286 | $43 | 85% |

Modeled cost per 1,000 turns on the 15 test threads (625 turns), at list prices and a five-minute cache. The policy's cost includes the modeled cost of fetching back what it archived. It keeps 15% of the tokens, and 75% of the 181 test turns that need history still have enough of it in view (80% on the training threads).

### Answers with Claude Sonnet 5.5

Sonnet 5.5 answered 30 test turns twice and Sonnet 5.5 graded each answer against the turn's key facts.

| Turns | Full history | Policy view |
|---|---|---|
| All 30 | 25 correct | 24 correct |
| 20 that need history | 16 correct | 14 correct |
| 10 that do not | 9 correct | 10 correct |
| Mean request | 120k tokens | 26k tokens |

- The model never called reopen. Where the policy's view lacked something, it answered from what it could see.
- The sample is small: one turn is three points. 24 of the 30 turns come from long threads.

Building the 30 threads took 6,262 writer calls and 1,227 judge calls, about $35 at list prices. The 30-turn answer run took 106 calls and $23, almost all of it the full-history side.

## Limits

- The turn mix, personas, lengths and clocks are assumptions. Report results by domain and turn type, not as one number.
- One retriever. A pipeline that rewrites the query from history fetches old evidence more often and needs history less.
- The writer and the withheld-history answerer are the same model, and the grader is a model. No person has checked the reference answers.
- Each alternative is a set that was enough, not proof that every part in it was needed.
