# Benchmarks

The table reports **modeled cost savings** against sending the full history on held-out conversations. Each workload was priced under three return patterns with median pauses of 30 seconds, 4 minutes and 30 minutes.

| Dataset | What it is | What the fit chose | Saving |
|---|---|---|---|
| [DocChat](docchat/README.md) | 30 document chats in five domains, 1,227 turns, with retrieval. Dependencies measured by withholding history | Keep questions and answers, keep retrieved text for one turn. 0 to 3 pings, one-hour cache entries for users who take half an hour | 43% to 89% |
| [BEAM](beam/README.md) | 90 personal chats of 130k to 1M tokens, no retrieval. Evidence from the dataset's graded questions | Keep every user message, drop earlier assistant text. 0 to 3 pings | 57% to 91% |
| LongDS | 68 data-analysis tasks of about 26k tokens. Each turn builds on the recorded work of earlier ones | Keep everything and tune the pings in five of six cases. Cut to the last 20 turns in one | 0% to 32% |

- **The policy follows the workload.** Where old retrieved text is the bulk and is rarely needed, it goes. Where assistant text is the bulk, that goes. Where almost every turn builds on earlier ones, nothing goes and only the pings are tuned.
- **Held out.** The fitted policy was the cheapest candidate on the unseen conversations in 18 of 21 cases across the three datasets, and within 3% in the other three.
- **Separate answer check.** On DocChat, Claude Sonnet 5.5 answered 24 of 30 held-out turns correctly from the policy's view and 25 from the full history, on a fifth of the tokens.

Costs are modeled: every request and ping is priced with the provider's cache rules at the threads' own timestamps, and a cut pays for fetching back what it archived. DocChat and BEAM rebuild from the commands on their pages. The LongDS conversion is not in the repository.

## Live

[Live model calls](live/README.md): DocChat conversations played at their users' pace through five ways in, and what the provider billed. On Claude Haiku 5.5 the fitted policies cut the bill by 42% to 49%. Across four integrations covering the same 52 graded turns each, full history scored 125/208, the trace-fitted policy 135/208 and the notes-fitted policy 132/208. On GPT-6 Luna, whose cache outlasts these users' breaks, they saved 2% to 5% and scored 23/52 and 20/52 respectively, against 24/52 for full history.

## Coding agents

[Agents](agents/README.md): Claude Code and Codex live, next to Headroom and Kompressor, and 174 public runs modeled. Long sessions with Claude Code cost 51% to 85% less, and all configurations passed the 13-prompt task’s 14 hidden tests. In short Claude Code sessions, compache cost 16% less than the baseline and solved 24/34 issues versus 26/34.

## Other tools

[Compare](compare/README.md): Headroom, Kompressor and the cheapest way to keep the whole history cached, against compache. Modeled, 30% to 70% under the cheapest cache setting. In five live chats, compache and full history both scored 33/52 at $1.48 and $2.90 per 1,000 turns respectively. [Read more](../docs/comparisons.md)
