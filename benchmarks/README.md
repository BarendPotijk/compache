# Benchmarks

One package, three workloads, three different policies. Cost is against keeping the full history, on conversations the fit did not see, under three declared return patterns: users back after 30 seconds, 4 minutes or 30 minutes at the median.

| Dataset | What it is | What the fit chose | Saving |
|---|---|---|---|
| [DocChat](docchat/README.md) | 30 document chats in five domains, 1,227 turns, with retrieval. Dependencies measured by withholding history | Keep questions and answers, keep retrieved text for one turn. 0 to 3 pings | 43% to 85% |
| [BEAM](beam/README.md) | 90 personal chats of 130k to 1M tokens, no retrieval. Evidence from the dataset's graded questions | Keep every user message, drop earlier assistant text. 0 to 3 pings | 57% to 91% |
| LongDS | 68 data-analysis tasks of about 26k tokens. Each turn builds on the recorded work of earlier ones | Keep everything and tune the pings in five of six cases. Cut to the last 20 turns in one | 0% to 32% |

- **The policy follows the workload.** Where old retrieved text is the bulk and is rarely needed, it goes. Where assistant text is the bulk, that goes. Where almost every turn builds on earlier ones, nothing goes and only the pings are tuned.
- **Held out.** The fitted policy was the cheapest candidate on the unseen conversations in 18 of 21 cases across the three datasets, and within 3% in the other three.
- **Answers hold.** On DocChat, Claude Sonnet 5.5 answered 24 of 30 held-out turns correctly from the policy's view and 25 from the full history, on a fifth of the tokens.

Costs are modeled: every request and ping is priced with the provider's cache rules at the threads' own timestamps, and a cut pays for fetching back what it archived. DocChat and BEAM rebuild from the commands on their pages. The LongDS conversion is not in the repository.

## Live

[Live model calls](live/README.md): DocChat conversations played at their users' pace through five ways in, and what the provider billed. On Claude Haiku 5.5 the fitted policies cut the bill by 42% to 49% with as many answers right, whether fitted on past traces or on the model's own notes. On GPT-6 Luna, whose cache outlasts these users' breaks, they saved 2% to 5%.
