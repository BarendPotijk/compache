# Fixed-schedule GoodAI LTM pilot

The adapter reads the published GPT-4o master log from GoodAI's Benchmark 3, 32k memory-span configuration, plus the three published definitions for each of Colours, NameList and Shopping. It verifies each source file against its pinned Git object. No benchmark records are bundled into the package. [Pinned source revision](https://github.com/GoodAI/goodai-ltm-benchmark/tree/188e7618413775f1ce783763d5ee0b5ccd4c31c9) · [Benchmark paper](https://papers.neurips.cc/paper_files/paper/2024/file/4aedf0cba303537fcb6cf948bb41b2df-Paper-Datasets_and_Benchmarks_Track.pdf)

```sh
git clone https://github.com/GoodAI/goodai-ltm-benchmark.git /tmp/goodai-ltm
git -C /tmp/goodai-ltm checkout --detach 188e7618413775f1ce783763d5ee0b5ccd4c31c9
python benchmarks/public_chat/goodai.py --checkout /tmp/goodai-ltm --output goodai.json
```

Each source reset starts a new independent case. An instance qualifies only if its entire published script appears in order in the same reset block. The adapter retains its live user messages, including interleaved updates to other qualifying tasks and explicitly requested memory resets. The sequence stops at the last scoreable check in each block. Each arm generates its own updates and scored answers. All original unscored trivia user/assistant pairs before each live turn enter the prompt as frozen conversation history. The original model outputs on selected tasks and source evaluations never enter the prompt. The checked-in adapter and the supplied questions form a paired replay of the three named task families. Native scheduler behavior, discarded task families and resetting filler lengths can produce different outcomes on the complete suite.

| Reset block | Live user turns | Frozen trivia exchanges | Graded probes |
|---|---:|---:|---:|
| 1 | 4 | 32 | 1 |
| 2 | 33 | 63 | 5 |

The terminal answer of each probe is graded. Colours uses the case-insensitive substring in the original colour evaluator. NameList awards expected names recovered divided by the larger of the submitted and expected list sizes, ignoring case. Shopping scores one third for item coverage, one third for matching quantities and one third for avoiding extra items, with the upstream substring rule for inflections. The portable parser accepts JSON arrays and fenced JSON arrays. Shopping credit is deduplicated by expected item, and a strict full pass requires exactly one matching quantity per expected item. Unscored updates contribute to the billed user-turn denominator. Answer quality uses the six probes as its denominator. [Colour grader](https://github.com/GoodAI/goodai-ltm-benchmark/blob/188e7618413775f1ce783763d5ee0b5ccd4c31c9/datasets/colours.py) · [NameList grader](https://github.com/GoodAI/goodai-ltm-benchmark/blob/188e7618413775f1ce783763d5ee0b5ccd4c31c9/datasets/name_list.py) · [Shopping grader](https://github.com/GoodAI/goodai-ltm-benchmark/blob/188e7618413775f1ce783763d5ee0b5ccd4c31c9/datasets/shopping.py)

GoodAI releases the framework under MIT and credits TriviaQA as its filler source under Apache 2.0. The selected task definitions come from the procedural Colours, Shopping and NameList generators. The frozen filler replies are the selected published source model's answers. GoodAI's README lists distinct conditions for additional task families. The adapted histories contain ordinary conversation messages and no new RAG documents. Report the observed input tokens, sampled return schedule and six-probe quality separately from GoodAI's full native score. [MIT notice](GOODAI-LICENSE.txt) · [GoodAI usage and attribution notes](https://github.com/GoodAI/goodai-ltm-benchmark/tree/188e7618413775f1ce783763d5ee0b5ccd4c31c9#licence-and-usage)
