# Public source adapters

Use these adapters when you need the original question order, public context and grading format for the **short development pilots**. The long Luna cost and tool-call records are available directly in the [portable trace bundle](../handoff/README.md). The two public suites already informed development.

## GoodAI LTM

The standard-library adapter verifies the pinned task definitions and log, preserves the published reset boundaries and emits the two fixed pilot conversations. Run the commands in the [adapter notes](GOODAI.md). The generated file also contains evaluator fields. Give the host only the current question, released history and visible sources. [Upstream revision](https://github.com/GoodAI/goodai-ltm-benchmark/tree/188e7618413775f1ce783763d5ee0b5ccd4c31c9)

## MemoryArena Progressive Search

Use Python 3.11 or later. Download the pinned `progressive_search/data.jsonl` as `progressive_search.jsonl`, then place all seven pinned BrowseComp-Plus parquet files in `corpus-shards/`. The search step requires PyArrow and checks the release and shard hashes. Both steps run locally on the downloaded files. The fixture records questions, frozen evidence, source identifiers and scoring metadata. Keep the grading material with the evaluator. [Question release](https://huggingface.co/datasets/ZexueHe/memoryarena/tree/67ec2ba295867dbb4ce001a1dd9cc5c0e95978a6/progressive_search) · [Seven source shards](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus/tree/b27b02bc3e45511b8b82a13e6f90ce761df726f6/data)

```sh
python3 benchmarks/public_chat/memoryarena.py fetch --release progressive_search.jsonl --shards corpus-shards --hits arena-hits.json
python3 benchmarks/public_chat/memoryarena.py build --release progressive_search.jsonl --hits arena-hits.json --out memoryarena.json
```

For a small example of the serving and grading contracts that runs without any downloads, start with the [fictional RAG adapter fixture](../handoff/example/README.md).
