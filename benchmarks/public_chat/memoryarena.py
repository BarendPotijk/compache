"""Build a paired, frozen-search adaptation of two public MemoryArena conversations.

Retrieval runs locally using the seven pinned BrowseComp-Plus corpus shards.
Queries depend exclusively on public user questions. The manifest
and benchmark references reside with the evaluator; contexts contain source
passages only. See the study README for the scope of the adapted score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from math import log1p

RELEASE = "67ec2ba295867dbb4ce001a1dd9cc5c0e95978a6"
SOURCE_DIGEST = "b445ee36fa3ccb9ad08eae9e7adda86bbc64f14f1e2a0682a8b2085cdb8e4c0e"
CORPUS_RELEASE = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
CORPUS = "Tevatron/browsecomp-plus-corpus"
SOURCE_URL = f"https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/{RELEASE}/progressive_search/data.jsonl"

SHARD_DIGESTS = {
    "train-00000-of-00007.parquet": "7c07f9e23b1ca548110fd831714cadc67d44db5223bace6e45fcaa795d3153d0",
    "train-00001-of-00007.parquet": "e92d8202e0f656a85b262153dbcd22ecf80ea2d0c96d9884f9c8e25480b869ab",
    "train-00002-of-00007.parquet": "0e4113a4503342527258d8f2c49877747435f3e65bfe1f7306b4f488c8d225fe",
    "train-00003-of-00007.parquet": "0ceea5e703332a2e3ce700f641273400d84583fad84b659d3248ed06d3a9fef3",
    "train-00004-of-00007.parquet": "15b62914ddc3de6946893c770f07d5d84d29646e833ca1447955668f2b57940c",
    "train-00005-of-00007.parquet": "a9a75708ad37c522e93a774e5a968a3129e12b0559971c8f950a5628e0201df0",
    "train-00006-of-00007.parquet": "290062b60c1a6ebba7d5469a37a431f0a2596e68788295284b1b2d35db07b62c"
}

# Registered before viewing the reference answers. Each bag uses only words
# already seen on that thread, including the current question. Manual query
# compression is shared across every policy and preserves the search budget.
QUERIES = {
    6: [
        "college major French established",
        "French Battalion Drum thesis graduated college student",
        "French Battalion Drum thesis writer broadcasting network major",
        "French Battalion Drum thesis writer college graduate",
        "French Battalion Drum thesis college writer broadcasting network French",
    ],
    29: [
        "composite novel",
        "workshop annually composite novel",
        "composite novel workshop intuition story interview",
        "published composite novel workshop story",
        "discontent intuition story composite novel workshop interview",
        "music awards nomination composite novel workshop",
        "international music awards shows nominations",
        "nomination category year discontent workshop composite novel international music awards",
        "composite novel discontent intuition nomination category workshop interview",
    ],
}
STOP = frozenset("a an as about and are at be been before by can certain did does for from had has have his how identify in individual is it may name of on or said some that the their them there these they this to used was were what when where which who with would year years".split())


def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 1 and word not in STOP]


def nominated(rows):
    openings = ("what is the name of the college", "what is a composite novel")
    selected = [min((r for r in rows if r["questions"][0].lower().startswith(start)), key=lambda r: r["id"])
                for start in openings]
    if tuple(r["id"] for r in selected) != tuple(QUERIES):
        raise ValueError("Changed pinned pilot selection")
    return selected


def read_release(source):
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_DIGEST:
        raise ValueError("Progressive Search release failed its pinned checksum")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 221 or sum(len(r["questions"]) for r in rows) != 1641:
        raise ValueError("Unexpected release dimensions")
    return nominated(rows)


def queries(row):
    observed = ""
    out = []
    for question, query in zip(row["questions"], QUERIES[row["id"]], strict=True):
        observed += " " + question
        if not set(tokens(query)).issubset(set(tokens(observed))):
            raise ValueError(f"Retrieval query includes unseen terms on case {row['id']}: {query}")
        current = " ".join(dict.fromkeys(tokens(question)))[:350]
        out.append((query, current))
    return out


def index_shard(args):
    import pyarrow.parquet as pq

    path, vocabulary = args
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["docid", "text"]):
        ids, texts = (column.to_pylist() for column in batch.columns)
        for docid, body in zip(ids, texts):
            words = re.findall(r"[a-z0-9]+", body.lower())
            counts = Counter(w for w in words if w in vocabulary)
            rows.append((docid, len(words), counts))
    return rows


def fetch(rows, path, shards):
    """BM25 on physical pinned shards. Only query terms need postings."""
    import pyarrow.parquet as pq

    if path.exists():
        raise FileExistsError(path)
    paths = sorted(shards.glob("*.parquet"))
    if {p.name for p in paths} != SHARD_DIGESTS.keys():
        raise ValueError("Provide all seven shards from the pinned corpus")
    for shard in paths:
        with shard.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != SHARD_DIGESTS[shard.name]:
                raise ValueError(f"Corpus shard failed its pinned checksum: {shard.name}")
    terms = sorted({q for r in rows for pair in queries(r) for q in pair})
    vocabulary = {w for q in terms for w in tokens(q)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        indexed = [doc for partition in pool.map(index_shard, [(p, vocabulary) for p in paths]) for doc in partition]
    if len(indexed) != 100195 or len(set(i for i, _, _ in indexed)) != len(indexed):
        raise ValueError("Unexpected corpus dimensions")
    frequencies = Counter(t for _, _, counts in indexed for t in counts)
    avgdl = sum(length for _, length, _ in indexed) / len(indexed)
    idf = {t: log1p((len(indexed)-df+.5)/(df+.5)) for t, df in frequencies.items()}
    scores = {q: [] for q in terms}
    term_sets = {q: set(tokens(q)) for q in terms}
    for docid, length, counts in indexed:
        norm = 1.2 * (.25 + .75 * length / avgdl)
        vec = {t: idf[t] * f * 2.2 / (norm + f) for t, f in counts.items()}
        for q, bag in term_sets.items():
            score = sum(vec[w] for w in sorted(bag & vec.keys()))
            if score > 0:
                scores[q].append((score, docid))
    ranked = {q: [i for _, i in sorted(hits, key=lambda h: (-h[0], h[1]))[:6]] for q,hits in scores.items()}
    wanted = {i for ids in ranked.values() for i in ids}
    found = {}
    for filename in paths:
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=128):
            ids = batch.column("docid").to_pylist()
            for pos, docid in enumerate(ids):
                if docid in wanted:
                    found[docid] = {k: batch.column(k)[pos].as_py() for k in ("docid","text","url")}
    result = {q: [found[i] for i in ids] for q, ids in ranked.items()}
    path.write_text(json.dumps(result, ensure_ascii=False))


def passage(doc, words, limit=3500):
    # Score overlapping paragraph-sized windows by distinctive query-term
    # coverage. The passage is a contiguous substring of the published row.
    body = doc["text"]
    if len(body) <= limit:
        return 0, body
    starts = range(0, len(body)-300, limit//2)
    def coverage(start):
        body_terms = set(tokens(body[start:start+limit]))
        return sum(min(10,len(term)) for term in words & body_terms)
    best = max(starts, key=coverage)
    return best, body[best:best+limit]


def contexts(terms, hits):
    # Reciprocal-rank fusion over the current and history-aware query.
    docs, weights = {}, {}
    for query in dict.fromkeys(terms):
        for rank, doc in enumerate(hits[query], 1):
            key = doc["docid"]
            docs[key] = doc
            weights[key] = weights.get(key, 0) + 1/(10+rank)
    result = []
    for key in sorted(docs, key=lambda k: (-weights[k], k))[:5]:
        doc = docs[key]
        start, chunk = passage(doc, set(tokens(" ".join(terms))))
        result.append(dict(id=f"bc-{key}", version=CORPUS_RELEASE[:12], chunk_id=str(start),
                           text=chunk, url=doc["url"], char_offset=start,
                           source_sha256=sha256(doc["text"].encode()).hexdigest()))
    return result


def build(rows, hits, fixture):
    cases = []
    for row in rows:
        turns = []
        for q, a, term in zip(row["questions"], row["answers"], queries(row), strict=True):
            docs = contexts(term, hits)
            if not docs:
                raise ValueError(f"Empty retrieval for {row['id']}, turn {len(turns)+1}")
            turns.append({"question":q, "contexts":docs, "search_queries":list(term),
                          "score":{"kind":"memoryarena_reference","reference":a}})
        cases.append({"id":f"arena-{row['id']:03}", "source_id":row["id"],
                      "instructions":"Give a direct, brief answer to the latest question using its earlier constraints. If several answers are requested, list them. Use the supplied search passages and conversation.",
                      "turns":turns})
    fixture.write_text(json.dumps({
        "benchmark":"memoryarena","release":RELEASE,"license":"CC BY 4.0 (benchmark annotations only)",
        "corpus":CORPUS,"corpus_release":CORPUS_RELEASE,"source_sha256":SOURCE_DIGEST,
        "comparison":"Fixed-search adaptation; retrieved passages are independent of released answers. No native agent-driven web search.",
        "cases":cases}, ensure_ascii=False, indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["fetch","build"])
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--hits",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--shards",type=Path)
    args=parser.parse_args()
    rows=read_release(args.release)
    if args.action=="fetch":
        if not args.shards: parser.error("--shards is required for fetch")
        fetch(rows,args.hits,args.shards)
    else:
        if not args.out: parser.error("--out is required for build")
        build(rows,json.loads(args.hits.read_bytes()),args.out)


if __name__=="__main__":
    main()
