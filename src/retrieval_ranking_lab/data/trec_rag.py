"""TREC RAG on the MS MARCO V2.1 segmented corpus (113,520,750 segments).

RAG25: 105 narrative topics; NIST qrels, sub-narratives and post-edited nuggets for 22 topics.
RAG24: 301 topics with UMBRELA qrels and 89 with NIST qrels, used for training;
56 of the NIST topics have track nuggets (CoverageBench release), used as the RAG dev set.
Segment text comes from the Pyserini BM25 index, which stores raw segments.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from retrieval_ranking_lab.java_setup import ensure_java_home
from retrieval_ranking_lab.types import Qrels, Queries, read_qrels

DEFAULT_DATA_DIR = Path("data/trec_rag")
BM25_INDEX = "msmarco-v2.1-doc-segmented"
SPLADE_INDEX = "msmarco-v2.1-doc-segmented.splade-v3"

ASSET_URLS = {
    "rag25_topics.jsonl": "https://trec-rag.github.io/assets/jsonl/trec_rag_2025_queries.jsonl",
    "rag25_qrels_nist.txt": "https://trec.nist.gov/data/rag/2025-rag-qrels.txt",
    "rag25_narratives_sub.json":
        "https://trec.nist.gov/data/rag/trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json",
    "rag25_nuggets.jsonl": "https://raw.githubusercontent.com/TREC-RAG/trec-rag-data/main/"
                           "trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl",
    # TREC 2024 RAG nuggets with the track's assignments to submitted answers, 56 topics
    "rag24_nuggets_assignments.jsonl": "https://huggingface.co/datasets/hltcoe/coveragebench/resolve/main/"
                                       "nuggets/rag24_nuggets.jsonl",
    # organizer RankQwen3-32B top-100, the AG task input
    "rag25_run_rankqwen3_32b.trec": "https://raw.githubusercontent.com/castorini/ragnarok_data/main/"
                                    "rag25/retrieve_results/MISC/run.rankqwen3_32b.rag25.txt",
}
PYSERINI_QRELS = {"rag24_qrels_nist.txt": "rag24.test", "rag24_qrels_umbrela.txt": "rag24.test-umbrela-all"}


def download_assets(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, url in ASSET_URLS.items():
        if not (data_dir / file_name).exists():
            urllib.request.urlretrieve(url, data_dir / file_name)
    ensure_java_home()
    from pyserini.search import get_qrels_file, get_topics

    for file_name, pyserini_name in PYSERINI_QRELS.items():
        if not (data_dir / file_name).exists():
            shutil.copy(get_qrels_file(pyserini_name), data_dir / file_name)
    if not (data_dir / "rag24_topics.tsv").exists():
        with open(data_dir / "rag24_topics.tsv", "w") as file:
            for query_id, topic in get_topics("rag24.test").items():
                file.write(f"{query_id}\t{topic['title']}\n")


class SegmentStore(Mapping):
    """Read-only doc_id -> {"text", "title", "url"} backed by the BM25 index."""

    def __init__(self, index_name: str = BM25_INDEX):
        ensure_java_home()
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        self.fetch = lru_cache(maxsize=200_000)(self._fetch)

    def _fetch(self, doc_id: str) -> dict[str, str]:
        doc = self.searcher.doc(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        record = json.loads(doc.raw())
        title = record.get("title", "")
        if record.get("headings"):
            title = f"{title} | {record['headings']}" if title else record["headings"]
        return {"text": record["segment"], "title": title, "url": record.get("url", "")}

    def __getitem__(self, doc_id: str) -> dict[str, str]:
        return self.fetch(doc_id)

    def __iter__(self):
        raise TypeError("SegmentStore is lookup-only")

    def __len__(self) -> int:
        return self.searcher.num_docs


@dataclass
class RagTopics:
    name: str
    queries: Queries
    qrels: Qrels = field(default_factory=dict)
    sub_narratives: dict[str, list[dict]] = field(default_factory=dict)  # {"text", "importance"}
    nuggets: dict[str, list[dict]] = field(default_factory=dict)  # {"text", "importance", "mapped_sub_narrative"}

    @property
    def judged_queries(self) -> Queries:
        return {query_id: text for query_id, text in self.queries.items() if query_id in self.qrels}


def load_rag25(data_dir: Path = DEFAULT_DATA_DIR) -> RagTopics:
    queries = {}
    with open(data_dir / "rag25_topics.jsonl") as file:
        for line in file:
            record = json.loads(line)
            queries[str(record["id"])] = record["title"]
    sub_narratives = {str(record["id"]): record.get("sub_narratives") or []
                      for record in json.loads((data_dir / "rag25_narratives_sub.json").read_text())}
    nuggets = {}
    with open(data_dir / "rag25_nuggets.jsonl") as file:
        for line in file:
            record = json.loads(line)
            nuggets[str(record["qid"])] = record["nuggets"]
    return RagTopics("rag25", queries, read_qrels(data_dir / "rag25_qrels_nist.txt"), sub_narratives, nuggets)


def load_rag24(data_dir: Path = DEFAULT_DATA_DIR, qrels_source: str = "umbrela") -> RagTopics:
    """qrels_source: umbrela (301 topics) | nist (89 topics)."""
    queries = {}
    with open(data_dir / "rag24_topics.tsv") as file:
        for line in file:
            query_id, text = line.rstrip("\n").split("\t", 1)
            queries[query_id] = text
    return RagTopics(f"rag24-{qrels_source}", queries, read_qrels(data_dir / f"rag24_qrels_{qrels_source}.txt"))


def load_rag24_nuggets(data_dir: Path = DEFAULT_DATA_DIR) -> RagTopics:
    """RAG dev set: RAG24 NIST topics with track nuggets ({"text", "importance"}); no sub-narratives."""
    topics = load_rag24(data_dir, "nist")
    topics.name = "rag24-dev"
    with open(data_dir / "rag24_nuggets_assignments.jsonl") as file:
        for line in file:  # one row per submitted answer; every row of a topic lists the same nuggets
            record = json.loads(line)
            topics.nuggets.setdefault(record["qid"], [{"text": nugget["text"].strip(), "importance": nugget["importance"]}
                                                      for nugget in record["nuggets"]])
    return topics


TOPIC_SETS = ("rag25", "rag24-dev", "rag24-train", "rag24-all")


def load_topic_set(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> RagTopics:
    """Evaluation protocol.
    rag25: test, 22 NIST-judged topics. rag24-dev: tuning, 89 NIST-judged topics.
    rag24-train: UMBRELA qrels of the other 212 topics. rag24-all: UMBRELA, 301 topics, final retrain only.
    """
    if name == "rag25":
        return load_rag25(data_dir)
    if name == "rag24-dev":
        topics = load_rag24(data_dir, "nist")
        topics.name = name
        return topics
    if name in ("rag24-train", "rag24-all"):
        topics = load_rag24(data_dir, "umbrela")
        if name == "rag24-train":
            dev_ids = set(load_rag24(data_dir, "nist").qrels)
            topics.qrels = {query_id: grades for query_id, grades in topics.qrels.items() if query_id not in dev_ids}
        topics.name = name
        return topics
    raise ValueError(f"unknown topic set {name!r}, choose from {TOPIC_SETS}")
