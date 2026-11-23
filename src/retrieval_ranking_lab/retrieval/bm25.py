from __future__ import annotations

from retrieval_ranking_lab.data.trec_rag import BM25_INDEX
from retrieval_ranking_lab.java_setup import ensure_java_home
from retrieval_ranking_lab.retrieval.base import Retriever, hits_to_run
from retrieval_ranking_lab.types import Queries, Run


class BM25Retriever(Retriever):
    """Pyserini prebuilt Lucene index. Anserini defaults bm25_k1=0.9, bm25_b=0.4."""

    name = "bm25"

    def __init__(self, index_name: str = BM25_INDEX, bm25_k1: float = 0.9, bm25_b: float = 0.4, threads: int = 8):
        ensure_java_home()
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        self.searcher.set_bm25(bm25_k1, bm25_b)
        self.threads = threads

    def search(self, queries: Queries, num_results: int = 100) -> Run:
        query_ids = list(queries)
        results = self.searcher.batch_search([queries[query_id] for query_id in query_ids], query_ids,
                                             k=num_results, threads=self.threads)
        return hits_to_run(query_ids, results)
