from __future__ import annotations

from retrieval_ranking_lab.data.trec_rag import SPLADE_INDEX
from retrieval_ranking_lab.java_setup import ensure_java_home
from retrieval_ranking_lab.retrieval.base import Retriever, hits_to_run
from retrieval_ranking_lab.types import Queries, Run


class SpladeRetriever(Retriever):
    """SPLADE-v3 impact index. Queries use Anserini's ONNX encoder (not gated)."""

    name = "splade"

    def __init__(self, index_name: str = SPLADE_INDEX, threads: int = 8):
        ensure_java_home()
        from pyserini.search.lucene import LuceneImpactSearcher

        self.searcher = LuceneImpactSearcher.from_prebuilt_index(index_name, query_encoder="SpladeV3",
                                                                 encoder_type="onnx")
        self.threads = threads

    def search(self, queries: Queries, k: int = 100) -> Run:
        query_ids = list(queries)
        results = self.searcher.batch_search([queries[query_id] for query_id in query_ids], query_ids,
                                             k=k, threads=self.threads)
        return hits_to_run(query_ids, results)
