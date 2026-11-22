from __future__ import annotations

from retrieval_ranking_lab.retrieval.base import Retriever
from retrieval_ranking_lab.types import Queries, Run


def reciprocal_rank_fusion(runs: list[Run], rrf_k: int = 60, depth: int = 1000) -> Run:
    """RRF (Cormack et al., 2009): score = sum over runs of 1 / (rrf_k + rank)."""
    fused: Run = {}
    for run in runs:
        for query_id, docs in run.items():
            scores = fused.setdefault(query_id, {})
            ranking = sorted(docs.items(), key=lambda item: -item[1])[:depth]
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return fused


class HybridRetriever(Retriever):
    """RRF over several retrievers, e.g. BM25 + SPLADE-v3."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, candidate_k: int = 1000):
        self.retrievers, self.rrf_k, self.candidate_k = retrievers, rrf_k, candidate_k

    def search(self, queries: Queries, k: int = 100) -> Run:
        runs = [retriever.search(queries, k=max(k, self.candidate_k)) for retriever in self.retrievers]
        fused = reciprocal_rank_fusion(runs, rrf_k=self.rrf_k)
        return {query_id: dict(sorted(docs.items(), key=lambda item: -item[1])[:k]) for query_id, docs in fused.items()}
