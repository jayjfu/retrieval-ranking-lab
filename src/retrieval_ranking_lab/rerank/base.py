from __future__ import annotations

from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from retrieval_ranking_lab.retrieval.base import doc_text
from retrieval_ranking_lab.types import Corpus, Queries, Run


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Scores for (query, doc text) pairs, higher is better."""

    def rerank(self, queries: Queries, corpus: Corpus, run: Run, depth: int = 100, show_progress: bool = True) -> Run:
        """Rescore the top-depth docs. Docs below depth keep their order after the reranked ones."""
        pairs, pair_keys = [], []
        tails: dict[str, list[str]] = {}
        for query_id, docs in run.items():
            if query_id not in queries:
                continue
            ranking = [doc_id for doc_id, _ in sorted(docs.items(), key=lambda item: -item[1])]
            for doc_id in ranking[:depth]:
                pairs.append((queries[query_id], doc_text(corpus[doc_id])))
                pair_keys.append((query_id, doc_id))
            tails[query_id] = ranking[depth:]

        scores = self.score_in_chunks(pairs, show_progress)
        reranked: Run = {}
        for (query_id, doc_id), score in zip(pair_keys, scores):
            reranked.setdefault(query_id, {})[doc_id] = float(score)
        for query_id, tail in tails.items():
            head = reranked.setdefault(query_id, {})
            floor = min(head.values(), default=0.0)
            for offset, doc_id in enumerate(tail, start=1):
                head[doc_id] = floor - offset
        return reranked

    def score_in_chunks(self, pairs, show_progress: bool, chunk_size: int = 4096) -> list[float]:
        scores: list[float] = []
        starts = range(0, len(pairs), chunk_size)
        for start in tqdm(starts, desc=f"rerank[{self.name}]", disable=not show_progress or len(pairs) <= chunk_size):
            scores.extend(self.score(pairs[start : start + chunk_size]))
        return scores
