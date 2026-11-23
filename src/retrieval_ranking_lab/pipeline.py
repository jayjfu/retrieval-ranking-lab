"""Two-stage search: retrieve top-k, then rerank top-n."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from retrieval_ranking_lab.rerank.base import Reranker
from retrieval_ranking_lab.retrieval.base import Retriever
from retrieval_ranking_lab.types import Corpus, Queries, Run


def make_retriever(name: str) -> Retriever:
    """bm25 | splade | hybrid (RRF of BM25 + SPLADE-v3)."""
    if name == "bm25":
        from retrieval_ranking_lab.retrieval.bm25 import BM25Retriever

        return BM25Retriever()
    if name == "splade":
        from retrieval_ranking_lab.retrieval.splade import SpladeRetriever

        return SpladeRetriever()
    if name == "hybrid":
        from retrieval_ranking_lab.retrieval.hybrid import HybridRetriever

        return HybridRetriever([make_retriever("bm25"), make_retriever("splade")])
    raise ValueError(f"unknown retriever {name!r}")


def make_reranker(name: str | None, model_name: str | None = None, adapter_path: str | None = None,
                  batch_size: int | None = None) -> Reranker | None:
    """none | cross-encoder | llm."""
    if name in (None, "none", ""):
        return None
    if name == "cross-encoder":
        from retrieval_ranking_lab.rerank.cross_encoder import DEFAULT_CROSS_ENCODER, CrossEncoderReranker

        return CrossEncoderReranker(model_name or DEFAULT_CROSS_ENCODER, batch_size=batch_size or 128)
    if name == "llm":
        from retrieval_ranking_lab.rerank.llm_reranker import DEFAULT_LLM_RERANKER, LLMReranker

        if adapter_path:  # base model comes from the adapter config
            return LLMReranker(model_name, adapter_path=adapter_path, batch_size=batch_size or 16)
        return LLMReranker(model_name or DEFAULT_LLM_RERANKER, batch_size=batch_size or 16)
    raise ValueError(f"unknown reranker {name!r}")


@dataclass
class TwoStagePipeline:
    retriever: Retriever | None
    reranker: Reranker | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def search(self, queries: Queries, corpus: Corpus | None, k_retrieve: int = 1000, k_rerank: int = 100,
               first_stage: Run | None = None) -> tuple[Run, Run]:
        """Returns (first-stage run, final run). Pass first_stage to rerank an existing run."""
        start = time.perf_counter()
        if first_stage is None:
            first_stage = self.retriever.search(queries, num_results=k_retrieve)
        self.timings["retrieve_s"] = time.perf_counter() - start
        if self.reranker is None:
            return first_stage, first_stage
        start = time.perf_counter()
        final = self.reranker.rerank(queries, corpus, first_stage, depth=k_rerank)
        self.timings["rerank_s"] = time.perf_counter() - start
        return first_stage, final
