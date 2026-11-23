from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval_ranking_lab.types import Queries, Run


def doc_text(doc: dict[str, str]) -> str:
    """Text shown to every model: 'title. text'."""
    title = doc.get("title") or ""
    return f"{title}. {doc['text']}" if title else doc["text"]


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def search(self, queries: Queries, num_results: int = 100) -> Run: ...


def hits_to_run(query_ids: list[str], results: dict) -> Run:
    return {query_id: {hit.docid: float(hit.score) for hit in results.get(query_id, [])} for query_id in query_ids}
