"""Shared types and TREC-format I/O.

Queries = {query_id: text}
Qrels   = {query_id: {doc_id: grade}}
Run     = {query_id: {doc_id: score}}, higher is better
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]
Corpus = Mapping[str, dict[str, str]]  # doc_id -> {"text", "title", "url"}


def read_qrels(path: str | Path) -> Qrels:
    qrels: Qrels = {}
    with open(path) as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4:
                query_id, _, doc_id, grade = parts[:4]
                qrels.setdefault(query_id, {})[doc_id] = int(grade)
    return qrels


def write_run(run: Run, path: str | Path, tag: str = "retrieval_ranking_lab", depth: int | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for query_id, docs in run.items():
            ranking = sorted(docs.items(), key=lambda item: -item[1])[:depth]
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                file.write(f"{query_id} Q0 {doc_id} {rank} {score:.9g} {tag}\n")


def read_run(path: str | Path) -> Run:
    run: Run = {}
    with open(path) as file:
        for line in file:
            query_id, _, doc_id, _rank, score, *_ = line.split()
            run.setdefault(query_id, {})[doc_id] = float(score)
    return run


def top_k(run: Run, depth: int) -> Run:
    return {query_id: dict(sorted(docs.items(), key=lambda item: -item[1])[:depth]) for query_id, docs in run.items()}


def ranked_doc_ids(run: Run, query_id: str, depth: int | None = None) -> list[str]:
    doc_ids = [doc_id for doc_id, _ in sorted(run.get(query_id, {}).items(), key=lambda item: -item[1])]
    return doc_ids if depth is None else doc_ids[:depth]
