"""IR metrics with trec_eval semantics.

Queries without relevant docs are skipped. Queries missing from the run score 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from retrieval_ranking_lab.types import Qrels, Run

METRIC_PATTERN = re.compile(r"^(ndcg|mrr|recall|p|map)@(\d+)$", re.I)


def ranked_ids(docs: dict[str, float]) -> list[str]:
    # ties broken by doc id descending, as in trec_eval
    return [doc_id for doc_id, _ in sorted(docs.items(), key=lambda item: (item[1], item[0]), reverse=True)]


def ndcg_at_k(ranking: list[str], grades: dict[str, int], k: int) -> float:
    """Linear gain, as in trec_eval ndcg_cut."""
    dcg = sum(grades.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(ranking[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranking[:k]):
        if doc_id in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranking[:k])) / k


def average_precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranking[:k]):
        if doc_id in relevant:
            hits += 1
            total += hits / (rank + 1)
    return total / len(relevant)


def evaluate(
    run: Run,
    qrels: Qrels,
    metrics: Iterable[str] = ("ndcg@30", "ndcg@100", "recall@100"),
    relevance_threshold: int = 1,
    per_query: bool = False,
):
    """Metric names: ndcg@k, mrr@k, recall@k, p@k, map@k."""
    parsed = []
    for metric in metrics:
        match = METRIC_PATTERN.match(metric)
        if not match:
            raise ValueError(f"Unknown metric {metric!r}")
        parsed.append((metric, match.group(1).lower(), int(match.group(2))))

    per_query_scores: dict[str, dict[str, float]] = {}
    for query_id, grades in qrels.items():
        relevant = {doc_id for doc_id, grade in grades.items() if grade >= relevance_threshold}
        if not relevant:
            continue
        ranking = ranked_ids(run.get(query_id, {}))
        scores = {}
        for name, kind, k in parsed:
            if kind == "ndcg":
                scores[name] = ndcg_at_k(ranking, grades, k)
            elif kind == "mrr":
                scores[name] = reciprocal_rank_at_k(ranking, relevant, k)
            elif kind == "recall":
                scores[name] = recall_at_k(ranking, relevant, k)
            elif kind == "p":
                scores[name] = precision_at_k(ranking, relevant, k)
            else:
                scores[name] = average_precision_at_k(ranking, relevant, k)
        per_query_scores[query_id] = scores

    num_queries = max(len(per_query_scores), 1)
    summary = {name: sum(row[name] for row in per_query_scores.values()) / num_queries for name, _, _ in parsed}
    summary["num_q"] = len(per_query_scores)
    return (summary, per_query_scores) if per_query else summary


def format_table(rows: dict[str, dict[str, float]], metrics: list[str] | None = None) -> str:
    """Markdown table, one row per system."""
    if not rows:
        return ""
    metrics = metrics or [name for name in next(iter(rows.values())) if name != "num_q"]
    width = max(len(system) for system in rows) + 2
    lines = [
        "| " + "system".ljust(width) + " | " + " | ".join(f"{name:>10}" for name in metrics) + " |",
        "|" + "-" * (width + 2) + "|" + "|".join("-" * 12 for _ in metrics) + "|",
    ]
    for system, row in rows.items():
        cells = []
        for name in metrics:
            value = row.get(name, float("nan"))
            cells.append(f"{value:>10.4f}" if isinstance(value, float) else f"{value!s:>10}")
        lines.append("| " + system.ljust(width) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)
