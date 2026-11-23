"""Ensemble teacher score caches: sum of per-topic z-scores. Writes <teachers>/<topics>/<name>.jsonl (+ .trec)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.types import write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@30", "recall@100"]


def load_z_scores(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for line in open(path):
        row = json.loads(line)
        run.setdefault(row["query_id"], {})[row["doc_id"]] = row["score"]
    for query_id, docs in run.items():
        mean, spread = statistics.mean(docs.values()), statistics.pstdev(docs.values()) or 1.0
        run[query_id] = {doc_id: (value - mean) / spread for doc_id, value in docs.items()}
    return run


@app.command()
def main(
    teacher: list[str] = typer.Option(..., help="cache name, repeatable, e.g. Qwen3.5-9B_yes_no"),
    name: str = typer.Option(..., help="e.g. ensemble_9b-yes_no_35b-a3b-graded"),
    topics: str = "rag24-dev",
    teacher_dir: Path = Path("outputs/llm_reranking/teachers"),
):
    caches = [load_z_scores(teacher_dir / topics / f"{cache}.jsonl") for cache in teacher]
    fused = {query_id: {doc_id: sum(cache[query_id][doc_id] for cache in caches) for doc_id in docs}
             for query_id, docs in caches[0].items()}
    out = teacher_dir / topics / name
    write_run(fused, out.with_suffix(".trec"), name)
    with open(out.with_suffix(".jsonl"), "w") as file:
        for query_id, docs in fused.items():
            for doc_id, score in docs.items():
                file.write(json.dumps({"query_id": query_id, "doc_id": doc_id, "score": score,
                                       "teacher_z_scores": [cache[query_id][doc_id] for cache in caches]}) + "\n")
    qrels = load_topic_set(topics).qrels
    if qrels:
        typer.echo(format_table({name: evaluate(fused, qrels, METRICS)}, METRICS))
    typer.echo(f"{sum(map(len, fused.values()))} pairs, {len(fused)} topics -> {out.with_suffix('.jsonl')}")


if __name__ == "__main__":
    app()
