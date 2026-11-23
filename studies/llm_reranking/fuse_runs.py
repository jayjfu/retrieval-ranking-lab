"""Fuse reranked runs over the same candidates: RRF or the sum of per-topic z-scores.

Writes outputs/llm_reranking/<topics>/fusion/<name>.trec and appends to summary.json there.
RAG25 fusions are logged in rag25_evaluations.jsonl like every other test evaluation.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.retrieval.hybrid import reciprocal_rank_fusion
from retrieval_ranking_lab.types import Run, read_run, top_k, write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@30", "recall@100"]


def z_score_sum(runs: list[Run]) -> Run:
    """Sum of per-topic z-scores; a document missing from a run gets that run's lowest z-score."""
    fused: Run = {}
    for run in runs:
        for query_id, docs in run.items():
            mean, spread = statistics.mean(docs.values()), statistics.pstdev(docs.values()) or 1.0
            z_scores = {doc_id: (score - mean) / spread for doc_id, score in docs.items()}
            fused.setdefault(query_id, {})
            for doc_id in set(fused[query_id]) | set(z_scores):
                fused[query_id][doc_id] = fused[query_id].get(doc_id, 0.0)
            floor = min(z_scores.values())
            for doc_id in fused[query_id]:
                fused[query_id][doc_id] += z_scores.get(doc_id, floor)
    return fused


@app.command()
def main(
    run: list[str] = typer.Option(..., help="member run as label=path.trec, repeatable"),
    method: str = typer.Option("rrf", help="rrf | zsum"),
    rrf_k: int = 60,
    depth: int = 100,
    topics: str = typer.Option("rag24-dev", help="rag24-dev (tuning) | rag25 (test, logged)"),
    name: str = typer.Option(None, help="default <method>_<labels joined by +>"),
    out_dir: Path = Path("outputs/llm_reranking"),
):
    if method not in ("rrf", "zsum"):
        raise typer.BadParameter("method is rrf or zsum")
    labels, paths = zip(*(member.split("=", 1) for member in run))
    members = [top_k(read_run(Path(path)), depth) for path in paths]
    fused = reciprocal_rank_fusion(members, rrf_k=rrf_k, depth=depth) if method == "rrf" else z_score_sum(members)
    name = name or f"{method}_{'+'.join(labels)}"

    target_dir = out_dir / topics / "fusion"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_run(fused, target_dir / f"{name}.trec", name, depth=depth)
    row = evaluate(fused, load_topic_set(topics).qrels, METRICS)
    summary_path = target_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[name] = row | {"members": dict(zip(labels, paths)), "method": method, "rrf_k": rrf_k, "depth": depth}
    summary_path.write_text(json.dumps(summary, indent=2))
    if topics == "rag25":
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "reranker": f"fusion:{name}",
                 "members": dict(zip(labels, paths)), "method": method, "depth": depth, **row}
        with open(out_dir / "rag25_evaluations.jsonl", "a") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    typer.echo(format_table({name: row}, METRICS))


if __name__ == "__main__":
    app()
