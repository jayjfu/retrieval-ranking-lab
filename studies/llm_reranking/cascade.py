"""Cascades: our reranker over a deep SPLADE-v3 list, then an LLM final stage on its top-k only.

head:    write our top-k as a run, ordered by our scores or by SPLADE-v3 scores (the final stage's start order).
combine: put the final stage's top-k (one run, or RRF of several) above the rest of our ranking,
         and record nDCG, the share of judged documents in the top 30, and the summed cost per query.
Outputs go to outputs/llm_reranking/<topics>/cascade/; RAG25 cascades are logged in rag25_evaluations.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.retrieval.hybrid import reciprocal_rank_fusion
from retrieval_ranking_lab.types import read_run, top_k, write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@30", "recall@100"]


def ranking(docs: dict[str, float]) -> list[str]:
    return sorted(docs, key=lambda doc_id: -docs[doc_id])


@app.command()
def head(
    base: Path = typer.Option(..., help="our reranked run"),
    depth: int = typer.Option(..., help="top-k of our ranking"),
    order: str = typer.Option("ours", help="ours | splade: score the head by our scores or SPLADE-v3 scores"),
    splade: Path = typer.Option(None, help="SPLADE-v3 run for --order splade"),
    out: Path = typer.Option(...),
):
    ours = top_k(read_run(base), depth)
    if order == "splade":
        first_stage = read_run(splade)
        ours = {query_id: {doc_id: first_stage[query_id].get(doc_id, float("-inf")) for doc_id in docs}
                for query_id, docs in ours.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(ours, out, f"head{depth}_{order}")


@app.command()
def combine(
    base: Path = typer.Option(..., help="our reranked run"),
    depth: int = typer.Option(0, help="final-stage depth; 0 = our ranking alone"),
    run: list[str] = typer.Option([], help="final-stage run over our top-k, label=path; several are fused by RRF"),
    cost: list[str] = typer.Option([], help="component cost as summary.json:key, read from its s/query; repeatable"),
    name: str = typer.Option(...),
    topics: str = "rag24-dev",
    out_dir: Path = Path("outputs/llm_reranking"),
):
    ours = read_run(base)
    members = [top_k(read_run(Path(member.split("=", 1)[1])), depth) for member in run]
    final = {}
    for query_id, docs in ours.items():
        order = ranking(docs)
        if members:
            head_docs = (reciprocal_rank_fusion(members, depth=depth) if len(members) > 1 else members[0]).get(query_id, {})
            head_order = ranking(head_docs)
            order = head_order + [doc_id for doc_id in order if doc_id not in head_docs]
        final[query_id] = {doc_id: float(len(order) - rank) for rank, doc_id in enumerate(order)}

    components = {}
    for source in cost:
        path, key = source.split(".json:", 1)
        components[source] = json.loads(Path(f"{path}.json").read_text())[key]["s/query"]
    qrels = load_topic_set(topics).qrels
    row = evaluate(final, qrels, METRICS)
    judged = [sum(doc_id in qrels[query_id] for doc_id in ranking(final[query_id])[:30]) / 30
              for query_id in qrels if query_id in final]
    row |= {"judged@30": sum(judged) / max(len(judged), 1), "s/query": sum(components.values())}

    target_dir = out_dir / topics / "cascade"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_run(final, target_dir / f"{name}.trec", name, depth=100)
    summary_path = target_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[name] = row | {"base": str(base), "depth": depth, "members": run, "components": components}
    summary_path.write_text(json.dumps(summary, indent=2))
    if topics == "rag25":
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "reranker": f"cascade:{name}",
                 "base": str(base), "depth": depth, "members": run, **row}
        with open(out_dir / "rag25_evaluations.jsonl", "a") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    typer.echo(format_table({name: row}, METRICS + ["judged@30", "s/query"]))


if __name__ == "__main__":
    app()
