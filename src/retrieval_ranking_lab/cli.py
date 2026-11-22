"""Usage: python -m retrieval_ranking_lab <command> --help"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import DEFAULT_DATA_DIR, load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate as evaluate_run
from retrieval_ranking_lab.eval.metrics import format_table
from retrieval_ranking_lab.types import read_run, top_k, write_run

app = typer.Typer(add_completion=False, help="Retrieval Ranking Lab")
METRICS = ["ndcg@30", "ndcg@100", "recall@100"]  # TREC RAG 2025 R task


@app.command("download-data")
def download_data(data_dir: Path = DEFAULT_DATA_DIR, indexes: bool = typer.Option(True, help="also fetch indexes")):
    """Fetch topics, qrels, nuggets, organizer runs and the BM25 / SPLADE-v3 indexes."""
    from retrieval_ranking_lab.data.trec_rag import BM25_INDEX, SPLADE_INDEX, download_assets

    download_assets(data_dir)
    if indexes:
        from pyserini.util import download_prebuilt_index

        for index_name in (BM25_INDEX, SPLADE_INDEX):
            typer.echo(download_prebuilt_index(index_name))


@app.command()
def run(
    topics: str = typer.Option("rag25", help="rag25 (test) | rag24-dev (tuning) | rag24-train | rag24-all"),
    retriever: str = typer.Option("bm25", help="bm25 | splade | hybrid"),
    first_stage_run: Path = typer.Option(None, help="rerank this run instead of retrieving"),
    reranker: str = typer.Option("none", help="none | cross-encoder | llm"),
    rerank_model: str = typer.Option(None, help="HF model id"),
    adapter_path: str = typer.Option(None, help="LoRA adapter for --reranker llm"),
    k_retrieve: int = 1000,
    k_rerank: int = 100,
    all_topics: bool = typer.Option(False, help="all topics, not only judged ones"),
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = typer.Option(Path("runs"), help="runs/<topics>/<name>.trec"),
    tag: str = typer.Option(None, help="run name"),
):
    """Retrieve (or load a run), optionally rerank, save top-100 TREC runs, print metrics."""
    from retrieval_ranking_lab.pipeline import TwoStagePipeline, make_reranker, make_retriever

    topic_set = load_topic_set(topics, data_dir)
    queries = topic_set.queries if all_topics else topic_set.judged_queries
    first_stage = None
    if first_stage_run:
        first_stage = {query_id: docs for query_id, docs in read_run(first_stage_run).items() if query_id in queries}
    pipeline = TwoStagePipeline(None if first_stage else make_retriever(retriever),
                                make_reranker(reranker, rerank_model, adapter_path))
    corpus = None
    if pipeline.reranker is not None:
        from retrieval_ranking_lab.data.trec_rag import SegmentStore

        corpus = SegmentStore()
    first, final = pipeline.search(queries, corpus, k_retrieve, k_rerank, first_stage=first_stage)

    base_name = first_stage_run.stem if first_stage_run else retriever
    output_dir = out_dir / topics
    rows = {base_name: evaluate_run(top_k(first, 100), topic_set.qrels, METRICS)}
    if not first_stage_run:
        write_run(first, output_dir / f"{base_name}.trec", tag=base_name, depth=k_retrieve)
    if pipeline.reranker is not None:
        run_name = tag or f"{base_name}+{pipeline.reranker.name.replace('/', '_').replace(':', '-')}@{k_rerank}"
        write_run(final, output_dir / f"{run_name}.trec", tag=run_name, depth=100)
        rows[run_name] = evaluate_run(top_k(final, 100), topic_set.qrels, METRICS)
    typer.echo(format_table(rows, METRICS))
    typer.echo(json.dumps({name: round(seconds, 1) for name, seconds in pipeline.timings.items()}))


@app.command()
def fuse(
    runs: list[Path],
    out: Path = typer.Option(..., help="fused run path"),
    rrf_k: int = 60,
    depth: int = 1000,
):
    """Reciprocal rank fusion of saved runs."""
    from retrieval_ranking_lab.retrieval.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([read_run(run_path) for run_path in runs], rrf_k=rrf_k, depth=depth)
    write_run(fused, out, tag=out.stem, depth=depth)
    typer.echo(f"{len(fused)} topics -> {out}")


@app.command()
def evaluate(
    runs: list[Path],
    topics: str = typer.Option("rag25", help="rag25 (test) | rag24-dev (tuning) | rag24-train | rag24-all"),
    metrics: str = ",".join(METRICS),
    with_overview: bool = typer.Option(True, help="add overview Table 5 reference rows"),
    data_dir: Path = DEFAULT_DATA_DIR,
):
    """Evaluate TREC run files (top 100) against NIST qrels."""
    qrels = load_topic_set(topics, data_dir).qrels
    metric_names = metrics.split(",")
    rows = {}
    if with_overview and topics == "rag25":
        from retrieval_ranking_lab.eval.overview_2025 import RETRIEVAL_NIST

        rows.update(RETRIEVAL_NIST)
    for run_path in runs:
        rows[run_path.stem] = evaluate_run(top_k(read_run(run_path), 100), qrels, metric_names)
    typer.echo(format_table(rows, metric_names))


if __name__ == "__main__":
    app()
