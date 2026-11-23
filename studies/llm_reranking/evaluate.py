"""Compare rerankers on the same first-stage candidates.

Tune on rag24-dev. Evaluate a frozen configuration on rag25 once; every rag25 evaluation is appended to
outputs/llm_reranking/rag25_evaluations.jsonl so all tried configurations can be reported.
Reranker specs: ce:<hf id>, llm:<hf id> (zero-shot), lora:<adapter dir>.
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import typer

from retrieval_ranking_lab.data.trec_rag import DEFAULT_DATA_DIR, SegmentStore, load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.eval.overview_2025 import RETRIEVAL_NIST
from retrieval_ranking_lab.types import read_run, top_k, write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@100"]


def load_reranker(spec: str, batch_size: int):
    kind, _, target = spec.partition(":")
    if kind == "ce":
        from retrieval_ranking_lab.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(target)
    from retrieval_ranking_lab.rerank.llm_reranker import LLMReranker

    if kind == "llm":
        return LLMReranker(target, batch_size=batch_size)
    if kind == "lora":
        return LLMReranker(None, adapter_path=target, batch_size=batch_size)
    raise ValueError(f"unknown reranker spec {spec!r}")


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@app.command()
def main(
    topics: str = typer.Option("rag24-dev", help="rag24-dev (tuning) | rag25 (test, logged)"),
    first_stage_run: Path = typer.Option(None, help="default: runs/<topics>/splade.trec"),
    rerankers: str = "ce:cross-encoder/ms-marco-MiniLM-L6-v2",
    depth: int = 100,
    batch_size: int = 16,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = typer.Option(Path("outputs/llm_reranking")),
    name: str = typer.Option(None, help="output folder name, e.g. the run name; default <first stage>@<depth>"),
):
    if topics not in ("rag24-dev", "rag25"):
        raise typer.BadParameter("evaluate on rag24-dev for tuning or rag25 for the final test")
    first_stage_run = first_stage_run or Path("runs") / topics / "splade.trec"
    topic_set = load_topic_set(topics, data_dir)
    queries = topic_set.judged_queries
    candidates = {query_id: docs for query_id, docs in top_k(read_run(first_stage_run), depth).items()
                  if query_id in queries}
    segments = SegmentStore()

    rows = dict(RETRIEVAL_NIST) if topics == "rag25" else {}
    rows[first_stage_run.stem] = evaluate(candidates, topic_set.qrels, METRICS)
    output_dir = out_dir / topics / (name or f"{first_stage_run.stem}@{depth}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in rerankers.split(","):
        reranker = load_reranker(spec, batch_size)
        synchronize_cuda()
        start = time.perf_counter()
        reranked = reranker.rerank(queries, segments, candidates, depth=depth)
        synchronize_cuda()
        elapsed = time.perf_counter() - start
        run_name = spec.replace("/", "_")
        write_run(reranked, output_dir / f"{run_name}.trec", tag=run_name, depth=100)
        rows[spec] = evaluate(reranked, topic_set.qrels, METRICS) | {"s/query": elapsed / max(len(queries), 1)}
        if topics == "rag25":
            entry = {"time": datetime.now().isoformat(timespec="seconds"), "reranker": spec,
                     "first_stage_run": str(first_stage_run), "depth": depth, **rows[spec]}
            with open(out_dir / "rag25_evaluations.jsonl", "a") as log_file:
                log_file.write(json.dumps(entry) + "\n")
        del reranker
        gc.collect()
        torch.cuda.empty_cache()

    table = format_table(rows, METRICS + ["s/query"])
    header = f"# {topics} reranking: {first_stage_run.stem} top-{depth}, {len(queries)} judged topics"
    (output_dir / "summary.md").write_text(f"{header}\n\n{table}\n")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    typer.echo(f"\n{table}\n-> {output_dir}")


if __name__ == "__main__":
    app()
