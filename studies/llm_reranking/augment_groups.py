"""Add unjudged first-stage documents to training groups as extra negatives, labeled only by the teacher.

Per topic: unjudged segments of the first-stage top-k, in rank order, optionally only those with a teacher
z-score <= --max-teacher-score. RAG25-judged segments are dropped. Each group gets the topic's list as
extra_negatives / extra_negative_ids; train.py --extra-negatives-per-group samples from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import SegmentStore, load_rag25, load_topic_set
from retrieval_ranking_lab.retrieval.base import doc_text
from retrieval_ranking_lab.types import read_run, top_k

app = typer.Typer(add_completion=False)


@app.command()
def main(
    groups: Path = Path("data/reranker_train/rag24-train.jsonl"),
    first_stage_run: Path = Path("runs/rag24-train/splade.trec"),
    teacher_scores: Path = Path("outputs/llm_reranking/teachers/rag24-train/ensemble_9b-yes_no_35b-a3b-graded.jsonl"),
    depth: int = 100,
    max_teacher_score: float = typer.Option(None, help="keep unjudged segments with teacher z-score <= this; none = all"),
    max_per_topic: int = 32,
    out: Path = typer.Option(..., help="e.g. data/reranker_train/rag24-train-unjudged.jsonl"),
):
    qrels = load_topic_set("rag24-train").qrels
    excluded_segments = {doc_id for grades in load_rag25().qrels.values() for doc_id in grades}
    teacher = {}
    for line in open(teacher_scores):
        row = json.loads(line)
        teacher[(row["query_id"], row["doc_id"])] = row["score"]
    ranked = top_k(read_run(first_stage_run), depth)
    records = [json.loads(line) for line in open(groups)]
    segments = SegmentStore()

    extras: dict[str, list[str]] = {}
    for query_id in {record["query_id"] for record in records}:
        candidates = [doc_id for doc_id in sorted(ranked[query_id], key=lambda doc_id: -ranked[query_id][doc_id])
                      if doc_id not in qrels.get(query_id, {}) and doc_id not in excluded_segments
                      and (max_teacher_score is None or teacher[(query_id, doc_id)] <= max_teacher_score)]
        extras[query_id] = candidates[:max_per_topic]
    texts = {doc_id: doc_text(segments[doc_id]) for doc_ids in extras.values() for doc_id in doc_ids}

    with open(out, "w") as file:
        for record in records:
            doc_ids = extras[record["query_id"]]
            record |= {"extra_negatives": [texts[doc_id] for doc_id in doc_ids], "extra_negative_ids": doc_ids}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = [len(doc_ids) for doc_ids in extras.values()]
    log = {"groups": str(groups), "max_teacher_score": max_teacher_score, "max_per_topic": max_per_topic,
           "num_topics": len(counts), "num_extra_segments": sum(counts), "min_per_topic": min(counts),
           "topics_below_8": sum(count < 8 for count in counts)}
    out.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
    typer.echo(f"{len(records)} groups, {sum(counts)} extra segments over {len(counts)} topics "
               f"(min {min(counts)} per topic, {log['topics_below_8']} topics below 8) -> {out}")


if __name__ == "__main__":
    app()
