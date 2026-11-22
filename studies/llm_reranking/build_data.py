"""Reranker training groups from RAG24 qrels on the same segmented corpus.

One group per (topic, positive). Positives: grade >= min_positive_grade. Negatives: judged grade 0.
Validation-loss groups: --split rag24-dev --max-negatives-per-topic 7 freezes 1 positive + 7 negatives.
Splits follow the protocol in load_topic_set: train on rag24-train, tune on rag24-dev, test on rag25.
Decontamination drops RAG24 topics similar to any RAG25 topic and every RAG25-judged segment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from tqdm.auto import tqdm

from retrieval_ranking_lab.data.trec_rag import DEFAULT_DATA_DIR, SegmentStore, load_rag25, load_topic_set
from retrieval_ranking_lab.retrieval.base import doc_text

app = typer.Typer(add_completion=False)


def similar_topics(train_queries: dict[str, str], test_queries: dict[str, str], threshold: float) -> dict[str, dict]:
    """Train topics whose TF-IDF cosine to any test topic exceeds threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_ids, test_ids = list(train_queries), list(test_queries)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([train_queries[q] for q in train_ids] + [test_queries[q] for q in test_ids])
    similarity = (vectorizer.transform([train_queries[q] for q in train_ids])
                  @ vectorizer.transform([test_queries[q] for q in test_ids]).T).toarray()
    removed = {}
    for row, query_id in enumerate(train_ids):
        best = int(similarity[row].argmax())
        if similarity[row, best] > threshold:
            removed[query_id] = {"similarity": round(float(similarity[row, best]), 3), "rag25_topic": test_ids[best],
                                 "rag24_text": train_queries[query_id], "rag25_text": test_queries[test_ids[best]]}
    return removed


@app.command()
def main(
    split: str = typer.Option("rag24-train", help="rag24-train | rag24-all (final retrain) | rag24-dev (validation loss)"),
    out: Path = typer.Option(None, help="default: data/reranker_train/<split>[-raw].jsonl"),
    decontaminate: bool = typer.Option(True, help="drop RAG25-similar topics and RAG25-judged segments"),
    similarity_threshold: float = 0.25,
    min_positive_grade: int = 2,
    max_positives_per_topic: int = 32,
    max_negatives_per_topic: int = 64,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 13,
):
    if split not in ("rag24-train", "rag24-all", "rag24-dev"):
        raise typer.BadParameter("groups come from rag24-train, rag24-all, or rag24-dev (validation loss only)")
    out = out or Path("data/reranker_train") / f"{split}{'' if decontaminate else '-raw'}.jsonl"
    rng = random.Random(seed)
    topics = load_topic_set(split, data_dir)
    excluded_topics, excluded_segments = {}, set()
    if decontaminate:
        rag25 = load_rag25(data_dir)
        excluded_topics = similar_topics({q: topics.queries[q] for q in topics.qrels}, rag25.queries,
                                         similarity_threshold)
        excluded_segments = {doc_id for grades in rag25.qrels.values() for doc_id in grades}

    segments = SegmentStore()
    missing_ids: set[str] = set()

    def text_or_none(doc_id: str) -> str | None:
        """Qrels contain placeholder ids absent from the corpus."""
        try:
            return doc_text(segments[doc_id])
        except KeyError:
            missing_ids.add(doc_id)
            return None

    num_groups, num_segments_dropped = 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as file:
        for query_id, grades in tqdm(topics.qrels.items(), desc="topics"):
            if query_id in excluded_topics:
                continue
            kept = {doc_id: grade for doc_id, grade in grades.items() if doc_id not in excluded_segments}
            num_segments_dropped += len(grades) - len(kept)
            positive_ids = sorted((d for d, g in kept.items() if g >= min_positive_grade), key=lambda d: -kept[d])
            negative_ids = [doc_id for doc_id, grade in kept.items() if grade == 0]
            if not positive_ids or not negative_ids:
                continue
            rng.shuffle(negative_ids)
            negatives = [text for text in map(text_or_none, negative_ids[:max_negatives_per_topic]) if text]
            for doc_id in positive_ids[:max_positives_per_topic]:
                positive = text_or_none(doc_id)
                if positive is None or not negatives:
                    continue
                record = {"query_id": query_id, "query": topics.queries[query_id],
                          "positives": [positive], "negatives": negatives}
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_groups += 1

    log = {"split": split, "decontaminate": decontaminate, "similarity_threshold": similarity_threshold,
           "num_topics": len(topics.qrels), "num_topics_removed": len(excluded_topics),
           "num_judgements_removed": num_segments_dropped, "missing_segment_ids": sorted(missing_ids),
           "num_groups": num_groups,
           "removed_topics": excluded_topics}
    out.with_suffix(".log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    typer.echo(f"{num_groups} groups, {len(excluded_topics)} topics and {num_segments_dropped} judgements removed -> {out}")


if __name__ == "__main__":
    app()
