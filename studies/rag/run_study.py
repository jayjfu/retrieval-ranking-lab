"""RAG study on RAG25 (test) or RAG24 (dev): generate cited answers from a run's top-k segments, then score nuggets and support.

Conditions are run files. With the organizer RankQwen3-32B run, top 20 / 50 contexts match the rag25_qwen3_20/50_ag baselines.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import DEFAULT_DATA_DIR, SegmentStore, load_rag24_nuggets, load_rag25
from retrieval_ranking_lab.eval.metrics import format_table
from retrieval_ranking_lab.eval.overview_2025 import NUGGETS_AUTO_ASSIGNMENT, SUPPORT_AUTO
from retrieval_ranking_lab.generators import make_generator
from retrieval_ranking_lab.retrieval.base import doc_text
from retrieval_ranking_lab.types import ranked_doc_ids, read_run
from studies.rag.nugget_eval import (NUGGET_BATCH_SIZE, nugget_prompts, nugget_scores, parse_nugget_labels, parse_support,
                                     support_prompt, support_scores)
from studies.rag.rag_components import (answer_text, build_messages, export_trec_rag, parse_cited_sentences,
                                        truncate_to_words)

app = typer.Typer(add_completion=False)
METRICS = ["strict_vital", "vital", "subnarrative_coverage", "support_precision", "support_recall", "words"]


def select_context(run, query_id: str, size: int, max_per_document: int) -> list[str]:
    """Top segments of a run; with max_per_document > 0, at most that many segments of one document."""
    if not max_per_document:
        return ranked_doc_ids(run, query_id, size)
    selected, per_document = [], {}
    for segment_id in ranked_doc_ids(run, query_id):
        document_id = segment_id.split("#")[0]
        if per_document.get(document_id, 0) < max_per_document:
            per_document[document_id] = per_document.get(document_id, 0) + 1
            selected.append(segment_id)
            if len(selected) == size:
                break
    return selected


def generate_answers(records: list[dict], segments, generator_spec: str) -> None:
    generator = make_generator(generator_spec, max_tokens=1024, max_model_len=32768)
    prompts = [build_messages(record["query"], [doc_text(segments[doc_id]) for doc_id in record["context_ids"]])
               for record in records]
    for record, answer in zip(records, generator.generate(prompts)):
        record["raw_answer"] = answer
        record["sentences"] = truncate_to_words(parse_cited_sentences(answer, len(record["context_ids"])))
    if hasattr(generator, "shutdown"):
        generator.shutdown()


def judge_answers(records: list[dict], topics, segments, judge_spec: str) -> None:
    nugget_jobs, support_jobs = [], []
    for record_index, record in enumerate(records):
        nuggets = topics.nuggets.get(record["query_id"], [])
        for batch_index, prompt in enumerate(nugget_prompts(record["query"], answer_text(record["sentences"]), nuggets)):
            nugget_jobs.append((record_index, batch_index, prompt))
        for sentence_index, sentence in enumerate(record["sentences"]):
            if sentence["citations"]:
                passage = doc_text(segments[record["context_ids"][sentence["citations"][0]]])
                support_jobs.append((record_index, sentence_index, support_prompt(sentence["text"], passage)))

    # 35B-A3B GPTQ weights take 24.4 GB; ~2 GB of the 32 GB GPU is used by the desktop
    judge = make_generator(judge_spec, max_tokens=128, max_model_len=8192, gpu_memory_utilization=0.90)
    outputs = judge.generate([job[2] for job in nugget_jobs] + [job[2] for job in support_jobs])
    if hasattr(judge, "shutdown"):
        judge.shutdown()

    nugget_outputs, support_outputs = outputs[: len(nugget_jobs)], outputs[len(nugget_jobs):]
    batch_texts: dict[int, dict[int, str]] = {}
    for (record_index, batch_index, _), text in zip(nugget_jobs, nugget_outputs):
        batch_texts.setdefault(record_index, {})[batch_index] = text
    for record_index, record in enumerate(records):
        nuggets = topics.nuggets.get(record["query_id"], [])
        labels = []
        for batch_index in range(0, -(-len(nuggets) // NUGGET_BATCH_SIZE)):
            count = len(nuggets[batch_index * NUGGET_BATCH_SIZE : (batch_index + 1) * NUGGET_BATCH_SIZE])
            labels += parse_nugget_labels(batch_texts.get(record_index, {}).get(batch_index, ""), count)
        record["nugget_labels"] = labels
        record["support_labels"] = [None] * len(record["sentences"])
    for (record_index, sentence_index, _), text in zip(support_jobs, support_outputs):
        records[record_index]["support_labels"][sentence_index] = parse_support(text)


def score(record: dict, topics) -> dict[str, float]:
    nuggets = topics.nuggets.get(record["query_id"], [])
    return (nugget_scores(nuggets, record["nugget_labels"]) | support_scores(record["support_labels"])
            | {"words": float(len(answer_text(record["sentences"]).split()))})


def nan_mean(values) -> float:
    values = [value for value in values if value == value]
    return statistics.fmean(values) if values else float("nan")


@app.command()
def main(
    runs: list[str] = typer.Option(["rankqwen3_32b=data/trec_rag/rag25_run_rankqwen3_32b.trec"],
                                   help="context run as label=path.trec, or a path (label = file stem)"),
    context_sizes: str = "20,50",
    max_per_document: str = typer.Option("0", help="comma list of segment caps per document, 0 = no cap"),
    generator: str = typer.Option("vllm:Qwen/Qwen3.5-9B", help="vllm:<hf id> | anthropic:<model id>"),
    judge: str = typer.Option("vllm:Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", help="nugget and support assessor"),
    all_topics: bool = typer.Option(False, help="generate for all 105 topics (only 22 are scored)"),
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = typer.Option(Path("outputs/rag")),
    team_id: str = "retrieval_ranking_lab",
    name: str = typer.Option("default", help="experiment name; outputs go to <out_dir>/<topic set>/<generator>/<name>/"),
    max_topics: int = typer.Option(0, help="first n scored topics only, 0 = all (smoke tests)"),
    topic_set: str = typer.Option("rag25", help="rag24-dev (56 topics with nuggets, tuning) | rag25 (22 topics, test)"),
):
    if topic_set not in ("rag24-dev", "rag25"):
        raise typer.BadParameter("topic set is rag24-dev or rag25")
    topics = load_rag25(data_dir) if topic_set == "rag25" else load_rag24_nuggets(data_dir)
    queries = topics.queries if all_topics else {query_id: text for query_id, text in topics.queries.items() if query_id in topics.nuggets}
    if max_topics:
        queries = dict(list(queries.items())[:max_topics])
    segments = SegmentStore()
    context_size_values = [int(size) for size in context_sizes.split(",")]
    cap_values = [int(cap) for cap in max_per_document.split(",")]

    conditions = {}
    for spec in runs:
        label, _, run_path = spec.rpartition("=")
        conditions[label or Path(run_path).stem] = Path(run_path)
    records = []
    for label, run_path in conditions.items():
        run = read_run(run_path)
        for context_size in context_size_values:
            for cap in cap_values:
                for query_id, query in queries.items():
                    records.append({"condition": label, "context_size": context_size, "max_per_document": cap,
                                    "query_id": query_id, "query": query,
                                    "context_ids": select_context(run, query_id, context_size, cap)})
    typer.echo(f"{len(records)} answers with {generator}")
    generate_answers(records, segments, generator)

    judged = [record for record in records if record["query_id"] in topics.nuggets]
    judge_answers(judged, topics, segments, judge)

    generator_tag = generator.split(":", 1)[1].split("/")[-1]
    output_dir = out_dir / topic_set / generator_tag / name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {} if topic_set != "rag25" else {key: NUGGETS_AUTO_ASSIGNMENT.get(key, {}) | SUPPORT_AUTO.get(key, {})
            for key in list(NUGGETS_AUTO_ASSIGNMENT) + [key for key in SUPPORT_AUTO if key not in NUGGETS_AUTO_ASSIGNMENT]}
    with open(output_dir / "answers.jsonl", "w") as file:
        for record in records:
            if "nugget_labels" in record:
                record["scores"] = score(record, topics)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    for label in conditions:
        for context_size, cap in [(size, cap) for size in context_size_values for cap in cap_values]:
            subset = [record for record in records if record["condition"] == label
                      and record["context_size"] == context_size and record["max_per_document"] == cap]
            run_id = f"{generator_tag}-{label}-top{context_size}" + (f"-cap{cap}" if cap else "")
            export_trec_rag(subset, output_dir / f"{run_id}.jsonl", team_id=team_id, run_id=run_id)
            scored = [record["scores"] for record in subset if "scores" in record]
            rows[run_id] = {name: nan_mean(scores[name] for scores in scored) for name in METRICS}

    table = format_table(rows, METRICS)
    judge_note = f"Judge: {judge}. Overview rows use GPT-4.1 (nuggets) and GPT-OSS-120B (support, 17 topics)."
    (output_dir / "summary.md").write_text(f"# {topic_set}: {generator}\n\n{judge_note}\n\n{table}\n")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    typer.echo(f"\n{judge_note}\n{table}\n\n-> {output_dir}")


if __name__ == "__main__":
    app()
