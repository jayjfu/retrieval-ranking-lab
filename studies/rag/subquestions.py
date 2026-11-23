"""Sub-question retrieval: an LLM splits each topic into sub-questions, SPLADE-v3 retrieves for each, ours reranks,
and the lists are merged with the original topic's ranking into one context run.

decompose: topics -> outputs/rag/<topic set>/subquestions/subquestions.jsonl
retrieve:  sub-questions -> SPLADE-v3 top-100 reranked by ours -> subquestions.trec (query ids "<topic>::<n>")
combine:   original ranking + sub-question rankings -> round-robin or RRF run per topic
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import SegmentStore, load_rag24_nuggets, load_rag25
from retrieval_ranking_lab.retrieval.hybrid import reciprocal_rank_fusion
from retrieval_ranking_lab.types import ranked_doc_ids, read_run, write_run

app = typer.Typer(add_completion=False)
DECOMPOSE_PROMPT = (
    "Split the information need below into 2 to 5 self-contained search questions that together cover every part of it. "
    "Each question must make sense on its own. Return only a JSON list of strings.\n\nInformation need: {query}"
)
OURS = "lora:checkpoints/reranker/listwise_lr1e-4_pos32_epoch1_cosine_clip1/step-290"


def topic_queries(topic_set: str) -> dict[str, str]:
    topics = load_rag25() if topic_set == "rag25" else load_rag24_nuggets()
    return {query_id: text for query_id, text in topics.queries.items() if query_id in topics.nuggets}


def parse_questions(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.S)
    try:
        questions = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        questions = []
    if not questions:  # one question per line as a fallback
        questions = [line.strip(" -*0123456789.") for line in text.splitlines() if line.strip().endswith("?")]
    return [str(question).strip() for question in questions if str(question).strip()][:5]


def subquestion_dir(topic_set: str, out_dir: Path) -> Path:
    return out_dir / topic_set / "subquestions"


@app.command()
def decompose(
    topic_set: str = typer.Option("rag24-dev", help="rag24-dev | rag25"),
    generator: str = "vllm:Qwen/Qwen3.5-9B",
    out_dir: Path = Path("outputs/rag"),
):
    from retrieval_ranking_lab.generators import make_generator

    queries = topic_queries(topic_set)
    model = make_generator(generator, max_tokens=512, max_model_len=4096)
    outputs = model.generate([[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}] for query in queries.values()])
    target = subquestion_dir(topic_set, out_dir)
    target.mkdir(parents=True, exist_ok=True)
    counts = []
    with open(target / "subquestions.jsonl", "w") as file:
        for (query_id, query), output in zip(queries.items(), outputs):
            questions = parse_questions(output)
            counts.append(len(questions))
            file.write(json.dumps({"query_id": query_id, "query": query, "sub_questions": questions, "raw": output}) + "\n")
    typer.echo(f"{len(counts)} topics, {sum(counts)} sub-questions, {sum(count == 0 for count in counts)} topics without any")


@app.command()
def retrieve(
    topic_set: str = typer.Option("rag24-dev", help="rag24-dev | rag25"),
    reranker: str = OURS,
    depth: int = 100,
    out_dir: Path = Path("outputs/rag"),
):
    from retrieval_ranking_lab.rerank.llm_reranker import LLMReranker
    from retrieval_ranking_lab.retrieval.splade import SpladeRetriever

    target = subquestion_dir(topic_set, out_dir)
    queries = {}
    for line in open(target / "subquestions.jsonl"):
        record = json.loads(line)
        for index, question in enumerate(record["sub_questions"], start=1):
            queries[f"{record['query_id']}::{index}"] = question
    candidates = SpladeRetriever().search(queries, num_results=depth)
    model = LLMReranker(None, adapter_path=reranker.split(":", 1)[1], batch_size=16)
    reranked = model.rerank(queries, SegmentStore(), candidates, depth=depth)
    write_run(reranked, target / "subquestions.trec", "subquestions_ours", depth=depth)
    typer.echo(f"{len(queries)} sub-questions -> {target / 'subquestions.trec'}")


@app.command()
def combine(
    topic_set: str = typer.Option("rag24-dev", help="rag24-dev | rag25"),
    base: Path = typer.Option(..., help="ours on the original topics"),
    method: str = typer.Option("roundrobin", help="roundrobin | rrf"),
    depth: int = 100,
    out_dir: Path = Path("outputs/rag"),
):
    target = subquestion_dir(topic_set, out_dir)
    original = read_run(base)
    sub_runs: dict[str, dict[str, dict[str, float]]] = {}
    for query_id, docs in read_run(target / "subquestions.trec").items():
        topic_id, _, index = query_id.partition("::")
        sub_runs.setdefault(topic_id, {})[index] = docs
    combined = {}
    for topic_id in topic_queries(topic_set):
        lists = [ranked_doc_ids(original, topic_id)] + [ranked_doc_ids({"q": docs}, "q")
                                                         for _, docs in sorted(sub_runs.get(topic_id, {}).items())]
        if method == "rrf":
            fused = reciprocal_rank_fusion([{"q": {doc_id: -rank for rank, doc_id in enumerate(ranking)}} for ranking in lists])
            order = ranked_doc_ids(fused, "q")
        else:  # take the next unseen segment of each list in turn, original ranking first
            order, seen = [], set()
            for position in range(max(map(len, lists))):
                for ranking in lists:
                    if position < len(ranking) and ranking[position] not in seen:
                        seen.add(ranking[position])
                        order.append(ranking[position])
        combined[topic_id] = {doc_id: float(depth - rank) for rank, doc_id in enumerate(order[:depth])}
    write_run(combined, target / f"subquestions_{method}.trec", f"subquestions_{method}", depth=depth)
    typer.echo(f"{len(combined)} topics -> {target / f'subquestions_{method}.trec'}")


if __name__ == "__main__":
    app()
