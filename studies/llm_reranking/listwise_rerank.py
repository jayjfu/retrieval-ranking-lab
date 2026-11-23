"""Zero-shot listwise reranking with a sliding window (RankGPT / RankZephyr style), served by vLLM.

Windows of --window passages move from the bottom of the candidate list to the top with --stride,
so the best passages bubble up. All topics advance one window at a time in a single vLLM batch.
Writes outputs/llm_reranking/<topics>/listwise/<name>.trec and appends to summary.json there;
RAG25 runs are logged in rag25_evaluations.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import SegmentStore, load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.retrieval.base import doc_text
from retrieval_ranking_lab.types import read_run, top_k, write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@30", "recall@100"]
SYSTEM = "You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query."
HEADER = ("I will provide you with {count} passages, each indicated by a numerical identifier []. "
          "Rank the passages based on their relevance to the search query: {query}\n\n")
FOOTER = ("\nSearch Query: {query}\nRank the {count} passages above based on their relevance to the search query. "
          "All the passages should be included and listed using identifiers, in descending order of relevance. "
          "The output format should be [] > [], e.g., [2] > [1]. Only respond with the ranking results, "
          "do not say any word or explain.")


def window_starts(length: int, window: int, stride: int) -> list[int]:
    """Back-to-front window offsets, the last one at 0."""
    starts = list(range(max(length - window, 0), 0, -stride))
    return starts + [0]


def parse_permutation(text: str, count: int) -> tuple[list[int], bool]:
    """Window positions in the model's order; unknown or repeated ids are dropped, missing ones keep their order."""
    seen: list[int] = []
    for match in re.findall(r"\[(\d+)\]", text) or re.findall(r"\d+", text):
        position = int(match) - 1
        if 0 <= position < count and position not in seen:
            seen.append(position)
    complete = len(seen) == count
    return seen + [position for position in range(count) if position not in seen], complete


@app.command()
def main(
    model: str = typer.Option("Qwen/Qwen3.5-9B", help="e.g. Qwen/Qwen3.5-9B, Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
    topics: str = typer.Option("rag24-dev", help="rag24-dev (tuning) | rag25 (test, logged)"),
    first_stage_run: Path = typer.Option(None, help="initial order; default runs/<topics>/splade.trec"),
    name: str = typer.Option(..., help="output name, e.g. qwen9b_splade"),
    depth: int = 100,
    window: int = 20,
    stride: int = 10,
    max_passage_tokens: int = 256,
    max_topics: int = typer.Option(0, help="first n topics only, 0 = all (smoke tests)"),
    gpu_memory_utilization: float = 0.85,
    out_dir: Path = Path("outputs/llm_reranking"),
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    topic_set = load_topic_set(topics)
    queries = topic_set.judged_queries
    first_stage_run = first_stage_run or Path("runs") / topics / "splade.trec"
    candidates = {query_id: docs for query_id, docs in top_k(read_run(first_stage_run), depth).items()
                  if query_id in queries}
    if max_topics:
        candidates = dict(list(candidates.items())[:max_topics])
    rankings = {query_id: sorted(docs, key=lambda doc_id: -docs[doc_id]) for query_id, docs in candidates.items()}

    segments = SegmentStore()
    tokenizer = AutoTokenizer.from_pretrained(model)
    passages = {}
    for doc_ids in rankings.values():
        for doc_id in doc_ids:
            if doc_id not in passages:
                ids = tokenizer.encode(doc_text(segments[doc_id]), add_special_tokens=False)[:max_passage_tokens]
                passages[doc_id] = tokenizer.decode(ids).replace("\n", " ").strip()

    def prompt(query: str, doc_ids: list[str]) -> str:
        body = "".join(f"[{position}] {passages[doc_id]}\n" for position, doc_id in enumerate(doc_ids, start=1))
        content = HEADER.format(count=len(doc_ids), query=query) + body + FOOTER.format(count=len(doc_ids), query=query)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    engine = LLM(model=model, max_model_len=window * (max_passage_tokens + 16) + 1024,
                 gpu_memory_utilization=gpu_memory_utilization, limit_mm_per_prompt={"image": 0, "video": 0},
                 enable_prefix_caching=True)
    sampling = SamplingParams(max_tokens=window * 6 + 16, temperature=0.0)

    start_time, calls, incomplete = time.perf_counter(), 0, 0
    longest = max(map(len, rankings.values()))
    for start in window_starts(longest, window, stride):
        batch = [(query_id, start) for query_id, ranking in rankings.items() if start < len(ranking)]
        prompts = [prompt(queries[query_id], rankings[query_id][start:start + window]) for query_id, start in batch]
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        for (query_id, start), output in zip(batch, outputs):
            current = rankings[query_id][start:start + window]
            order, complete = parse_permutation(output.outputs[0].text, len(current))
            rankings[query_id][start:start + window] = [current[position] for position in order]
            incomplete += not complete
        calls += len(batch)
    elapsed = time.perf_counter() - start_time

    run = {query_id: {doc_id: float(len(ranking) - rank) for rank, doc_id in enumerate(ranking)}
           for query_id, ranking in rankings.items()}
    target_dir = out_dir / topics / "listwise"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_run(run, target_dir / f"{name}.trec", name, depth=depth)
    row = evaluate(run, topic_set.qrels, METRICS) | {"s/query": elapsed / max(len(run), 1)}
    summary_path = target_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    settings = {"model": model, "first_stage_run": str(first_stage_run), "depth": depth, "window": window,
                "stride": stride, "max_passage_tokens": max_passage_tokens, "num_topics": len(run),
                "calls": calls, "incomplete_permutations": incomplete}
    summary[name] = row | settings
    summary_path.write_text(json.dumps(summary, indent=2))
    if topics == "rag25" and not max_topics:
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "reranker": f"listwise:{name}", **settings, **row}
        with open(out_dir / "rag25_evaluations.jsonl", "a") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    typer.echo(format_table({name: row}, METRICS + ["s/query"]))
    typer.echo(f"{calls} window calls, {incomplete} incomplete permutations -> {target_dir / name}.trec")


if __name__ == "__main__":
    app()
