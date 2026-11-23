"""Zero-shot teacher scores with vLLM: yes/no margin or expected 0-3 grade for each (topic, segment) pair.

Writes a TREC run and a JSONL score cache to outputs/llm_reranking/teachers/<topics>/<teacher>_<prompt>.*
and prints dev metrics when qrels exist. The cache is reused for distillation and augmentation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import typer

from retrieval_ranking_lab.data.trec_rag import SegmentStore, load_topic_set
from retrieval_ranking_lab.eval.metrics import evaluate, format_table
from retrieval_ranking_lab.rerank.llm_reranker import PROMPT_PREFIX, PROMPT_SUFFIX, format_content
from retrieval_ranking_lab.retrieval.base import doc_text
from retrieval_ranking_lab.types import read_run, top_k, write_run

app = typer.Typer(add_completion=False)
METRICS = ["ndcg@30", "ndcg@100", "recall@30", "recall@100"]
GRADES = ["0", "1", "2", "3"]
GRADED_SYSTEM = "You are a search quality rater judging how relevant a passage is to a user's information need."
GRADED_USER = """Rate the passage on this scale:
3 = dedicated to the information need and contains the exact answer
2 = answers part of the information need, or the answer is mixed with unrelated content
1 = related to the topic but does not answer it
0 = unrelated

Information need: {query}

Passage: {doc}

Answer with a single digit: 0, 1, 2, or 3."""


def token_logprobs(top: dict, token_ids: list[int]) -> tuple[list[float], int]:
    """Logprobs of token_ids from a top-k dict; a missing token gets the smallest listed logprob minus 1."""
    floor = min(entry.logprob for entry in top.values()) - 1.0
    values = [top[token_id].logprob if token_id in top else floor for token_id in token_ids]
    return values, sum(token_id not in top for token_id in token_ids)


@app.command()
def main(
    model: str = typer.Option(..., help="e.g. Qwen/Qwen3.5-9B, Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
    prompt: str = typer.Option("yes_no", help="yes_no (student template) | graded (0-3, expected grade)"),
    topics: str = typer.Option("rag24-dev", help="rag24-dev | rag24-train"),
    first_stage_run: Path = typer.Option(None, help="default runs/<topics>/splade.trec"),
    depth: int = 100,
    max_topics: int = typer.Option(0, help="score only the first n topics, 0 = all (smoke tests)"),
    groups: Path = typer.Option(None, help="training groups jsonl; their documents are scored too"),
    max_doc_tokens: int = 400,
    gpu_memory_utilization: float = 0.85,
    dtype: str = typer.Option("float16", help="float16 keeps 8x finer logprobs than bfloat16, so fewer ties"),
    out_dir: Path = Path("outputs/llm_reranking/teachers"),
    name: str = typer.Option(None, help="output name; default <model>_<prompt>"),
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    topic_set = load_topic_set(topics)
    first_stage_run = first_stage_run or Path("runs") / topics / "splade.trec"
    queries = topic_set.judged_queries if topic_set.qrels else topic_set.queries
    candidates = {query_id: docs for query_id, docs in top_k(read_run(first_stage_run), depth).items()
                  if query_id in queries}
    if groups:  # group documents outside the first-stage top-k go after it
        for line in open(groups):
            group = json.loads(line)
            docs = candidates.setdefault(group["query_id"], {})
            for doc_id in group["positive_ids"] + group["negative_ids"]:
                docs.setdefault(doc_id, float("-inf"))
        queries = topic_set.queries
    if max_topics:
        candidates = dict(list(candidates.items())[:max_topics])
    segments = SegmentStore()
    tokenizer = AutoTokenizer.from_pretrained(model)

    def truncate(text: str) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_doc_tokens]
        return tokenizer.decode(ids)

    pairs, prompts = [], []
    prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
    for query_id, docs in candidates.items():
        for doc_id in docs:
            doc = truncate(doc_text(segments[doc_id]))
            if prompt == "yes_no":
                ids = prefix_ids + tokenizer.encode(format_content(queries[query_id], doc), add_special_tokens=False) + suffix_ids
            else:
                messages = [{"role": "system", "content": GRADED_SYSTEM},
                            {"role": "user", "content": GRADED_USER.format(query=queries[query_id], doc=doc)}]
                text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=False)
                ids = tokenizer.encode(text, add_special_tokens=False)
            pairs.append((query_id, doc_id))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
    typer.echo(f"{len(pairs)} pairs, {len(candidates)} topics, longest prompt {max(len(p['prompt_token_ids']) for p in prompts)} tokens")

    engine = LLM(model=model, max_model_len=1024, gpu_memory_utilization=gpu_memory_utilization, dtype=dtype,
                 limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=True)
    start_time = time.perf_counter()
    outputs = engine.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=True)
    seconds_per_query = (time.perf_counter() - start_time) / max(len(candidates), 1)

    target_ids = ([tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")] if prompt == "yes_no"
                  else [tokenizer.convert_tokens_to_ids(grade) for grade in GRADES])
    run, cache, missing = {}, [], 0
    for (query_id, doc_id), output in zip(pairs, outputs):
        values, missed = token_logprobs(output.outputs[0].logprobs[0], target_ids)
        missing += missed
        if prompt == "yes_no":
            score = values[1] - values[0]  # equals logit(yes) - logit(no)
        else:
            weights = [math.exp(value) for value in values]
            score = sum(grade * weight for grade, weight in enumerate(weights)) / sum(weights)
        run.setdefault(query_id, {})[doc_id] = score
        cache.append({"query_id": query_id, "doc_id": doc_id, "score": score, "logprobs": values})

    tied = sum(len(docs) - len(set(docs.values())) for docs in run.values())
    for query_id, docs in run.items():  # break the remaining ties by first-stage rank
        first_stage = list(candidates[query_id])
        for rank, doc_id in enumerate(first_stage):
            docs[doc_id] += 1e-6 * (len(first_stage) - rank)
    typer.echo(f"tied scores before the first-stage tie-break: {tied} of {len(pairs)}")
    name = name or f"{model.split('/')[-1]}_{prompt}"
    target_dir = out_dir / topics
    target_dir.mkdir(parents=True, exist_ok=True)
    write_run(run, target_dir / f"{name}.trec", name)
    with open(target_dir / f"{name}.jsonl", "w") as file:
        file.writelines(json.dumps(entry) + "\n" for entry in cache)
    typer.echo(f"target tokens outside the top-20: {missing} of {len(pairs) * len(target_ids)}; {seconds_per_query:.3f} s/query")
    if topic_set.qrels and not groups:
        rows = {"splade": evaluate(candidates, topic_set.qrels, METRICS),
                name: evaluate(run, topic_set.qrels, METRICS) | {"s/query": seconds_per_query}}
        (target_dir / f"{name}_summary.json").write_text(json.dumps(rows, indent=2))
        typer.echo(format_table(rows, METRICS))


if __name__ == "__main__":
    app()
