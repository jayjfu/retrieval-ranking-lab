# RAG Study

Cited, report-style answers (at most 400 words) for RAG25 narratives, in the official AG/RAG JSONL format.

## Setup

- Context: top-k segments of a run. The organizer RankQwen3-32B run with k = 20 / 50 matches the `rag25_qwen3_20/50_ag` baselines.
- Generator: Qwen3.5-9B. Judge: Qwen3.5-35B-A3B-GPTQ-Int4.

## Metrics

- `strict_vital`: vital nuggets fully supported by the answer (overview Table 8).
- `vital`: the same, with partial support counted as 0.5.
- `subnarrative_coverage`: sub-narratives with at least one fully supported nugget.
- `support_precision`, `support_recall`: first citation per sentence, weights 1 / 0.5 / 0 (overview Table 12). Uncited sentences count as no support.

The official assessors are GPT-4.1 (nuggets) and GPT-OSS-120B (support), so our scores are close but not identical in setup.

## Run

```bash
python -m studies.rag.run_study --runs data/trec_rag/rag25_run_rankqwen3_32b.trec --context-sizes 20,50
python -m studies.rag.run_study --runs runs/rag25/hybrid+llm-qwen3.5-0.8b-rag24@100.trec --context-sizes 20
```

Outputs are written to `outputs/rag/rag25/<generator>/`.

## Next

- Retrieval quality to answer quality: same generator over BM25, SPLADE-v3, hybrid and reranked runs.
- Generator scale: Qwen3.5 0.8B / 2B / 4B / 9B (27B does not fit on one 32 GB GPU).
- Judge agreement: Qwen3.5 judge sizes against the overview's assessors.

## Reference

### Papers, Reports and Blogs

- TREC 2024 RAG Support Evaluation, human vs LLM judges (202504): [arXiv](https://arxiv.org/abs/2504.15205)
- Great Nugget Recall, automatic nugget evaluation (202504): [arXiv](https://arxiv.org/abs/2504.15068)
- AutoNuggetizer, TREC 2024 RAG nugget results (202411): [arXiv](https://arxiv.org/abs/2411.09607)
- Ragnarök, TREC RAG baselines framework (202406): [arXiv](https://arxiv.org/abs/2406.16828)
- vLLM, PagedAttention serving (202309): [arXiv](https://arxiv.org/abs/2309.06180)
- Lost in the Middle, long-context position effects (202307): [arXiv](https://arxiv.org/abs/2307.03172)
- ALCE, generation with citations (202305): [arXiv](https://arxiv.org/abs/2305.14627)
- Verifiability in Generative Search, citation recall and precision (202304): [arXiv](https://arxiv.org/abs/2304.09848)
- RAG, retrieval-augmented generation (202005): [arXiv](https://arxiv.org/abs/2005.11401)

### Code

- castorini/nuggetizer: [GitHub](https://github.com/castorini/nuggetizer)
- vllm-project/vllm: [GitHub](https://github.com/vllm-project/vllm)
- castorini/ragnarok: [GitHub](https://github.com/castorini/ragnarok)
