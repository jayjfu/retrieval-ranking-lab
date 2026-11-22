# Retrieval Ranking Lab

Two-stage retrieval and two studies on the TREC 2025 RAG track (MS MARCO V2.1 segmented corpus, 113.5M segments).
Results are reported next to the track overview (arXiv:2603.09891) where the setups are comparable.

- [LLM reranking](studies/llm_reranking/README.md): Qwen3.5 rerankers fine-tuned on RAG24, tested on RAG25.
- [RAG](studies/rag/README.md): cited report-style answers, scored with nuggets and citation support.

```
src/retrieval_ranking_lab/
  data/trec_rag.py      topics, NIST qrels, sub-narratives, nuggets, segment lookup
  retrieval/            bm25.py, splade.py (Pyserini prebuilt indexes), hybrid.py (RRF)
  rerank/               cross_encoder.py, llm_reranker.py (yes/no scoring, LoRA adapters)
  pipeline.py           retrieve then rerank
  eval/metrics.py       trec_eval-compatible nDCG, Recall, MRR, P, MAP
  eval/overview_2025.py reference scores from the overview
  generators.py         vLLM and Claude backends
  cli.py                download-data, run, evaluate
studies/llm_reranking/
studies/rag/
```

## Quick Start

Needs Java 21 (Pyserini) and about 110 GB for the BM25 and SPLADE-v3 indexes.

```bash
pip install -r requirements.txt
export PYTHONPATH=src         # run all commands from the repo root
export HF_HUB_DISABLE_XET=1   # hf_xet downloads stall on this machine

python -m retrieval_ranking_lab download-data
python -m retrieval_ranking_lab run --retriever hybrid --k-retrieve 1000
python -m retrieval_ranking_lab run --first-stage-run runs/rag25/hybrid.trec --reranker llm --rerank-model Qwen/Qwen3-Reranker-4B
```

`run` scores the 22 NIST-judged RAG25 topics by default. Add `--all-topics` for all 105.

## Data

| Item | Source |
|---|---|
| Corpus | MS MARCO V2.1 segmented, read through the Pyserini BM25 index |
| RAG25 topics | 105 narratives, `trec_rag_2025_queries.jsonl` |
| RAG25 qrels | NIST, 22 topics, grades 0–4 |
| RAG25 nuggets | post-edited, 22 topics, vital / okay, mapped to sub-narratives |
| RAG25 candidates | organizer RankQwen3-32B top-100 (AG task input) |
| RAG24 qrels | UMBRELA (301 topics) and NIST (89 topics), training only |

## Comparability with the Overview

- Retrieval (Table 5): same topics, qrels and metrics. Our metrics reproduce `rag25_test_rankqwen3` exactly (0.6462 / 0.5844 / 0.2274).
  Our runs were not in the judgment pool, so unjudged segments count as non-relevant. This biases scores downward.
- Rerankers are trained on RAG24 and tuned on RAG24 NIST topics, never on RAG25. See the [study protocol](studies/llm_reranking/README.md#protocol).
- Nuggets (Table 8): same post-edited nuggets and assignment prompt. The assessor is Qwen3.5 instead of GPT-4.1.
- Support (Table 12): same first-citation protocol. The assessor is Qwen3.5 instead of GPT-OSS-120B, on 22 topics instead of 17.

## Models

Qwen3.5 for all LLM roles: generator Qwen3.5-9B, judge Qwen3.5-35B-A3B-GPTQ-Int4, fine-tuning base Qwen3.5-0.8B.
Qwen3.5 has no reranker model, so Qwen3-Reranker is the purpose-built reference.

## Results

Non-LLM baselines on the 22 NIST-judged RAG25 topics (`scripts/run_baselines.sh`):

| System | nDCG@30 | nDCG@100 | R@100 |
|---|---|---|---|
| BM25 | 0.3468 | 0.2900 | 0.1107 |
| SPLADE-v3 | **0.5957** | **0.5389** | **0.2105** |
| BM25 + SPLADE-v3 (RRF) | 0.5292 | 0.5002 | 0.2003 |
| SPLADE-v3 → MiniLM-L6 @100 | 0.5319 | 0.5220 | 0.2105 |
| RRF → MiniLM-L6 @100 | 0.5486 | 0.5009 | 0.2003 |
| *overview* `anserini_bm25_top100` | 0.3468 | 0.2900 | 0.1107 |
| *overview* `rag25_test_splade-v3` | 0.5957 | 0.5387 | 0.2103 |
| *overview* `rag25_test_rankqwen3` | 0.6462 | 0.5844 | 0.2274 |
| *overview* best (`4method_merge`) | 0.6934 | 0.6134 | 0.2331 |

- BM25 and SPLADE-v3 reproduce the organizer baselines.
- Equal-weight RRF with BM25 hurts SPLADE-v3. The organizers fused SPLADE-v3 with Arctic-Embed-L instead.
- The MiniLM cross-encoder hurts SPLADE-v3. It was trained on short MS MARCO queries, not multi-sentence narratives.

LLM reranker tuning results on RAG24 dev, with training and dev curves, are in the [study README](studies/llm_reranking/README.md#dev-results). RAG25 test results follow once a configuration is frozen.

## Reference

### Papers, Reports and Blogs

- TREC 2025 RAG Overview, track results (202603): [arXiv](https://arxiv.org/abs/2603.09891)
- Qwen3.5, native multimodal agents (202602): [Blog](https://qwen.ai/blog?id=qwen3.5)
- TREC 2025 RAG Corpus, segmented corpus and qrels (202505): [Website](https://trec-rag.github.io/annoucements/2025-rag25-corpus/)
- Arctic-Embed, dense text embeddings (202405): [arXiv](https://arxiv.org/abs/2405.05374)
- SPLADE-v3, learned sparse retrieval (202403): [arXiv](https://arxiv.org/abs/2403.06789)
- Pyserini, reproducible sparse and dense IR (202102): [arXiv](https://arxiv.org/abs/2102.10073)
- MS MARCO, web search dataset (201611): [arXiv](https://arxiv.org/abs/1611.09268)
- RRF, reciprocal rank fusion (200907): [ACM](https://dl.acm.org/doi/10.1145/1571941.1572114)

### Code

- castorini/pyserini: [GitHub](https://github.com/castorini/pyserini)
- TREC-RAG/trec-rag-data: [GitHub](https://github.com/TREC-RAG/trec-rag-data)
