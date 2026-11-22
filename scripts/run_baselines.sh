#!/usr/bin/env bash
# Non-LLM baselines on the 22 NIST-judged RAG25 topics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
LAB="python -m retrieval_ranking_lab"

$LAB run --retriever bm25 --k-retrieve 1000
[ -f runs/rag25/splade.trec ] || $LAB run --retriever splade --k-retrieve 1000
$LAB fuse runs/rag25/bm25.trec runs/rag25/splade.trec --out runs/rag25/hybrid.trec
$LAB run --first-stage-run runs/rag25/splade.trec --reranker cross-encoder --k-rerank 100 --tag splade+minilm
$LAB run --first-stage-run runs/rag25/hybrid.trec --reranker cross-encoder --k-rerank 100 --tag hybrid+minilm

$LAB evaluate runs/rag25/*.trec
