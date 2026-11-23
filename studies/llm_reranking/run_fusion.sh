#!/usr/bin/env bash
# Fusion: reranker-stage RRF / z-score sums on dev, BM25 + SPLADE-v3 first stage on dev,
# then teacher scores on RAG25 and the dev-best fusion, tested once.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"
PLAIN=listwise_lr1e-4_pos32_epoch1_cosine_clip1

fuse() {  # topics method member...
  local topics=$1 method=$2; shift 2
  local args=(); for member in "$@"; do args+=(--run "$member"); done
  python -m studies.llm_reranking.fuse_runs --topics $topics --method $method "${args[@]}" | grep -E "^\| (rrf|zsum)"
}

echo "== reranker-stage fusion on dev"
TEACHERS=outputs/llm_reranking/teachers/rag24-dev
OURS=ours=outputs/llm_reranking/rag24-dev/$PLAIN/lora:checkpoints_reranker_${PLAIN}_step-290.trec
TEACHER=teacher=$TEACHERS/ensemble_9b-yes_no_35b-a3b-graded.trec
QWEN9B=qwen9b=$TEACHERS/Qwen3.5-9B_yes_no.trec
QWEN35B=qwen35b=$TEACHERS/Qwen3.5-35B-A3B-GPTQ-Int4_graded.trec
SPLADE=splade=runs/rag24-dev/splade.trec
for method in rrf zsum; do
  fuse rag24-dev $method $OURS $TEACHER
  fuse rag24-dev $method $OURS $QWEN9B $QWEN35B
  fuse rag24-dev $method $OURS $QWEN9B
  fuse rag24-dev $method $OURS $SPLADE
  fuse rag24-dev $method $OURS $TEACHER $SPLADE
  fuse rag24-dev $method $OURS $QWEN9B $SPLADE
done

echo "== first stage: BM25 + SPLADE-v3 on dev"
python -m retrieval_ranking_lab run --topics rag24-dev --retriever hybrid --k-retrieve 1000 2>&1 | grep "^|"

echo "== teachers on RAG25 (SPLADE-v3 top-100)"
python -m studies.llm_reranking.score_teacher --topics rag25 --model Qwen/Qwen3.5-9B --prompt yes_no \
  --gpu-memory-utilization 0.85 2>&1 | grep -vE "$FILTER" | grep "^|"
python -m studies.llm_reranking.score_teacher --topics rag25 --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --prompt graded \
  --gpu-memory-utilization 0.90 2>&1 | grep -vE "$FILTER" | grep "^|"
python -m studies.llm_reranking.ensemble_teachers --topics rag25 --name ensemble_9b-yes_no_35b-a3b-graded \
  --teacher Qwen3.5-9B_yes_no --teacher Qwen3.5-35B-A3B-GPTQ-Int4_graded | grep "^|"

echo "== RAG25: the dev-best fusion"
TEACHERS=outputs/llm_reranking/teachers/rag25
OURS=ours=outputs/llm_reranking/rag25/splade@100/lora:checkpoints_reranker_${PLAIN}_step-290.trec
fuse rag25 rrf $OURS teacher=$TEACHERS/ensemble_9b-yes_no_35b-a3b-graded.trec splade=runs/rag25/splade.trec
echo "== done"
