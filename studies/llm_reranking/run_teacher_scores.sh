#!/usr/bin/env bash
# Teacher score cache for rag24-train: SPLADE-v3 top-100 plus every training-group document,
# scored by Qwen3.5-9B (yes/no) and Qwen3.5-35B-A3B (graded).
# Decontamination is applied later, when training groups are built.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"

if [ ! -f runs/rag24-train/splade.trec ]; then
  echo "== SPLADE-v3 rag24-train"
  python -m retrieval_ranking_lab run --topics rag24-train --retriever splade --k-retrieve 1000 2>&1 | grep -vE "$FILTER" | tail -3
fi

score() {  # model prompt gpu_memory_utilization
  local name=$(basename $1)_$2
  if [ ! -f outputs/llm_reranking/teachers/rag24-train/${name}.jsonl ]; then
    echo "== $name"
    python -m studies.llm_reranking.score_teacher --topics rag24-train --groups data/reranker_train/rag24-train.jsonl --model $1 --prompt $2 --gpu-memory-utilization $3 \
      2>&1 | grep -vE "$FILTER" | grep -E "pairs|^\||tied|top-20|Traceback|Error" || true
  fi
}
score Qwen/Qwen3.5-9B yes_no 0.85
score Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 graded 0.90
echo "== done"
