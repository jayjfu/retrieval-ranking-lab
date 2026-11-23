#!/usr/bin/env bash
# Teacher gate: zero-shot Qwen3.5 teachers on rag24-dev (SPLADE-v3 top-100). A teacher must clearly beat the student (0.545).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"

# exact yes/no margin with a float32 head, same scorer as the student
if [ ! -f outputs/llm_reranking/rag24-dev/teachers/summary.json ]; then
  echo "== Qwen3.5-9B yes_no (transformers, float32 head)"
  python -m studies.llm_reranking.evaluate --topics rag24-dev --depth 100 --name teachers --batch-size 8 \
    --rerankers llm:Qwen/Qwen3.5-9B 2>&1 | grep -vE "$FILTER" | grep "^|" || true
fi

score() {  # model prompt gpu_memory_utilization
  local name=$(basename $1)_$2
  if [ ! -f outputs/llm_reranking/teachers/rag24-dev/${name}_summary.json ]; then
    echo "== $name (vLLM, float16)"
    python -m studies.llm_reranking.score_teacher --model $1 --prompt $2 --gpu-memory-utilization $3 \
      2>&1 | grep -vE "$FILTER" | grep -E "^\||tied|top-20|Traceback|Error" || true
  fi
}
score Qwen/Qwen3.5-9B yes_no 0.85
score Qwen/Qwen3.5-9B graded 0.85
score Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 yes_no 0.90
score Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 graded 0.90
echo "== done"
