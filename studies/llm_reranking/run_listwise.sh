#!/usr/bin/env bash
# Listwise LLM reranking (window 20, stride 10) on dev: Qwen3.5-9B and 35B-A3B,
# starting from the SPLADE-v3 order or from our fine-tuned reranker's order; then the RAG25 test.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1 MKL_THREADING_LAYER=GNU
FILTER="WARN|SLF4J|Warning|INFO|it/s\]|s/it\]|EngineCore"
PLAIN=listwise_lr1e-4_pos32_epoch1_cosine_clip1

listwise() {  # topics model short_name start_name start_run gpu_memory_utilization
  local name=$3_$4
  if ! grep -q "\"$name\"" outputs/llm_reranking/$1/listwise/summary.json 2>/dev/null; then
    echo "== $1 $name"
    python -m studies.llm_reranking.listwise_rerank --topics $1 --model $2 --name $name --first-stage-run $5 \
      --gpu-memory-utilization $6 2>&1 | grep -vE "$FILTER" | grep -E "^\||window calls|Traceback|Error" || true
  fi
}

OURS=outputs/llm_reranking/rag24-dev/$PLAIN/lora:checkpoints_reranker_${PLAIN}_step-290.trec
for start in "splade runs/rag24-dev/splade.trec" "ours $OURS"; do
  set -- $start
  listwise rag24-dev Qwen/Qwen3.5-9B qwen9b $1 $2 0.85
  listwise rag24-dev Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 qwen35b $1 $2 0.90
done

# RAG25, tested once: listwise 35B-A3B from ours, and the dev-best fusion (its listwise member starts from SPLADE-v3)
OURS=outputs/llm_reranking/rag25/splade@100/lora:checkpoints_reranker_${PLAIN}_step-290.trec
listwise rag25 Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 qwen35b ours $OURS 0.90
listwise rag25 Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 qwen35b splade runs/rag25/splade.trec 0.90
python -m studies.llm_reranking.fuse_runs --topics rag25 --run ours=$OURS \
  --run list35bspl=outputs/llm_reranking/rag25/listwise/qwen35b_splade.trec \
  --run qwen9b=outputs/llm_reranking/teachers/rag25/Qwen3.5-9B_yes_no.trec | grep "^| rrf"
echo "== done"
