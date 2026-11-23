#!/usr/bin/env bash
# Cascades on one topic set: our reranker over the SPLADE-v3 top-D (D = 100, 500), then on its top-k (k = 30, 100)
# either nothing, listwise Qwen3.5-35B-A3B (from our order), or RRF of ours + listwise 35B-A3B (from SPLADE-v3 order)
# + pointwise Qwen3.5-9B. Cost is the sum of the components' s/query, model loading excluded.
# Usage: run_cascade.sh [topics] [system ...]   (no systems = all)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1 MKL_THREADING_LAYER=GNU
FILTER="WARN|SLF4J|Warning|INFO|it/s\]|s/it\]|EngineCore"
TOPICS=${1:-rag24-dev}; shift || true
WANTED=" $* "
PLAIN=listwise_lr1e-4_pos32_epoch1_cosine_clip1
SPEC=lora:checkpoints/reranker/$PLAIN/step-290
OUT=outputs/llm_reranking/$TOPICS
HEADS=$OUT/cascade/heads
SPLADE=runs/$TOPICS/splade.trec

wanted() { [ "$WANTED" = "  " ] || [[ "$WANTED" == *" $1 "* ]]; }

ours() {  # depth -> our run and its timing source
  local name=cascade_ours_depth$1
  if [ ! -f $OUT/$name/summary.json ]; then
    python -m studies.llm_reranking.evaluate --topics $TOPICS --depth $1 --name $name --rerankers $SPEC 2>&1 | grep -E "^\| lora" || true
  fi
  OURS_RUN=$OUT/$name/${SPEC//\//_}.trec
  OURS_COST=$OUT/$name/summary.json:$SPEC
}

listwise() {  # name head depth
  if ! grep -q "\"$1\"" $OUT/listwise/summary.json 2>/dev/null; then
    python -m studies.llm_reranking.listwise_rerank --topics $TOPICS --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --name $1 \
      --first-stage-run $2 --depth $3 --gpu-memory-utilization 0.90 2>&1 | grep -vE "$FILTER" | grep -E "Traceback|Error" || true
  fi
}

pointwise() {  # name head depth
  if [ ! -f outputs/llm_reranking/teachers/$TOPICS/$1_summary.json ]; then
    python -m studies.llm_reranking.score_teacher --topics $TOPICS --model Qwen/Qwen3.5-9B --prompt yes_no --name $1 \
      --first-stage-run $2 --depth $3 --gpu-memory-utilization 0.85 2>&1 | grep -vE "$FILTER" | grep -E "Traceback|Error" || true
  fi
}

combine() {  # name depth [--run ... --cost ...]
  local name=$1 depth=$2; shift 2
  python -m studies.llm_reranking.cascade combine --topics $TOPICS --base $OURS_RUN --depth $depth --name $name \
    --cost $OURS_COST "$@" | grep -E "^\| $name"
}

for depth in 100 500; do
  ours $depth
  wanted ours_depth$depth && combine ours_depth$depth 0
  for top in 30 100; do
    system=depth${depth}_top${top}
    if wanted ${system}_listwise || wanted ${system}_fusion; then
      python -m studies.llm_reranking.cascade head --base $OURS_RUN --depth $top --order ours --out $HEADS/${system}_ours.trec
      python -m studies.llm_reranking.cascade head --base $OURS_RUN --depth $top --order splade --splade $SPLADE \
        --out $HEADS/${system}_splade.trec
    fi
    if wanted ${system}_listwise; then
      listwise cascade_${system}_ours $HEADS/${system}_ours.trec $top
      combine ${system}_listwise $top --run listwise=$OUT/listwise/cascade_${system}_ours.trec \
        --cost $OUT/listwise/summary.json:cascade_${system}_ours
    fi
    if wanted ${system}_fusion; then
      listwise cascade_${system}_splade $HEADS/${system}_splade.trec $top
      pointwise cascade_${system}_qwen9b $HEADS/${system}_ours.trec $top
      combine ${system}_fusion $top --run ours=$HEADS/${system}_ours.trec \
        --run listwise=$OUT/listwise/cascade_${system}_splade.trec \
        --run qwen9b=outputs/llm_reranking/teachers/$TOPICS/cascade_${system}_qwen9b.trec \
        --cost $OUT/listwise/summary.json:cascade_${system}_splade \
        --cost outputs/llm_reranking/teachers/$TOPICS/cascade_${system}_qwen9b_summary.json:cascade_${system}_qwen9b
    fi
  done
done
echo "== done"
