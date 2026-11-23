#!/usr/bin/env bash
# Distillation on the existing groups: the selected configuration plus a KL term to the two-teacher ensemble.
# Loss: alpha * listwise CE + (1 - alpha) * tau^2 * KL(teacher || student). Finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"
BASE=listwise_lr1e-4_pos32_epoch1_cosine_clip1
DATA=data/reranker_train/rag24-train.jsonl
DEV=data/reranker_train/rag24-dev-val.jsonl
TEACHER=outputs/llm_reranking/teachers/rag24-train/ensemble_9b-yes_no_35b-a3b-graded.jsonl

run() {  # name train_args...
  local name=$1; shift
  local out=checkpoints/reranker/$name
  if [ ! -f $out/adapter_config.json ]; then
    echo "== train $name"
    python -m studies.llm_reranking.train --data $DATA --dev-data $DEV --out $out --epochs 1 --save-every 58 \
      --teacher-scores $TEACHER "$@" 2>&1 | grep -vE "$FILTER" | tail -1
  fi
  if [ ! -f outputs/llm_reranking/rag24-dev/$name/summary.json ]; then
    echo "== dev $name"
    local specs=$(ls -d $out/step-* | sort -t- -k2 -n | sed 's/^/lora:/' | paste -sd,)
    python -m studies.llm_reranking.evaluate --topics rag24-dev --depth 100 --name $name \
      --rerankers "$specs,lora:$out" 2>&1 | grep -vE "$FILTER" | grep "^| lora" || true
  fi
}

run ${BASE}_distill_alpha0.5_tau1 --distill-weight 0.5 --distill-temperature 1
run ${BASE}_distill_alpha0.5_tau2 --distill-weight 0.5 --distill-temperature 2
run ${BASE}_distill_alpha0_tau1 --distill-weight 1 --distill-temperature 1
run ${BASE}_distill_alpha0_tau2 --distill-weight 1 --distill-temperature 2
echo "== done"
