#!/usr/bin/env bash
# Augmentation: pure distillation (alpha 0, tau 1) on 16-document groups,
# 1 positive + 8 unjudged SPLADE-v3 top-100 segments + 7 judged negatives. Finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"
BASE=listwise_lr1e-4_pos32_epoch1_cosine_clip1_distill_alpha0_tau1
DEV=data/reranker_train/rag24-dev-val.jsonl
TEACHER=outputs/llm_reranking/teachers/rag24-train/ensemble_9b-yes_no_35b-a3b-graded.jsonl

[ -f data/reranker_train/rag24-train-unjudged.jsonl ] || \
  python -m studies.llm_reranking.augment_groups --out data/reranker_train/rag24-train-unjudged.jsonl
[ -f data/reranker_train/rag24-train-unjudged-z0.jsonl ] || \
  python -m studies.llm_reranking.augment_groups --max-teacher-score 0 --out data/reranker_train/rag24-train-unjudged-z0.jsonl

run() {  # name data
  local name=$1 data=$2
  local out=checkpoints/reranker/$name
  if [ ! -f $out/adapter_config.json ]; then
    echo "== train $name"
    python -m studies.llm_reranking.train --data $data --dev-data $DEV --out $out --epochs 1 --save-every 58 \
      --group-size 16 --queries-per-step 4 --grad-accum 4 --extra-negatives-per-group 8 \
      --teacher-scores $TEACHER --distill-weight 1 --distill-temperature 1 2>&1 | grep -vE "$FILTER" | tail -1
  fi
  if [ ! -f outputs/llm_reranking/rag24-dev/$name/summary.json ]; then
    echo "== dev $name"
    local specs=$(ls -d $out/step-* | sort -t- -k2 -n | sed 's/^/lora:/' | paste -sd,)
    python -m studies.llm_reranking.evaluate --topics rag24-dev --depth 100 --name $name \
      --rerankers "$specs,lora:$out" 2>&1 | grep -vE "$FILTER" | grep "^| lora" || true
  fi
}

run ${BASE}_group16_unjudged8 data/reranker_train/rag24-train-unjudged.jsonl
run ${BASE}_group16_unjudged8_z0 data/reranker_train/rag24-train-unjudged-z0.jsonl
echo "== done"
