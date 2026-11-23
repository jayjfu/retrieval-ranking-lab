#!/usr/bin/env bash
# Full-data sweep: listwise, LR 1e-4, 32 positives per topic, one config per run.
# Each run trains, logs validation loss, and scores every checkpoint on rag24-dev. Finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
FILTER="WARN|SLF4J|Warning|it/s\]|s/it\]"
PREFIX=listwise_lr1e-4_pos32
DATA=data/reranker_train/rag24-train.jsonl
DEV=data/reranker_train/rag24-dev-val.jsonl

run() {  # name train_args...
  local name=$1; shift
  local out=checkpoints/reranker/$name
  if [ ! -f $out/adapter_config.json ]; then
    echo "== train $name"
    python -m studies.llm_reranking.train --data $DATA --dev-data $DEV --out $out --save-every 58 "$@" \
      2>&1 | grep -vE "$FILTER" | tail -1
  fi
  if [ ! -f outputs/llm_reranking/rag24-dev/$name/summary.json ]; then
    echo "== dev $name"
    local specs=$(ls -d $out/step-* | sort -t- -k2 -n | sed 's/^/lora:/' | paste -sd,)
    python -m studies.llm_reranking.evaluate --topics rag24-dev --depth 100 --name $name \
      --rerankers "$specs,lora:$out" 2>&1 | grep -vE "$FILTER" | grep "^| lora" || true
  fi
}

run ${PREFIX}_epoch1_wsd_clip1 --epochs 1 --schedule wsd --decay-ratio 0.2
run ${PREFIX}_epoch1_cosine_clip1_loraplus4 --epochs 1 --lora-plus-ratio 4
run ${PREFIX}_epoch1_cosine_clip1_beta2_0.95_wd0.01 --epochs 1 --adam-beta2 0.95 --weight-decay 0.01
run ${PREFIX}_epoch1_cosine_clip10 --epochs 1 --max-grad-norm 10
run ${PREFIX}_epoch1_cosine_noclip --epochs 1 --max-grad-norm 0
run ${PREFIX}_epoch1_cosine_clip1_temperature2 --epochs 1 --temperature 2
run ${PREFIX}_epoch1_cosine_clip1_smoothing0.1 --epochs 1 --label-smoothing 0.1
run ${PREFIX}_epoch1_cosine_clip1_softcap20 --epochs 1 --margin-softcap 20
run ${PREFIX}_epoch1_cosine_clip1_zloss1e-3 --epochs 1 --score-zloss 1e-3
run ${PREFIX}_epoch1_cosine_clip1_ema0.99 --epochs 1 --ema-decay 0.99
run ${PREFIX}_epoch1_cosine_noclip_zloss1e-3 --epochs 1 --max-grad-norm 0 --score-zloss 1e-3
run ${PREFIX}_epoch1_cosine_noclip_zloss1e-3_beta2_0.95_wd0.01 --epochs 1 --max-grad-norm 0 --score-zloss 1e-3 \
  --adam-beta2 0.95 --weight-decay 0.01
run ${PREFIX}_epoch1_cosine_clip1 --epochs 1 --seed 21
echo "== done"
