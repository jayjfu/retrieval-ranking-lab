#!/usr/bin/env bash
# RAG experiments, each condition measured once (no tuning on RAG25):
#   dev_context_runs RAG24 dev (56 topics with nuggets): Qwen3.5-9B, top 20, SPLADE-v3 / ours / dev-best RRF
#   dev_context_diversity  RAG24 dev: ours, top 20 / 30 / 50, with and without at most 2 segments per document
#   dev_subquestions       RAG24 dev: top 50 of ours vs sub-question contexts (round-robin, RRF)
#   test_verification      RAG25, once: ours, ours over the organizers' candidates, and RankQwen3-32B, top 20 / 50
#   calibration     organizer RankQwen3-32B context, top 20 / 50, against the overview's AG baselines
#   context_runs    Qwen3.5-9B, top 20, over four context runs (SPLADE-v3, ours, dev-best RRF, RankQwen3-32B)
#   generator_scale Qwen3.5 0.8B / 2B / 4B / 9B, top 20, over the dev-best RRF context
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU
FILTER="WARN|INFO|SLF4J|Warning|it/s|EngineCore|^\s*$"
RERANKING=outputs/llm_reranking/rag25
SPLADE=splade=runs/rag25/splade.trec
OURS="ours=$RERANKING/splade@100/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec"
RRF="ours_listwise_rrf=$RERANKING/fusion/rrf_ours+list35bspl+qwen9b.trec"
RANKQWEN3=rankqwen3_32b=data/trec_rag/rag25_run_rankqwen3_32b.trec

study() {  # name generator context_sizes run...   (TOPIC_SET: rag25 or rag24-dev; CAPS: segments per document)
  local name=$1 generator=$2 sizes=$3; shift 3
  local tag=${generator##*/}
  if [ ! -f outputs/rag/${TOPIC_SET:-rag25}/$tag/$name/summary.json ]; then
    echo "== ${TOPIC_SET:-rag25} $name $generator"
    local args=(); for run in "$@"; do args+=(--runs "$run"); done
    python -m studies.rag.run_study --topic-set ${TOPIC_SET:-rag25} --name $name --generator $generator \
      --context-sizes $sizes --max-per-document ${CAPS:-0} "${args[@]}" 2>&1 \
      | grep -vE "$FILTER" | grep -E "^\| $tag|Traceback|Error" || true
  fi
}

study calibration vllm:Qwen/Qwen3.5-9B 20,50 "$RANKQWEN3"
study context_runs vllm:Qwen/Qwen3.5-9B 20 "$SPLADE" "$OURS" "$RRF" "$RANKQWEN3"
for size in 0.8B 2B 4B 9B; do
  study generator_scale vllm:Qwen/Qwen3.5-$size 20 "$RRF"
done

DEV=outputs/llm_reranking/rag24-dev
TOPIC_SET=rag24-dev study dev_context_runs vllm:Qwen/Qwen3.5-9B 20 splade=runs/rag24-dev/splade.trec \
  "ours=$DEV/listwise_lr1e-4_pos32_epoch1_cosine_clip1/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec" \
  "ours_listwise_rrf=$DEV/fusion/rrf_ours+list35bspl+qwen9b.trec"
TOPIC_SET=rag24-dev CAPS=0,2 study dev_context_diversity vllm:Qwen/Qwen3.5-9B 20,30,50 \
  "ours=$DEV/listwise_lr1e-4_pos32_epoch1_cosine_clip1/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec"
SUBQ=outputs/rag/rag24-dev/subquestions
if [ ! -f $SUBQ/subquestions_rrf.trec ]; then
  python -m studies.rag.subquestions decompose --topic-set rag24-dev
  python -m studies.rag.subquestions retrieve --topic-set rag24-dev
  for method in roundrobin rrf; do
    python -m studies.rag.subquestions combine --topic-set rag24-dev --method $method \
      --base "$DEV/listwise_lr1e-4_pos32_epoch1_cosine_clip1/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec"
  done
fi
TOPIC_SET=rag24-dev study dev_subquestions vllm:Qwen/Qwen3.5-9B 50 \
  "ours=$DEV/listwise_lr1e-4_pos32_epoch1_cosine_clip1/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec" \
  "subq_roundrobin=$SUBQ/subquestions_roundrobin.trec" "subq_rrf=$SUBQ/subquestions_rrf.trec"
if [ ! -f $RERANKING/rankqwen3_candidates/summary.json ]; then
  python -m studies.llm_reranking.evaluate --topics rag25 --first-stage-run data/trec_rag/rag25_run_rankqwen3_32b.trec \
    --name rankqwen3_candidates --rerankers lora:checkpoints/reranker/listwise_lr1e-4_pos32_epoch1_cosine_clip1/step-290
fi
study test_verification vllm:Qwen/Qwen3.5-9B 20,50 "$OURS" "$RANKQWEN3" \
  "ours_on_rankqwen3=$RERANKING/rankqwen3_candidates/lora:checkpoints_reranker_listwise_lr1e-4_pos32_epoch1_cosine_clip1_step-290.trec"
echo "== done"
