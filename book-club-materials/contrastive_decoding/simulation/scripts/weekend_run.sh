#!/bin/bash
# Weekend run: 3 base models x (single-story sim + comparative sim + Opus 4.6
# summaries for each).  Idempotent — re-running skips any stage whose primary
# output already exists.  Launch detached, survives terminal close:
#
#   cd /project/jevans/maxzhuyt/book-club-v3/book-club-materials/contrastive_decoding/simulation
#   setsid nohup bash scripts/weekend_run.sh > scripts/weekend.log 2>&1 < /dev/null &
#   disown
#
# Monitor:
#   tail -f scripts/weekend.log
#   ls -la outputs_*/discussion_transcript.md outputs_*/group_summary_opus.md
#
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

OPUS_MODEL="anthropic/claude-opus-4.6"

# (model_id, output_tag)
MODELS=(
  "Qwen/Qwen3-14B|qwen3_14b"
  "Qwen/Qwen3.5-9B|qwen3.5_9b"
  "google/gemma-3-12b-it|gemma3_12b"
)

# Stage helpers --------------------------------------------------------------

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
log() { echo "$(ts) $*"; }

run_single() {
  local model_id="$1"
  local out="$2"
  if [[ -f "$out/discussion_transcript.md" ]]; then
    log "skip single ($out): discussion_transcript.md already exists"
    return 0
  fi
  log "BEGIN single-story sim: $model_id -> $out"
  python3 -u src/bookclub/simulate.py \
    --model-id "$model_id" \
    --output-dir "$out" \
    --seed 7 --rounds 2 \
    2>&1 | sed "s|^|[$out] |"
  log "END   single-story sim: $model_id -> $out (exit=$?)"
}

run_compare() {
  local model_id="$1"
  local out="$2"
  if [[ -f "$out/discussion_transcript.md" ]]; then
    log "skip compare ($out): discussion_transcript.md already exists"
    return 0
  fi
  log "BEGIN compare sim: $model_id -> $out"
  python3 -u src/bookclub/simulate_compare.py \
    --model-id "$model_id" \
    --output-dir "$out" \
    --seed 13 --rounds 2 \
    2>&1 | sed "s|^|[$out] |"
  log "END   compare sim: $model_id -> $out (exit=$?)"
}

run_opus() {
  local out="$1"
  if [[ ! -f "$out/manifest.json" ]]; then
    log "skip opus ($out): no manifest.json (sim must have failed)"
    return 0
  fi
  if [[ -f "$out/group_summary_opus.md" ]]; then
    log "skip opus ($out): group_summary_opus.md already exists"
    return 0
  fi
  log "BEGIN opus summary: $out"
  python3 -u src/bookclub/summarize_opus.py \
    --outputs-dir "$out" \
    --model-id "$OPUS_MODEL" \
    2>&1 | sed "s|^|[opus:$out] |"
  log "END   opus summary: $out (exit=$?)"
}

# Main loop ------------------------------------------------------------------

log "=== weekend run starting ==="
log "OPUS_MODEL=$OPUS_MODEL"
log "transformers=$(python3 -c 'import transformers; print(transformers.__version__)')"
log "torch=$(python3 -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read l; do log "gpu: $l"; done

for spec in "${MODELS[@]}"; do
  model_id="${spec%%|*}"
  tag="${spec##*|}"
  log ""
  log "########## $model_id (tag=$tag) ##########"

  single_out="outputs_${tag}"
  compare_out="outputs_${tag}_compare"

  run_single  "$model_id" "$single_out"
  run_opus    "$single_out"
  run_compare "$model_id" "$compare_out"
  run_opus    "$compare_out"
done

log ""
log "=== weekend run complete ==="
log "outputs:"
ls -d outputs_* 2>/dev/null | while read d; do
  have_sim=$([[ -f "$d/discussion_transcript.md" ]] && echo "yes" || echo "NO")
  have_sum=$([[ -f "$d/group_summary_opus.md" ]] && echo "yes" || echo "NO")
  log "  $d   sim=$have_sim summary=$have_sum"
done
