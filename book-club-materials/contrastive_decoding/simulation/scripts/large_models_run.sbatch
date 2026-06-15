#!/bin/bash
#SBATCH --job-name=bookclub_large
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=47:00:00
#SBATCH --output=/project/jevans/maxzhuyt/book-club-v3/book-club-materials/contrastive_decoding/simulation/scripts/large_models_%j.log
#SBATCH --error=/project/jevans/maxzhuyt/book-club-v3/book-club-materials/contrastive_decoding/simulation/scripts/large_models_%j.log

# Reproduce the weekend_run.sh pipeline for 30B-range models on 2x H200.
# Each branch of contrastive decoding occupies one H200 (cuda:0 / cuda:1).
# Models are loaded from local paths — no HuggingFace network fetch needed.

export PYTHONUNBUFFERED=1
export HF_HOME="/project/jevans/maxzhuyt/.cache/huggingface"

# Load DeepSeek API key for summarization
set -a; source ~/.env; set +a

module load cuda

PYTHON="/project/jevans/maxzhuyt/honest_llama_env/bin/python"

cd /project/jevans/maxzhuyt/book-club-v3/book-club-materials/contrastive_decoding/simulation

MODELS_DIR="/project/jevans/maxzhuyt/models"
OPUS_MODEL="anthropic/claude-opus-4.6"

# (local_model_path, output_tag)
MODELS=(
  "$MODELS_DIR/Qwen3-32B|qwen3_32b"
  "$MODELS_DIR/Mistral-Small-24B-Instruct-2501|mistral_small_24b"
)

# --------------------------------------------------------------------------

ts()  { date +"[%Y-%m-%d %H:%M:%S]"; }
log() { echo "$(ts) $*"; }

run_single() {
  local model_id="$1" out="$2"
  if [[ -f "$out/discussion_transcript.md" ]]; then
    log "skip single ($out): discussion_transcript.md already exists"
    return 0
  fi
  log "BEGIN single-story sim: $model_id -> $out"
  "$PYTHON" -u src/bookclub/simulate.py \
    --model-id "$model_id" \
    --output-dir "$out" \
    --seed 7 --rounds 2 \
    2>&1 | sed "s|^|[$out] |"
  log "END   single-story sim: $model_id -> $out (exit=${PIPESTATUS[0]})"
}

run_compare() {
  local model_id="$1" out="$2"
  if [[ -f "$out/discussion_transcript.md" ]]; then
    log "skip compare ($out): discussion_transcript.md already exists"
    return 0
  fi
  log "BEGIN compare sim: $model_id -> $out"
  "$PYTHON" -u src/bookclub/simulate_compare.py \
    --model-id "$model_id" \
    --output-dir "$out" \
    --seed 13 --rounds 2 \
    2>&1 | sed "s|^|[$out] |"
  log "END   compare sim: $model_id -> $out (exit=${PIPESTATUS[0]})"
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
  "$PYTHON" -u src/bookclub/summarize_opus.py \
    --outputs-dir "$out" \
    --backend deepseek \
    2>&1 | sed "s|^|[opus:$out] |"
  log "END   opus summary: $out (exit=${PIPESTATUS[0]})"
}

# --------------------------------------------------------------------------

log "=== large_models_run starting ==="
log "torch=$("$PYTHON" -c 'import torch; print(torch.__version__)')"
log "transformers=$("$PYTHON" -c 'import transformers; print(transformers.__version__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  | while read l; do log "gpu: $l"; done

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
log "=== large_models_run complete ==="
log "outputs:"
ls -d outputs_* 2>/dev/null | while read d; do
  have_sim=$([[ -f "$d/discussion_transcript.md" ]] && echo "yes" || echo "NO")
  have_sum=$([[ -f "$d/group_summary_opus.md" ]] && echo "yes" || echo "NO")
  log "  $d   sim=$have_sim  opus=$have_sum"
done
