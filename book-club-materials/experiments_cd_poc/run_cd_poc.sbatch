#!/bin/bash
#SBATCH --job-name=cd_poc_qwen3
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/project/jevans/maxzhuyt/book-club-v3/book-club-materials/experiments_cd_poc/cd_poc_%j.log
#SBATCH --error=/project/jevans/maxzhuyt/book-club-v3/book-club-materials/experiments_cd_poc/cd_poc_%j.log

# CD proof-of-concept: Qwen3-32B (bf16) dual-copy contrastive decoding.
# Two GPU pairs run two cell-shards in parallel to halve wall-clock:
#   shard 0 -> cuda:0 (pos) / cuda:1 (neg)
#   shard 1 -> cuda:2 (pos) / cuda:3 (neg)
# bf16 needs no FP8/DeepGEMM kernels — device-correct by construction.
# Each shard runs its own smoke gate (coherence + throughput + CD-active) and
# aborts that shard if it fails. Resumable via pass2.txt existence.

export PYTHONUNBUFFERED=1
export HF_HOME="/project/jevans/maxzhuyt/.cache/huggingface"
module load cuda/12.8

PYTHON="/project/jevans/maxzhuyt/venvs/cdv4/bin/python"
MODEL="/project/jevans/maxzhuyt/models/Qwen3-32B"

cd /project/jevans/maxzhuyt/book-club-v3/book-club-materials/experiments_cd_poc

echo "[$(date)] node=$(hostname) gpus:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Shard 0 on GPUs 0,1
"$PYTHON" -u run_cd_poc.py --model-path "$MODEL" \
  --device-pos cuda:0 --device-neg cuda:1 \
  --shard 0 --nshards 2 2>&1 | sed "s|^|[s0] |" &
PID0=$!

# Shard 1 on GPUs 2,3
"$PYTHON" -u run_cd_poc.py --model-path "$MODEL" \
  --device-pos cuda:2 --device-neg cuda:3 \
  --shard 1 --nshards 2 2>&1 | sed "s|^|[s1] |" &
PID1=$!

wait $PID0; echo "[$(date)] shard0 exited $?"
wait $PID1; echo "[$(date)] shard1 exited $?"
echo "[$(date)] all shards done"
