#!/bin/bash
#SBATCH --job-name=eval_offline
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=lowprio

# Usage:
#   sbatch scripts/eval_offline.slurm <checkpoint> <dataset> [<extra_args>]
#
# Example:
#   sbatch scripts/eval_offline.slurm \
#       checkpoints/eval1_v2/<run>/step_2000 \
#       ethrl2026/so101_pickup_20260503_153511_task1 \
#       "--frames-per-episode 4 --overlay-dir reports/eval1_step2000_overlay --out reports/eval1_step2000.json"

set -euo pipefail

CKPT=${1:?"usage: sbatch eval_offline.slurm <checkpoint> <dataset> [extra_args]"}
DATASET=${2:?"usage: sbatch eval_offline.slurm <checkpoint> <dataset> [extra_args]"}
EXTRA=${3:-}

source /apps/u24/opt/x86_64_v3/miniforge3-25.3.0-3-parbelp6xzi5ketwmb6iavf6xur4z55o/etc/profile.d/conda.sh
conda activate lerobot

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache
if [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

mkdir -p logs

echo "=== eval_offline job $SLURM_JOB_ID  ckpt=$CKPT  dataset=$DATASET  host=$(hostname) ==="
nvidia-smi
python -u scripts/eval_offline.py --checkpoint "$CKPT" --dataset "$DATASET" $EXTRA
echo "=== done ==="
