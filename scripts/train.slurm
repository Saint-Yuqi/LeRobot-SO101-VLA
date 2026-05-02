#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                 # any GPU; pin (gpu:a100:1 / gpu:h100:1) only for big runs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch scripts/train.slurm <config.yaml> [<script.py>]
# Examples:
#   sbatch scripts/train.slurm configs/train/overfit.yaml scripts/overfit_test.py
#   sbatch scripts/train.slurm configs/train/full_eval1.yaml scripts/train.py
#
# Tip from the brief: "you may use cheaper GPUs than H100s to extend your
# allocated compute." A single A100 80GB + grad-accum=2 fits SmolVLA
# fine-tuning at batch 64 with reasonable speed. Save H100s for the final
# big runs.

set -euo pipefail

CONFIG=${1:?"usage: sbatch train.slurm <config.yaml> [<script.py>]"}
SCRIPT=${2:-scripts/train.py}

# --- Conda env ---
source /apps/u24/opt/x86_64_v3/miniforge3-25.3.0-3-parbelp6xzi5ketwmb6iavf6xur4z55o/etc/profile.d/conda.sh
conda activate lerobot

# Ignore ~/.local/lib user-site — there is a broken torch there that
# shadows the conda-env copy and silently fails with libtorch_cuda map errors.
export PYTHONNOUSERSITE=1

# torchcodec dlopens libavutil at video-decode time. Make sure the FFmpeg
# libs we installed via `conda install -c conda-forge ffmpeg` are findable
# by the system linker — conda activation alone does not set this.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Keep HF model cache + pip wheels on scratch (home quota is tight).
export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache
export PIP_CACHE_DIR=/home/yuqyan/scratch/Lerobot/pip_cache

mkdir -p logs

echo "=== job $SLURM_JOB_ID  config=$CONFIG  script=$SCRIPT  host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi
python --version
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')"

python -u "$SCRIPT" --config "$CONFIG"

echo "=== done ==="
