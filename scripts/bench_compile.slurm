#!/bin/bash
#SBATCH --job-name=vla_bench_compile
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err
#SBATCH --time=0:35:00
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage-B (compile only — freeze_vision_encoder is off the table for SmolVLA
# on this task). Single run at b=128 with torch.compile(mode="reduce-overhead").
# Compile cold-start adds ~1-2 min, hence warmup=60.
#
# Compare against the b=128 c0_f0 row from logs/bench/2679976/ for the speedup.
#
# Usage: sbatch scripts/bench_compile.slurm

set -euo pipefail

source /apps/u24/opt/x86_64_v3/miniforge3-25.3.0-3-parbelp6xzi5ketwmb6iavf6xur4z55o/etc/profile.d/conda.sh
conda activate lerobot

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache
export PIP_CACHE_DIR=/home/yuqyan/scratch/Lerobot/pip_cache

mkdir -p logs

echo "=== compile bench job ${SLURM_JOB_ID}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES} ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: ${GPU_NAME} (${GPU_MEM_MB} MiB)"
if ! echo "${GPU_NAME}" | grep -qi "A100"; then
    echo "SKIP: this task wants A100 but got ${GPU_NAME}; exiting cleanly."
    exit 0
fi
if [ "${GPU_MEM_MB}" -lt 70000 ]; then
    echo "SKIP: A100 with <70GB memory (${GPU_MEM_MB} MiB) — bench targets 80GB cards."
    exit 0
fi

python -u scripts/benchmark_batch.py \
    --config configs/train/bench.yaml \
    --batch-size 128 \
    --warmup 60 \
    --num-steps 260 \
    --compile \
    --tag "stageB_compile_${SLURM_JOB_ID}"

echo "=== compile bench done ==="
