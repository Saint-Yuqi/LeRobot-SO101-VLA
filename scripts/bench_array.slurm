#!/bin/bash
#SBATCH --job-name=vla_bench
#SBATCH --output=logs/bench-%A_%a.out
#SBATCH --error=logs/bench-%A_%a.err
#SBATCH --time=0:25:00
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6
# A100 80GB nodes (u24-chaiam0-611..615) live in lowprio. Standard partition's
# A100s are draining, so jobs there sit PENDING forever. GRES name is
# case-sensitive: `gpu:A100:1`. Lowprio means we can be preempted, fine for
# a 25-min benchmark.

# Stage-A batch sweep. Each array task runs 200 steps at one batch size.
# OOM in one task does not kill the others.
#
# To run Stage-B (fixed batch, sweep compile/freeze) or Stage-C (refined
# batch list on the stage-B winner), submit a different array with a
# different BATCH_SIZES list and the appropriate --compile / --freeze-ve
# flags below.
#
# Usage:
#   sbatch scripts/bench_array.slurm
#   # then:
#   python scripts/aggregate_bench.py logs/bench/$SLURM_ARRAY_JOB_ID

set -euo pipefail

BATCH_SIZES=(128 160 192 224 256 288 320)
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

source /apps/u24/opt/x86_64_v3/miniforge3-25.3.0-3-parbelp6xzi5ketwmb6iavf6xur4z55o/etc/profile.d/conda.sh
conda activate lerobot

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache
export PIP_CACHE_DIR=/home/yuqyan/scratch/Lerobot/pip_cache

mkdir -p logs

echo "=== bench job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}  batch=${BATCH}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES} ==="
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
    --batch-size "${BATCH}" \
    --tag "${SLURM_ARRAY_JOB_ID}"

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
