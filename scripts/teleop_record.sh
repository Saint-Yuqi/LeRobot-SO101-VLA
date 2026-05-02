#!/bin/bash
# Wrapper around `lerobot record` for consistent teleop sessions.
#
# Usage:
#   bash scripts/teleop_record.sh <output_dir> [num_episodes] [task_prompt]
#
# Example:
#   bash scripts/teleop_record.sh data/raw/eval1_session1 20 "Put the banana in the red colored bowl."

set -euo pipefail

OUT=${1:?"usage: teleop_record.sh <output_dir> [num_episodes] [task_prompt]"}
N=${2:-1}
TASK=${3:-"Put the banana in the colored bowl."}

mkdir -p "$OUT"

# Adjust the lerobot CLI call to match the version you've installed.
# The flag names below reflect lerobot ~v0.2.x. Check `lerobot record --help`.
lerobot record \
    --robot.type=so101 \
    --dataset.repo_id="local/$(basename "$OUT")" \
    --dataset.root="$OUT" \
    --dataset.num_episodes="$N" \
    --dataset.single_task="$TASK"

echo "Recorded $N episode(s) into $OUT"
echo "Validate with:"
echo "  python scripts/teleop_validate.py --dataset $OUT"
