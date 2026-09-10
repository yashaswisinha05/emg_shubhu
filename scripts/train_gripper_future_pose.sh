#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pass --root <directories...>; optional common trainer options follow.
# Each setting trains three independent modality models with the same split.
experiment_dir="${FUTURE_POSE_OUTPUT:-runs/gripper_future_pose_200ms}"
for setting in control future; do
  weight=0
  if [[ "$setting" == future ]]; then weight=0.1; fi
  python scripts/train_gripper_state_pose.py \
    --pixel-architecture goal-consistent --pixel-weight 0.35 \
    --final-pose-weight 0.2 \
    "$@" \
    --models emg imu emg+imu \
    --future-pose-ms 200 --future-pose-weight "$weight" \
    --output-dir "$experiment_dir/$setting"
done
