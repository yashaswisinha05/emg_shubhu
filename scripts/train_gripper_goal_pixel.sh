#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_gripper_state_pose.py \
  --pixel-architecture goal-consistent \
  --models emg+imu \
  --pixel-weight 0.35 \
  --final-pose-weight 0.2 \
  "$@"
