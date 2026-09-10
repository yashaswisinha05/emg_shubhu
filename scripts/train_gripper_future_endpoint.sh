#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_gripper_state_pose.py \
  --models emg+imu \
  --pixel-architecture goal-consistent \
  --pixel-weight 0.35 \
  --final-pose-weight 0.2 \
  --endpoint-loss-space physical \
  --final-position-multiplier 2.5 \
  --endpoint-progress-weight 1.0 \
  --future-pose-ms 200 \
  --future-pose-weight 0.1 \
  "$@"
