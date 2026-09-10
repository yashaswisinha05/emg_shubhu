#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_gripper_neuromuscular_future.py \
  --models emg imu emg+imu \
  --epochs 60 \
  --motion-reconstruction-weight 0.05 \
  "$@"
