#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_gripper_timesiam_future.py \
  --models emg imu emg+imu \
  --timesiam-pretrain-epochs 15 \
  --epochs 60 \
  "$@"
