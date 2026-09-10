#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_gripper_state_attention.py \
  --models emg imu emg+imu \
  --epochs 60 \
  --state-condition-weight 0.5 \
  --future-consistency-weight 0.1 \
  "$@"
