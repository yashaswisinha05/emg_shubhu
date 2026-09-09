#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_reach_grasp_masked_reconstruction.py \
  --root data/1dc1acaa5827 \
  --device cuda \
  --epochs 60 \
  --batch-size 8 \
  --seed 42 \
  --raw-rate-hz 1259.4 \
  --models imu emg emg+imu \
  --output-dir runs/reach_grasp_masked_reconstruction_seed42
