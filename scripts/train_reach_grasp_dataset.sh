#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_reach_grasp.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --device cuda --epochs 40 --batch-size 8 \
  --models imu emg emg+imu \
  --output-dir runs/reach_grasp_seed42
