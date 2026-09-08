#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_reach_grasp_orientation.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --models imu emg emg+imu --device cuda --epochs 40 --batch-size 4 \
  --orientation-weight 0.5 \
  --output-dir runs/reach_grasp_orientation_hybrid_seed42
