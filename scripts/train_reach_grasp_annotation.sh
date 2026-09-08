#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_reach_grasp.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --annotation-aware --uncertainty-ms 200 --selection-tolerance-ms 200 \
  --models imu emg emg+imu --device cuda --epochs 40 --batch-size 8 \
  --output-dir runs/reach_grasp_annotation_seed42
