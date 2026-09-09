#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_reach_grasp_react_intent.py \
  --root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \
  --device cuda --epochs 60 --batch-size 4 --seed 42 \
  --raw-rate-hz 1259.4 --uncertainty-ms 500 \
  --output-dir runs/reach_grasp_react_intent_seed42 "$@"
