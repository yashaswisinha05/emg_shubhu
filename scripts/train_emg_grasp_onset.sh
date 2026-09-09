#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/train_emg_grasp_onset.py \
  --root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \
  --device cuda --epochs 60 --batch-size 8 --seed 42 \
  --raw-rate-hz 1259.4 --rate-hz 200 \
  --uncertainty-ms 200 --selection-tolerance-ms 200 \
  --output-dir runs/emg_grasp_onset_seed42

