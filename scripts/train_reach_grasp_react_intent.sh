#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Capacity now matches the EMG/IMU patch encoders (width/layers/heads were
# previously hardcoded and the ReactEMG-style intent branch silently stayed
# at its class defaults -- layers=2, context=100 -- no matter what depth was
# requested for the rest of the network; see reach_grasp_react_intent.py).
# --warmup-epochs matches the paper's stated recipe of linear LR warmup then
# linear decay (arXiv:2506.19815 sec 3.4); the masking-mode implementation
# now also matches its four named regimes exactly (sec 3.2).
python scripts/train_reach_grasp_react_intent.py \
  --root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \
  --device cuda --epochs 60 --batch-size 4 --seed 42 \
  --raw-rate-hz 1259.4 --uncertainty-ms 500 \
  --width 128 --layers 4 --heads 4 --dropout .1 \
  --react-context 100 --warmup-epochs 4 \
  --output-dir runs/reach_grasp_react_intent_seed42 "$@"
