#!/usr/bin/env bash
set -euo pipefail

candidate_root="/home/nahar3/shubham/emg_shubhu/shubham_3930d150/32e00ff16111"
candidate_prefix="32e00ff16111"
candidate_cache="artifacts/tracked_cache_candidate_scratch_32e00ff16111"
candidate_seed="42"

cd "$(dirname "$0")/.."

echo "[1/3] Training candidate-only soft-routed base from random weights"
python scripts/train_candidate_scratch_01_soft_routed.py \
  --root "$candidate_root" \
  --config configs/tracked_soft_routed_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "$candidate_prefix" \
  --device cuda --seed "$candidate_seed" \
  --teacher-epochs 25 --epochs 50 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/candidate_32e00ff16111_scratch_01_soft_routed

echo "[2/3] Adding the temporal EMG residual"
python scripts/train_candidate_scratch_02_emg_residual.py \
  --root "$candidate_root" \
  --initial-checkpoint \
    runs/candidate_32e00ff16111_scratch_01_soft_routed/final.pt \
  --config configs/tracked_emg_residual_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "$candidate_prefix" \
  --device cuda --seed "$candidate_seed" \
  --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/candidate_32e00ff16111_scratch_02_emg_residual

echo "[3/3] Adding EMG acceleration dynamics"
python scripts/train_candidate_scratch_03_acceleration.py \
  --root "$candidate_root" \
  --initial-checkpoint \
    runs/candidate_32e00ff16111_scratch_02_emg_residual/final.pt \
  --config configs/tracked_emg_acceleration_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "$candidate_prefix" \
  --device cuda --seed "$candidate_seed" \
  --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/candidate_32e00ff16111_scratch_03_acceleration

echo "Scratch training complete"
echo "Final checkpoint: runs/candidate_32e00ff16111_scratch_03_acceleration/final.pt"
echo "Live calibration: runs/candidate_32e00ff16111_scratch_03_acceleration/live_calibration.npz"
