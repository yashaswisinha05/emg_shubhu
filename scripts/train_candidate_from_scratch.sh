#!/usr/bin/env bash
set -euo pipefail

candidate_root="/home/nahar3/shubham/emg_shubhu"
candidate_prefixes=(
  "32e00ff16111"
  "shubhamcal1_b0f8c99b"
  "f5a69f99ddeb"
)
candidate_cache="artifacts/tracked_cache_three_candidate_scratch"
candidate_seed="42"

cd "$(dirname "$0")/.."

echo "Candidates: ${candidate_prefixes[*]}"
echo "[1/3] Training three-candidate soft-routed base from random weights"
python scripts/train_candidate_scratch_01_soft_routed.py \
  --root "$candidate_root" \
  --config configs/tracked_soft_routed_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "${candidate_prefixes[@]}" \
  --device cuda --seed "$candidate_seed" \
  --teacher-epochs 25 --epochs 50 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/three_candidate_scratch_01_soft_routed

echo "[2/3] Adding the temporal EMG residual"
python scripts/train_candidate_scratch_02_emg_residual.py \
  --root "$candidate_root" \
  --initial-checkpoint \
    runs/three_candidate_scratch_01_soft_routed/final.pt \
  --config configs/tracked_emg_residual_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "${candidate_prefixes[@]}" \
  --device cuda --seed "$candidate_seed" \
  --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/three_candidate_scratch_02_emg_residual

echo "[3/3] Adding EMG acceleration dynamics"
python scripts/train_candidate_scratch_03_acceleration.py \
  --root "$candidate_root" \
  --initial-checkpoint \
    runs/three_candidate_scratch_02_emg_residual/final.pt \
  --config configs/tracked_emg_acceleration_complete_reach.yaml \
  --cache-dir "$candidate_cache" \
  --session-prefixes "${candidate_prefixes[@]}" \
  --device cuda --seed "$candidate_seed" \
  --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/three_candidate_scratch_03_acceleration

echo "Scratch training complete"
echo "Final checkpoint: runs/three_candidate_scratch_03_acceleration/final.pt"
echo "Per-candidate live calibrations are in runs/three_candidate_scratch_03_acceleration/"
