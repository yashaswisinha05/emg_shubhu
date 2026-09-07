#!/usr/bin/env bash
set -euo pipefail

candidate_root="/home/nahar3/shubham/emg_shubhu"
candidate_prefixes=(
  "32e00ff16111"
  "shubhamcal1_b0f8c99b"
  "f5a69f99ddeb"
  "ca6ad138de88"
)
candidate_cache="artifacts/imu_to_emg_four_recordings_cache"
candidate_seed="42"

cd "$(dirname "$0")/.."

echo "Dataset prefixes: ${candidate_prefixes[*]}"
echo "Training IMU teacher, EMG baseline, then EMG distilled student"
python scripts/train_imu_to_emg.py \
  --root "$candidate_root" \
  --session-prefixes "${candidate_prefixes[@]}" \
  --cache-dir "$candidate_cache" \
  --device cuda --seed "$candidate_seed" \
  --teacher-epochs 30 --epochs 50 \
  --latent-weight 0.1 --output-weight 0.25 \
  --output-dir runs/imu_to_emg_four_recordings_seed42

echo "Training complete"
echo "EMG-only checkpoint: runs/imu_to_emg_four_recordings_seed42/emg_student.pt"
echo "Comparison results: runs/imu_to_emg_four_recordings_seed42/results.json"
