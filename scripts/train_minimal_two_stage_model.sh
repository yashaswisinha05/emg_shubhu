#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 STAGE1_OUTPUT STAGE2_OUTPUT DATA_ROOT [DATA_ROOT ...]" >&2
  exit 2
fi

stage1_output="$1"
stage2_output="$2"
shift 2
dataset_roots=("$@")

cd "$(dirname "$0")/.."

python scripts/train_gripper_classifier_minimal.py \
  --root "${dataset_roots[@]}" \
  --models emg+imu \
  --device "${DEVICE:-cuda}" \
  --raw-rate-hz "${RAW_RATE_HZ:-1259.4}" \
  --seed "${SEED:-42}" --split-seed "${SPLIT_SEED:-42}" \
  --epochs "${STAGE1_EPOCHS:-60}" --batch-size "${BATCH_SIZE:-4}" \
  --patience 12 \
  --output-dir "$stage1_output"

bash scripts/train_shared_encoder_residual_gru.sh \
  "$stage1_output/emg_imu_best.pt" \
  "$stage2_output" \
  "${dataset_roots[@]}"

echo "Final checkpoint: $stage2_output/emg_imu_best.pt"
