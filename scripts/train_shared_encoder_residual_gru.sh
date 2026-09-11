#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 CLASSIFIER_CHECKPOINT OUTPUT_DIR DATA_ROOT [DATA_ROOT ...]" >&2
  exit 2
fi

classifier_checkpoint="$1"
output_dir="$2"
shift 2
dataset_roots=("$@")

cd "$(dirname "$0")/.."

python scripts/train_shared_encoder_residual_gru.py \
  --classifier-checkpoint "$classifier_checkpoint" \
  --root "${dataset_roots[@]}" \
  --models emg+imu \
  --device "${DEVICE:-cuda}" \
  --raw-rate-hz "${RAW_RATE_HZ:-1259.4}" \
  --seed "${SEED:-42}" --split-seed "${SPLIT_SEED:-42}" \
  --epochs "${EPOCHS:-60}" --batch-size "${BATCH_SIZE:-4}" --patience 12 \
  --position-weight 1.0 --pixel-weight 0.35 \
  --future-pose-ms 200 --future-pose-weight 0.5 \
  --reconstruction-horizon-ms 1000 --reconstruction-step-ms 100 \
  --reconstruction-decay-ms 500 \
  --long-position-weight 0.05 --emg-to-future-imu-weight 0.05 \
  --output-dir "$output_dir"

echo "Checkpoint: $output_dir/emg_imu_best.pt"
