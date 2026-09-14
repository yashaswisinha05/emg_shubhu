#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 OUTPUT_ROOT DATA_ROOT [DATA_ROOT ...]" >&2
  exit 2
fi

output_root="$1"
shift
dataset_roots=("$@")

cd "$(dirname "$0")/.."

for modality in emg imu; do
  classifier_dir="$output_root/${modality}_classifier"
  motion_dir="$output_root/${modality}_motion"

  echo "[$modality 1/2] Training modality-specific open/close classifier"
  python scripts/train_gripper_classifier_minimal.py \
    --root "${dataset_roots[@]}" \
    --models "$modality" \
    --device "${DEVICE:-cuda}" \
    --raw-rate-hz "${RAW_RATE_HZ:-1259.4}" \
    --seed "${SEED:-42}" --split-seed "${SPLIT_SEED:-42}" \
    --epochs "${STAGE1_EPOCHS:-60}" --batch-size "${BATCH_SIZE:-4}" \
    --patience 12 \
    --output-dir "$classifier_dir"

  echo "[$modality 2/2] Training modality-specific motion model"
  python scripts/train_shared_encoder_residual_gru.py \
    --classifier-checkpoint "$classifier_dir/${modality}_best.pt" \
    --root "${dataset_roots[@]}" \
    --models "$modality" \
    --device "${DEVICE:-cuda}" \
    --raw-rate-hz "${RAW_RATE_HZ:-1259.4}" \
    --seed "${SEED:-42}" --split-seed "${SPLIT_SEED:-42}" \
    --epochs "${STAGE2_EPOCHS:-60}" --batch-size "${BATCH_SIZE:-4}" \
    --patience 12 \
    --position-weight 1.0 --pixel-weight 0.35 \
    --future-pose-ms 200 --future-pose-weight 0.5 \
    --reconstruction-horizon-ms 1000 --reconstruction-step-ms 100 \
    --reconstruction-decay-ms 500 \
    --emg-to-future-imu-weight 0.05 \
    --output-dir "$motion_dir"
done

echo "EMG checkpoint: $output_root/emg_motion/emg_best.pt"
echo "IMU checkpoint: $output_root/imu_motion/imu_best.pt"
