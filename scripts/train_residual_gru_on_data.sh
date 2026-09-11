#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 CLASSIFIER_CHECKPOINT OUTPUT_DIR DATA_ROOT [DATA_ROOT ...]" >&2
  echo "Optional environment: DEVICE, RAW_RATE_HZ, EPOCHS, BATCH_SIZE, SEED" >&2
  exit 2
fi

classifier_checkpoint="$1"
output_dir="$2"
shift 2
dataset_roots=("$@")

device="${DEVICE:-cuda}"
raw_rate_hz="${RAW_RATE_HZ:-1259.4}"
epochs="${EPOCHS:-60}"
batch_size="${BATCH_SIZE:-4}"
seed="${SEED:-42}"

cd "$(dirname "$0")/.."

if [[ ! -f "$classifier_checkpoint" ]]; then
  echo "Classifier checkpoint not found: $classifier_checkpoint" >&2
  exit 2
fi
if [[ -e "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory must be absent or empty: $output_dir" >&2
  exit 2
fi

echo "Training frozen-classifier residual GRU"
echo "Roots: ${dataset_roots[*]}"
echo "Raw rate: $raw_rate_hz Hz"

python scripts/train_neuromuscular_residual_gru.py \
  --classifier-checkpoint "$classifier_checkpoint" \
  --root "${dataset_roots[@]}" \
  --models emg+imu \
  --device "$device" \
  --raw-rate-hz "$raw_rate_hz" \
  --seed "$seed" \
  --split-seed "$seed" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --patience 12 \
  --pixel-architecture direct \
  --pixel-weight 0.35 \
  --position-weight 1.0 \
  --orientation-weight 0 \
  --final-pose-weight 0 \
  --future-pose-weight 0.5 \
  --future-pose-ms 200 \
  --reconstruction-horizon-ms 1000 \
  --reconstruction-step-ms 100 \
  --reconstruction-decay-ms 500 \
  --future-imu-weight 0 \
  --long-state-weight 0 \
  --output-dir "$output_dir"

echo "Training complete"
echo "Live checkpoint: $output_dir/emg_imu_best.pt"
echo "Inference: python scripts/infer_neuromuscular_residual_gru.py --checkpoint $output_dir/emg_imu_best.pt --trial-csv DATA_ROOT/trial_001.csv --device $device"
