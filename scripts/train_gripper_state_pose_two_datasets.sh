#!/usr/bin/env bash
# Train gripper-state pose model pooling two dataset directories into one run.
#
#   bash scripts/train_gripper_state_pose_two_datasets.sh \
#     /path/to/dataset_a /path/to/dataset_b \
#     --output-dir runs/gripper_state_pose_combined
#
# Both directories are searched recursively for trial_*.csv and pooled into a
# single train/validation/test split (see train_gripper_state_pose.py --root,
# which now accepts more than one directory). Trials that are byte-identical
# across the two roots are still deduplicated by content hash, same as
# duplicates within a single dataset. Any extra arguments are forwarded
# as-is to train_gripper_state_pose.py, including --models to change or
# disable the emg/imu/emg+imu ablation.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <dataset_a> <dataset_b> [train_gripper_state_pose.py args...]" >&2
  exit 1
fi

DATASET_A="$1"
DATASET_B="$2"
shift 2

for dataset in "$DATASET_A" "$DATASET_B"; do
  if [ ! -d "$dataset" ]; then
    echo "error: dataset directory not found: $dataset" >&2
    exit 1
  fi
done

python scripts/train_gripper_state_pose.py \
  --root "$DATASET_A" "$DATASET_B" \
  --device cuda --epochs 60 --batch-size 4 --seed 42 \
  --raw-rate-hz 1259.4 \
  --output-dir runs/gripper_state_pose_combined_seed42 "$@"
