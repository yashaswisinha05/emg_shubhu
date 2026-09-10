#!/usr/bin/env bash
# Train the gripper-state pose model pooling any number of dataset directories
# into one run.
#
#   bash scripts/train_gripper_state_pose_two_datasets.sh \
#     /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c \
#     --output-dir runs/gripper_state_pose_combined
#
# Every leading argument is treated as a dataset directory, up to the first
# one starting with "-"; the rest are forwarded as-is to
# train_gripper_state_pose.py, including --models to change or disable the
# emg/imu/emg+imu ablation and --pixel-weight to control click-target
# regression. All directories are searched recursively for trial_*.csv and
# pooled into a single train/validation/test split. Trials that are
# byte-identical across roots are deduplicated by content hash, same as
# duplicates within a single dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=()
while [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; do
  if [ ! -d "$1" ]; then
    echo "error: dataset directory not found: $1" >&2
    exit 1
  fi
  DATASETS+=("$1")
  shift
done

if [ "${#DATASETS[@]}" -lt 1 ]; then
  echo "usage: $0 <dataset> [<dataset>...] [train_gripper_state_pose.py args...]" >&2
  exit 1
fi

python scripts/train_gripper_state_pose.py \
  --root "${DATASETS[@]}" \
  --device cuda --epochs 60 --batch-size 4 --seed 42 \
  --raw-rate-hz 1259.4 \
  --output-dir runs/gripper_state_pose_combined_seed42 "$@"
