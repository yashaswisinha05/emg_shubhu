# Complete-Trajectory Configuration Experiment

This is the recommended first experiment: given the complete recorded trajectory, measure
how accurately each sensor configuration predicts the touch location. Signal `time_s=0` is
the start of the pre-buffered recording, while `reaction_time_s` is a cue-to-click duration;
they are not directly comparable timestamps. The complete files generally extend a few tens
of milliseconds beyond the recorded click/save callback. Use `EMG_TEMPORAL_STUDY.md` for
precise touch-aligned causal conclusions.

It is separate from the earlier early-prediction/participant-held-out pipeline.

## Protocol

- Keep every valid trajectory at its natural duration; never crop it to 320 samples.
- Resample at the common rate, apply a causal 3-sample EMG median filter, and normalize with
  a scaler fitted only on the training partition.
- Pad only to the longest trajectory in a batch. Padding is masked from model pooling.
- Split trials separately inside every participant and coarse target region into five folds:
  60% train, 20% validation, and 20% test.
- Test every valid trajectory exactly once across the five folds.
- Pool out-of-fold predictions by configuration, not by participant.
- Allow configurations to have different participant and trial counts.

This first protocol measures performance for participants represented in training. It does
not estimate generalization to a completely unseen participant; use the original
participant-held-out pipeline for that later question.

Trajectories outside 0.25 to 10.0 seconds are treated as likely corrupt and excluded, not
truncated. Each run lists them in `data_report.json`.

## Models

- `emg_tcn`: EMG-only temporal convolution baseline.
- `emg_patch`: EMG-only PatchTST-style encoder.
- `imu_patch`: IMU-only diagnostic model.
- `multimodal`: EMG+IMU patch encoders with cross-attention.

All use the same deterministic coordinate head and robust regression loss. Add distillation,
MDN, or CVAE only after these models demonstrate that the signals beat center prediction.

## Environment

Run all commands from the project directory:

```bash
conda activate smss
cd /Users/yashaswi/phd_iisc/emg_shubhu/emg_touch
python -m pip install -e .
```

## One-time data preparation

Skip this if the manifest and cache already exist.

```bash
python scripts/build_manifest.py --config configs/full_trajectory.yaml
python scripts/prepare_cache.py \
  --config configs/full_trajectory.yaml \
  --workers 4
```

## First run: a1, fold 0

```bash
python scripts/make_trajectory_folds.py \
  --config configs/full_trajectory.yaml \
  --configuration a1

fold=artifacts/trajectory_cv/a1/fold-0

python scripts/fit_scaler.py \
  --config configs/full_trajectory.yaml \
  --split "$fold/split.json" \
  --output "$fold/scaler.npz"

python scripts/evaluate_full_trajectory_mean.py \
  --config configs/full_trajectory.yaml \
  --split "$fold/split.json" \
  --scaler "$fold/scaler.npz" \
  --output-dir runs/full_trajectory/a1/fold-0/mean_baseline

python scripts/train_full_trajectory.py \
  --config configs/full_trajectory.yaml \
  --kind emg_tcn \
  --split "$fold/split.json" \
  --scaler "$fold/scaler.npz" \
  --output-dir runs/full_trajectory/a1/fold-0
```

Train the three patch models after the TCN command succeeds:

```bash
for kind in emg_patch imu_patch multimodal; do
  python scripts/train_full_trajectory.py \
    --config configs/full_trajectory.yaml \
    --kind "$kind" \
    --split "$fold/split.json" \
    --scaler "$fold/scaler.npz" \
    --output-dir runs/full_trajectory/a1/fold-0
done
```

Training automatically evaluates the best checkpoint and writes these files below the
model-kind directory: `best.pt`, `history.csv`, `test_metrics.json`, `predictions.csv`,
`data_report.json`, and `config.yaml`.

## Run all five folds for a1

Complete fold 0 first. Then run:

```bash
for fold_dir in artifacts/trajectory_cv/a1/fold-*; do
  fold_name="$(basename "$fold_dir")"

  python scripts/fit_scaler.py \
    --config configs/full_trajectory.yaml \
    --split "$fold_dir/split.json" \
    --output "$fold_dir/scaler.npz"

  python scripts/evaluate_full_trajectory_mean.py \
    --config configs/full_trajectory.yaml \
    --split "$fold_dir/split.json" \
    --scaler "$fold_dir/scaler.npz" \
    --output-dir "runs/full_trajectory/a1/$fold_name/mean_baseline"

  for kind in emg_tcn emg_patch imu_patch multimodal; do
    python scripts/train_full_trajectory.py \
      --config configs/full_trajectory.yaml \
      --kind "$kind" \
      --split "$fold_dir/split.json" \
      --scaler "$fold_dir/scaler.npz" \
      --output-dir "runs/full_trajectory/a1/$fold_name"
  done
done
```

Summarize the five out-of-fold files separately for each model:

```bash
mkdir -p evaluation/full_trajectory

for kind in mean_baseline emg_tcn emg_patch imu_patch multimodal; do
  python scripts/compare_configs.py \
    runs/full_trajectory/a1/fold-*/"$kind"/predictions.csv \
    --output "evaluation/full_trajectory/a1_${kind}.csv"
done
```

The comparison command rejects duplicate trial predictions. `held_out_trials` should equal
the number of valid a1 trajectories across all participants.

## Run all configurations

Create all folds by omitting `--configuration`:

```bash
python scripts/make_trajectory_folds.py \
  --config configs/full_trajectory.yaml
```

Start with one model across every configuration and fold:

```bash
kind=emg_tcn

for configuration_dir in artifacts/trajectory_cv/*; do
  configuration="$(basename "$configuration_dir")"

  for fold_dir in "$configuration_dir"/fold-*; do
    fold_name="$(basename "$fold_dir")"

    python scripts/fit_scaler.py \
      --config configs/full_trajectory.yaml \
      --split "$fold_dir/split.json" \
      --output "$fold_dir/scaler.npz"

    python scripts/train_full_trajectory.py \
      --config configs/full_trajectory.yaml \
      --kind "$kind" \
      --split "$fold_dir/split.json" \
      --scaler "$fold_dir/scaler.npz" \
      --output-dir "runs/full_trajectory/$configuration/$fold_name"
  done
done
```

Compare configurations using every out-of-fold prediction for that model:

```bash
python scripts/compare_configs.py \
  runs/full_trajectory/*/fold-*/emg_tcn/predictions.csv \
  --output evaluation/full_trajectory/emg_tcn_configuration_accuracy.csv
```

Repeat with `emg_patch`, `imu_patch`, and `multimodal` after the TCN pipeline is verified.

## Re-evaluate a checkpoint

Training already evaluates its best checkpoint. To regenerate the evaluation files:

```bash
python scripts/evaluate_full_trajectory.py \
  --config configs/full_trajectory.yaml \
  --kind emg_tcn \
  --checkpoint runs/full_trajectory/a1/fold-0/emg_tcn/best.pt \
  --split artifacts/trajectory_cv/a1/fold-0/split.json \
  --scaler artifacts/trajectory_cv/a1/fold-0/scaler.npz \
  --output-dir evaluation/full_trajectory/a1/fold-0/emg_tcn
```

## Interpret the diagnostic models

- If `imu_patch` and `multimodal` stay near center prediction, inspect time/label alignment
  and whether recordings really end at the touch.
- If IMU works but both EMG models fail, movement information exists but the four RMS EMG
  channels may not identify location by themselves.
- If multimodal beats IMU, EMG adds information beyond motion.
- If `emg_patch` beats `emg_tcn`, long-range patch attention is useful here.

Primary metrics are target-box accuracy, 8-by-5 screen-region accuracy, accuracy within
50/100 pixels, and median/p90 pixel error.
