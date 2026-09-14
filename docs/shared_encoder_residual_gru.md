# Shared-encoder residual GRU

This is the single-encoder version of the final model. The selected Stage-I
checkpoint supplies one frozen EMG patch encoder, one frozen IMU patch encoder,
and the authoritative open/close classifier. Stage I uses only class-balanced
open/close cross-entropy. Those
features are computed once and reused by two-layer causal GRU adapters for
current XYZ, pixel XY, and 10--200 ms future XYZ.

The training-only future-IMU decoder receives the EMG representation only. It
therefore tests whether present muscle activity predicts subsequent mechanics
without directly copying the current IMU representation. Masked-EMG
reconstruction is disabled. Stage II directly regresses normalized pixel XY;
grid classification, within-grid offsets, future consistency, and correction
regularization are removed.

## Train

Train both simplified stages together:

```bash
git pull origin main

bash scripts/train_minimal_two_stage_model.sh \
  runs/gripper_classifier_minimal \
  runs/shared_encoder_residual_gru_minimal \
  data/shubham_open data/shubham_close \
  data/mukund_open data/mukund_closed \
  data/gazania_open data/gazania_closed
```

The deployable checkpoint is:

```text
runs/shared_encoder_residual_gru_minimal/emg_imu_best.pt
```

To override runtime settings:

```bash
DEVICE=cuda EPOCHS=60 BATCH_SIZE=4 SEED=42 SPLIT_SEED=42 \
  bash scripts/train_shared_encoder_residual_gru.sh \
  CLASSIFIER_CHECKPOINT OUTPUT_DIR DATA_ROOT [DATA_ROOT ...]
```

Use a new, empty output directory for every run.

## Train proper modality ablations

The following command trains independent EMG-only and IMU-only Stage-I and
Stage-II models. It uses the same split seed, widths, heads, horizons, and loss
weights as the fused model; it is therefore the appropriate modality ablation.
Test-time zeroing of a fused checkpoint is only a sensor-removal stress test.

```bash
bash scripts/train_minimal_unimodal_ablations.sh \
  runs/shared_encoder_unimodal_ablations \
  data/shubham_open data/shubham_close \
  data/mukund_open data/mukund_closed \
  data/gazania_open data/gazania_closed
```

The resulting checkpoints are
`emg_motion/emg_best.pt` and `imu_motion/imu_best.pt` beneath the selected
output root.

## Infer on an unseen CSV

```bash
python scripts/infer_shared_encoder_residual_gru.py \
  --checkpoint runs/shared_encoder_residual_gru_minimal/emg_imu_best.pt \
  --trial-csv data/unseen_person/trial_001.csv \
  --canvas-px 1920 1080 \
  --device cuda \
  --speed 0
```

`--speed 0` evaluates without real-time waiting; `--speed 1` replays according
to recorded timestamps.

## Infer from a live source

Omit `--trial-csv` and send one JSON object per line on standard input:

```json
{"time_s": 1.234, "emg": [0, 0, 0, 0], "imu": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

```bash
sensor_program | python scripts/infer_shared_encoder_residual_gru.py \
  --checkpoint runs/shared_encoder_residual_gru_minimal/emg_imu_best.pt \
  --canvas-px 1920 1080 --device cuda
```

Each emitted JSON record contains open/close probabilities and hysteretic
state, current position, normalized and absolute pixel coordinates, dense
future positions through 200 ms, and the learned EMG correction gate.
