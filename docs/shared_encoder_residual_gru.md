# Shared-encoder residual GRU

This is the single-encoder version of the final model. The selected Stage-I
neuromuscular-future checkpoint supplies one frozen EMG patch encoder, one
frozen IMU patch encoder, and the authoritative open/close classifier. Those
features are computed once and reused by two-layer causal GRU adapters for
current XYZ, pixel XY, 10--200 ms future XYZ, and 0.1--1.0 s intent XYZ.

The training-only future-IMU decoder receives the EMG representation only. It
therefore tests whether present muscle activity predicts subsequent mechanics
without directly copying the current IMU representation. Masked-EMG
reconstruction is disabled.

## Train

The classifier checkpoint must come from
`train_gripper_neuromuscular_future.py` and have format
`gripper_neuromuscular_future_v1`.

```bash
git pull origin main

bash scripts/train_shared_encoder_residual_gru.sh \
  runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  runs/shared_encoder_residual_gru_multi_person \
  data/shubham_open data/shubham_close \
  data/mukund_open data/mukund_closed \
  data/gazania_open data/gazania_closed
```

The deployable checkpoint is:

```text
runs/shared_encoder_residual_gru_multi_person/emg_imu_best.pt
```

To override runtime settings:

```bash
DEVICE=cuda EPOCHS=60 BATCH_SIZE=4 SEED=42 SPLIT_SEED=42 \
  bash scripts/train_shared_encoder_residual_gru.sh \
  CLASSIFIER_CHECKPOINT OUTPUT_DIR DATA_ROOT [DATA_ROOT ...]
```

Use a new, empty output directory for every run.

## Infer on an unseen CSV

```bash
python scripts/infer_shared_encoder_residual_gru.py \
  --checkpoint runs/shared_encoder_residual_gru_multi_person/emg_imu_best.pt \
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
  --checkpoint runs/shared_encoder_residual_gru_multi_person/emg_imu_best.pt \
  --canvas-px 1920 1080 --device cuda
```

Each emitted JSON record contains open/close probabilities and hysteretic
state, current position, normalized and absolute pixel coordinates, dense
future positions through 200 ms, intent positions through 1 s, and the learned
EMG correction gate.
