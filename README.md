# Minimal causal EMG+IMU model

This branch contains only the final causal model and the code required to
train and run it. The model predicts open/closed gripper state, current XYZ,
continuous screen XY, and future XYZ at 10--200 ms. It has no grid head.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Arrange the data

Directories may have any name and are searched recursively for `trial_*.csv`:

```text
data/
├── participant_01_open/
│   ├── trial_001.csv
│   └── trial_002.csv
├── participant_01_closed/
│   └── trial_001.csv
└── participant_02/
    └── trial_001.csv
```

Each CSV must contain:

- `time_perf_counter`
- `EMG 1_S0`, `EMG 1_S4`, `EMG 1_S8`, `EMG 1_S12`
- `ACC X/Y/Z_*` and `GYRO X/Y/Z_*` for `S0`, `S4`, `S8`, and `S12`
- `gripper_state` with `open` or `close`
- `VIVE_T0_pos_x_m`, `VIVE_T0_pos_y_m`, `VIVE_T0_pos_z_m`
- `canvas_width_px`, `canvas_height_px`
- `click_x_norm`, `click_y_norm`, or alternatively `click_x_pos`, `click_y_pos`

## Train

```bash
python minimal_emg_imu/train.py \
  --root data/participant_01_open data/participant_01_closed \
         data/participant_02 \
  --device cuda \
  --epochs 60 \
  --output-dir runs/minimal_emg_imu
```

Use `--raw-rate-hz` if the raw sampling rate is not 1259.4 Hz. The output
directory must be new or empty. The best checkpoint is written to
`runs/minimal_emg_imu/emg_imu_best.pt`.

## Run inference on a CSV

```bash
python minimal_emg_imu/infer.py \
  --checkpoint runs/minimal_emg_imu/emg_imu_best.pt \
  --trial-csv data/unseen/trial_001.csv \
  --canvas-px 1920 1080 \
  --device cuda
```

Inference reads only timestamps, four EMG channels, and 24 IMU channels.
VIVE positions, screen targets, and gripper labels are never model inputs.

## Test

```bash
pytest -q
```
