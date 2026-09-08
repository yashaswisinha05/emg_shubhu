# Robust causal reach–grasp model

This experiment leaves the previous orientation-hybrid model unchanged. It
trains a new `reach_grasp_robust_v1` checkpoint with three additions:

1. causal, physically motivated wearable augmentation during training only;
2. frame-resolution grasp/release time-to-event distributions;
3. learned position and orientation uncertainty.

VIVE position and orientation are supervision labels. They are never model
inputs. Deployment consumes four EMG channels and 24 IMU axes only.

## Architecture

The local causal EMG/IMU convolutional path predicts grasp/release pulses and
six time classes for each event: approximately 0, 100, 200, 300, and 400 ms in
the future, plus no event in the prediction horizon. The patch-transformer
context predicts holding state, XYZ, continuous 6D orientation, XYZ log
variance, and orientation log variance. A conservatively initialized learned
gate lets the event-time estimate amend the existing local event head.

Augmentation simulates per-electrode EMG gain, sensor noise, short missing
spans, complete sensor dropout, IMU bias drift, and small physical IMU mounting
rotations shared by each sensor's accelerometer and gyroscope. It does not move
future measurements into the past.

## Train

From the repository root:

```bash
git pull origin main
bash scripts/train_reach_grasp_robust.sh
```

Or specify the recording explicitly:

```bash
python scripts/train_reach_grasp_robust.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --device cuda \
  --epochs 60 \
  --batch-size 8 \
  --seed 42 \
  --raw-rate-hz 1259.4 \
  --models imu emg emg+imu \
  --output-dir runs/reach_grasp_robust_seed42
```

The main deployment checkpoint is:

```text
runs/reach_grasp_robust_seed42/emg_imu_best.pt
```

`results.json` reports the existing event, holding, position, orientation, and
ablation metrics plus:

- event-time MAE inside the 450 ms prediction horizon;
- event-within-horizon recall;
- mean predicted position/orientation one-sigma uncertainty;
- empirical one-sigma coverage.

Compare against the existing orientation-hybrid run using identical trial
splits and at least three seeds. Augmentation is useful only if held-out results
improve; training loss is not evidence of robustness.

## Recorded or live Franka inference

The existing deployment scripts recognize the new checkpoint format:

```bash
python scripts/live_franka_pybullet.py \
  --checkpoint runs/reach_grasp_robust_seed42/emg_imu_best.pt \
  --trial-csv /path/to/unseen/trial_001.csv \
  --device cuda \
  --speed 0.5
```

JSON predictions include `position_uncertainty_cm`,
`orientation_uncertainty_deg`, and `event_time_estimate_ms`. The current Franka
controller continues using its validated state machine; these uncertainty
values are exposed for the next confidence-aware controls experiment.
