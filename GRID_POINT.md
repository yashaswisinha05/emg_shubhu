# Calibrated Grid-and-Offset Touch Model

This pipeline predicts one touch point with an 8×5 CenterNet-style heatmap and a
continuous local offset. It is touch-aligned and uses no post-touch signal.

## Architecture

```text
Raw IMU → initial-rest calibration → full causal encoder ─────┐
                              └→ final 500 ms encoder ────────┤
Raw EMG → final 500 ms encoder ───────────────────────────────┤
        └→ final 300 ms encoder → lookback/reliability gate ──┤
                                                              ↓
                                              8×5 heatmap + offsets
                                                              ↓
                                                   continuous screen (x,y)
```

Per sensor, IMU calibration creates:

- baseline-subtracted acceleration;
- gyro-bias-corrected angular velocity;
- initial gravity direction (resting roll/pitch) and integrated relative orientation;
- acceleration-magnitude change and gyro magnitude;
- jerk magnitude and angular-acceleration magnitude.

The first 300 ms of the pre-cue recording estimates the resting baseline and
gravity direction. Four sensors produce 64 calibrated features.

Three models use identical splits and scaling:

- `grid_imu`: full plus endpoint calibrated IMU.
- `grid_emg`: nested final-500/final-300-ms EMG.
- `grid_fusion`: exact frozen `grid_imu` plus a reliability-gated EMG residual.

The fusion residual is initialized to zero, so before training its predictions are
exactly equal to the pretrained IMU model. This makes the paired fusion-versus-IMU
comparison interpretable.

## Loss

The target heatmap is a normalized Gaussian over neighbouring cells. Offsets are
relative to cell centres and lie in approximately `[-0.5, 0.5]`. All loss terms are
added:

```text
heatmap soft cross-entropy
+ target-cell offset Smooth-L1
+ pixel-scaled Smooth-L1
+ radial Charbonnier distance
```

Loss magnitudes are normalized by the 80-pixel target size. Evaluation uses hard
heatmap argmax plus that cell's offset. Prediction files also contain heatmap
confidence, entropy, selected cell, EMG reliability, and 300/500-ms gate weights.

## Prerequisites

Run from the project directory in the `smss` environment:

```bash
cd /Users/yashaswi/phd_iisc/emg_shubhu/emg_touch
conda activate smss
python -m pip install -e .
```

Ensure the manifest contains touch alignment:

```bash
python scripts/build_manifest.py --config configs/grid_point.yaml
```

It should report `Touch-aligned timestamps: 13200/13200`. Existing signal caches
and trajectory folds are reused.

## One-fold pilot

```bash
python scripts/run_grid_point_sweep.py \
  --config configs/grid_point.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

This fits a leakage-safe calibrated-feature scaler and trains all three models.

## Complete mix7 five-fold run

The runner is resumable and skips the completed pilot fold:

```bash
caffeinate -i python scripts/run_grid_point_sweep.py \
  --config configs/grid_point.yaml \
  --configuration mix7 \
  --device mps \
  2>&1 | tee -a runs/grid_point_mix7.log
```

## All configurations

After confirming mix7:

```bash
caffeinate -i python scripts/run_grid_point_sweep.py \
  --config configs/grid_point.yaml \
  --device mps \
  2>&1 | tee -a runs/grid_point_all.log
```

Different participant and trajectory counts per configuration are supported.

## Outputs

- `runs/grid_point/sweep_status.json`: progress and failures.
- `runs/grid_point/<configuration>/fold-<n>/<model>/best.pt`: checkpoints.
- Each model directory contains validation/test metrics and trial predictions.
- `evaluation/grid_point/*_configuration_accuracy.csv`: configuration rankings.
- `evaluation/grid_point/fusion_vs_imu.csv`: paired participant-blocked EMG gain.

Positive `paired_mean_gain_px` in `fusion_vs_imu.csv` means the endpoint EMG
residual improved the exact IMU reference. Require its 95% interval to remain above
zero and check practical metrics such as within-100-pixel and target-box accuracy.

Do not move to 16×8 or a CVAE until the 8×5 model beats direct regression under the
same folds. A 16×8 ablation can then be made by copying the configuration and
changing `model.grid_size` to `[16, 8]`.
