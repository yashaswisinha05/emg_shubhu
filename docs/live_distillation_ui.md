# Live wearable-intent UI

This dashboard performs fixed-weight inference on newly arriving raw EMG and
IMU samples. It does not replay dataset trials. Multiple checkpoints read the
same causal buffer, making their screen predictions directly comparable.

## 1. Build the normalization calibration

The tracked models were trained with per-session EMG and IMU normalization.
Build the matching arrays from one session recorded with the same wearer and
unchanged electrode placement:

```bash
python scripts/build_live_distillation_calibration.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_channel_horizon_distillation.yaml \
  --session-prefix dev_a1 \
  --output artifacts/live_dev_a1_calibration.npz
```

Do not reuse a calibration after moving or reapplying the sensors. The model
results use session-specific normalization, so doing so would silently change
the input distribution.

## 2. Start the dashboard and models

```bash
python scripts/run_live_distillation_ui.py \
  --checkpoint "Factor baseline=runs/latent_distillation_pixel_v2/final.pt" \
  --checkpoint "Channel + horizon=runs/channel_horizon_distillation/final.pt" \
  --calibration artifacts/live_dev_a1_calibration.npz \
  --device cuda \
  --screen-width 1920 --screen-height 1080 \
  --interval-ms 40
```

Open <http://127.0.0.1:8765>. Add another `--checkpoint "Name=path"` argument
for any other latent-distillation checkpoint with the same preprocessing. A
virtual-leader checkpoint is technically loadable because it has the same
student architecture, but the 245.5 px run was worse and is not included in
the recommended comparison.

## 3. Send live samples

POST JSON to `http://127.0.0.1:8765/api/event`. At the beginning of each new
movement:

```json
{"event":"start","canvas":[1920,1080]}
```

At the raw acquisition rate, send batches rather than one HTTP request per
sample. A batch has this shape:

```json
{
  "event": "samples",
  "time_s": [100.0000, 100.0008],
  "emg": [
    [0.01, -0.02, 0.03, -0.01],
    [0.02, -0.01, 0.04, -0.02]
  ],
  "imu": [
    [0,0,9.8,0,0,0, 0,0,9.8,0,0,0, 0,0,9.8,0,0,0, 0,0,9.8,0,0,0],
    [0,0,9.8,0,0,0, 0,0,9.8,0,0,0, 0,0,9.8,0,0,0, 0,0,9.8,0,0,0]
  ]
}
```

Raw channel order is:

- EMG: `S0, S4, S8, S12`.
- IMU: `ACC X/Y/Z, GYRO X/Y/Z` for S0, then S4, S8, and S12.

The server accepts a single row using `event: "sample"` as well, but batches
of roughly 25–50 ms are more efficient at the approximately 1.26 kHz raw rate.
After the resting/pre-buffer samples and at the actual movement onset, send:

```json
{"event":"movement_start","time_s":100.300}
```

This retains the causal resting history needed by the IMU posture features,
but clears any warm-up predictions so the displayed improvement curve begins
at movement onset. If no movement-start event is available, elapsed time begins
at the first received sample.

## 4. Supply ground truth separately

The target is optional. It can be sent when the task presents it or only after
touch:

```json
{"event":"target","x_px":1440,"y_px":360,"canvas":[1920,1080]}
```

Or finish the movement and supply the target together:

```json
{"event":"touch","time_s":101.24,"x_px":1440,"y_px":360}
```

The inference history is not recomputed when this arrives. The dashboard only
measures the already-generated predictions against the target, preserving the
strict separation between model inputs and evaluation labels.

## What the UI shows

- Current screen-coordinate prediction and growing prediction trail per model.
- Current pixel error and improvement from the model's first live prediction.
- Pixel error over elapsed time.
- Fixed-weight inference latency.
- Predicted time-to-touch for the channel+horizon checkpoint.
- Live S0/S4/S8/S12 attention over the most recent 100 ms.

The browser polls the local server for display state. Model inference happens
in Python on the selected CUDA device, not in the browser.
