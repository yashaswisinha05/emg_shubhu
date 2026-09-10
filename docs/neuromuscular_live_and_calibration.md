# Live inference and 30-trial calibration

## Python function

The function consumes raw samples in the trained channel order: four EMG
values (`S0`, `S4`, `S8`, `S12`) and 24 IMU values (ACC XYZ then GYRO XYZ for
each sensor). It performs the same causal filtering, feature construction,
100 Hz resampling, validity masking, and normalization used in training.

```python
from emg_touch.neuromuscular_inference import NeuromuscularStream

predictor = NeuromuscularStream(
    "runs/gripper_neuromuscular_future_fused/emg_imu_best.pt",
    device="cuda",
)

def on_sensor_sample(timestamp_s, emg_4, imu_24):
    result = predictor.update(
        timestamp_s, emg_4, imu_24, canvas_px=(1440, 900))
    if result is not None:
        return result
```

Call `predictor.reset()` between recordings. Large values in
`normalization_diagnostics`—especially a meaningful fraction beyond 10
standard deviations—indicate an input scale/order/domain mismatch. Do not
try to hide that failure with calibration.

## Pixel and gripper calibration

The calibration script freezes current pose, orientation, final endpoint,
future trajectory, and the entire encoder. It learns only grouped FiLM for
the pixel/gripper branches, a 2D pixel affine correction, and gripper
temperature/bias.

```bash
python scripts/calibrate_pixel_gripper_film.py \
  --checkpoint runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --root data/new_candidate_calibration \
  --device cuda --epochs 100 --film-groups 16 \
  --output runs/new_candidate_pixel_gripper_film.pt
```

For 30 valid trials, the deterministic split is approximately 20 train, 5
validation, and 5 untouched test trials. Use the resulting calibration
checkpoint with exactly the same inference function:

```python
predictor = NeuromuscularStream(
    "runs/new_candidate_pixel_gripper_film.pt", device="cuda")
```

Calibration trials should cover the screen and include both open and closed
states. Judge the adapter by the printed untouched-test comparison. If it
does not improve late pixel error without preserving gripper F1, retain the
uncalibrated checkpoint.
