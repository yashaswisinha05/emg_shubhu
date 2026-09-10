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

Calibration CSVs do **not** need any VIVE columns. Required data are the
timestamp, four EMG channels, 24 IMU channels, `gripper_state`, canvas size,
and either normalized or pixel click coordinates.

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

## Gripper-only hard-contrastive calibration

When 20 recordings are available for each static state, use the dedicated
calibrator below. It needs no VIVE or pixel columns and leaves the pixel head
as well as every pose/future output unchanged.

```bash
python scripts/calibrate_gripper_hard_contrastive.py \
  --checkpoint runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --open-root data/candidate_open \
  --close-root data/candidate_close \
  --trials-per-class 20 --device cuda --epochs 100 \
  --contrastive-weight 0.5 --contrastive-margin 0.35 \
  --output runs/candidate_gripper_contrastive.pt
```

The split is stratified by complete trial: 14 open + 14 close for training,
3 + 3 for validation, and 3 + 3 for the untouched test. The loss constructs
one feature centroid per class per trial and applies hardest-positive versus
closest-negative cosine triplet loss. It therefore cannot inflate the sample
count with overlapping frames from the same recording.

## Nine-grid two-parameter calibration

For exactly one open and one close recording at each of nine screen cells,
keep the classifier itself frozen and calibrate only its probability output:

```bash
python scripts/calibrate_gripper_logits_9grid.py \
  --checkpoint runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --open-root data/candidate_open \
  --close-root data/candidate_close \
  --grid-count 9 --device cuda \
  --output runs/candidate_gripper_logits.pt
```

The script pairs open/close recordings using their click coordinate, performs
nine-fold leave-one-grid-out evaluation, and then fits one deployment adapter
on all 18 trials. It learns only a scalar temperature and one close-class
bias. Every recording receives equal loss weight regardless of duration.
The pixel coordinate is used only as the grid identifier; the pixel head is
not trained. VIVE is not required.

Use `runs/candidate_gripper_logits.pt` with `NeuromuscularStream` exactly like
the other checkpoints. Trust the adapter only if the aggregated
leave-one-grid-out result improves over its printed baseline.
