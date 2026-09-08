# Reach–grasp–release: a new experiment

This pipeline does NOT modify the pointing, distillation or manipulator models.
It trains causal wearable encoders for interaction events plus the **current
VIVE T0 XYZ position in metres**. It is not finger pose, an arm skeleton, a future
trajectory, or an orientation estimator. The physical meaning of T0 depends on
where you mounted it; this code does not assume a mounting location.

## Run on your training computer

```bash
git pull origin main
bash scripts/train_reach_grasp_dataset.sh
```

The launcher uses `/home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827` and trains
IMU-only, EMG-only, then EMG+IMU from scratch. It uses the existing `emg_env`
dependencies; no Chronos download, old checkpoint or VAE is required.

Equivalent explicit command:

```bash
python scripts/train_reach_grasp.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --models imu emg emg+imu \
  --device cuda --epochs 40 --batch-size 8 \
  --output-dir runs/reach_grasp_seed42
```

For a smoke run, use `--epochs 1 --output-dir runs/reach_grasp_smoke`.
Nonempty output directories are rejected. All `trial_*.csv` files under root
are discovered recursively; `.npy` and `.pkl` companions are not loaded.

## Labels: clock origin must be correct

Default uses repeated `t_grasp_perf` and `t_release_perf`, on the same absolute
clock as `time_perf_counter`. Nonempty copies must agree, release must follow
grasp, and both events must lie inside the recording. Pre/post buffers remain
in the sequence. Invalid trials and exact duplicate CSV files are listed in
`data_audit.json`; the script requires at least 20 accepted trials.

If absolute event fields are missing, default rejects the trial rather than
guess. **Only after confirming** that `grasp_onset_s` and `grasp_offset_s` are
relative to `t_start_perf`, supply `--event-origin start`. The fallback then
uses `t_start_perf + offset`. Do not use that switch merely to bypass an error.

Grasp/release labels are 100 ms positive pulses STARTING at the event, with a
separate holding label between events. These are annotation-defined events:
if the annotations are button presses, performance is relative to those presses,
not automatically physical contact or force establishment. No label column,
phase code, event time, elapsed trial time, VIVE value or trial number enters
the network. Event times can still correlate with repeatable trial motion, so
the report includes a training-median, schedule-only event baseline.

## Preprocessing

- Sort timestamps, remove duplicate timestamps; keep the time axis across gaps.
- Default raw rate **1259.4 Hz**; set `--raw-rate-hz` to your actual configured
  acquisition rate if different. A conflicting declared rate rejects a trial.
  Filtering assumes approximately regular sampling within uninterrupted blocks;
  it is not a full timestamp-jitter correction algorithm.
- Forward-fill sensor gaps for at most 20 ms. Longer gaps become zero with an
  explicit feature-availability mask. Never backfill from future samples.
- EMG: causal fourth-order 20–450 Hz band-pass (upper cutoff reduced for lower
  raw rates), then trailing 20 and 50 ms RMS per sensor. No median filter on raw
  EMG. This first model uses RMS features, not a raw-waveform encoder or ICA/PCA.
- IMU: trailing three-sample median and causal 15 Hz low-pass; retain gravity/DC.
- Reset filter state at long gaps. Read the most recent past sample on a 100 Hz
  output grid; mask stale samples. These are initial engineering settings, not
  experimentally optimized filters or a guarantee against all aliasing.
- Normalize each feature using valid **training samples only**, saved with the
  checkpoint. Missing features remain zero after normalization with mask bits.
- Position supervision uses finite VIVE XYZ, sync error <=20 ms and tracking
  age <=50 ms when those quality columns exist. Missing tracker labels mask only
  pose loss, not otherwise valid interaction supervision.

## Model and training

Each modality has an equal-width causal temporal-convolution encoder. Fusion
feeds an interaction head (holding, grasp and release logits) and a separate
XYZ head. At 100 Hz the convolutional receptive field is about 1.27 seconds;
all predictions use past/current inputs, not later frames. Whole trials are
batched with right-padding, and padding is excluded from losses.

Training loss is weighted binary cross-entropy plus 0.2 times standardized XYZ
Huber loss. Positive-class weights come from training labels. A frame needs at
least 75% available features in its selected modality (both for fusion).
Evaluation uses the SAME common EMG+IMU availability mask across models.
Missing-input event periods remain potential misses in event recall; holding
and pose metrics exclude invalid frames and report valid coverage.

Trials split 60/20/20 before normalization, with seed 42 by default. This is a
within-dataset trial split, NOT held-out-person/session evidence. Byte-identical
duplicates are removed before splitting; semantically duplicated recordings
with different bytes still require a dataset audit.

Validation chooses event thresholds from a small predefined grid and selects
the checkpoint primarily by grasp/release macro-F1, then holding F1. No test
threshold tuning. Pose error is reported separately, not used to conceal event
failure. All three models are selected before test evaluation.

## Read the results

`runs/reach_grasp_seed42/results.json` reports:

- Grasp and release event precision, recall and F1 with +/-150 ms matching.
- Timing MAE and signed latency for **matched events only**. Always read recall
  alongside latency: missing events are not included in timing MAE.
- False triggers/minute across the full recording (including buffers).
- Holding-state F1 and current XYZ Euclidean error in cm on valid frames.
- `fusion_zero_emg`: removal diagnostic, not a substitute for independently
  trained EMG-only/IMU-only comparisons.
- `schedule_only_baseline`: predicts training-median event times without any
  wearable measurements, to expose an overly scripted timing protocol.

The event detector triggers on upward threshold crossings with a 400 ms
refractory period. It does not search future peaks, constrain event order, or
silently allow only one trigger per trial. There is one annotated grasp and one
release per trial in this initial implementation.

Good evidence for EMG is better event accuracy/timing versus independently
trained IMU-only, not necessarily better XYZ position. Repeat with new seeds and
confirm on untouched recordings. High schedule-only performance calls for more
variation in grasp/release timing and no-grasp motion controls.

Files:

- `imu_best.pt`, `emg_best.pt`, `emg_imu_best.pt`: independent selected models.
- `*_history.json`: validation history and selected thresholds.
- `splits.json`, `data_audit.json`: exact data membership and exclusions.
- Checkpoints contain model weights, normalization, filter settings, thresholds
  and annotation-clock setting for reproducibility.

These checkpoint formats are new. Previous pointing/manipulator UI loaders will
not load them. A hardware-live wrapper must reuse the same causal filters and
training normalization, with an input-only buffering implementation; the offline
CSV loader itself requires annotation columns and is not that wrapper.
