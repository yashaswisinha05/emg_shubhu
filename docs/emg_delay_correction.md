# Frozen IMU + EMG delay sweep

Run from the repository using the completed four-recording teacher:

```bash
python scripts/train_emg_delay_correction.py \
  --teacher-checkpoint runs/imu_to_emg_four_recordings_seed42/imu_teacher.pt \
  --root /home/nahar3/shubham/emg_shubhu \
  --device cuda --epochs 30 \
  --delays-ms 0 25 50 75 100 150 \
  --output-dir runs/emg_delay_correction_seed42
```

Dataset prefixes, seed, preprocessing and calibration settings come from the
teacher checkpoint. The script requires the original `splits.json` alongside it
and rejects changed splits. Keep the original root and dataset unchanged.

The teacher is frozen throughout. The correction starts at exactly zero and
predicts normalized screen and onset-relative 3D-path residuals. Training uses
the existing scaled task loss plus a small squared-correction penalty. No VIVE
data or true lead is passed to either network. IMU remains required at inference.

Delay d means IMU history ends at t and EMG history ends at t-d. Extra past data
is fetched to preserve a full context window, rather than just truncating its
left edge. The zero-delay arm retains the latest EMG. Missing early EMG is masked.
Delays are rounded to samples at the configured effective rate: this inherits
the existing loader's sampling assumptions, not timestamp-exact realignment.
This is a lookback/truncation experiment, not a learned physiological-delay
estimator or proof of electromechanical delay. Preprocessing is unchanged.

A matched-capacity IMU-context control receives zero sensor inputs to the same
correction encoder. The sweep selects EMG delay and epoch by validation
`screen_px + 5 * path_cm`; epoch zero is eligible. Test is then evaluated only
for the selected EMG branch, the selected control, and the original IMU baseline.
Zero/shuffled-EMG tests are also reported (shuffle is within minibatches; singleton
batches cannot be shuffled). Per-lead arrays are [screen px, path cm, endpoint cm].

Read `results.json`. A useful gain requires the selected EMG branch to improve
over both IMU baseline and control, not just have a large shuffle penalty.
Repeat across seeds before claiming a reliable gain. Existing data has already
been used for other experiments, so it is not a fresh confirmatory test set.

`selected_emg.pt` includes the frozen teacher and correction weights, config,
delay and model constructor settings. Other checkpoints are validation winners
for each sweep arm. Existing live UI loaders do not support this new format.
No original model or checkpoint is overwritten, and 30 px improvement is not
guaranteed. This script uses a small GRU, not Chronos.
