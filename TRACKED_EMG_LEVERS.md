# Wearable EMG contribution experiments

These experiments ask whether EMG adds **trial-specific predictive information**
to the wearable trajectory model. They do not treat attention weights or a
forced latent allocation as evidence. The primary intervention is the change in
held-out error when the same trained checkpoint receives intact, zeroed, or
trial-shuffled EMG.

All commands below use VIVE only to define movement onset, construct labels and
score predictions. In `--task wearable`, VIVE position and velocity are forced
to zero before every model call.

## Fast software check

```bash
python scripts/run_ablation_sweep.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_trajectory_emg_enhanced.yaml \
  --cache-dir artifacts/tracked_cache \
  --output-dir runs/emg_levers_smoke \
  --task wearable --models trajectory --ablations all-inputs \
  --cutoff-offset-ms -100 -50 0 50 100 \
  --epochs 1 --seeds 1
```

This is only a wiring check. Do not report its accuracy.

## Controlled sequence

Use the same seeds, cutoff offsets, split and epoch budget in every stage. Give
every stage a new output directory: the sweep resumes existing directories and
will intentionally reuse an existing `results.json`.

### E0: corrected timing, legacy fusion

```bash
python scripts/run_ablation_sweep.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_trajectory.yaml --cache-dir artifacts/tracked_cache \
  --output-dir runs/emg_E0_timing --task wearable \
  --cutoff-offset-ms -100 -50 0 50 100 \
  --paired-modality-interventions --seeds 1 2 3
```

This fixes the earlier test, which started about 127 ms after onset and assigned
every row in a batch the final row's timing bucket. Evidence for anticipatory
EMG is an EMG-removal or EMG-shuffle penalty concentrated at -100 to +50 ms.

### E1: equal-width modality encoders

Add `--separate-modalities` and use `runs/emg_E1_separate`. This tests whether
the old single projection's 4 EMG versus 24 IMU channel imbalance caused the
model to ignore EMG. A useful change increases the EMG removal/shuffle penalty
without increasing intact-input test error.

### E2: EMG-only auxiliary objective

```bash
... --separate-modalities --emg-only-weight 0.5 \
    --output-dir runs/emg_E2_auxiliary
```

This adds

`loss = fused trajectory loss + 0.5 * EMG-only trajectory loss`.

It succeeds only if `emg_only_mean_m` improves on held-out sessions and the
intact fused model stays equal or improves. Sweep weights 0.25, 0.5 and 1.0 in
separate output directories, selecting the weight on validation results.

### E3: IMU modality dropout

```bash
... --separate-modalities --emg-only-weight 0.5 --imu-dropout 0.5 \
    --output-dir runs/emg_E3_dropout
```

This removes the entire IMU latent for a random subset of training trials. Test
0.25, 0.5 and 0.75 separately. It succeeds when EMG-only performance and the
EMG-removal/shuffle penalties improve without sacrificing intact-input error.
Do not select a setting merely because it makes the ablation penalty large.

### E4: causal high-rate EMG feature bank

```bash
... --separate-modalities --emg-only-weight 0.5 --imu-dropout 0.5 \
    --emg-feature-windows-ms 10 25 50 \
    --emg-feature-kinds rms waveform_length log_energy derivative \
    --output-dir runs/emg_E4_features
```

RMS, waveform length, log energy and RMS derivative are calculated causally at
the raw ~1.26 kHz EMG rate before decimation. Compare progressively:

1. `rms`
2. `rms waveform_length`
3. `rms waveform_length log_energy`
4. all four kinds

Keep a feature only when it improves held-out EMG-only or fused accuracy across
seeds. More feature dimensions alone are not evidence of more EMG information.

## Combined experiment

The enhanced config enables separate encoders, EMG-only weight 0.5, IMU dropout
0.5, all causal EMG features and paired modality interventions:

```bash
python scripts/run_ablation_sweep.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_trajectory_emg_enhanced.yaml \
  --cache-dir artifacts/tracked_cache \
  --output-dir runs/sweep_wearable_emg_enhanced \
  --task wearable --cutoff-offset-ms -100 -50 0 50 100 --seeds 1 2 3
```

## Reading the output

- `model`: intact EMG+IMU prediction.
- `emg_only`: auxiliary EMG pathway through the shared trajectory decoder.
- `without_emg - model`: same-checkpoint EMG removal cost; positive is useful.
- `shuffled_emg - model`: trial-pairing cost; positive shows the model uses
  information about this trial rather than an average EMG pattern.
- `without_imu - model`: corresponding IMU removal cost.
- `mean_reach`: no-input population-average reference.

The strongest support for the project is: intact and EMG-only models beat the
mean-reach reference, both EMG interventions worsen the same checkpoint, the
effect is largest before/around movement onset, and it repeats across seeds and
held-out sessions. If only the training loss or latent attribution changes, EMG
importance has been forced architecturally rather than demonstrated.
