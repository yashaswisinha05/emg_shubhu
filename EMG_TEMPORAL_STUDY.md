# Touch-Aligned EMG Temporal Contribution Study

This study asks **when EMG adds information beyond IMU**. It is separate from the
completed full-recording sweep and writes to new artifact, run, and evaluation
directories, so it does not overwrite existing checkpoints.

## Experimental control

- Every input is cropped at `touch_time_s`, reconstructed by mapping the recorded
  GUI click/save monotonic timestamp into the signal time axis; post-touch samples
  are discarded. `reaction_time_s` is not used directly as a signal timestamp
  because it excludes the pre-buffer preceding the cue.
- The IMU input is the same complete causal trajectory in every comparison.
- Only the retained EMG time window changes.
- Every window uses the same five folds and the same causal training-fold scaler.
- `imu_patch` is trained once per fold and reused as the reference.
- `emg_residual` starts from that IMU checkpoint, freezes the exact reference IMU
  encoder/head, and learns a correction whose input is EMG only. It deliberately
  has no second trainable IMU path, preventing extra IMU capacity from being
  mistaken for EMG value.
- `emg_patch` measures how predictive each EMG window is by itself.
- Window selection is performed separately inside each fold using validation data;
  the associated outer test fold is used only for final reporting.
- Confidence intervals resample participants as blocks. Unequal participant and
  trajectory counts across configurations are allowed.

The predefined touch-relative windows are in
`configs/emg_temporal_study.yaml`. Positive `paired_mean_gain_px` means that adding
the EMG window reduced pixel error relative to causal IMU alone.

## Environment

```bash
cd /Users/yashaswi/phd_iisc/emg_shubhu/emg_touch
conda activate smss
python -m pip install -e .
```

Rebuild the manifest once to add `touch_time_s`. This reads existing trial metadata;
the signal cache and `artifacts/trajectory_cv` folds remain valid and are reused.

```bash
python scripts/build_manifest.py --config configs/emg_temporal_study.yaml
```

The command should report a touch-aligned timestamp for every trial. New causal
scalers are fitted automatically.

## Recommended pilot: mix7

Run all seven windows and all five folds for the current best configuration first:

```bash
caffeinate -i python scripts/run_emg_temporal_study.py \
  --config configs/emg_temporal_study.yaml \
  --configuration mix7 \
  --device mps \
  2>&1 | tee -a runs/emg_temporal_touch_mix7.log
```

The command is resumable. A model is skipped only after its checkpoint, validation
predictions, and test predictions all exist.

For a one-fold software check before the pilot:

```bash
python scripts/run_emg_temporal_study.py \
  --config configs/emg_temporal_study.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

## Full configuration study

Omit `--configuration` after the pilot is complete. Existing mix7 results are
automatically reused.

```bash
caffeinate -i python scripts/run_emg_temporal_study.py \
  --config configs/emg_temporal_study.yaml \
  --device mps \
  2>&1 | tee -a runs/emg_temporal_touch_all.log
```

This is a large experiment: 13 configurations × 5 folds × 7 windows. Run the mix7
pilot first and, if desired, confirm only the most promising windows across every
configuration by repeating `--window`, for example:

```bash
caffeinate -i python scripts/run_emg_temporal_study.py \
  --config configs/emg_temporal_study.yaml \
  --window causal_all \
  --window final_300ms \
  --window final_500ms \
  --device mps \
  2>&1 | tee -a runs/emg_temporal_touch_confirm.log
```

## Outputs

The main outputs are:

- `evaluation/emg_temporal_touch/temporal_window_results.csv`: validation and
  held-out test results for every configuration/window.
- `evaluation/emg_temporal_touch/fold_window_selection.csv`: window selected from
  validation data inside each fold.
- `evaluation/emg_temporal_touch/validation_selected_test_results.csv`: unbiased
  pooled test performance after fold-specific validation selection.
- `runs/emg_temporal_touch/study_status.json`: progress and failures.

The earlier `runs/emg_temporal_study` directory is retained only as an alignment
diagnostic. It cropped at cue-to-click duration on a recording-start time axis and
must not be used for scientific conclusions.

Rebuild the analysis without retraining:

```bash
python scripts/analyze_emg_temporal_study.py \
  --run-root runs/emg_temporal_touch \
  --output-dir evaluation/emg_temporal_touch
```

Use the sign and participant-block confidence interval of `paired_mean_gain_px` as
the primary incremental-value result. Attention weights alone are not treated as
evidence that a time window is useful.
