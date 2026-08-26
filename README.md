# EMG/IMU Touch-Location Prediction

This project implements a causal, participant-safe research pipeline for predicting normalized screen-touch coordinates from EMG and IMU recordings in `../MERGED DATA`.

For the current **complete-trajectory, configuration-level** experiment, start with
[`FULL_TRAJECTORY.md`](FULL_TRAJECTORY.md). It uses every valid trajectory at its natural
duration, five-fold within-participant trial cross-validation, and pooled configuration
accuracy. The remainder of this README describes the stricter early-prediction and
participant-held-out protocol.

To determine which pre-touch EMG segment adds value beyond IMU, use the causal,
touch-aligned protocol in [`EMG_TEMPORAL_STUDY.md`](EMG_TEMPORAL_STUDY.md).

For the calibrated CenterNet-style 8×5 heatmap plus local-offset architecture, use
[`GRID_POINT.md`](GRID_POINT.md).

For the centre-safe direct-coordinate model with auxiliary grid supervision and
raw-plus-calibrated IMU, use [`HYBRID_POINT.md`](HYBRID_POINT.md).

For repeated movement-onset/200-ms/touch predictions with trajectory-dependent
EMG-channel and hierarchical IMU sensor/feature attention, use
[`CONTINUAL_ATTENTION.md`](CONTINUAL_ATTENTION.md).

For the EMG-dominant variant, which gives the EMG encoder the full causal prefix
and admits IMU only through a priced, zero-initialized residual gate, see
[`EMG_FIRST.md`](EMG_FIRST.md).

For multi-scale patching plus patch-level cross-variate attention over the four
EMG electrodes and four IMU sensors, adapted from MCV-PatchTST, see
[`CROSS_VARIATE.md`](CROSS_VARIATE.md).

The default final model is:

- a multi-scale causal EMG encoder;
- a sensor-grouped IMU encoder;
- PatchTST-style temporal patching;
- EMG-conditioned IMU lookback selection;
- cross-modal teacher fusion;
- future-IMU auxiliary prediction;
- a probabilistic MDN coordinate head;
- EMG-only student distillation.

The project also contains TCN and Hugging Face PatchTST baselines and an optional CVAE coordinate head.

## Important dataset assumptions

- The available EMG channels are RMS envelopes, not raw high-frequency EMG.
- The effective rate is approximately `148.148 Hz`.
- Rows are sorted by `time_perf_counter`; `sample_rate_hz_declared` is not used.
- Only explicitly whitelisted EMG/ACC/GYRO columns enter the cache.
- Click, target, reaction-time, and button-position columns cannot enter model inputs.
- Missing sensors are zero-filled and accompanied by observation masks.
- Evaluation cutoffs are relative to the recorded cue (`time_s = 0`), not an inferred movement onset.
- Splits are participant-disjoint.

The manifest derives the configuration from `participant_id`, not the current directory nesting. Therefore, it indexes the misplaced Akash sessions correctly even before physical layout repair.

The enclosing participant folder is treated as the canonical participant ID. This safely corrects known summary-metadata typos such as `priyan_4` versus folder `priyan_a4` and `dev_mix77` versus folder `dev_mix7`, without modifying the original JSON files. The audit reports every such mismatch.

## Repository layout and the data path

Every config uses `data_root: "../MERGED DATA"`, a path relative to the
repository root. Clone the repository so that its root sits **beside** the
`MERGED DATA` directory:

```text
<parent>/
├── <repo root>/        # this repository
└── MERGED DATA/        # a1 a2 a3 a4 b1 b2 b3 mix1 mix2 mix3 mix5 mix6 mix7
```

For example, on a machine where the data lives at
`/home/<user>/shubham/MERGED DATA`, clone into `/home/<user>/shubham/emg_shubhu`
and the relative path resolves with no config edits.

The recorded data is not part of this repository, and neither are
`artifacts/`, `runs/`, or `evaluation/`. On a new machine, rebuild the manifest
and cache (steps 3 onward) before training; `artifacts/manifest.csv` stores
absolute paths and is therefore machine-specific.

## Installation

Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 1. Audit the dataset

This operation is read-only.

```bash
python scripts/audit_data.py --output artifacts/audit.json
```

Eight Akash sessions are expected to be reported as logically misplaced beneath the `b3/gazania...` tree.

## 2. Optionally repair the physical layout

The default command is a dry run:

```bash
python scripts/repair_layout.py
```

Review every proposed move. Apply only when satisfied:

```bash
python scripts/repair_layout.py --apply
```

The manifest does not require this repair, so this step is optional. If applied after creating artifacts, rebuild the manifest and cache because paths change.

## 3. Build the manifest and signal cache

```bash
python scripts/build_manifest.py
python scripts/prepare_cache.py --workers 4
```

The cache contains only:

- monotonic cue-relative timestamps;
- four canonical EMG RMS channels and masks;
- 24 canonical IMU channels and masks.

## 4. Create one participant-held-out fold

Example for configuration `a1`, testing `dev` and validating on `nihil`:

```bash
python scripts/make_split.py \
  --configuration a1 \
  --test-subject dev \
  --val-subject nihil \
  --output artifacts/a1_test-dev/split.json
```

If `yash1` and `yash2` identify the same physical person, enable the alias mapping in `configs/default.yaml` before building the manifest.

## 5. Fit normalization on training participants only

```bash
python scripts/fit_scaler.py \
  --split artifacts/a1_test-dev/split.json \
  --output artifacts/a1_test-dev/scaler.npz
```

The default EMG pipeline is:

```text
timestamp sorting
→ strictly causal resampling
→ 3-sample trailing median filter
→ log1p
→ training-only robust scaling
```

Set `median_kernel: 1`, `3`, or `5` in separate configs for the filtering ablation.

To generate every rotating LOSO split for a configuration:

```bash
python scripts/make_loso_splits.py --configuration a1
```

Configurations with at least three participants use a separate validation participant. A configuration with exactly two participants uses one completely held-out test participant and takes validation trials only from the remaining training participant. Test trials never enter training or validation.

## 6. Train baselines

Mean-coordinate and handcrafted EMG ridge baselines:

```bash
python scripts/train_ridge.py \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev/ridge
```

TCN:

```bash
python scripts/train_baseline.py \
  --kind tcn \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev
```

PatchTST regression:

```bash
python scripts/train_baseline.py \
  --kind patchtst \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev
```

PatchTST is a baseline, not the final multimodal model.

## Running on Linux with CUDA

`choose_device` prefers CUDA when it is available, so **omit `--device`** and the
right accelerator is selected automatically. Passing `--device cuda` is
equivalent; `--device mps` is Apple-only.

Two settings are tuned for macOS and are worth raising on a Linux GPU host:

- `training.num_workers: 0` avoids a macOS `torch_shm_manager` restriction. On
  Linux, 4-8 workers is usually faster.
- `training.amp: true` only takes effect on CUDA (`train_grid_model.py` gates it
  on `device.type == "cuda"`), so mixed precision activates automatically there
  and is inert on MPS.

`caffeinate -i` in the documented commands is a macOS sleep-inhibitor. Drop it on
Linux, or use `nohup ... &` / `tmux` instead.

If `build_manifest.py` or `prepare_cache.py` fails with
`ModuleNotFoundError: No module named 'numpy._core...'`, the environment has
NumPy <2.0. The recorded `.pkl` trial files were written under NumPy 2.x, whose
private module layout (`numpy._core`) does not exist before 2.0, so an older
NumPy cannot unpickle them. `pip install -e .` pulls in a compatible NumPy from
`pyproject.toml`; if a pre-existing environment (e.g. a `base` conda env with an
older NumPy already installed) is being reused instead of a fresh one, upgrade
it directly:

```bash
pip install -U "numpy>=2,<3"
```

## 7. Masked multimodal pretraining

Pretraining uses only the training partition of the fold, preventing test-participant leakage.

```bash
python scripts/pretrain.py \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev
```

The loss combines masked EMG reconstruction, masked IMU reconstruction, and aligned EMG/IMU contrastive learning.

## 8. Train the EMG+IMU teacher

Without pretraining:

```bash
python scripts/train_teacher.py \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev
```

With pretraining:

```bash
python scripts/train_teacher.py \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --pretrained runs/a1_test-dev/pretraining/best.pt \
  --output-dir runs/a1_test-dev
```

The teacher uses only past and current IMU at the selected cutoff. Future IMU is a training target, never an input.

## 9. Distill the EMG-only student

```bash
python scripts/distill_student.py \
  --teacher-checkpoint runs/a1_test-dev/teacher/best.pt \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --output-dir runs/a1_test-dev
```

The student is initialized from the teacher's EMG encoder and learns from ground truth, teacher predictions, teacher representations, and future-IMU auxiliary targets. Its inference path uses EMG only.

## 10. Evaluate early prediction

```bash
python scripts/evaluate.py \
  --kind student \
  --checkpoint runs/a1_test-dev/student/best.pt \
  --split artifacts/a1_test-dev/split.json \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --cutoffs 0.0 0.1 0.2 0.3 0.5 -1 \
  --output-dir evaluation/a1_test-dev/student
```

`-1` means the full trial. Output includes trial-level predictions and aggregate metrics.

## 11. Compare configurations fairly

Repeat the participant-held-out folds for every configuration, then pass every `predictions.csv` file to:

```bash
python scripts/compare_configs.py \
  evaluation/*/student/predictions.csv \
  --output evaluation/configuration_comparison.csv
```

Every configuration uses all of its available held-out predictions; participant identity is not a comparison axis. Participants are used only when constructing leakage-safe train/validation/test folds. Pool the test predictions from all completed folds so that every trial is evaluated exactly once when it is held out.

The configuration report contains:

- target-box accuracy;
- 8×5 screen-region accuracy;
- accuracy within 50 and 100 pixels;
- median, mean, and 90th-percentile pixel error;
- mean normalized coordinate error;
- trial-bootstrap confidence intervals;
- the number of held-out trials contributing to each result.

Unequal participant and trial counts across configurations are allowed and are reported through `held_out_trials`. The script rejects duplicate predictions so a trial cannot accidentally be counted twice.

## 12. Predict one new trial

For the trained continual EMG-only and EMG+IMU grid models, including live
newline-delimited JSON input and recorded-CSV replay, see
[`docs/realtime_deployment.md`](docs/realtime_deployment.md).

EMG-only deployment:

```bash
python scripts/predict_csv.py path/to/trial.csv \
  --kind student \
  --checkpoint runs/a1_test-dev/student/best.pt \
  --scaler artifacts/a1_test-dev/scaler.npz \
  --cutoff 0.3
```

The output contains normalized coordinates, pixel coordinates, and samples from the predictive distribution.

## Switching from MDN to CVAE

The recommended default is:

```yaml
model:
  head: mdn
  mdn_components: 3
```

For the generative ablation, copy the config and change:

```yaml
model:
  head: cvae
  cvae_latent_dim: 8
```

Train a separate teacher and student. Do not compare an MDN checkpoint with a CVAE configuration; their parameter layouts differ.

## Experimental matrix

At minimum, run these ablations with identical participant splits and seeds:

1. TCN, EMG only.
2. PatchTST, EMG only.
3. Student without distillation.
4. EMG+IMU teacher.
5. Distilled EMG-only student.
6. MDN versus CVAE head.
7. Median kernels 1, 3, and 5.
8. Teacher without future-IMU loss.
9. Teacher without EMG-conditioned lookback gating.

Primary selection metrics should be subject-aggregated median pixel error, 90th-percentile pixel error, target-box hit rate, and performance versus cue-relative cutoff.
