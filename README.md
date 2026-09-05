# EMG/IMU Touch-Location Prediction

This project contains two causal EMG/IMU research pipelines: the current
tracked EMG+IMU+VIVE experiments documented below, and the earlier
participant-safe screen-touch pipeline for recordings in `../MERGED DATA`.

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

## Current tracked-dataset model: soft-routed complete reach

The current tracked-data experiment is implemented by
[`scripts/train_soft_routed_complete_reach.py`](scripts/train_soft_routed_complete_reach.py),
[`src/emg_touch/models/soft_routed_complete_reach.py`](src/emg_touch/models/soft_routed_complete_reach.py),
and
[`configs/tracked_soft_routed_complete_reach.yaml`](configs/tracked_soft_routed_complete_reach.yaml).
It is intentionally separate from the older models so every earlier result
remains reproducible.

### Question and inference boundary

The model asks whether a causal wearable history can predict both:

1. the final touchscreen coordinate; and
2. the complete movement-onset-to-touch 3D hand path.

The deployable student accepts only:

```text
incoming EMG + incoming IMU + causal padding mask
```

VIVE position, velocity, screen target, future trajectory, and true time to
touch are never student inputs. VIVE is used in two training-only roles:

- as privileged input to the teacher; and
- as the label for screen and 3D losses.

At test time the teacher is not called. The red screen point and black 3D path
in the visualizers are display-only ground truth.

### Why this model was introduced

The architecture followed directly from two conflicting experiments:

| Model | Screen error | Complete 3D path error | What it showed |
| --- | ---: | ---: | --- |
| Task-separated | 207.5 px | 7.67 cm | Shared learning retained screen accuracy, but 3D tracking remained weak. |
| Hard asymmetric | 251.0 px | **5.55 cm** | Hard stop-gradients protected motion, but removing cross-task adaptation damaged the screen head. |
| **Soft-routed** | **203.4 px** | 5.89 cm | Small cross-task gradients retained screen accuracy while preserving most of the 3D improvement. |

These are exploratory one-seed results from the same development sequence, not
the final multi-seed paper estimate. They motivated replacing hard
`detach()` boundaries with soft gradient routing rather than adding another
independent head or restoring a stochastic student VAE.

### Architecture and logic

```text
EMG history -> channel/time attention -> EMG intent context -----+
                                                               |
IMU history -> motion encoder ----------> motion context -------+-> fused factors
                                                               |
                              privileged VIVE teacher ----------+  training only

fused factors -> teacher-coordinate bridge -> screen head -> pixel x,y

IMU motion base + bounded EMG-intent correction
    -> complete 3D path head
    -> explicit 3D endpoint head
    -> signed x/y/z direction head
```

The screen head receives the EMG-owned intent factors with their full gradient.
It can also read motion and residual factors, but their screen gradients are
scaled to `0.10`. The 3D head receives the IMU motion route at full strength
and the EMG intent route with a `0.10` gradient scale. The operation is:

```python
def scale_gradient(x, scale):
    return x.detach() + scale * (x - x.detach())
```

The forward value is exactly `x`; only its backward gradient changes. Therefore
`scale=0` reproduces hard task isolation and `scale=1` reproduces unrestricted
joint training. The configured `0.10` is the compromise tested here.

The 3D prediction starts from the IMU-derived relative path and learns a
bounded EMG-conditioned correction. A smooth progress constraint makes the
predicted path start at movement onset, and the final path sample is forced to
agree with the explicit 3D endpoint head. Auxiliary direction losses penalize
mirrored up/down or left/right reaches.

### What remains from the VAE

The privileged teacher retains its VAE representation and is trained with VIVE
available. The wearable student is deterministic:

- student sampling noise is `0.0`;
- student latent-distillation weight is `0.0`;
- the student reports zero log-variance; and
- deployment uses a single deterministic forward pass.

The useful retained component is the teacher-aligned coordinate bridge and
hierarchical output guidance, not stochastic sampling in the deployed model.
Consequently, this experiment supports a claim about privileged teacher
guidance, but does not by itself prove that a student VAE is better than a
deterministic student.

### Objective

The student objective combines:

```text
screen coordinate and hierarchical screen losses
+ selective privileged-teacher output guidance
+ complete 3D Cartesian path loss
+ explicit 3D endpoint loss
+ path/endpoint consistency
+ endpoint, path, and velocity direction losses
+ signed x/y/z direction classification
+ EMG-only screen and direction auxiliaries
+ IMU-base preservation and correction regularization
```

IMU modality dropout and physical-sensor dropout remain active during training.
The EMG-only auxiliary prevents the fused model from discarding its EMG route.

### Current one-seed result

The run in `runs/soft_routed_complete_reach` produced:

| Measurement | Result |
| --- | ---: |
| Wearable student screen error, all evaluated leads | **203.4 px** |
| Late-window screen score, 0/48/103 ms to touch | **176.9 px** |
| Complete 3D path error | **5.89 cm** |
| Mean-target screen baseline | 440.8 px |
| Training-only privileged teacher | 61.4 px |
| Remove EMG | +51.5 px |
| Shuffle EMG across trials | +120.7 px |
| Remove IMU | +138.1 px |
| Wrong-way 3D reach fraction | 5.9% |
| Endpoint direction angle | 28.2 degrees |

Per-axis endpoint results were:

| Axis | Sign accuracy | MAE | Correlation |
| --- | ---: | ---: | ---: |
| x | 83.9% | 5.00 cm | +0.923 |
| y | 67.8% | 3.62 cm | +0.806 |
| z | 69.0% | 2.57 cm | +0.939 |

The modality interventions are the strongest evidence that the wearable model
uses both inputs. Shuffling EMG is substantially worse than removing it,
consistent with the network using trial-specific EMG timing rather than only a
global amplitude prior.

Two limitations must remain explicit:

- the learned 3D correction improves the IMU path by only `0.09 cm`, so IMU is
  still responsible for most continuous 3D tracking while EMG contributes
  more strongly to destination inference; and
- time-to-touch prediction remains weak (`116.6 ms` MAE and `20.6%` bin
  accuracy) and should not be used as a primary claim.

The `61.4 px` teacher is not a deployable baseline because it receives
privileged VIVE information. The honest result is the wearable-only student
versus wearable ablations and the mean-target baseline.

### Train the model

```bash
python scripts/train_soft_routed_complete_reach.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_soft_routed_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_posture \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
  --device cuda \
  --teacher-epochs 25 \
  --epochs 50 \
  --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/soft_routed_complete_reach
```

The run writes:

- `best_screen.pt`: best validation screen checkpoint;
- `best_3d.pt`: best validation 3D checkpoint;
- `best.pt`: best joint screen-plus-3D checkpoint; and
- `final.pt`: joint-selected model plus final test metrics.

Use `final.pt` for reporting both tasks from one unified model. Using
`best_screen.pt` for pixels and `best_3d.pt` for motion produces two separately
selected systems and must be reported as such.

### Random held-out-test visualizations

Pixel prediction from a growing EMG+IMU history, automatically advancing
through shuffled test trials without replacement:

```bash
python scripts/live_best_model_ui.py \
  --trial-root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --checkpoint runs/soft_routed_complete_reach/final.pt \
  --device cuda --split test --random-trials --random-seed 7 \
  --auto-next-ms 1200 --speed 1.0 --prediction-delay-ms 600 \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4
```

Complete predicted 3D path and model-driven inverse kinematics on randomly
sampled test trials:

```bash
python scripts/visualize_complete_reach_manipulator.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --checkpoint runs/soft_routed_complete_reach/final.pt \
  --cache-dir artifacts/tracked_cache_posture \
  --device cuda --split test --observation-lead-ms 0 \
  --num-trials 10 --random-trials --random-seed 7 --auto-next \
  --fps 20 --output-dir runs/soft_routed_manipulator_random
```

### Experimental successor: temporal EMG residual for 3D

[`scripts/train_emg_residual_complete_reach.py`](scripts/train_emg_residual_complete_reach.py)
tests whether temporally resolved EMG can predict the part of the 3D path left
unexplained by the soft-routed model. It does not replace or overwrite the
current result.

The new branch uses 16 learned path queries to attend to causal EMG tokens and
predicts a bounded residual:

```text
soft-routed base path B(t) + gated temporal-EMG residual R(t) -> final path
```

Its path and endpoint output layers are zero-initialized. Before training, the
new model therefore exactly reproduces the supplied soft-routed checkpoint.
Training first freezes that checkpoint for a 10-epoch residual warm-up, then
jointly fine-tunes for `--epochs` epochs at one-tenth of the original learning
rate.

```bash
python scripts/train_emg_residual_complete_reach.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/soft_routed_complete_reach/final.pt \
  --config configs/tracked_emg_residual_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_posture \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
  --device cuda --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/emg_residual_complete_reach
```

The final report adds paired 3D interventions for full EMG+IMU, zeroed EMG,
trial-shuffled EMG, and zeroed IMU. For each condition it prints complete-path
error, endpoint error, direction angle, and wrong-way fraction. A positive
`remove EMG path cost` or `shuffle EMG path cost` is the direct measurement of
EMG helping 3D motion. This successor should be retained only if it improves
those paired metrics without materially degrading the unified screen result.

### Experimental successor: one shared screen/3D goal

[`scripts/train_goal_prototype_complete_reach.py`](scripts/train_goal_prototype_complete_reach.py)
loads the temporal-EMG-residual checkpoint and tests whether the two output
tasks improve when they share one wearable-predicted 8x5 goal distribution.
The implementation is in
[`src/emg_touch/models/goal_prototype_complete_reach.py`](src/emg_touch/models/goal_prototype_complete_reach.py)
and its isolated configuration is
[`configs/tracked_goal_prototype_complete_reach.yaml`](configs/tracked_goal_prototype_complete_reach.yaml).

For each screen cell, the model learns a complete-path and endpoint residual
prototype from training labels. At inference its EMG+IMU screen heatmap softly
selects and mixes those prototypes; no true target id is supplied. In the
reverse direction, the wearable-predicted 3D endpoint produces a bounded
screen correction. Thus screen and 3D supervise a shared goal without making
the two decoders identical:

```text
wearable goal probabilities -> screen coordinate
                            -> target-conditioned 3D residual prototype

IMU motion + temporal EMG correction + goal prototype -> complete 3D path
predicted 3D endpoint -> bounded screen correction
```

All prototype tensors and the final geometry-correction layer start at zero.
The supplied checkpoint is therefore reproduced exactly before training. The
first 8 epochs train only the new bridge; `--epochs` then controls low-rate
joint fine-tuning.

```bash
python scripts/train_goal_prototype_complete_reach.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/emg_residual_complete_reach/final.pt \
  --config configs/tracked_goal_prototype_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_posture \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
  --device cuda --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/goal_prototype_complete_reach
```

The final diagnostics report screen, complete-path, and endpoint results both
before and after the goal bridge, goal-cell accuracy, bridge gates, and the
same paired EMG 3D interventions. Retain this branch only when screen error is
below `201.0 px`, path error is below `5.81 cm`, endpoint error is below
`7.42 cm`, and EMG removal/shuffling costs remain positive.

Before treating the one-seed values as paper results, repeat at minimum seeds
`1`, `2`, and `3` and report the mean, spread, and paired trial-level
confidence intervals. A gradient-routing ablation should compare scales
`0`, `0.05`, `0.10`, `0.25`, and `1.0` while holding the split, losses, and
checkpoint criterion fixed.

### Experimental successor: EMG acceleration dynamics

The shared-goal experiment did not earn its added complexity on the first
run: its bridge changed screen error by only `+0.2 px` and complete-path error
by `+0.00 cm`; the final `5.95 cm` path and `7.65 cm` endpoint were also worse
than the earlier temporal-EMG-residual checkpoint (`5.81 cm`, `7.42 cm`). Keep
that run as a negative ablation, not as the base for the next model.

[`scripts/train_emg_acceleration_complete_reach.py`](scripts/train_emg_acceleration_complete_reach.py)
therefore starts from `runs/emg_residual_complete_reach/final.pt`. Its new
branch predicts a temporally resolved acceleration residual from causal EMG,
then uses differentiable trapezoidal integration to turn acceleration into a
velocity residual and a complete-path correction:

```text
causal EMG tokens -> acceleration residual a_EMG(t)
                  -> integrate -> velocity residual
                  -> integrate -> 3D path correction

previous EMG+IMU path + integrated correction -> final complete reach
```

VIVE position and velocity construct supervision labels only. Neither VIVE
velocity, acceleration, position, nor target enters `student_forward`. The
true acceleration label is computed by smoothing the resampled VIVE velocity
and differentiating it; the model must infer that acceleration from EMG+IMU.
The new acceleration output is zero-initialized, so the supplied checkpoint is
reproduced exactly before training. A 10-epoch dynamics-only warm-up is
followed by low-rate joint fine-tuning, and each phase retains its incoming
checkpoint unless validation actually improves.

```bash
python scripts/train_emg_acceleration_complete_reach.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/emg_residual_complete_reach/final.pt \
  --config configs/tracked_emg_acceleration_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_posture \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
  --device cuda --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/emg_acceleration_complete_reach
```

The final diagnostics print before/after dynamics path and endpoint errors,
acceleration RMSE, duration MAE, integrated correction magnitude, gate value,
and paired zeroed/shuffled-EMG costs. Retain this model only if it beats both
`5.81 cm` path and `7.42 cm` endpoint while keeping screen error at or below
about `201 px` and preserving positive EMG intervention costs. A smaller
training loss alone is not evidence that acceleration helped.

### Personalizing to one new candidate with 200 trials

Do not train the complete network from scratch on 200 trials. The isolated
[`scripts/train_personalized_complete_reach.py`](scripts/train_personalized_complete_reach.py)
trainer retains the 800-trial acceleration model and adds a zero-initialized,
rank-12 candidate adapter. It randomly reserves 20% validation and 20% test,
fits normalization and PCA on the remaining 60% only, freezes the wearable
encoders, trains the adapter, and finally tunes only the established output
heads at one-tenth the learning rate. The untouched test split is evaluated
once after validation selection.

Replace `newperson_a1` with the unique prefix of the new candidate's folder:

```bash
python scripts/train_personalized_complete_reach.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/emg_acceleration_complete_reach/final.pt \
  --config configs/tracked_personalized_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_personalized \
  --session-prefixes newperson_a1 \
  --device cuda --epochs 20 --finetune-epochs 10 \
  --lead-window-ms 0 400 \
  --output-dir runs/personalized_newperson_a1
```

The run also writes `live_calibration.npz` from the candidate training split.
Use that exact file with the personalized checkpoint in true live inference:

```bash
python scripts/run_live_distillation_ui.py \
  --checkpoint "Personalized=runs/personalized_newperson_a1/final.pt" \
  --calibration runs/personalized_newperson_a1/live_calibration.npz \
  --device cuda --interval-ms 40
```

For the paper, compare this against (1) the unadapted population checkpoint
and (2) the same architecture trained from scratch using the identical
train/validation/test trial indices. Calibration and model selection must not
use the reserved test trials.

### Candidate-only training from scratch

The scratch control uses the same candidate selection, seed, 60/20/20 split,
training-only normalization, and causal labels as personalization. It is a
three-stage architectural build, but it never loads the 800-trial population
model. Run these commands in order, using the same unique candidate prefix in
all three commands.

Stage 1 trains the teacher and soft-routed EMG+IMU student from random weights:

```bash
python scripts/train_candidate_scratch_01_soft_routed.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_soft_routed_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_candidate_scratch \
  --session-prefixes newperson_a1 \
  --device cuda --teacher-epochs 25 --epochs 50 \
  --finetune-epochs 0 --lead-window-ms 0 400 \
  --output-dir runs/candidate_scratch_01_soft_routed
```

Stage 2 adds and trains the zero-initialized temporal EMG residual:

```bash
python scripts/train_candidate_scratch_02_emg_residual.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/candidate_scratch_01_soft_routed/final.pt \
  --config configs/tracked_emg_residual_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_candidate_scratch \
  --session-prefixes newperson_a1 \
  --device cuda --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/candidate_scratch_02_emg_residual
```

Stage 3 adds and trains the zero-initialized acceleration dynamics:

```bash
python scripts/train_candidate_scratch_03_acceleration.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --initial-checkpoint runs/candidate_scratch_02_emg_residual/final.pt \
  --config configs/tracked_emg_acceleration_complete_reach.yaml \
  --cache-dir artifacts/tracked_cache_candidate_scratch \
  --session-prefixes newperson_a1 \
  --device cuda --epochs 30 --finetune-epochs 0 \
  --lead-window-ms 0 400 \
  --output-dir runs/candidate_scratch_03_acceleration
```

The final scratch checkpoint is
`runs/candidate_scratch_03_acceleration/final.pt`. Each stage also writes the
same training-only `live_calibration.npz`. Keep `--seed` identical across all
stages if overriding the default seed, otherwise the checkpoint chain would
not represent one fixed experimental split.

## Legacy `MERGED DATA` model

The older `MERGED DATA` final model is:

- a multi-scale causal EMG encoder;
- a sensor-grouped IMU encoder;
- PatchTST-style temporal patching;
- EMG-conditioned IMU lookback selection;
- cross-modal teacher fusion;
- future-IMU auxiliary prediction;
- a probabilistic MDN coordinate head;
- EMG-only student distillation.

The project also contains TCN and Hugging Face PatchTST baselines and an optional CVAE coordinate head.

## Legacy `MERGED DATA` dataset assumptions

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
