# ReactEMG-inspired current and future intent

This experiment adapts ideas from **ReactEMG: Zero-Shot, Low-Latency Intent
Detection via sEMG**, arXiv:2506.19815v1, to our four-sensor reach/grasp/release
data. It retains current position/orientation, holding, grasp/release and
100/250/500/750/1000 ms future position/orientation and interaction predictions.

Sources inspected: [paper](https://arxiv.org/abs/2506.19815),
[project](https://roamlab.github.io/reactemg/), and
[official implementation](https://github.com/roamlab/reactemg/blob/main/reactemg/nn_models.py)
(`Any2Any_Model`, inspected 2026-09-09). This is an independent implementation
inspired by the method, not a reproduction, copied source, or pretrained ReactEMG.

## What transfers and what changes

| Paper idea | This implementation |
| --- | --- |
| EMG and intent as synchronized tokens | EMG feature tokens and not-holding/holding/MASK tokens, with shared time encoding and separate modality embeddings |
| Contiguous masking and reconstruction | Random 100 ms spans; reconstruct masked EMG features and classify hidden holding labels |
| Fully masked intent as main task | Always compute supervised deployment outputs with every label hidden |
| Mixed auxiliary objectives | All-label-hidden EMG reconstruction, aligned masking, and independently masked labels/signals |
| Stable maintenance after transitions | Small consistency penalty only inside certain, valid, unchanged holding states; report maintenance flips/minute |
| Limited lookahead | Zero future-sample access; causal attention at every layer. No lookahead smoothing or backdated predictions |

The paper uses eight Myo channels at 200 Hz. Our existing pipeline produces
eight causal RMS features (two scales for each of FOUR physical sensors) at
100 Hz, alongside 24 IMU axes and masks. These are not eight independent EMG
electrodes, and the paper's weights are not directly compatible.

The new EMG branch contributes to interaction logits and the context used for
future pose and events. The existing pose/IMU encoder remains part of the model.
An added future velocity head predicts interval-average velocity in standardized
position units/second. Integrating with each horizon's actual duration gives
displacement relative to the **model's current predicted position**. A learned
blend combines this with direct future positions. Displacement supervision and
direct/integrated agreement provide a motion constraint. This is our extension,
not a ReactEMG result or physical integration of measured accelerometer signals.

VIVE current/future position, orientation, and manual labels are supervision only.
No initial VIVE pose or action label is required by inference. Absolute pose
still depends on the learned recording coordinate system; this change does not
solve arbitrary new world frames or unknown sensor placement.

## Train on the existing data

From the repository root on Linux:

```bash
git pull origin main
bash scripts/train_reach_grasp_react_intent.sh
```

Default data: `/home/nahar3/shubham/emg_shubhu/data/184a6ef69b83`.
For an explicit matched split (recommended when comparing to the old run):

```bash
bash scripts/train_reach_grasp_react_intent.sh \
  --split-file runs/reach_grasp_future_intent_seed42/splits.json
```

Overrides are accepted at the end of the launcher:

```bash
bash scripts/train_reach_grasp_react_intent.sh \
  --root /path/to/previous/data --batch-size 2 \
  --output-dir runs/react_intent_comparison
```

Train from random weights; previous checkpoints and output directories are
preserved. Normalization is fit on training trials only. Whole trials are split
before any auxiliary masking; no windows from one trial enter multiple splits.
The split-file option requires exactly the same accepted absolute trial paths.
All new checkpoint options are recorded in `training_options`.

For several recording folders, `--split-by recording --root /path/to/recordings`
holds out entire CSV parent directories (at least three required). A recording
split is not necessarily a participant split: recordings from the same participant
must be grouped separately when making new-user claims.

## Visualize the trained model with Franka

The existing visualizer now recognizes both checkpoint formats:

```bash
python scripts/visualize_future_intent_franka.py \
  --checkpoint runs/reach_grasp_react_intent_seed42/best.pt \
  --trial-csv /path/to/unseen/trial_006.csv \
  --device cuda --speed 1.0 --control-horizon-ms 250
```

The visualizer uses model outputs for pose and interaction. Any VIVE stream is
withheld comparison data. This is streaming inference on a recorded trial;
future CSV rows are not exposed to the model before their timestamps.

## How to decide whether this helped

`results.json` reports the same current/future pose and event metrics as the old
model, including EMG/IMU removal tests. `stability.json` adds holding accuracy
outside uncertain boundaries and maintenance flips/minute at a fixed 0.5 holding
threshold. This is a descriptive metric, **not** the paper's exact Transition
Accuracy. A constant holding output can have zero flips, so always read accuracy,
event recall and latency alongside it.

Use the same split and at least three seeds. Compare with the original trainer,
and with this model using `--masked-weight 0` to isolate the masking objective.
Use `--motion-weight 0` and `--stability-weight 0` to isolate those losses.
These flags disable objectives, not the associated network components.
Select settings on validation. After repeated test-driven development, reserve
new recordings for a fresh final test. Removal ablations are not independently
trained IMU-only baselines.

The method aims for robustness, not elimination of all noise. Reconstruction
targets are recorded RMS features, not noise-free EMG. Missing frames stay
masked; uncertain labels stay hidden and are excluded from dense holding loss.
No automatic relabeling, test-time fitting, PCA/ICA, or future-data interpolation
is added. Small feature noise is used in auxiliary training only. The paper's
cross-user results also relied on diverse training participants; borrowing its
architecture does not give our dataset the same diversity.
