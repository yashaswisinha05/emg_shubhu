# IMU teacher → EMG-only student

This is a new isolated experiment. Existing trainers and checkpoints are not
modified. It is **not Chronos, not a VAE, and not a VIVE-input teacher**. The
first experiment isolates cross-modal distillation using a small two-layer GRU.
Chronos is a future encoder ablation, not required to run this script.

## Train

Run from the repository directory inside the existing `emg_env` environment:

```bash
python scripts/train_imu_to_emg.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
  --cache-dir artifacts/imu_to_emg_cache \
  --device cuda --seed 42 \
  --teacher-epochs 30 --epochs 50 \
  --latent-weight 0.1 --output-weight 0.25 \
  --output-dir runs/imu_to_emg_seed42
```

For the new candidate, replace the root and prefixes with:

```bash
--root "/home/nahar3/shubham/emg_shubhu/shubham_3930d150/32e00ff16111" \
--session-prefixes 32e00ff16111
```

Prefixes match actual ancestor directory names. For multiple recordings, use a
common root and list their actual folder prefixes. An existing nonempty output
directory is rejected so no previous results are overwritten. For an initial
smoke run use `--teacher-epochs 1 --epochs 1` and a separate output directory.

The default config supplies only the established dataset preprocessing/splits
and loader settings; old VAE, routing, teacher and loss sections are NOT used.
This script defines its own architecture/loss settings and saves them in each
checkpoint. Existing dependencies (PyTorch, NumPy, pandas, SciPy, PyYAML) suffice;
no pretrained weights or Chronos installation is needed.

## Three stages, one command

1. Independently train an **IMU-only teacher** on screen and full-path labels.
2. Train a directly supervised **EMG-only baseline**.
3. Train an **EMG-only student** with the same initial EMG weights, supervised
   losses, plus the frozen teacher's motion latent and output predictions.

Baseline/student use the same RNG seeds and window sampling schedule; validation
early stopping can select different epochs. The teacher receives the same
observation cutoff as the student, not the future IMU recording. Neither forward
interface accepts VIVE, screen labels, session identity or true time-to-touch.

The motion latent drives the 3D head. The screen head sees that latent and a
separate modality-specific feature vector. Only the motion latent is aligned.
The loss is screen Huber (100 px scale) + path Huber (10 cm scale) + 0.5 endpoint
Huber, plus 0.1 normalized-motion-latent MSE and 0.25 teacher screen/path Huber.
These weights are starting hypotheses, not validated optima.

The network predicts normalized screen xy and 16 **onset-relative 3D positions
in metres**, from movement onset to touch. The first position is fixed to zero;
the last path position is the predicted endpoint. This is not absolute tracker
position or a current-position tracking head. Inputs are up to two seconds of
history, trained at 0–400 ms before the recording's touch/end sample. Earlier
streaming phases are not validated by this experiment.

## Outputs and interpretation

- `imu_teacher.pt`: selected teacher, IMU input only.
- `emg_baseline.pt`: selected directly supervised EMG baseline.
- `emg_student.pt`: selected distilled student, **EMG input only**.
- `results.json`: test screen px, path cm, endpoint cm, overall and per lead.
- `*_history.json`: training loss and validation metrics for each epoch.
- `splits.json`: exact train/validation/test trial paths.
- `live_calibration*.npz`: per-selected-candidate training normalization.

Checkpoint selection minimizes validation `screen_px + 5 * path_cm`; test is
evaluated after all selections. This is a within-candidate trial split, **not**
evidence of unseen-person generalization. Compare the student with both baseline
and teacher across seeds (e.g. 1, 2, 3), using new output directories. Do not
choose hyperparameters based on test results. Matching teacher latents alone is
not success: screen/path errors must improve over the supervised EMG baseline.

VIVE supplies supervision and the existing loader's validity/onset definitions,
not network inputs. Offline preprocessing retains the existing estimated sample
rate and tracker-valid sample selection, so this is not a full raw-stream latency
benchmark. Train-only candidate normalization must be reused at inference. Canvas
dimensions must come from recordings; missing dimensions raise an error.

## Loading the EMG-only network

```python
import torch
from emg_touch.models.imu_to_emg import WearableTaskNetwork

# Only load checkpoints you trust.
ckpt = torch.load("runs/imu_to_emg_seed42/emg_student.pt",
                  map_location="cuda", weights_only=False)
model = WearableTaskNetwork(**ckpt["model_args"]).cuda().eval()
model.load_state_dict(ckpt["state_dict"])
with torch.no_grad():
    out = model(emg_history, valid_mask)
# emg_history: [batch, time, preprocessed_EMG_features], using saved calibration
# valid_mask: [batch, time] bool; no IMU tensor is passed
# out['screen']: normalized xy; multiply by the current matching canvas width/height
# out['path']: [batch, 16, 3] onset-relative metres
```

The old live/manipulator UI does not yet recognize this checkpoint format. Do
not pass it to the previous UI loader expecting compatibility. Real deployment
also needs a known coordinate anchor to turn relative positions into robot
coordinates; it must not secretly use the test VIVE pose.
