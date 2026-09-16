# Minimal EMG+IMU model

This branch isolates the selected final model. Both training and inference use
EMG+IMU only. VIVE position and screen coordinates are training/evaluation
targets and are never model inputs.

## Retained objectives

1. Class-balanced cross-entropy for open/close state.
2. Smooth-L1 for current Cartesian XYZ.
3. Late-weighted physical-pixel Smooth-L1 for normalized pixel XY, with width
   and height scaled independently.
4. Smooth-L1 for 20 future XYZ points from 10 to 200 ms.

There is no orientation, endpoint, grid/offset, state conditioning, masked
reconstruction, future-IMU reconstruction, one-second intent, consistency, or
correction-regularization loss.

## Train

```bash
python minimal_emg_imu/train.py \
  --root data/shubham_open data/shubham_close \
         data/mukund_open data/mukund_closed \
         data/gazania_open data/gazania_closed \
  --device cuda --epochs 60 \
  --output-dir runs/minimal_emg_imu
```

## Replay an unseen CSV

```bash
python minimal_emg_imu/infer.py \
  --checkpoint runs/minimal_emg_imu/emg_imu_best.pt \
  --trial-csv data/unseen/trial_001.csv \
  --canvas-px 1920 1080 --device cuda
```

For a live source, omit `--trial-csv` and send newline-delimited objects of the
form `{"time_s": ..., "emg": [4 values], "imu": [24 values]}` to stdin, or
import `MinimalEMGIMUStream` and call `update` directly.

## PatchTST encoder ablation

This controlled ablation replaces only the two primary causal encoders with
randomly initialized Hugging Face PatchTST encoders. Preprocessing, split,
heads, loss weights, optimizer, and evaluation remain unchanged. An explicit
causal attention mask prevents later frames from leaking into earlier outputs.

```bash
python minimal_emg_imu/train_patchtst.py \
  --root data/participant_01_open data/participant_01_closed \
         data/participant_02_open data/participant_02_closed \
         data/participant_03_open data/participant_03_closed \
  --device cuda --epochs 60 \
  --output-dir runs/patchtst_encoder_ablation
```

After training, compare it with the selected model:

```bash
python minimal_emg_imu/compare_patchtst.py \
  --baseline runs/minimal_emg_imu/results.json \
  --patchtst runs/patchtst_encoder_ablation/results.json \
  --output runs/patchtst_encoder_ablation/comparison.json
```
