# Masked-EMG reconstruction experiment

This is a separate representation-learning experiment built on the robust
causal reach-grasp model. It does not replace or modify an existing checkpoint.

The normal task forward pass receives complete EMG+IMU input and retains all
existing losses for holding, grasp/release, event horizon, XYZ, orientation,
and pose uncertainty. A second training-only forward pass hides contiguous
blocks from genuinely observed EMG features. A decoder attached directly to
the causal EMG encoder reconstructs only those hidden normalized features:

```text
L = L_task + lambda_rec * MSE(hidden EMG reconstruction)
```

The default settings use `lambda_rec=0.1`, a mask ratio of `0.35`, and blocks
of five frames (50 ms at the 100 Hz processed rate). True missing samples and
padded frames are excluded from reconstruction targets. IMU features are not
available to the reconstruction decoder, so IMU cannot satisfy this auxiliary
loss on behalf of EMG.

VIVE position and orientation remain supervision labels for the task heads.
VIVE is never an encoder input. The reconstruction decoder is unused during
deployment.

## Train

From the repository root:

```bash
git pull origin main

python scripts/train_reach_grasp_masked_reconstruction.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --device cuda \
  --epochs 60 \
  --batch-size 8 \
  --seed 42 \
  --raw-rate-hz 1259.4 \
  --models imu emg emg+imu \
  --output-dir runs/reach_grasp_masked_reconstruction_seed42
```

The shortcut uses the repository-local data path:

```bash
bash scripts/train_reach_grasp_masked_reconstruction.sh
```

The deployment checkpoint is:

```text
runs/reach_grasp_masked_reconstruction_seed42/emg_imu_best.pt
```

Each history file records `masked_emg_reconstruction_mse`. Checkpoint selection
still uses validation task quality, preventing a low reconstruction error from
winning when grasp/release or orientation becomes worse.

## Required comparison

Compare this run with `reach_grasp_robust_v1` using the same trials and seeds.
Report the test event macro F1, holding F1, position error, orientation error,
and `fusion_zero_emg`. A useful reconstruction objective must improve held-out
task metrics or produce a larger reproducible EMG ablation cost. Reconstruction
MSE alone is not evidence that the representation helps the task.

For a loss-weight ablation, run separate output directories with:

```text
--masked-emg-reconstruction-weight 0.05
--masked-emg-reconstruction-weight 0.10
--masked-emg-reconstruction-weight 0.25
```
