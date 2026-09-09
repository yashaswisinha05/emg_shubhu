# One-second reach, grasp and release intent

This experiment predicts both current state and future intent from causal
EMG+IMU input. VIVE is used only to supervise translation and orientation
during training. It is not an encoder input.

The future heads predict position and 6D orientation at 100, 250, 500, 750 and
1000 ms, the one-second endpoint, whether grasp/release will occur within the
next second, and categorical time-to-event distributions. The current pose,
holding, grasp and release heads remain present.

```bash
bash scripts/train_reach_grasp_future_intent.sh
```

The held-out report includes current position, orientation, holding and event
metrics; future position/orientation error by horizon; grasp/release intent
AUROC and average precision; time-to-event MAE; and without-EMG/without-IMU
ablations. Both `best.pt` and the identical deployment alias `final.pt` are
written. All splits are by complete trial.
