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

## Franka future-intent replay

After training, replay any recorded trial causally in PyBullet:

```bash
python scripts/visualize_future_intent_franka.py \
  --checkpoint runs/reach_grasp_future_intent_seed42/final.pt \
  --trial-root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \
  --device cuda --trial-seed 7 --speed 1
```

Black is withheld VIVE for visual comparison, cyan is the continuously
recomputed 250 ms command, magenta is the complete rolling one-second forecast,
and orange is the actual Franka end effector. The robot never consumes VIVE.
Use `--control-horizon-ms 0` to follow the current estimate or choose one of
`100 250 500 750 1000` for receding-horizon control.

Grasp and release use probability mass within `--gripper-lookahead-ms` rather
than the coarse “event anywhere in the next second” probability. This prevents
the gripper from actuating a full second early. Both actuation thresholds default
to 0.9 and the gripper state is latched until the opposite event fires.

Every emitted prediction contains three distinct error measurements:

- `model_vs_vive_error_by_horizon`: model versus synchronized VIVE at the
  current instant and each trained future horizon, in centimetres and degrees.
- `control_horizon_model_vs_vive_error`: the selected command horizon only.
- `franka.position_tracking_error_cm` and
  `franka.orientation_tracking_error_deg`: Franka versus the model command.
- `franka.vs_vive_future_euclidean_cm` and
  `franka.vs_vive_future_angle_deg`: end-to-end Franka versus future VIVE.

VIVE comparisons are attached only after inference and are never read by the
model or used to generate the robot command.
