# Reach/grasp orientation experiment

This is a new experiment built on the hybrid model. It keeps the existing
event, holding, and XYZ heads and adds a separate orientation head driven by
the same global patch context. Earlier checkpoints and run directories are not
modified.

```bash
git pull origin main
bash scripts/train_reach_grasp_orientation.sh
```

Outputs go to `runs/reach_grasp_orientation_hybrid_seed42`. The CSV quaternion
columns must be named `VIVE_T0_quat_w`, `VIVE_T0_quat_x`,
`VIVE_T0_quat_y`, and `VIVE_T0_quat_z`.

## Representation and metrics

VIVE orientation is a training/evaluation label only. It is never part of the
wearable encoder input. Quaternions are converted to the continuous 6D
representation formed by the first two columns of the rotation matrix. This
avoids the `q` versus `-q` ambiguity and discontinuities of direct Euler-angle
regression. The predicted axes are orthonormalized before evaluation.

Every modality result reports:

- `orientation_geodesic_deg`: mean full 3D rotation error;
- `orientation_geodesic_median_deg`: median full rotation error;
- `yaw_mae_deg`: circular mean absolute yaw error;
- `yaw_median_ae_deg`: circular median absolute yaw error;
- `valid_orientation_frames`: evaluated tracker frames.

Yaw uses the ZYX convention and is rotation around the VIVE/world z axis. The
three independently trained models and `fusion_zero_emg` are all evaluated,
so an EMG orientation claim requires both EMG+IMU beating IMU-only and the
zero-EMG ablation worsening. Because the sensors have accelerometers and
gyroscopes but no magnetometer, absolute yaw is only identifiable from the
standardized initial pose and integrated motion history; report this limitation.

The default orientation loss weight is 0.5. Do not tune it on test results.
First run the fixed setting across multiple seeds.
