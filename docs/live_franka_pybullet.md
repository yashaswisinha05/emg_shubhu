# Live EMG+IMU control of a PyBullet Franka

This program transfers the trained model outputs directly to a simulated
Franka Panda:

```text
4 EMG + 24 IMU -> trained model -> predicted XYZ + orientation quaternion
                                -> Panda 7-DoF pose IK -> arm joint commands
                                -> grasp/release triggers -> finger commands
```

VIVE is not read at inference. It was used to supply pose labels during
training. PyBullet IK is deterministic robot kinematics; it is not another
learned model and its output is not fed back into the predictor.

## Install

On the Linux inference machine:

```bash
python -m pip install -r requirements-pybullet.txt
```

## Run from the Delsys stream

```bash
python scripts/live_franka_pybullet.py \
  --checkpoint runs/reach_grasp_orientation/emg_imu_best.pt \
  --delsys-sdk-path /path/to/delsys/sdk \
  --device cuda
```

The default opens the PyBullet GUI. Add `--headless` for a non-visual run.
Each JSON prediction reports the requested robot pose, solved joint angles,
actual simulated end-effector pose, and IK/FK tracking errors.

## Run on an unseen recorded trial

No Delsys connection is needed. Point the same program at a held-out CSV:

```bash
python scripts/live_franka_pybullet.py \
  --checkpoint runs/reach_grasp_orientation/emg_imu_best.pt \
  --trial-csv /path/to/held_out_recording/trial_069.csv \
  --device cuda \
  --speed 1.0
```

`--speed 1.0` follows the recorded timing, `--speed 2.0` runs twice as fast,
and `--speed 0` runs without waiting. To choose a reproducible random trial
recursively from a held-out directory:

```bash
python scripts/live_franka_pybullet.py \
  --checkpoint runs/reach_grasp_orientation/emg_imu_best.pt \
  --trial-root /path/to/held_out_recording \
  --trial-seed 7 \
  --device cuda \
  --speed 1.0
```

The replay reader passes only `time_perf_counter`, the four trained EMG
channels, and the 24 trained accelerometer/gyroscope channels to the model.
When VIVE XYZ is present it is drawn as a black, comparison-only trajectory;
it is never passed to the predictor. Missing wearable values retain the same
causal, bounded-gap handling used by live inference, and the Panda holds the
last valid target rather than receiving invented pose values.

## Coordinate calibration

For meaningful absolute robot motion, measure the rigid transform from the
VIVE/world frame used for model labels to the Panda base frame and pass it as:

```bash
python scripts/live_franka_pybullet.py \
  --checkpoint runs/reach_grasp_orientation/emg_imu_best.pt \
  --delsys-sdk-path /path/to/delsys/sdk \
  --robot-from-vive-translation TX TY TZ \
  --robot-from-vive-quaternion-wxyz W X Y Z
```

The mapping is `p_robot = R_calibration p_model + translation` and
`R_robot = R_calibration R_model`. Model and command-line quaternions use
`w x y z`; the integration converts them to PyBullet's `x y z w` convention.

The Franka script additionally applies a `+90°` anticlockwise rotation around
the robot Z axis by default. It rotates position about the anchored home point,
so the starting point remains fixed, and applies the same rotation to the
predicted end-effector orientation. Override or disable it with
`--trajectory-z-rotation-deg ANGLE` (use `0` to disable).

Without an explicit calibration, the first predicted pose is anchored at
`--home-position 0.45 0 0.50` and subsequent relative position and orientation
changes are preserved. This synthetic anchor is useful for visualization, but
it is not a substitute for robot/world calibration on physical hardware.

The fingers start open. Grasp probability at or above `0.9` commands both
Panda finger joints to `0.00 m`, then latches the gripper closed. Only release
probability at or above `0.9` opens each finger to `0.04 m`. The confirmed
probability at the decoder's event timestamp is preserved even when the
current pose frame is invalid. Override the thresholds with
`--grasp-probability-threshold` and `--release-probability-threshold`.

The PyBullet GUI draws the entire withheld VIVE path in black, the growing
model-requested path in cyan, and the actual Franka end-effector path in
orange. Crosses label model start, grasp, and release, and large text reports
`GRIPPER: OPEN` or `GRIPPER: CLOSED`.

During missing/invalid wearable frames the Panda keeps its last valid arm
target and the simulation continues stepping, instead of freezing. Recorded
replay also runs 240 final simulation steps by default so the arm and fingers
can converge to the last model command. Use `--final-settle-steps` to change
this duration. Each prediction uses 24 physics settling steps by default;
change this with `--simulation-steps`. IK arrays include all nine movable
Panda DoFs, preventing
PyBullet from discarding joint damping because the two finger DoFs were absent.
