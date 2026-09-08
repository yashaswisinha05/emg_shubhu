# Live EMG+IMU reach/grasp/orientation inference

Use the fused checkpoint produced by the orientation experiment:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --device cuda
```

The default mode reads newline-delimited JSON from stdin. Print the exact
protocol and channel order with:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --print-protocol
```

Each raw sample contains only four EMG and 24 IMU measurements. VIVE is not an
accepted input. The process emits causal predictions every 40 ms after a 200 ms
warmup: holding/grasp/release probabilities and triggers, XYZ in metres,
quaternion `(w,x,y,z)`, and ZYX yaw/pitch/roll in degrees. Send `start` between
trials to clear filter, transformer, and event-decoder history.

The same output now contains analytical 3R inverse kinematics and a two-finger
gripper. `joint_angles_deg` contains base yaw, shoulder, and elbow angles;
`chain_m` contains base=P1, elbow, and end-effector points. A validated grasp
trigger closes the gripper and a release trigger opens it. Between triggers the
previous gripper state is retained. The returned jaw points are placed at the
model-driven end effector.

For physically meaningful angles, measure the shoulder/base in the same VIVE
world frame and pass it explicitly:

```bash
... --base-world X Y Z --link-lengths 0.50 0.60 \
    --axis-order xyz --axis-signs 1 1 1
```

Without `--base-world`, the program anchors the first predicted model position
to `--initial-joint-deg 0 20 90`. This is useful for visualizing motion shape,
but those angles are synthetic and must not be reported as physical arm angles.
Targets outside the robot workspace are radially projected and marked with
`workspace_projected: true`. The 3R IK uses XYZ only; the separately predicted
wrist orientation is reported but cannot be imposed by a position-only 3R arm.

On a machine using the same `EMGCollector.py` Delsys interface as data
collection, the script can connect directly:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --device cuda \
  --delsys-sdk-path /path/to/Example-Applications/Python
```

The live scan must expose the exact trained names `EMG 1_S0/S4/S8/S12` and the
ACC/GYRO XYZ channels for those sensors. It fails loudly on missing or duplicate
channels rather than silently changing sensor order.

The online path reproduces the training transformations causally: raw EMG
band-pass, 20/50 ms trailing RMS, IMU trailing median and low-pass, 100 Hz
previous-sample resampling, checkpoint normalization, and validity masks.
Predictions are fixed-weight inference; no VIVE, annotation, calibration target,
or online parameter update is used.
