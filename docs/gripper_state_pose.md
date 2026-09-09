# Gripper state and pose model

This task predicts current 3D position, 6D orientation, and one dense gripper
class at every timestep: `open` or `close`. The encoder receives EMG+IMU only.
VIVE pose and `gripper_state` are training labels and are not model inputs.

The architecture retains the causal EMG and IMU patch encoders, gated fusion,
and ReactEMG-inspired masked EMG/state branch. The obsolete grasp/release pulse
and future-event heads are absent. The local classification route detects state
changes, while the context route stabilizes sustained states.

The CSV must contain `time_perf_counter`, the existing four EMG and 24 IMU
columns, VIVE position/orientation columns, and a per-row `gripper_state` column
whose nonempty values are exactly `open` or `close`. Missing labels are filled
only from the past for at most 20 ms, then excluded from loss and metrics.
Grasp/release timestamp columns are not required.

```bash
git pull origin main
bash scripts/train_gripper_state_pose.sh \
  --root /home/nahar3/shubham/emg_shubhu/data/YOUR_NEW_RECORDING \
  --output-dir runs/gripper_state_pose_new_data
```

Outputs are `best.pt`, `final.pt`, `results.json`, `history.json`, `splits.json`,
and `data_audit.json`. `results.json` reports macro F1, accuracy, the open/close
confusion matrix, position error in cm, orientation error in degrees, and zeroed
EMG/IMU ablations. The split unit is a complete trial.
