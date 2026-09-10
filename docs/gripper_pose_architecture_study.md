# Gripper/pose architecture comparison

This benchmark compares causal architectures under one data protocol. Every
model receives the same 100 Hz EMG+IMU tensors, trial split, current XYZ,
1920x1080-aware pixel target, open/close target and withheld 10--200 ms XYZ
targets. No baseline uses bidirectional recurrence, centred convolution or
unmasked attention.

Run all baselines plus the proposed frozen-classifier hybrid:

```bash
python scripts/run_gripper_pose_architecture_study.py \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --classifier-checkpoint \
    runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --device cuda --epochs 60 --split-seed 42 --seeds 42 \
  --output-dir runs/gripper_pose_architecture_study
```

For five-seed publication results, use `--seeds 11 22 33 44 55`. The fixed
`--split-seed 42` keeps the train/validation/test trials identical while only
weight initialization changes. This is five
full trainings per architecture and can take substantial GPU time. First use
`--dry-run` to inspect every generated command without starting training.

Outputs:

- `comparison.md`: paper-readable table;
- `comparison.csv`: architecture-level mean and standard deviation;
- `comparison_runs.csv`: raw seed-level rows;
- `comparison.json`: both summary and raw rows;
- `<architecture>/seed<seed>/`: checkpoint, history, split and audit.

`constant` is initialized strictly from training trials. The other comparison
models share the same four heads and objective. The proposed model retains its
frozen neuromuscular classifier and architectural auxiliary losses because
those are part of the method being evaluated.

For a publishable subject-independent study, run the command once per held-out
subject using roots/splits that exclude that subject from training. The frozen
classifier checkpoint must have been trained on exactly that fold's training
subjects; reusing a classifier that saw the held-out subject is leakage.
