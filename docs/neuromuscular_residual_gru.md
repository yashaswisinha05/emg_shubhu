# State-conditioned neuromuscular residual GRU

This model extends the strongest causal GRU comparison without changing the
dataset protocol. It has separate heads for EMG-only open/close state, current
XYZ, 3x3 screen-cell plus within-cell pixel offset, and 200 ms future XYZ.
Training-only heads reconstruct masked EMG spans and future IMU changes.

IMU is the mechanical base representation. A bounded, zero-initialized EMG
residual can correct that representation, conditioned on the detached soft
open/close state. Thus pose losses cannot corrupt the EMG state classifier and
the correction initially behaves as a no-op.

To append it to an already completed architecture study, use the same command
and add `--resume`. Complete prior runs are reused and only the new
`residual_gru` directory is trained:

```bash
git pull origin main

python scripts/run_gripper_pose_architecture_study.py \
  --root data/shubham_open data/shubham_close \
         data/mukund_closed data/mukund_open \
         data/gazania_open data/gazania_closed \
         data/yashaswi_open data/yashaswi_close \
  --classifier-checkpoint \
    runs/gripper_neuromuscular_future_paper_split42/emg_imu_best.pt \
  --device cuda --epochs 60 --split-seed 42 --seeds 42 \
  --resume \
  --output-dir runs/gripper_pose_architecture_study_paper_final
```

The standalone training entry point is:

```bash
python scripts/train_neuromuscular_residual_gru.py \
  --root data/shubham_open data/shubham_close \
         data/mukund_closed data/mukund_open \
         data/gazania_open data/gazania_closed \
         data/yashaswi_open data/yashaswi_close \
  --models emg+imu --device cuda --epochs 60 \
  --seed 42 --split-seed 42 \
  --pixel-weight .35 --future-pose-ms 200 --future-pose-weight .5 \
  --grid-weight .15 --grid-residual-weight .1 \
  --masked-emg-weight .05 --future-imu-weight .05 \
  --future-consistency-weight .1 --correction-weight .01 \
  --output-dir runs/neuromuscular_residual_gru_seed42
```

For publication, run at least three seeds. The result only supports an EMG
contribution if zeroing and trial-shuffling EMG worsen held-out performance.
