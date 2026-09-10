# TimeSiam-inspired future-pose experiment

This is a separate experiment. It does not replace or modify the selected
`train_gripper_state_pose.py` or `train_gripper_future_endpoint.py` models.

## What changes

The existing causal EMG/IMU encoders and all current-task heads are retained.
The flat future projector is replaced by:

1. a shared encoder for causal past tokens and masked future queries;
2. learned lineage embeddings for horizons 10–200 ms;
3. cross-attention from future queries to past context only;
4. future wearable reconstruction pre-training;
5. fine-tuning for current pose, gripper state, screen target, endpoint, and
   200 ms future pose.

All future queries are fully masked. Future EMG, IMU, and VIVE samples are
targets only and never model inputs, so inference remains causal.

This is TimeSiam-inspired rather than an exact reproduction: its lineage is
the prediction horizon, and fully masked future queries are used because no
future wearable samples exist at deployment.

## Recommended first run

Test the fused model before spending time on the full ablation:

```bash
git pull origin main
python scripts/train_gripper_timesiam_future.py \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --models emg+imu --device cuda \
  --timesiam-pretrain-epochs 15 --epochs 60 \
  --output-dir runs/gripper_timesiam_future_fused
```

For the EMG, IMU, and EMG+IMU ablation:

```bash
bash scripts/train_gripper_timesiam_future.sh \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --device cuda --output-dir runs/gripper_timesiam_future_all
```

## Honest comparison

Use the identical held-out split and compare against the current model on:

- `future_pose_by_ms["200"].position_cm` (primary future-intent metric);
- the 200 ms persistence baseline;
- `position_cm` and `final_position_cm`;
- last-quarter pixel error;
- gripper macro F1.

The current reference is 4.24 cm at 200 ms versus 8.25 cm persistence. Keep
the new model only if it improves held-out future error without materially
damaging current pose, endpoint, pixel, or gripper results.

The output directory contains a reconstruction-pretrained checkpoint, the
best fine-tuned checkpoint, training history, split manifest, and
`results.json`.
