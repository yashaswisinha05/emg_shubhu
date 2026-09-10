# EMG-state-conditioned position attention

This is a new experiment; it does not modify earlier architectures or
checkpoints. An EMG-only head predicts soft open/close state. That probability
conditions a causal EMG-to-IMU attention correction, while IMU remains the
residual base. Both the state FiLM transform and EMG correction are initialized
as exact no-ops.

Deployment heads are open/close, current XYZ, and pixel XY. A training-only
decoder predicts withheld future XYZ for the next 200 ms. Its prediction is
supervised by future VIVE labels and must agree with the current-position head
when that future frame arrives. Future VIVE is never an input.

Run the fused model first:

```bash
python scripts/train_gripper_state_attention.py \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --models emg+imu --device cuda --epochs 60 \
  --output-dir runs/gripper_state_attention_fused
```

Run capacity-matched modality ablations with:

```bash
bash scripts/train_gripper_state_attention.sh \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --device cuda --output-dir runs/gripper_state_attention_all
```

The important comparisons are fused versus IMU-only current/future XYZ,
fused versus EMG-only state F1, and fused zero-EMG degradation. Attention
weights are descriptive only; causal channel removal remains the attribution
test.
