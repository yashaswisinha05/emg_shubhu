# Neuromuscular future model

This is the proposed project-specific experiment. It leaves every earlier
model unchanged.

The model treats future movement as two components:

- an IMU-conditioned mechanical-state rollout;
- a horizon-dependent EMG innovation that can correct the rollout before the
  corresponding mechanical change is visible.

The innovation starts with a small gate, preventing noisy EMG from destroying
the strong IMU baseline. During training, an auxiliary causal objective asks
the future latent to predict the change in IMU over the next 10–200 ms. Future
IMU and VIVE values are supervision only, never inference inputs.

This is not TimeSiam: there are no Siamese windows, shared past/future encoder,
lineage matching, masked-token decoder, or cross-attention reconstruction.
Withheld-future supervision is the only retained general principle.

Run the fused experiment first:

```bash
git pull origin main
python scripts/train_gripper_neuromuscular_future.py \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --models emg+imu --device cuda --epochs 60 \
  --motion-reconstruction-weight 0.05 \
  --output-dir runs/gripper_neuromuscular_future_fused
```

Run all three modality ablations with:

```bash
bash scripts/train_gripper_neuromuscular_future.sh \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --device cuda --output-dir runs/gripper_neuromuscular_future_all
```

The paper claim should be conditional on held-out evidence. Report the 200 ms
future-pose error against persistence, IMU-only, and the existing future model.
Also report zero-EMG/shuffled-EMG degradation and the learned EMG innovation
gate. Do not call the architecture novel until a proper related-work search
rules out closely matching prior models.
