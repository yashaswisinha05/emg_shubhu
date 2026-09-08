# Hybrid causal reach/grasp model

This experiment preserves the patch transformer's global context for current
3D position and holding state, but does not ask 160 ms patch aggregation to
locate sharp grasp/release boundaries. Its dedicated event head reads the
frame-resolution, left-padded causal convolution features from the EMG and IMU
branches before patch aggregation.

Run it without changing any earlier run directory:

```bash
git pull origin main
bash scripts/train_reach_grasp_hybrid.sh
```

Outputs are written to `runs/reach_grasp_hybrid_seed42`. The three independent
models (IMU, EMG, and EMG+IMU) use the same trial splits and preprocessing as the
previous patch-transformer experiment.

## Two event estimates and validation-only combination

The first event estimate is the dedicated local grasp/release head. The second
is generated when the global holding probability completes a causal hysteresis
transition. A confirmed transition creates only a short forward-time pulse; it
is never backdated. On validation, the code selects the pulse duration, local
head weight, threshold, and causal persistence independently for grasp and
release. Those parameters are stored under `hybrid_event_decoder` in each
checkpoint before the test trials are loaded.

`results.json` retains all original metrics and adds
`hybrid_event_decoding`, which reports the local head, holding transitions, the
combined decoder, tolerance sensitivity, and a zero-EMG fusion ablation. Use the
`combined` result as the primary result for this architecture. The validation
search is exploratory model selection; it is not a test-set optimization.

The model remains fully causal. VIVE position and human event annotations are
supervision only and are not wearable inputs.
