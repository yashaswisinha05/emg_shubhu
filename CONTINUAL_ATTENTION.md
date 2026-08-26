# Continual causal touch prediction with channel attention

This experiment predicts the final touch point repeatedly from information that
would actually be available online:

- movement onset: signal through `reaction_time_s`;
- `0.2s`: signal through `reaction_time_s + 0.2`;
- `0.4s`: the same for trajectories that last at least 400 ms;
- `touch`: the complete causal trajectory through `touch_time_s`.

The label is the same final touch coordinate for all prefixes. Early-prefix
losses receive lower weights because their irreducible uncertainty is larger.
No post-prefix signal samples enter the encoders or the dynamic attention
statistics. The implementation currently recomputes the causal prefix at each
step; it represents online inference correctly, though encoder-state caching can
be added later to reduce latency.

Checkpoint selection uses the same causal validation views and their time-aware
weights. The history also records endpoint-only validation error, so improvement
at early cutoffs cannot hide a degraded touch-time estimate.

## Channel attention

IMU attention is hierarchical: the model first weights the four physical
sensors (`S0`, `S4`, `S8`, `S12`), then the raw/calibrated features within each
sensor. EMG attention independently weights the four EMG electrodes. Attention
is mask-aware and trajectory-dependent, and a residual gate prevents a newly
initialized attention block from destroying signal amplitude.

Every prediction row saves sensor and channel attention probabilities. The sweep
also creates `channel_attention_summary.csv` with the mean, median, standard
deviation, and top-attention frequency per configuration and time cutoff.
Attention is descriptive rather than proof of causal importance; use held-out
sensor/channel ablations before making physiological claims.

## One-fold smoke experiment

Run from the `emg_touch` directory in the `smss` environment:

```bash
conda activate smss

python scripts/run_grid_point_sweep.py \
  --config configs/continual_attention.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

The runner fits the matching 88-feature scaler, trains IMU, EMG, and fusion
models, evaluates every causal cutoff, and writes results below:

- `runs/continual_attention/mix7/fold-0/`
- `evaluation/continual_attention/`

## Full configuration comparison

After the one-fold run is sound, launch all configurations and all five folds:

```bash
caffeinate -i /opt/homebrew/Caskroom/miniconda/base/envs/smss/bin/python \
  scripts/run_grid_point_sweep.py \
  --config configs/continual_attention.yaml \
  --device mps
```

The configuration comparison remains out-of-fold and participant-blocked. A
configuration may contain any number of participants or trajectories.
