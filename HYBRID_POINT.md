# Centre-Safe Hybrid Touch Model

This experiment fixes the inward collapse observed in the original 8×5 grid
model while preserving its useful spatial supervision. Existing `grid_point`
checkpoints and results are not overwritten.

## What changed

The final touch coordinate comes from a continuously supervised direct X/Y head.
The 8×5 heatmap and local offsets remain auxiliary tasks that encourage the
encoder to distinguish screen regions.

```text
raw IMU (24) ───────────────┐
calibrated IMU (64) ────────┼─> full + final-500-ms encoders ──> direct X/Y
                            │                                  ├─> 8×5 heatmap
final-500/final-300-ms EMG ─┘                                  └─> cell offsets
```

The loss no longer computes the distance of the probability-weighted grid
coordinate. That calculation allowed errors on opposite sides of the screen to
cancel at the centre. Instead it uses:

- mostly hard cell supervision plus 15% Gaussian neighbour smoothing;
- expected candidate-to-target distance, so distant probability mass is costly;
- direct pixel Smooth-L1 and radial Charbonnier losses;
- modest edge weighting so edge and corner examples cannot be cheaply ignored;
- target-cell offset regression as an auxiliary task.

Checkpoint selection, learning-rate reduction and early stopping use validation
mean pixel error—not the combined auxiliary loss.

For fusion, the complete hybrid IMU model is loaded and frozen. Zero-initialized
EMG residuals modify its heatmap, offsets and direct-coordinate logits. Therefore,
fusion initially produces exactly the same prediction as the IMU checkpoint.

## One-fold mix7 test

Run from the project directory in the `smss` environment:

```bash
cd /Users/yashaswi/phd_iisc/emg_shubhu/emg_touch
conda activate smss
python -m pip install -e .

python scripts/run_grid_point_sweep.py \
  --config configs/hybrid_point.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

Outputs are written separately to:

- `artifacts/hybrid_point/`
- `runs/hybrid_point/`
- `evaluation/hybrid_point/`

The sweep also writes `evaluation/hybrid_point/directional_error.csv`. The key
anti-collapse fields are `mean_inward_error_px`, its participant-blocked interval,
and `edge_prediction_gap`. Values closer to zero are better.

Five-fold mix7 references:

- old grid fusion: 70.21 px mean inward error and −11.88 percentage-point edge gap;
- direct multimodal model: 20.71 px mean inward error.

## Decision rule after fold 0

Compare against the matched historical fold-0 results:

- direct IMU Patch: 106.25 px median;
- direct multimodal: 97.36 px median.

Proceed to all five folds only if hybrid IMU approaches the direct IMU baseline
and the inward component is materially smaller than the old grid model. Do not
judge only by total validation loss because it combines several auxiliary terms.

## Five-fold mix7 run

```bash
caffeinate -i python scripts/run_grid_point_sweep.py \
  --config configs/hybrid_point.yaml \
  --configuration mix7 \
  --device mps \
  2>&1 | tee -a runs/hybrid_point_mix7.log
```

The sweep is resumable. Its progress file is
`runs/hybrid_point/sweep_status.json`.
