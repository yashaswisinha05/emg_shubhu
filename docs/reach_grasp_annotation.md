# Annotation-aware reach–grasp–release

Run on the same dataset and root:

```bash
git pull origin main
bash scripts/train_reach_grasp_annotation.sh
```

This trains **new** IMU-only, EMG-only and fusion models into
`runs/reach_grasp_annotation_seed42`, without overwriting the original run.
The architecture, filters and default seed/split remain the same. Keep the CSV
files unchanged to preserve the original split; inspect both `splits.json` files
when comparing experiments. The original launcher still uses hard labels.

## What changed

- `--annotation-aware --uncertainty-ms 200`: event targets are Gaussian bumps
  centered on each manual marker, with sigma 100 ms, truncated at +/-200 ms.
  This is a working assumption about annotation uncertainty, not a measured
  physiological delay or a probabilistic guarantee that an event lies there.
- Holding labels within +/-200 ms of either marker are excluded from the
  holding loss. Event supervision and valid position supervision remain active.
  Short holds may have few/no certain holding frames; inspect reported coverage.
- `--selection-tolerance-ms 200`: event-head thresholds and checkpoint selection
  use validation event scores at +/-200 ms. Test data never tunes thresholds.
- After selecting each model checkpoint, validation selects a separate causal
  holding-transition decoder: hysteresis pairs (0.3,0.7)/(0.4,0.6), persistence
  30/60/100 ms. Both decoders are reported; neither is chosen using test scores.
- Transition timestamps are the time persistence is confirmed, not an earlier
  backdated boundary. Initial state is unknown, and establishing the initial
  holding/open state emits no event. Missing data interrupts pending evidence.
- The same thresholds/decoder parameters are evaluated at +/-100, +/-150,
  +/-200 and +/-300 ms. They are NOT retuned for each tolerance or ablation.

The model remains causal. Future timestamps can define supervision targets but
never enter the input. Soft targets before the annotation may encourage early
responses relative to that marker; this is not proof of physical anticipation.

## Results

`results.json` retains the main metrics (event heads at selection tolerance).
Each model additionally has `by_tolerance_ms`, containing `event_heads` and
`holding_transitions`. The schedule-only baseline is also evaluated at every
tolerance. Compare models at the SAME tolerance and with the SAME decoder type.

`holding_f1` excludes uncertain boundaries in this mode. For comparison with the
old model, `holding_f1_all_valid_frames` retains all available frames, and
`holding_certain_fraction_of_valid` shows how much evaluation remains. Position
evaluation does not exclude uncertain event boundaries. Timing MAE is still
relative to manual annotations and only includes matched events, so always
report recall, false triggers/minute and timing MAE together.

A score increase at +/-300 ms compared with +/-150 ms is a relaxed criterion,
not improved physical timing. The old/new runs also have different checkpoint
selection tolerances by default. For a stricter controlled comparison, train
with `--selection-tolerance-ms 150` in a different output directory.

Detailed command (also useful for a one-epoch smoke run):

```bash
python scripts/train_reach_grasp.py \
  --root /home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827 \
  --annotation-aware --uncertainty-ms 200 --selection-tolerance-ms 200 \
  --models imu emg emg+imu --device cuda --epochs 40 --batch-size 8 \
  --output-dir runs/reach_grasp_annotation_seed42
```

Print a compact comparison after training:

```bash
python - <<'PY'
import json
with open('runs/reach_grasp_annotation_seed42/results.json') as f:
    r = json.load(f)
for name in ['imu', 'emg', 'emg+imu']:
    print('\n', name)
    for ms, decoders in r[name]['by_tolerance_ms'].items():
        for decoder, m in decoders.items():
            print(ms, decoder, 'event F1=', round(m['event_macro_f1'], 3),
                  'grasp recall=', round(m['grasp']['recall'], 3),
                  'release recall=', round(m['release']['recall'], 3))
PY
```

Recheck a small sample against video/contact evidence if available. Manual
annotation uncertainty remains unresolved by changing a loss alone. This is
still exploratory development on the existing split, not a new confirmatory
test or a guarantee that EMG will outperform IMU.
