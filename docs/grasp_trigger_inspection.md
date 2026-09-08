# Full-trial plots and trigger stability

No retraining. Original checkpoints are read-only. Run on the machine holding
the saved annotation-aware run and original CSV files:

```bash
git pull origin main
python scripts/inspect_grasp_triggers.py \
  --run-dir runs/reach_grasp_annotation_seed42 \
  --device cuda --split validation --plots 6 \
  --output-dir runs/grasp_trigger_inspection
```

Requires matplotlib (`python -m pip install matplotlib` if missing). Files are
loaded from saved `splits.json`; the script does not invent a new split or refit
normalization. Use unchanged CSV recordings. PNGs work on headless systems;
no browser or display server is required.

Each of the three models gets the SAME seeded randomly selected trial subset:
EMG RMS, accelerometer norms, holding probability, grasp probability, release
probability. Green/red vertical lines are manual annotations. Colored bands
show annotation uncertainty; grey shading marks unavailable wearable inputs.
Original event triggers are grey crosses; validation-selected event triggers
are blue triangles. Holding-transition triggers have their own panel. Complete
trials are plotted, not just short slices around successful detections.

The trial plots use recorded units for sensor features. They are offline model
inference on recordings, not a live hardware interface. Physical contact timing
cannot be verified from uncertain annotations alone.

## Decoder study

The existing event-head thresholds remain fixed. A small validation sweep adds
hysteresis (rearm below 0.5 or 0.8 times the high threshold), persistence
(30/60/100 ms), and the original 400 ms refractory period. The original decoder
is also eligible, so validation can select no change. Parameters are selected
separately for each model with the same search space, using event macro-F1 at
the checkpoint's saved selection tolerance. One parameter set covers both events.

These are causal triggers: every emitted timestamp is the confirmation time,
not the start of the persistence interval. Missing samples interrupt pending
evidence. Grasp-before-release ordering is NOT forced; no ground-truth event or
trial duration enters decoding. Both too-early and too-late triggers remain errors.

`summary.md` compares original event heads, validation-selected event heads, and
the saved holding decoder at +/-100/150/200/300 ms. `decoder_report.json` stores
the selected parameters, search results, recall, timing and false-trigger metrics.
Validation gains are selection-biased: they are not new evidence of generalization.

`*_triggers.json` lists each trial's annotations, triggers and missing events.
For inspection only, unmatched detections are categorized as duplicates within
the scoring tolerance, nearby (within 500 ms), or distant. The 500 ms category
is a descriptive heuristic, not a new acceptance tolerance. Ground truth is
only used in this report, never to remove a trigger from predictions.

To evaluate fixed validation-selected settings on the existing test split:

```bash
python scripts/inspect_grasp_triggers.py \
  --run-dir runs/reach_grasp_annotation_seed42 \
  --device cuda --split test --plots 6 \
  --output-dir runs/grasp_trigger_inspection_test
```

It selects parameters using validation first, then loads test trials. This test
set has already been explored in the project; confirmation still needs fresh
recordings or a preregistered held-out evaluation. This tool changes neither
training labels nor model weights. It does not guarantee better EMG performance.
