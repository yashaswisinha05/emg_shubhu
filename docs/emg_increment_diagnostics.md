# Where does EMG add information?

This does not train, change weights, choose a new checkpoint, or prove that EMG
contains all information needed for motion. It diagnoses the already-selected
delay-correction model against its original IMU baseline and correction control.

```bash
python scripts/diagnose_emg_increment.py \
  --root /home/nahar3/shubham/emg_shubhu \
  --run-dir runs/emg_delay_correction_seed42 \
  --device cuda --split validation \
  --output-dir runs/emg_increment_validation
```

Use the unchanged dataset/root. All four recording prefixes and the original
split come from saved checkpoints; the regenerated split is verified against
`splits.json`. This uses the same training-only normalization as training.
The original run directory must contain `selected_emg.pt`, `imu_control.pt`,
and `splits.json`. Use the existing training environment; no new packages needed.

Read `report.md` for the table, `report.json` for all metrics, and
`paired_trials.json` for individual errors and shuffle donor paths. Reports are
grouped by recording, prediction lead, screen quadrant (normalized xy split at
0.5), and recording-by-lead. A recording prefix is not necessarily a unique
person. Separate reporting does NOT train a separate within-person model.

Positive `imu_control_minus_emg` means the intact EMG correction beats the
capacity control. Positive `shuffled_emg_minus_emg` means correct EMG pairing
helps that model. Interpret BOTH together; a shuffle penalty without improved
accuracy over the control is not evidence of incremental predictive benefit.

Shuffling exchanges whole histories and masks at the same lead within a
recording, across different trials, including across minibatch boundaries.
There are no self-pairs. This avoids mixing different recording calibrations,
but also changes EMG history length/phase when donor trials differ in duration.
It is one deterministic permutation, not a physiological causal intervention.

Paired 95% bootstrap intervals use trial-averaged errors, clustering all observed
leads within a trial. Overall reports weight trials equally; this can differ from
the old observation-weighted aggregate if some trials lack long leads. Intervals
do not capture training-seed variability or adjust for the many subgroup tests.
Pooled intervals are not person-held-out generalization estimates. Treat positive
subgroups as hypotheses to confirm on new data, not discoveries by themselves.

## Earlier phases (optional exploratory stress test)

Add `--extra-leads-ms 600 800` and use a new output directory. Results outside
the trained 0–400 ms lead range appear under `OUT_OF_TRAINING_RANGE` and are
excluded from the trained-range aggregate. Trials too short for a lead are
skipped by the original window builder; report counts reflect this. Failure at
these leads cannot establish absence of early EMG information. Testing early
EMG usefulness fairly would require training matched models at those cutoffs.

Default is validation because this report is for developing hypotheses. To
inspect the previously evaluated test split, explicitly use `--split test` and
a new output directory. The test set is already reused/explored in this project;
do not describe these subgroup results as a new confirmatory test.

If no subgroup shows a meaningful advantage, next steps are a raw recording
synchronization/channel-quality audit and matched single-recording training.
This script performs neither raw signal-quality analysis nor a synchronization
audit, and does not force the model to rely on EMG.
