# Final-model ablations

`run_final_model_ablations.py` retrains every selected variant once with the
same data roots, trial split, seed, optimizer, and epoch budget. The default
seed is 42; this runner does not perform a multi-seed sweep.

```bash
python scripts/run_final_model_ablations.py \
  --classifier-checkpoint runs/gripper_classifier_200ms/emg_imu_best.pt \
  --root data/shubham_open data/shubham_close \
         data/mukund_open data/mukund_closed \
         data/gazania_open data/gazania_closed \
  --seed 42 \
  --split-seed 42 \
  --epochs 60 \
  --device cuda \
  --output-dir runs/final_model_ablation_seed42
```

Use the actual Stage-I/template checkpoint path if it differs. The complete
machine-readable summary is written to
`runs/final_model_ablation_seed42/ablation_results.json`. It includes current
XYZ, overall pixel error, pixel error at 50--75% and 75--100% trial progress,
200-ms future XYZ, the persistence baseline, gripper macro-F1, and parameter
count.

For the final component-only study, with future-IMU reconstruction and state
conditioning disabled in the reference model and no unimodal experiments, add:

```bash
--suite components-no-aux
```

The default variants cover modality, state conditioning, fusion, residual GRU,
local/context features, every retained supervised loss, pixel time weighting,
future-IMU horizon, the no-copying constraint, and frozen/joint/single-stage
encoder training. To continue after completed runs, add `--resume`. To run only
a subset, add for example:

```bash
--variants full emg_only imu_only no_state_conditioning no_future_imu_loss
```
