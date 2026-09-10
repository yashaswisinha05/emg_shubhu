# Frozen-classifier state-attention hybrid

This model retains only the empirically reliable open/close classifier from
`gripper_neuromuscular_future_v1`. That classifier is frozen and supplies both
the deployed class result and the soft state condition used by the V2
XYZ/future/pixel model. The old classifier's pixel head is not used.

The model converts V2-normalized inputs into the frozen classifier's original
normalization before evaluating it. Pixel axes remain independent:
`x_norm=x_px/canvas_width` and `y_norm=y_px/canvas_height`; training loss is
computed after converting both axes back to physical pixels.

```bash
python scripts/train_neuro_classifier_state_attention.py \
  --classifier-checkpoint \
    runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --device cuda --epochs 60 \
  --output-dir runs/neuro_classifier_state_attention
```

The resulting deployment checkpoint is
`runs/neuro_classifier_state_attention/emg_imu_best.pt`.

Run it on a recorded trial with the actual 16:9 canvas dimensions:

```bash
python scripts/infer_gripper_state_attention.py \
  --checkpoint runs/neuro_classifier_state_attention/emg_imu_best.pt \
  --trial-csv data/unseen_recording/trial_001.csv \
  --device cuda --canvas-px 1920 1080
```

Streaming callers may omit `canvas_px`; the default is `(1920, 1080)`.
Normalized pixel output is converted independently as
`(x_px, y_px) = (x_norm * 1920, y_norm * 1080)`.
