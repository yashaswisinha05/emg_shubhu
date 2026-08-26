# Exact Hugging Face PatchTST study

This experiment uses the installed
`transformers.models.patchtst.modeling_patchtst.PatchTSTModel` as the temporal
backbone. It does not use the project's custom patch transformer.

The controlled `mix7/fold-0` study trains three randomly initialized models:

- `hf_patchtst_imu`: 88 raw and calibrated IMU features;
- `hf_patchtst_emg`: four EMG channels;
- `hf_patchtst_fusion`: all 92 channels in one PatchTST encoder.

All three retain the same causal preprocessing, continual prefixes, screen loss,
split and evaluation metrics as the previous continual-attention experiment.
PatchTST receives a fixed 256-sample (1.728-second) right-aligned context. The
most recent samples are retained when a recording is longer than the context.

The PatchTST configuration enables its built-in temporal attention and
cross-channel attention. A CLS token from each channel is flattened in the same
way as Hugging Face's classification head, then projected into the existing
grid-and-coordinate task head. The reported X/Y remains the direct coordinate;
the 8x5 grid is an auxiliary anti-collapse objective.

Run from the `emg_touch` directory:

```bash
conda activate smss

caffeinate -i python scripts/run_hf_patchtst_study.py \
  --config configs/hf_patchtst_exact.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

Each model writes `model_report.json`, including the fully qualified backbone
class, installed Transformers version, exact `PatchTSTConfig` and parameter
counts. Results are written below:

- `runs/hf_patchtst_exact/mix7/fold-0/`
- `evaluation/hf_patchtst_exact/`
