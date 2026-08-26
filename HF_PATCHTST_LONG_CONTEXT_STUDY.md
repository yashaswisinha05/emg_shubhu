# Exact PatchTST long-context control study

This is the complementary full-history control for `mix7/fold-0`. It keeps the
same split, preprocessing, continual prefix construction, task head, loss and
metrics as the 256-sample exact PatchTST study, but increases the fixed context
to 2,304 samples (15.552 seconds at 148.148 Hz).

The exact Hugging Face `transformers.PatchTSTModel` receives 64-sample patches
with stride 32, producing 71 patches per channel. All prefixes are right-aligned
and causally padded. Any recording longer than 15.552 seconds is excluded by the
data policy instead of being silently truncated.

Run from `emg_touch`:

```bash
conda activate smss

caffeinate -i python scripts/run_hf_patchtst_study.py \
  --config configs/hf_patchtst_long_context.yaml \
  --configuration mix7 \
  --fold 0 \
  --device mps
```

Outputs:

- `runs/hf_patchtst_long_context/mix7/fold-0/`
- `evaluation/hf_patchtst_long_context/`

