# Causal patch-transformer reach/grasp experiment

This is a separate architecture experiment. The original TCN code, checkpoints
and outputs remain valid. Run:

```bash
git pull origin main
bash scripts/train_reach_grasp_patch_transformer.sh
```

It uses the existing dataset path, annotation-aware labels, corrected mask-aware
evaluation, and trains independent IMU-only, EMG-only and EMG+IMU models. Outputs
go to `runs/reach_grasp_patch_transformer_seed42`. Batch size defaults to 4;
reduce it to 2 if CUDA memory is insufficient. No pretrained download or new
Python package is needed.

## What “Chronos-style” means here

This uses multivariate overlapping patches, sinusoidal temporal positions and a
four-layer Transformer per modality, inspired by patch-based time-series models.
At the 100 Hz processed rate, a 16-sample patch covers 160 ms and a four-sample
stride emits a context token every 40 ms. A parallel causal local-convolution
path retains frame-level changes. EMG and IMU receive equal latent width (128),
then a learned per-frame gate and interaction layer fuse them. Event and current
VIVE XYZ heads remain separate.

Every patch is left-padded and ends at its output timestamp. Transformer attention
is causally masked; patch context is held forward between patch endpoints. Later
samples cannot affect earlier outputs. Labels, true events and VIVE values do
not enter the wearable encoder.

This is **not the pretrained Amazon Chronos-2 model**. Chronos-2 is a 120M-parameter
universal forecasting model with 16-sample patches. Its public high-level
`embed()` method is no-gradient and produces final encoder embeddings. Applying
it causally to this frame-level detector would require recomputing embeddings for
thousands of rolling prefixes or using whole-trial embeddings that leak future
data. We do neither. “Chronos-style” must not be reported as “pretrained Chronos.”

The architecture has materially more capacity than the TCN, so the scientific
comparison is whether it improves held-out event/holding/position metrics—not
whether it sounds more sophisticated. Use the same seed first, then repeat both
architectures across seeds. Parameter count and training time differ.

The gate is trained but does not backpropagate its routing choice into either
encoder through the gate-input path; task gradients still train both encoded
feature paths. Gate weights are not causal attribution. Modality ablations and
independently trained unimodal models remain the evidence for sensor contribution.

Full-trial inspection supports this checkpoint format:

```bash
python scripts/inspect_grasp_triggers.py \
  --run-dir runs/reach_grasp_patch_transformer_seed42 \
  --device cuda --split validation --plots 6 \
  --output-dir runs/grasp_patch_transformer_inspection
```

The checkpoint still uses the original fixed trial split procedure. To compare
membership directly, compare `splits.json` with the TCN run. Test data has already
been explored in this project; this is architecture development, not a fresh
confirmatory test.

The attached TimeSiam paper motivates learning temporally structured patch
representations, but this implementation does not reproduce TimeSiam's Siamese
pretraining or lineage embeddings. A self-supervised pretraining ablation should
be added only after the supervised architecture establishes a reliable baseline.
