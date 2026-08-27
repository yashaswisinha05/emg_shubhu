# Cross-variate multi-scale patch model (`grid_crossvar`)

Adapts MCV-PatchTST (Qarni et al., *Sci Rep* 2026, doi:10.1038/s41598-026-67495-w)
to EMG/IMU touch prediction. Running log; newest entries at the bottom. All
numbers are `a1`, fold 0, 239 held-out trials, MPS, seed 42, mean-coordinate
baseline **433.4 px**.

## What the paper proposes

| ablation variant (UCI hourly) | MAE | vs PatchTST | params |
|---|---|---|---|
| PatchTST baseline | 0.265 | - | 1.85 M |
| + multi-scale only | 0.251 | **-5.3%** | 2.02 M |
| + cross-variate only | 0.257 | -3.0% | 1.92 M |
| + global positional only | 0.260 | -1.9% | 1.86 M |
| full MCV-PatchTST | 0.238 | -10.2% | 2.15 M (+16.2%) |

## Why this architecture, from measurements on a1

### The motivating fact: EMG and IMU errors are nearly independent

Per-trial pixel errors of the two baseline models:

| cutoff | r(err_IMU, err_EMG) | IMU | oracle per-trial best | headroom |
|---|---|---|---|---|
| 0.0s | 0.213 | 219.4 | 198.7 | **+20.7** |
| 0.2s | 0.129 | 172.8 | 146.8 | **+26.0** |
| 0.4s | 0.090 | 170.7 | 146.3 | +24.4 |
| touch | 0.167 | 152.4 | 137.0 | +15.4 |

EMG is individually ~2x worse but **wins on 22-32% of trials**, and on different
trials than IMU fails. `grid_fusion` captures none of this (153.6 vs 152.4 px at
touch). The oracle peeks at labels so it is an upper bound, but a nonzero gain at
r about 0.1 means the complementarity is real.

### Why the existing fusion cannot capture it

`GridFusionRegressor` adds a zero-initialized EMG logit residual scaled by a
learned **scalar** reliability. Testing that class of router directly - route to
EMG when IMU heatmap entropy is high:

| cutoff | AUC(IMU entropy -> EMG wins) | hard entropy gate | IMU alone |
|---|---|---|---|
| 0.0s | 0.677 | 242.5 | **219.4** |
| 0.2s | 0.584 | 176.7 | **172.8** |
| touch | 0.546 | 170.7 | **152.4** |

A scalar-confidence router is **worse than IMU alone at every cutoff**. The
headroom is per-trial and per-time, so routing must happen in the representation,
not on one scalar. That is the argument for patch-level cross-variate attention.

### Multi-scale, but coarse - the paper's direction inverts here

The EMG RMS envelope on `a1` has a median autocorrelation timescale of
**540 ms**, five times slower than the existing 108 ms patch (length 16 at
148 Hz). The signal is already smoother than the tokenization, so finer patches
would resolve noise. Fine timing carries nothing (5-fold ridge, baseline 433 px):

| feature set | median px |
|---|---|
| EMG amplitude only (4) | 421.7 |
| EMG onset latency only (4) | 427.7 |
| EMG pairwise onset differences (6) | 428.6 |

So the paper's downward scales {8,16,24} are replaced by **upward {16,32,64}** =
{108, 216, 432} ms, bracketing the 540 ms mode.

### Eight variate tokens, not 92 channels

Cross-variate attention costs O(N C^2 d). The paper has C about 7 (C^2 = 49);
92 raw channels would give C^2 = **8,464**, a 170x larger term, and would almost
certainly overfit 719 training trials. Pooling to physical entities - 4 EMG
electrodes + 4 IMU sensors - gives C = 8, C^2 = 64, matching the paper's own
cost, and yields an interpretable 8x8 map.

A caution recorded from the probe: between-sensor **relative** IMU features gave
290.4 px against 260.4 px for per-sensor absolute (both 256.6 px). IMU-internal
channel mixing is worth only a few px linearly, so this component is justified on
**EMG<->IMU** grounds, not IMU-internal grounds.

### Positional encoding keyed to seconds, not normalized index

The paper indexes a fixed-length lookback axis. These are variable-length causal
prefixes at four cutoffs, so a normalized index makes the same token mean
different physical times across trials, destroying the fixed ~60 ms
electromechanical delay. The encoding is therefore keyed to **physical seconds
remaining to the cutoff**, via Fourier features of that continuous quantity.

## Architecture

```
EMG 4ch                                  IMU 88ch
  |                                        |
  | -> 4 variate streams                   | -> 4 variate streams (by sensor, 22ch each)
  |                                        |
  +----------------+-----------------------+
                   |
   MultiScalePatchEmbedder, shared per modality
     stem -> 3 patch projections {16,32,64}, common stride 8
     -> adaptive-avg-pool align to N = min N_k
     -> softmax scale gate                            [component 1]
                   |
   + modality embedding + seconds-to-cutoff encoding   [component 3]
                   |
   cross-variate MHA over C = 8, 2 heads, ONCE, residual + LayerNorm
                                                       [component 2]
                   |
   channel-independent Transformer encoder (shared weights across variates)
                   |
   masked mean over patches -> attention pool over 8 variates -> context
                   |
   SpatialPointHead (unchanged: 8x5 heatmap + offsets + direct XY)
```

Split, preprocessing, continual prefixes, loss weights and metrics are unchanged
from `continual_attention`, so the comparison is controlled.

## Files

- `src/emg_touch/models/cross_variate.py` - `MultiScalePatchEmbedder`,
  `TimeToCutoffEncoding`, `CrossVariateBackbone`.
- `src/emg_touch/models/grid_point.py` - `GridCrossVariateRegressor`, kind
  `grid_crossvar` registered in `GRID_MODEL_KINDS`.
- `src/emg_touch/grid_training.py` - logs `variate_attention_*`,
  `cv_<from>_to_<to>` (the full 8x8 map), `cross_emg_to_imu`,
  `cross_imu_to_emg`, and `scale_*` gate weights per trial.
- `configs/crossvar.yaml` - `patch_lengths: [16,32,64]`,
  `cross_variate_heads: 2`.

## How to run

Activate the environment first. Omit `--device` to let CUDA be
selected automatically on a GPU host; use `--device mps` on Apple silicon.

```bash
conda activate smss
python scripts/run_grid_point_sweep.py --config configs/crossvar.yaml --configuration a1 --fold 0 --models grid_crossvar
```

## Verification before running

Forward + backward on synthetic batches at three trajectory lengths, including
one shorter than the longest patch:

| case | T | params | prediction | loss | grad norm | cross-attn |
|---|---|---|---|---|---|---|
| normal | 200 | 4,633,601 | (4, 2) | 17.240 | 48.82 | (4, 8, 8), rows sum 1.000 |
| short (T < 64) | 40 | 4,633,601 | (4, 2) | 11.739 | 28.73 | (4, 8, 8), rows sum 0.994 |
| long | 900 | 4,633,601 | (4, 2) | 10.855 | 28.01 | (4, 8, 8), rows sum 0.998 |

No NaNs, gradients flow, softmaxes normalized, short trajectories handled by
causal left-padding that stays masked.

Parameter context: `grid_imu` 2,049,752; `grid_emg` 2,052,247; `grid_fusion`
4,101,999; `grid_crossvar` 4,633,601 = **1.13x grid_fusion**, close to the
paper's own +16.2% overhead. If it wins, bracket with `d_model=112`
(3,552,657 = 0.87x) to separate mechanism from capacity - the lever-1 experiment
showed a +1.0 M parameter change doubled the generalization gap.

## Success criterion

Beat `grid_fusion` at **0.0s (213.7 px) and 0.2s (166.7 px)**, where the measured
oracle headroom is largest (+20.7 and +26.0 px). Touch-time is near-saturated for
kinematics and is not the target.

## MPS portability fix

The first run on Apple Silicon (Mac, MPS backend) crashed immediately:

```
RuntimeError: Adaptive pool MPS: input sizes must be divisible by output
sizes. Non-divisible input sizes are not implemented on MPS device yet.
```

`_align`/`_align_mask` originally used `F.adaptive_avg_pool1d` /
`F.adaptive_max_pool1d`, and MPS refuses non-divisible adaptive pooling -
which stride-based multi-scale patch counts rarely satisfy. The same run
completed cleanly on the Linux/CPU machine, confirming the failure was
backend-specific, not a logic bug.

Replaced both with an explicit, cached bin-partition + matmul/any
implementation (`_bin_boundaries`, rewritten `_align`, `_align_mask`) that
reproduces the same "aligned contiguous interval" semantics the paper's
Eq. 6-8 describes, without depending on any adaptive-pool kernel. A second
MPS gap surfaced immediately after (`float64` is unsupported on MPS); bin
edges are now computed as plain Python integers via floor division, which is
both exact and avoids the float64 path entirely.

Verified before accepting the fix:

- **Exact match to reshape-mean** for every divisible (source_length, length)
  pair - the unambiguous ground truth case.
- **Partition correctness** for five non-divisible pairs: bins are
  contiguous, non-overlapping, gap-free, and cover the full source range;
  spot-checked against a plain per-bin mean/any.
- Confirms `F.adaptive_avg_pool1d` was never bit-exact to begin with in the
  non-divisible case - it uses PyTorch's own overlapping-window algorithm,
  not Eq. 7's clean partition - so the new implementation is a closer match
  to the paper's stated equation than the code it replaced, not just a
  portability patch.
- **`grid_crossvar` runs the real `train_grid_model.py` script end to end on
  MPS**: 2 epochs on `a1` fold-0, train + validation + test-cutoff evaluation,
  checkpoint written, exit code 0, no NaNs.
- One caveat found and not chased further: an adhoc synthetic script that
  loops three very different trajectory lengths (200, 40, 173) through one
  model instance back-to-back, switching `.train()`/`.eval()` between them,
  triggered a separate Metal kernel assertion
  (`MPSNDArrayConvolutionA14.mm: Weights tensor and ndArray input channel
  mismatch`) not reproduced by the real training script or by a same-shape
  train-then-eval repro. Real per-fold training already varies trajectory
  length continuously across bucketed batches within an epoch and completed
  cleanly, so this is not treated as blocking, but is recorded here in case
  it resurfaces on a long multi-configuration sweep.

## Results (before the token-norm fix below)

The Linux 5-fold `a1` pooled sweep (1200 trials, 6 participants) completed
before the fix documented in the next section:

| cutoff | median px | mean px | p90 px |
|---|---|---|---|
| 0.0s | 300.0 | 359.3 | 690.3 |
| 0.2s | 194.4 | 230.0 | 433.9 |
| 0.4s | 189.4 | 228.4 | 451.5 |
| touch | 178.2 | 203.0 | 376.4 |

Not yet comparable to the `grid_fusion`/`grid_imu` fold-0-only numbers quoted
earlier in this document - those are single-fold, this is 5-fold pooled. A
matched 5-fold `grid_imu`/`grid_fusion` baseline is running to make that
comparison fair, and is superseded in priority by the fix below.

Directional error on the same pooled predictions showed a real, independently
interpretable pattern regardless of the pooling question: predictions are
inward-biased 65-81% of trials and under-predict edge targets by 8-17 points
at every cutoff (`edge_prediction_gap` -0.08 to -0.17) - classic
regression-to-center hedging, worst at the 0.2s cutoff.

## Bug found: unnormalized token-scale asymmetry biases cross-variate attention

`analyze_channel_attention.py` on that same 5-fold sweep showed the final
`variate_attention` pooling had collapsed almost entirely onto the four IMU
sensor tokens:

| variate | mean attention (touch) | top_attention_fraction |
|---|---|---|
| S8 | 0.382 | 0.487 |
| S4 | 0.205 | 0.202 |
| S12 | 0.197 | 0.138 |
| S0 | 0.185 | 0.174 |
| AD | 0.008 | 0.000 |
| LD | 0.008 | 0.000 |
| TB | 0.008 | 0.000 |
| BB | 0.008 | 0.000 |

All four EMG electrodes: essentially zero weight, never once the argmax
across 1200 trials, at every cutoff. This alone does not distinguish "EMG is
ignored" from "EMG's contribution was already absorbed into IMU tokens
upstream, through the cross-variate residual" - both would look similar at
final pooling. Checking further surfaced something else.

The multi-scale gate result was also worth recording honestly against my own
prior reasoning: I argued for coarse EMG patch scales because linear timing
features (onset, onset differences) carry no signal at any granularity. The
learned gate instead strongly prefers the *shortest* EMG scale (p16, 108 ms:
mean weight 0.77-0.78, top_attention_fraction 0.93-0.97) and gives p64
essentially nothing (~0.07, top-fraction ~0). This is not a contradiction of
the linear-feature finding - a self-attention temporal encoder integrates
across the patch sequence itself and generally prefers more, finer tokens
over fewer coarse ones, since attention does the aggregation a coarse patch
would otherwise pre-compute. "Informative timescale for a hand-built linear
feature" and "tokenization a self-attention model prefers" are different
questions; I had conflated them.

**Checked directly, on the real trained checkpoint** (epoch 14 of the local
`a1` fold-0 run, real validation data, not synthetic): token norms entering
cross-variate attention, which has no pre-attention normalization.

| | mean token norm |
|---|---|
| EMG tokens | 19.475 |
| IMU tokens | 42.975 |
| ratio | **2.21x** |

`MultiScalePatchEmbedder`'s stem ends in `ChannelLayerNorm1d`, but the
per-scale `Conv1d` projections and the gated-fusion sum that follow it do
not, so the returned `fused` tokens carry no normalized scale. Two
independently parameterized instances (`emg_embedder`, `imu_embedder`, with
different fan-ins: 2 input channels vs 44) can drift to different output
norms over training, and `nn.MultiheadAttention` has no internal pre-norm to
correct for it - `Q @ K^T` is directly biased toward whichever modality's
keys have larger norm, independent of content. This is inconsistent with the
project's own convention: `PatchTransformerEncoder` ends in
`self.norm = nn.LayerNorm(d_model)` for exactly this reason.

Given the near-total (not just skewed) collapse pattern - uniformly ~0.008
for all four EMG positions, at every cutoff, across 1200 trials - the
2.21x norm gap is very unlikely to be the sole cause; a `variate_score`
head can also learn to discount a variate by its position/modality
identity (via the added `modality_embedding`) rather than its content, which
a 2.2x norm gap alone would not fully explain. Both are real; the fix below
addresses the norm confound so the remaining collapse, if any, can be
attributed to content.

### Fix

Added `self.output_norm = nn.LayerNorm(d_model)` to `MultiScalePatchEmbedder`,
applied to `fused` before it is returned - matching
`PatchTransformerEncoder`'s existing convention. Verified on a fresh
(untrained) model: token norms are now **exactly 1.000x** (both
`sqrt(128) = 11.31`, as expected for an untrained LayerNorm with unit affine
weight). The unfixed checkpoint (epoch 14, ratio 2.21x) is preserved at
`runs/crossvar_a1_fold0_BEFORE_norm_fix/` for a direct before/after
comparison once the fixed run completes.

## Results: the fix works mechanically, and the outcome is negative

Full 5-fold `a1` sweep re-run on CUDA with the token-norm fix (1200 held-out
trials, 6 participants), against the identical sweep before the fix.

**The fix un-collapsed EMG.** Variate attention at the touch cutoff:

| variate | mean before | mean after | top-fraction before | top-fraction after |
|---|---|---|---|---|
| AD | 0.0079 | 0.0366 | 0.000000 | 0.0633 |
| LD | 0.0082 | 0.0331 | 0.000000 | 0.0292 |
| BB | 0.0076 | 0.0316 | 0.000000 | 0.0317 |
| TB | 0.0078 | 0.0342 | 0.000000 | 0.0458 |

EMG went from never once being the argmax variate across 1200 trials to
winning roughly 17% of them. The 2.21x token-norm asymmetry was real and the
LayerNorm fix genuinely restored EMG participation.

**Accuracy got worse at every cutoff:**

| cutoff | before fix | after fix | delta |
|---|---|---|---|
| 0.0s | 300.0 | 329.3 | +29.4 |
| 0.2s | 194.4 | 199.8 | +5.5 |
| 0.4s | 189.4 | 193.3 | +3.9 |
| touch | 178.2 | 188.3 | +10.1 |

Center-regression also worsened: `edge_prediction_gap` -0.134 -> -0.168 and
`fraction_errors_inward` 0.654 -> 0.673 at touch.

**Interpretation.** When EMG actually participates, it hurts. The original
collapse-to-zero was not a pathology the model suffered - it was the model
correctly learning that these four RMS-envelope channels do not carry usable
signal for this task, and the unnormalized-token bug was what allowed it to
act on that. This is consistent with every other measurement in this project:
EMG-only at 303 px against IMU's 152 px, three architecturally unrelated
encoders all plateauing in the 250-320 px band, no direction information in
fine EMG timing, and the raw 2148 Hz stream never saved to disk.

### Correction: the coarse-scale argument was retracted on bad evidence

Earlier in this document the "use coarse patch scales" reasoning was walked
back because the learned gate strongly preferred p16 (108 ms), mean weight
0.77-0.78 with top-fraction 0.93-0.97. That retraction was based on the
**buggy** model. Post-fix the EMG scale gate flips to preferring **p32**
(216 ms): mean 0.638, top-fraction 0.897, while p16 collapses to mean 0.174
with top-fraction 0.0008. The p16 preference was an artifact of unnormalized
token scales feeding the gate. The original argument from the ~540 ms
envelope autocorrelation timescale was closer to correct than its retraction.

### Outstanding comparison

The numbers above are 5-fold pooled (1200 trials). Every `grid_imu` /
`grid_fusion` number quoted earlier in this document is fold-0 only (239
trials), so the two are not directly comparable. A matched 5-fold
`grid_imu` / `grid_fusion` run on `a1` is required to state the size of the
gap rather than only its direction.

## Log

- **2026-08-26 12:38** - Session restart had killed the previous background
  runs; `grid_emg_first` never finished and its chained control never started.
  Restarted.
- **2026-08-26 12:40** - Measured EMG envelope timescale (540 ms), fine-timing
  feature probes, EMG/IMU error decorrelation, oracle headroom, and the failure
  of scalar-confidence routing.
- **2026-08-26 12:52** - Implemented `grid_crossvar`; smoke-tested at three
  trajectory lengths; chained the `a1` fold-0 run behind `grid_emg_first`.
