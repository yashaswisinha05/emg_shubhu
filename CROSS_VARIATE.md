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

## Results

_Pending - the a1 fold-0 sweep is running on both the Linux/CPU machine and
locally on MPS after the fix above._

## Log

- **2026-08-26 12:38** - Session restart had killed the previous background
  runs; `grid_emg_first` never finished and its chained control never started.
  Restarted.
- **2026-08-26 12:40** - Measured EMG envelope timescale (540 ms), fine-timing
  feature probes, EMG/IMU error decorrelation, oracle headroom, and the failure
  of scalar-confidence routing.
- **2026-08-26 12:52** - Implemented `grid_crossvar`; smoke-tested at three
  trajectory lengths; chained the `a1` fold-0 run behind `grid_emg_first`.
