# EMG-first touch prediction

Running log of an attempt to make EMG the dominant signal and admit IMU only
where it demonstrably buys accuracy. Every number below is `a1`, fold 0, 239
held-out trials, MPS, seed 42. Newest entries are appended at the bottom.

## Goal

The deployment target is EMG-only. In every experiment so far, IMU carries the
prediction and EMG contributes almost nothing. This study asks whether that is a
property of the signal or of the architecture, and changes the architecture to
find out.

## Why: what the baseline actually shows

`a1` fold-0, `configs/continual_attention.yaml`. Train-mean baseline on the same
test trials is **433.4 px**.

| cutoff | n | `grid_imu` | `grid_emg` | `grid_fusion` |
|---|---|---|---|---|
| 0.0s | 239 | 219.4 | 390.4 | **213.7** |
| 0.2s | 239 | 172.8 | 350.6 | **166.7** |
| 0.4s | 58 | 170.7 | 317.7 | **165.3** |
| touch | 239 | **152.4** | 303.2 | 153.6 |

Median pixel error. Two facts drive this study:

1. **EMG is ~2× worse than IMU at every cutoff**, recovering only ~30% of the
   distance from the 433 px baseline to the IMU result.
2. **Fusion ≈ IMU.** Adding EMG to IMU changes the touch-time result by 1.2 px.

The same pattern holds on `mix7` and, per `evaluation/hf_patchtst_exact/`, holds
across three unrelated encoders (TCN 314 px, custom patch 252-281 px, HF
PatchTST 320 px). That consistency is what makes an architecture explanation
worth testing before concluding the signal is empty.

## Constraint found in the data: there is no raw EMG

The source CSVs carry four EMG columns only (`EMG RMS 1_S0/S4/S8/S12`); the
`.npy` is the same 52 columns and the `.pkl` is metadata. But the recording
reports:

```
sample_rate_hz          =  148.148     # what was saved
sample_rate_hz_declared = 2148.148     # what the sensor was streaming
```

The hardware ran at ~2.1 kHz and only the 148 Hz RMS envelope was written to
disk. Median frequency, MUAP shape, zero-crossing and waveform length are
unrecoverable for this dataset. **Saving raw EMG in future sessions is the
highest-leverage change available and nothing here substitutes for it.**

## Defect found in the code: the EMG branch is blindfolded

`EMGEndpointBackbone` had two encoders, both reading through `gather_tail`:
0.5 s and 0.3 s, right-aligned. `IMUGridBackbone` has a full-prefix encoder
*plus* a 0.5 s tail encoder. So EMG saw only the end of the trajectory while IMU
saw all of it.

Measured on the `a1` baseline `grid_emg` predictions:

- median touch-prefix duration: **1.41 s**
- trials longer than the 0.5 s EMG window: **100%**
- median fraction of the prefix EMG never saw: **64%**

And the learned lookback gate is pinned against its ceiling — it shifts
monotonically toward the longest window it is permitted to have as more signal
becomes available:

| cutoff | w(0.5 s) | w(0.3 s) |
|---|---|---|
| 0.0s | 0.444 | 0.556 |
| 0.2s | 0.524 | 0.476 |
| 0.4s | 0.605 | 0.395 |
| touch | **0.607** | 0.393 |

Two supporting signals, both consistent with EMG being starved rather than
merely weak:

- **Channel attention learned nothing**: AD/LD/BB/TB weights 0.256 / 0.249 /
  0.242 / 0.253, uniform to within 1.4% at every cutoff.
- **The heatmap is nearly uniform**: EMG entropy 3.19-3.42 against a 40-cell
  maximum of ln(40) = 3.689; IMU sits at 1.99.

This matters for EMG specifically because the informative part of a reach is the
anticipatory pre-movement burst, which leads motion by the electromechanical
delay and therefore sits at the *start* of the trajectory.

## Changes made

### Lever 1 - full-prefix EMG encoder

`src/emg_touch/models/grid_point.py`, `EMGEndpointBackbone`. Behind
`model.emg_full_context` (default `false`, so existing checkpoints still load):

- adds a third `MaskAwarePatchEncoder` over the entire causal prefix;
- widens the lookback gate from 2 to 3 windows, ordered `[full, 500ms, 300ms]`;
- generalises the fusion input to `[selected, pairwise differences, quality]`,
  so dimensions follow the window count instead of being hard-coded.

### Lever 2 - `grid_emg_first`, a new model kind

New `GridEMGFirstRegressor`: `GridFusionRegressor` with the modalities swapped.
EMG is the base predictor; IMU enters as a **zero-initialized, gated residual**:

```python
logits = emg_logits + gate * imu_residual_logits
```

At initialization the model is *exactly* `grid_emg` (verified: predictions are
bit-identical to the EMG base before training). IMU can only add what EMG cannot
supply, and the learned gate becomes a direct measure of how much IMU the task
requires.

`src/emg_touch/grid_training.py`:

- `loss.imu_gate_weight` charges an L1 price on the gate, keeping IMU a last
  resort;
- per-trial `imu_gate` is written to `predictions.csv`;
- lookback-weight logging generalised to N windows (`emg_weight_full`,
  `emg_weight_500ms`, `emg_weight_300ms`), preserving the legacy 2-window names.

### Config

`configs/emg_first.yaml`, a copy of `continual_attention.yaml` with
`experiment_name: emg_first`, `emg_full_context: true`,
`imu_gate_weight: 0.05`, and sweep budgets for the new kind. Everything else -
split, preprocessing, prefixes, loss weights, metrics - is unchanged, so the
comparison against the baseline table above is controlled.

## How to run

```bash
caffeinate -i /opt/homebrew/Caskroom/miniconda/base/envs/smss/bin/python scripts/run_grid_point_sweep.py --config configs/emg_first.yaml --configuration a1 --fold 0 --device mps --models grid_emg grid_emg_first
```

Outputs land in `runs/emg_first/a1/fold-0/` and `evaluation/emg_first/`.

## Verification before running

Forward + backward on a synthetic batch, both configs:

| config | kind | full ctx | params | lookback | loss | grad norm |
|---|---|---|---|---|---|---|
| `continual_attention` | `grid_emg` | no | 2,052,247 | (B, 2) | 13.57 | 26.50 |
| `emg_first` | `grid_emg` | yes | 3,032,352 | (B, 3) | 11.39 | 30.74 |
| `emg_first` | `grid_emg_first` | yes | 5,099,129 | (B, 3) | 11.50 | 31.13 |

`grid_emg_first` at init: `imu_gate` mean 0.513, and `base_prediction ==
prediction` exactly, confirming the zero-init residual.

## Results

### Lever 1 - full-prefix EMG encoder: partial, and confounded

`grid_emg`, `a1` fold-0, median pixel error. Mean-coordinate baseline 433.4 px;
`grid_imu` reference 219.4 / 172.8 / 170.7 / 152.4 px.

| cutoff | n | 2-window baseline | 3-window full-prefix | delta |
|---|---|---|---|---|
| 0.0s | 239 | 390.4 | **357.9** | **-32.5** |
| 0.2s | 239 | 350.6 | **343.7** | **-6.8** |
| 0.4s | 58 | 317.7 | 328.9 | +11.2 |
| touch | 239 | 303.2 | 329.5 | +26.3 |

**It helps early and hurts late.** That is a coherent shape: at the 0.0s cutoff
the prefix is short, so the full encoder adds genuinely new signal; by touch the
prefix is 1.41 s of mostly uninformative pre-movement baseline, and summarising
it dilutes the endpoint information the tail encoders were built to capture.

**The gate confirms the branch was starved.** With a full-prefix window
available, the model spends its budget on it, and the 0.5 s window collapses to
near-irrelevance:

| cutoff | w(full) | w(0.5 s) | w(0.3 s) |
|---|---|---|---|
| 0.0s | 0.397 | 0.060 | 0.543 |
| 0.2s | 0.456 | 0.078 | 0.466 |
| 0.4s | 0.461 | 0.118 | 0.422 |
| touch | **0.473** | 0.102 | 0.425 |

So the diagnosis was right - the branch wanted more context and was being denied
it. The intervention is nonetheless not a win at touch, and the reason is
visible in the training curves:

| model | epochs | best epoch | train@best | val@best | gap |
|---|---|---|---|---|---|
| 2-window baseline | 35 | 25 | 11.224 | 11.760 | **+0.536** |
| 3-window full-prefix | 35 | 31 | 10.334 | 11.545 | **+1.211** |

The generalisation gap more than doubled. The third encoder added ~1.0 M
parameters (2,052,247 -> 3,032,352) on 719 training trials, and much of the
extra capacity went into memorising. Validation loss did improve (11.760 ->
11.545), but the selection metric is a cutoff-weighted mean dominated by the
early prefixes, which is exactly where the gain sits - so checkpoint selection
preferred a model that is worse at touch.

**Verdict: context and capacity are confounded.** Running the capacity-matched
control before drawing any conclusion.

### Capacity-matched control

`configs/emg_first_matched.yaml`: three EMG windows at `d_model=104`,
`ffn_dim=312` -> **2,040,816** parameters, 0.99x the 2,052,247 of the 2-window
baseline. Same data, same loss, same schedule. If the early-cutoff gain survives
at constant capacity, the full-prefix context is real; if it evaporates, the
gain was capacity all along.

_Pending._

### Lever 2 - `grid_emg_first`

_Pending._

## Log

- **2026-08-25 17:33** - Ran `continual_attention` on `a1` fold-0 to establish a
  baseline. Completed in 15.4 min, zero failures. Numbers in the table above.
- **2026-08-25 17:45** - Confirmed no raw EMG exists in the dataset; found the
  declared-vs-saved sample-rate discrepancy.
- **2026-08-25 17:47** - Found the EMG full-prefix asymmetry and the saturated
  lookback gate.
- **2026-08-25 17:49** - Implemented levers 1 and 2, smoke-tested, launched
  `emg_first` on `a1` fold-0.
