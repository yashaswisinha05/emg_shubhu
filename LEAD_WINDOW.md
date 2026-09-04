# Lead-window pointing: what was measured, what changed, what failed

This documents the diagnostic-and-model sequence that produced the `176 px`
screen result and the `artifacts/tracked_cache_posture` feature set.

**Where this sits.** It precedes the soft-routed complete-reach model
documented in [`README.md`](README.md), which is built on top of it and
inherits both artefacts from it: `--cache-dir artifacts/tracked_cache_posture`
(the gravity-referenced posture features below) and `--lead-window-ms` (the
training change that produced the result). Nothing here is required to
reproduce that model; it is the record of how its operating point was arrived
at, and of the directions that were closed off along the way.

Every entry states the reason it was built and the number it produced, since
neither is useful without the other. Several entries record failures or
withdrawn conclusions. Those are kept deliberately: the negative results
constrain what is worth trying next as much as the positive ones do, and two
of them look like successes if only their final numbers are read.

---

## The problem this started from

Six architectures had landed between `390` and `430 px` on the screen-touch
task, ranked roughly *inversely* with how much machinery each one added:

| Model | Screen error |
| --- | ---: |
| Small GRU, direct regression | 392.7 px |
| Patch transformer + grid/offset head | 406.3 px |
| VAE + gradient-reversal latent split | 426.1 px |

Adding capacity was reliably making it worse. The next step was therefore to
measure what the data permits before building a seventh model.

---

## Diagnostics built before changing any model

| Script | Question it answers | What it found |
| --- | --- | --- |
| [`diagnose_screen_geometry.py`](scripts/diagnose_screen_geometry.py) | If the 3-D endpoint were known perfectly, what pixel error remains? | Pooled across sessions **331 px**; within a session **54.6 px**. Reach start positions vary by **163.9 cm**. |
| [`evaluate_by_lead_time.py`](scripts/evaluate_by_lead_time.py) | How does error depend on how far ahead the prediction is made? | `100 ms`: **219.9 px** mean / `191.7 px` median (**+50.6%** over guessing). `1000 ms`: **440.2 px** (**+1.1%**, chance). |
| [`diagnose_horizon_feasibility.py`](scripts/diagnose_horizon_feasibility.py) | Can this data support a longer forecast horizon at all? | Trial length p10/median/p90 = `1269`/`1362`/`1985 ms`. Onset is **0 ms at every percentile**. A `1000 ms` horizon is 100% feasible; `2000 ms` is **8.4%**. |
| [`plot_trajectory_rollout.py`](scripts/plot_trajectory_rollout.py) | How far can EMG+IMU dead-reckon global position with no tracker feedback? | Drift `5.65` → `8.28` → `17.23` → `19.12 cm` across the reach. |

Three of these changed what was built afterwards.

**The lead-time curve is the reason for the main result.** Every number
reported before it averaged cutoffs drawn uniformly between onset and touch,
pooling "where will they touch?" asked `1.2 s` out with the whole reach
uncommitted against the same question asked `120 ms` out with the hand nearly
there. Those cannot have the same answer, and one average was hiding a curve
running from strong signal to chance:

| Lead before touch | Mean | Median | Versus guessing |
| ---: | ---: | ---: | ---: |
| 100 ms | 219.9 px | 191.7 px | +50.6% |
| 200 ms | 230.4 px | 211.8 px | +48.3% |
| 400 ms | 334.4 px | 318.5 px | +24.9% |
| 1000 ms | 440.2 px | 426.1 px | +1.1% |

**Onset is `0 ms` at every percentile**, so recordings begin essentially
already in motion: the reach *is* the trial, with no still pre-buffer to
borrow horizon room from. This is why a `2 s` forecast horizon is unavailable
on this data regardless of model.

**Dead reckoning stays under `~8 cm` through the first half of a reach and
reaches `~19 cm` by touch.** That is a concrete re-anchoring interval —
roughly every `0.6`–`0.8 s` — rather than a general claim that wearable
tracking does or does not work.

---

## The change that produced the result

`--lead-window-ms` on
[`train_grid_reach_model.py`](scripts/train_grid_reach_model.py) restricts
training cutoffs to a window before touch instead of drawing them uniformly.

**Logic.** Measured on the sampler itself, the majority of training cutoffs
landed beyond `600 ms` before touch — about `57%` for a typical `1.4 s` reach,
with a median cutoff near `700 ms`. The lead-time sweep had just shown that
regime is at chance. Most of the gradient was therefore spent on examples
carrying no recoverable signal, and the only way to fit them is to hedge
toward the mean target — a hedge the model then also applies near contact,
where the signal is strong.

**Result.** `176.3 px` test error, **+60.4%** against the mean-target
baseline, from `390`–`406 px`. This was the first result under the `200 px`
goal.

```bash
python scripts/train_grid_reach_model.py \
  --root "/media/nahar3/Extreme SSD/emg2pose_dataset/emg_imu_vive" \
  --config configs/tracked_grid_within.yaml \
  --cache-dir artifacts/tracked_cache_posture \
  --device cuda --epochs 20 --inputs emg+imu \
  --lead-window-ms 50 400 \
  --output-dir runs/grid_leadwindow_emg_imu
```

Validation and test use the same window, so the reported score *is* the
operating point and is not comparable to uniform-cutoff runs. The script
prints this on startup rather than leaving it to be inferred.

---

## Gravity-referenced IMU posture features

Enabled by `data.imu_posture_features`; produces
`artifacts/tracked_cache_posture`. Implemented as `imu_posture_bank` in
[`tracked_dataset.py`](src/emg_touch/data/tracked_dataset.py).

**Logic.** Integrating acceleration to locate the hand drifts — `~19 cm` over
one reach, measured above. A low-passed accelerometer instead reads the
direction of gravity, which does not drift at all: absolute limb tilt at every
instant, no integration. Per sensor it emits tilt change from the trial's own
resting posture; per pair, the angle between two sensors' gravity directions,
which is a joint angle when the sensors sit on different segments. 24 IMU
channels become 46.

**Measured, not assumed.** Re-running the bank on one recording rotated as if
the band had been worn 52° around: the angles are invariant (`3e-06` and
`3e-07`), the 3-vectors are **not** — they rotate by 87% of their own scale.
The vectors are kept anyway, because direction is what separates a leftward
reach from a rightward one and the scalar angles discard exactly that. This
cannot be fixed by cleverer referencing: rotation about the gravity axis is
unobservable to an accelerometer. They are dependable within a session and
suspect across one.

Also verified: strictly causal (truncating the recording changes earlier
features by exactly `0.00e+00`); the baseline window is taken from leading
samples and never from the tracker-derived movement onset, so it cannot
smuggle the tracker into the encoder; and the features track reach progress at
`r = 0.97`.

---

## Follow-ups built on that model

### Uncertainty at every instant

[`train_uncertainty_head.py`](scripts/train_uncertainty_head.py) trains a
~17K-parameter `UncertaintyHead` on a fully frozen base.

**Logic.** A sampled-latent VAE was rejected for this. The earlier
`PointingIntentVAE` reached `426 px` because KL pushes sigma toward the
prior's `1.0` whenever reconstruction does not fight it, injecting growing
noise into the decoder every step. What "mu and sigma at every instant" needs
is uncertainty on the *point prediction*, which requires no sampling at all.
The head is a separate module on a base whose parameters are never in the
optimizer, because this repository's own `SpatialPointHead.log_sigma` records
a case where detaching a value checked clean on `.grad` but still cost
`184` → `204 px` in a real 35-epoch run: a shared optimizer step and shared
gradient-clip norm couple parameters together even where one term's gradient
into a given weight is exactly zero.

**Result.** Coverage `65.5%` at 1σ (target `~68%`) and `93.1%` at 2σ (target
`~95%`), with the frozen base's own point error unchanged at `176.5 px`
against its `176.3 px` — direct evidence the freeze held.

### Making EMG measurably matter

[`train_vae_discriminator_model.py`](scripts/train_vae_discriminator_model.py)
adds an IMU-only critic and a one-directional hinge.

**Logic.** A gradient-reversal discriminator on "which modality produced this
latent" would train the encoder to make EMG's presence *undetectable* —
EMG-invariance, the opposite of the goal — and it would fail silently, since
training converges and the discriminator looks fooled while the encoder
quietly discards EMG. The critic instead reads `imu_context.detach()` through
its own head to measure what IMU alone can do, and a hinge requires the fused
prediction to beat that ceiling by a margin, with gradient flowing only
through the fused path.

**Result.**

| Lead window | Fused | IMU-only critic | EMG margin |
| --- | ---: | ---: | ---: |
| 50–400 ms | 186.6 px | 194.3 px | **+7.7 px** |
| 50–1000 ms | 339.8 px | 343.7 px | **+3.9 px** |

EMG's edge is small but survives both regimes. It shrinks proportionally
(`4.1%` → `1.1%`), consistent with EMG's advantage being concentrated near
contact rather than spread evenly across the reach.

### Letting the model choose its own horizon

[`adaptive_horizon.py`](src/emg_touch/models/adaptive_horizon.py), with
[`train_adaptive_horizon_model.py`](scripts/train_adaptive_horizon_model.py)
and [`train_adaptive_lead_model.py`](scripts/train_adaptive_lead_model.py),
predicts `tau` — how far ahead this particular trial can be forecast —
instead of fixing it by hand. Built as a differentiable reward rather than a
discriminator: "predict further ahead" has no natural adversary, it is a
trade-off inside one network between more reach and more accuracy.

**Result: mechanism verified in isolation, not yet run on real data.**
Interpolation is exact, gradient reaches the head, and a ramp is required —
un-ramped, two synthetic groups with genuinely different achievable accuracy
collapsed to identical `tau` (`100.0` vs `100.0`) as the constant reach-reward
saturated the sigmoid before the accuracy-dependent gradient could
differentiate anything; ramped, the same groups separated to `99.7` vs `10.4`.

### Live replay and visualisation

[`live_inference.py`](scripts/live_inference.py) walks a trial sample by
sample with a rolling causal buffer, emitting a prediction at every instant
rather than at one cutoff.
[`live_prediction_ui.py`](scripts/live_prediction_ui.py) is its visual
counterpart, reusing the data-collection app's own dark theme and target
animations with a cyan marker tracking the causal prediction against the red
true touch point.

Causality is verified by tampering rather than asserted: replacing every
sample after a chosen cutoff with noise leaves every prediction strictly
before that cutoff bit-identical, and changes every prediction at or after it.

---

## Failures and withdrawn conclusions

Recorded because they were measured, and because each closes off a direction
that otherwise looks reasonable. Two of them read as successes if only the
final numbers are looked at.

| Attempt | Outcome |
| --- | --- |
| Within-session training would collapse toward the `54.6 px` geometric ceiling | **Wrong, and withdrawn.** Within-session gave `390 px` against `406 px` held out by session — the frame shift costs about `16 px`, not the `~280 px` inferred from it. The frame shift caps approaches that go *through* 3-D coordinates; this model maps EMG+IMU straight to pixels and never touches that pathway. |
| A compressed `256`→`40` bottleneck before the point head, as capacity regularisation | **Removed after measurement.** An 8-example overfit check that `GridReachModel` collapses to `24 px` plateaued at `250 px` through the bottleneck — it constrained even trivial memorisation. |
| [`finetune_emg_necessity.py`](scripts/finetune_emg_necessity.py): train the already-reported `without_emg` gap directly | **Failed on the real run.** Validation was worse every one of 10 epochs (`244.7` → `259.8 px` against a `212.2 px` baseline) and `train_student_phase`'s safeguard reverted the entire phase. The unchanged final numbers were a *reverted* run, not a neutral one. `--freeze-decoder` and explicit revert detection were added afterwards. |
| A `2 s` forecast horizon | **Not available on this data.** Only `8.4%` of trials can supply one valid cutoff. Requesting it does not error — it silently trains on zero usable batches (`loss=0.0000`, `val=nan`). |

One latent bug was found and fixed along the way. `make_window` drops trials
too short to supply a cutoff, but the training loop passed `batch["session"]`
unfiltered into the adversarial loss, so the labels and the model outputs
disagreed on batch size whenever a row was dropped. Invisible at the `254 ms`
default, where rows are essentially never dropped, and near-universal at
`1000 ms`.

---

## Honest reading of these numbers

The pointing result is a **lead-time-conditional** claim, not a single
average: `~176 px` at the late window, degrading to chance by `~1 s` before
touch. Reporting one averaged number across all leads pools two regimes that
do not have the same answer, which is precisely the mistake the lead-time
sweep exposed.

EMG's marginal contribution to the *screen* target is small but real and
reproducible — `+7.7 px` and `+3.9 px` by the critic margin at two different
lead windows, and `+25.5 px` by the removal ablation in the channel/horizon
model. It is not comparable to IMU's contribution, which is several times
larger by the same measurements.
