# Model history: why the project ended at future reach–grasp intent

This is the short history of the project. It records what each major model
family was meant to test, what we learned, and why the next model was built.
The numbers below are development results, usually from one seed, unless stated
otherwise. They are not yet a multi-participant paper estimate.

## The rule that stayed fixed

The deployable models receive only causal wearable measurements:

```text
past and current EMG + past and current IMU -> prediction
```

VIVE position and orientation are labels during training and comparison data
during evaluation. They are never encoder inputs at deployment. Human grasp
and release timestamps are also labels, never inputs.

## Phase 1: pointing and complete-reach prediction

The original question was whether EMG+IMU could predict a screen destination
and the 3D hand path of a pointing movement.

| Stage | Model or experiment | Why it was tried | What it taught us |
| --- | --- | --- | --- |
| 1 | Small GRU and direct regression | Establish a simple wearable baseline. | Errors around `390–400 px` showed that adding capacity alone would not solve the task. |
| 2 | Grid/offset and patch models | Represent the screen spatially instead of as two unrestricted coordinates. | They remained around `400 px`; architecture complexity was not the main limitation. |
| 3 | VAE and disentangled latent models | Separate intent from kinematics and regularize limited data. | The stochastic bottleneck often worsened point error (`~426 px`). KL regularization and sampled noise made precise regression harder. |
| 4 | Lead-window model | Stop mixing predictable near-touch frames with nearly impossible one-second-ahead frames. | Restricting training to `50–400 ms` before touch reduced error to about `176 px`. This established that prediction quality depends strongly on lead time. |
| 5 | Honest wearable-only trajectory test | Remove VIVE from the encoder and test whether wearables predict the individual trial. | Mean trajectory error was `4.40 cm`, versus `6.78 cm` for the average-reach baseline: `35.1%` better. |
| 6 | Latent teacher–student distillation | Use a VIVE-aware teacher during training while keeping the student wearable-only. | It improved representation learning, but a large teacher–student gap remained. The teacher was an oracle, not a deployable result. |
| 7 | Channel/horizon model | Learn which EMG channel matters and how far ahead the model can predict. | A representative run reached `198.1 px`; removing EMG cost `36.8 px`, while removing IMU cost `233.9 px`. EMG helped screen intent, but IMU dominated motion. |
| 8 | Complete-reach dual heads | Predict final screen point and the entire 3D path together. | A first version reached `176.1 px` at touch but `11.56 cm` path error. Endpoint accuracy did not automatically produce a good path. |
| 9 | Task-separated, asymmetric and deterministic heads | Prevent the screen and 3D objectives from damaging one another. | Hard separation improved 3D but hurt screen prediction. Removing the VAE alone also made results worse. |
| 10 | Soft-routed complete reach | Share one encoder while controlling how strongly each task changes each latent route. | It gave about `203.4 px` and `5.91 cm`, a better compromise than hard separation. This became the strongest complete-reach architecture. |
| 11 | Goal prototypes and EMG residual/acceleration heads | Make EMG describe goal, direction and muscle-driven acceleration on top of an IMU motion base. | Gains in 3D were small: acceleration improved path by only `0.05 cm` and endpoint by `0.21 cm`; removing/shuffling EMG cost `0.35/0.55 cm`. |
| 12 | IMU correction and EMG delay sweep | Test whether time-shifted EMG adds information to a strong IMU baseline. | IMU correction reached `109.17 px`; EMG+IMU was `109.25 px`, and zeroed/shuffled EMG was nearly identical. In this task, the apparent improvement came from retraining the correction head, not EMG. |

### Why we stopped forcing EMG into pointing

The incremental-information study found an overall EMG gain of only
`+0.17 px`, with a trial-bootstrap 95% interval of `[-0.06, +0.40] px`.
There were small recording- and lead-specific effects, but not a convincing
general improvement over IMU. This changed the scientific question: instead of
forcing EMG to duplicate IMU kinematics, use EMG for the interaction changes it
is physiologically suited to reveal.

The detailed pointing history is in [`LEAD_WINDOW.md`](LEAD_WINDOW.md), and the
selected complete-reach design is described in the main [`README.md`](README.md).

## Phase 2: reach, grasp, hold and release

The new experiment has four equally spaced forearm EMG sensors, four IMUs,
VIVE end-effector pose, and human grasp/release timestamps. The task is now:

```text
EMG + IMU -> current pose + orientation + holding state
          -> grasp/release transitions
          -> future pose, orientation and interaction intent
```

| Stage | Model or experiment | Why it was tried | Decision |
| --- | --- | --- | --- |
| 13 | Causal reach–grasp baseline | Create a clean model for holding, grasp, release and current XYZ with missing-data masks. | Kept as the minimum baseline. Independent EMG, IMU and fusion models expose which modality helps. |
| 14 | Annotation-uncertainty labels | Human button annotations may be roughly `±200–500 ms`. | Kept. Soft event labels and tolerance sweeps avoid pretending the labels are exact. |
| 15 | Chronos-style causal patch transformer | Give pose and holding a longer temporal context without importing a large pretrained forecasting model. | Kept. Domain-specific causal patches were more defensible than unrelated pretrained time-series weights. |
| 16 | Hybrid local event head | Patch aggregation can blur short transitions, so grasp/release also need frame-resolution causal features. | Kept. Pose/holding use long patch context; transitions use a local high-resolution head plus holding-transition evidence. |
| 17 | Continuous 6D orientation head | Position alone cannot control an end effector. Direct Euler or quaternion regression has discontinuities. | Kept. A 6D rotation representation predicts full orientation and reports geodesic and yaw error. |
| 18 | Robust model | Simulate sensor gain/noise/dropout and predict event time and pose uncertainty. | Useful for robustness and control experiments, but uncertainty does not replace accurate prediction. |
| 19 | Masked EMG reconstruction | Encourage the EMG encoder to retain physiological structure despite limited labelled data. | Kept as an ablation. Reconstruction is training-only and is valuable only if downstream test metrics improve. |
| 20 | Dedicated EMG grasp-onset model | Test the narrowest task where EMG should be strongest. | Kept as a diagnostic and optional grasp controller; it does not supply arm pose. |
| 21 | One-second future-intent model | Turn recognition into intent prediction for both motion and interaction. | This is the current main model. It unifies the strongest causal encoder with current and future SE(3) and event-time heads. |

## Why the current model looks the way it does

The current model is deterministic rather than a VAE. Precise pose regression
did not benefit consistently from latent sampling, while the patch/local split
had a direct reason tied to the signals:

```text
4-channel EMG -> causal local features -> EMG context -----+
                                                        |
24-axis IMU  -> causal local features -> IMU context ----+-> gated fusion
                                                        |
local fused features ------------------------------------+-> grasp/release timing
long patch context --------------------------------------+-> holding and current SE(3)
                                                        +-> future SE(3) waypoints
                                                        +-> future grasp/release intent
```

It predicts current XYZ and 6D orientation, plus future XYZ and orientation at
`100, 250, 500, 750, and 1000 ms`. It also predicts whether grasp/release will
occur within one second and a categorical time-to-event distribution.

## Current held-out result

On the trial-held-out test split used for the future-intent run:

| Metric | EMG+IMU | Zero EMG | Increment from EMG |
| --- | ---: | ---: | ---: |
| Current position | `2.62 cm` | `3.94 cm` | `1.32 cm` better (`33.5%`) |
| Current orientation | `7.81°` | `10.42°` | `2.62°` better (`25.1%`) |
| Mean future position | `5.30 cm` | `6.46 cm` | `1.16 cm` better (`18.0%`) |
| Mean future orientation | `11.55°` | `13.63°` | `2.08°` better (`15.2%`) |
| Future grasp/release mAP | `0.953` | `0.931` | `+0.022` |
| Grasp timing MAE | `104 ms` | `115 ms` | `11 ms` better |
| Release timing MAE | `97 ms` | `102 ms` | `4 ms` better |

The current grasp/release macro-F1 was `0.847` with EMG+IMU and `0.888` after
zeroing EMG. Therefore the honest claim is not that EMG improves every head.
It improves pose, orientation, holding and one-second future intent in this
run, but the present instantaneous event decoder still needs work.

These are within-recording trial splits. They do not establish
participant-independent generalization. On one different recording,
`7d88abddc3f3/trial_011.csv`, the 250 ms model error rose to `11.63 cm` and
`25.54°`, compared with `3.60 cm` and `9.95°` on the original held-out test.
This is evidence for session/person calibration shift and motivates
leave-one-participant-out evaluation and lightweight calibration.

## Robot integration

The project first used a 3R arm to verify trajectory-to-IK mapping, then moved
to the Franka Panda. The current PyBullet path is:

```text
rolling EMG+IMU
      -> one-second model forecast
      -> selected receding-horizon SE(3) command (default: 250 ms)
      -> coordinate-frame mapping (+90° around Z by default)
      -> Franka inverse kinematics
      -> latched grasp/release gripper state
```

The controller recomputes the future every update; it does not execute a stale
one-second path open-loop. Black VIVE is comparison-only, cyan is the selected
model command, magenta is the full rolling forecast, and orange is the actual
Franka path.

On the different-recording trial above, Franka tracked the model command to
`0.29 cm` and `0.36°` on average, while Franka-to-VIVE error was `11.75 cm`
and `25.54°`. The robot controller was therefore accurate; the remaining error
was almost entirely model/domain error. The gripper closed when grasp exceeded
`0.9`, but release peaked at `0.876`, so it correctly did not cross the fixed
`0.9` release threshold. Thresholds must be selected on validation data.

## What we can defend

The current defensible story is:

1. Causal EMG+IMU predicts current and future end-effector SE(3), holding and
   interaction intent without VIVE input.
2. EMG adds measurable information beyond IMU for pose, orientation and
   future intent on the current within-recording split.
3. Local high-resolution features and long patch context solve different
   temporal parts of the same task inside one architecture.
4. Receding-horizon predictions can be transferred to a standard Franka IK
   controller, with model error separated from robot tracking error.
5. Cross-recording performance is not solved and must not be hidden. It is the
   next experimental problem.

We should not claim participant-independent performance, precise physical
contact detection from human annotations, or that EMG universally beats IMU.

## Reproduce the current endpoint

Train:

```bash
bash scripts/train_reach_grasp_future_intent.sh
```

The main checkpoint is:

```text
runs/reach_grasp_future_intent_seed42/final.pt
```

Visualize an unseen recorded trial on Franka:

```bash
python scripts/visualize_future_intent_franka.py \
  --checkpoint runs/reach_grasp_future_intent_seed42/final.pt \
  --trial-root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \
  --device cuda --trial-seed 7 --speed 1 \
  --control-horizon-ms 250 --gripper-lookahead-ms 250
```

Implementation details are in
[`docs/reach_grasp_future_intent.md`](docs/reach_grasp_future_intent.md).
