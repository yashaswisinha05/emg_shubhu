# Auxiliary future pose reconstruction

The existing gripper model predicts current pose, trial-final endpoint and click,
and open/close state. This optional experiment adds a decoder of causal context
into VIVE position and orientation at every 10 ms through 200 ms ahead.
At time t the whole future block is withheld from the encoder by causal attention;
it is used only as a supervision target. No future EMG, IMU, or gripper labels
are passed into this decoder. This reconstructs future **pose**, not EMG waveforms.

Loss = existing loss + 0.1 * (future standardized-position SmoothL1 +
future 6D-orientation SmoothL1). Missing labels and targets beyond the end of a
trial are excluded. Pose normalization is fitted on training trials only.
The shared encoder receives gradients, so the auxiliary loss can indirectly
affect other tasks. Their existing heads and losses are retained.

```bash
git pull origin main
bash scripts/train_gripper_future_pose.sh \
  --root data/shubham_open data/shubham_close \
    data/shubham1_open data/shubham1_closed \
    data/gazania_open data/gazania_closed
```

This runs six models: EMG, IMU, EMG+IMU, each with auxiliary loss disabled
(`runs/gripper_future_pose_200ms/control`) and enabled (`.../future`).
Set FUTURE_POSE_OUTPUT to a new directory for another experiment. Existing
nonempty output directories are rejected. Same seed/data ordering produces the
same trial split. Decoder initialization preserves the RNG state so common
encoder and task heads start identically; repeat across seeds before conclusions.
Checkpoint selection keeps the existing joint criterion for both settings.

Compare current position/orientation, endpoint error, click error by quarter,
and classification on the unchanged held-out trials. Future runs additionally
report `future_pose_by_ms`, including 200 ms position error (cm), orientation
geodesic error (degrees), valid target counts, and a persistence baseline that
holds the current model-predicted position. Better future reconstruction alone
does not establish improvement in the original tasks. The split is by trial,
not a held-out-person generalization evaluation.

Each modality saves `<modality>_best.pt` with `model_args.future_steps`, a distinct
format suffix and timing metadata. Recreate GoalConsistentGripperPoseModel using
those model_args for inference. Future positions use the saved position
normalization; future orientations use the existing 6D representation. Existing
visualizers need explicit support for these new future outputs.

## Improving the final 3D endpoint

The first goal-consistent screen model allowed click loss to backpropagate into
the final 3D endpoint. On the six-dataset run this reduced late screen error but
made the endpoint act like a screen-coordinate code: fused endpoint error rose
from 9.75 cm to about 16.4 cm and stopped converging late in the trial.

The protected endpoint experiment detaches that gradient, trains endpoint
position in equal physical-axis units, increases only endpoint-position loss,
and moderately emphasizes observations near the end. The screen projector still
learns from endpoint values, but cannot change them. Future-pose loss stays on.

```bash
git pull origin main
bash scripts/train_gripper_future_endpoint.sh \
  --root data/shubham_open data/shubham_close \
    data/shubham1_open data/shubham1_closed \
    data/gazania_open data/gazania_closed \
  --output-dir runs/gripper_future_endpoint
```

Compare overall and quarter-4 `final_position_cm` against the future run, while
checking current pose, 200 ms future pose, late click error, and gripper F1. The
expected direction is recovery toward the earlier 9.75 cm overall endpoint and
2.74 cm late endpoint; the held-out result determines the actual improvement.
