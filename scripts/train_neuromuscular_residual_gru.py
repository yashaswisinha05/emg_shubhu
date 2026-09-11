#!/usr/bin/env python3
"""Train the multi-head state-conditioned neuromuscular residual GRU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_residual_gru import NeuromuscularResidualGRU
from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.models.reach_grasp_neuro_classifier_attention import (
    NeuroClassifierConditionedAttention,
)
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.models.state_attention_loss import future_position_losses, masked_mean
from scripts import train_gripper_state_pose as base


extra = None
trained_parameter_count = None
intent_horizons_steps = ()
classifier_state = None
active_normalization = None


def future_imu_loss(output, batch, usable):
    total = output["future_imu_delta"].sum() * 0
    count = output["future_imu_delta"].new_zeros(())
    for horizon in range(1, output["future_imu_delta"].shape[2] + 1):
        if horizon >= usable.shape[1]:
            break
        valid = usable[:, :-horizon] & usable[:, horizon:]
        prediction = output["future_imu_delta"][:, :-horizon, horizon - 1]
        target = batch["imu"][:, horizon:, :24] - batch["imu"][:, :-horizon, :24]
        total = total + F.smooth_l1_loss(
            prediction[valid], target[valid], reduction="sum")
        count = count + valid.sum() * target.shape[-1]
    return total / count.clamp_min(1)


def long_intent_loss(model, output, batch, usable, class_weight):
    """Reconstruct low-rate motion summaries, not noisy raw future samples."""
    zero = output["position"].sum() * 0
    if output["intent_position_delta"] is None:
        return zero, zero, zero
    position_total = imu_total = state_total = zero
    weight_total = output["position"].new_zeros(())
    for index, horizon in enumerate(model.intent_horizons_steps):
        if horizon >= usable.shape[1]:
            continue
        horizon_ms = horizon * 10
        weight = torch.exp(output["position"].new_tensor(
            -horizon_ms / extra.reconstruction_decay_ms))
        source_usable = usable[:, :-horizon]
        target_usable = usable[:, horizon:]
        pose_valid = (source_usable & target_usable
                      & batch["pose_mask"][:, :-horizon, 0].bool()
                      & batch["pose_mask"][:, horizon:, 0].bool())
        position_target = batch["pose"][:, horizon:] - batch["pose"][:, :-horizon]
        position_prediction = output["intent_position_delta"][:, :-horizon, index]
        position_total = position_total + weight * masked_mean(F.smooth_l1_loss(
            position_prediction, position_target, reduction="none"),
            pose_valid[..., None])

        previous = 0 if index == 0 else model.intent_horizons_steps[index - 1]
        interval = horizon - previous
        length = usable.shape[1] - horizon
        interval_valid = source_usable & target_usable
        interval_sum = torch.zeros_like(batch["imu"][:, :length, :24])
        for offset in range(previous + 1, horizon + 1):
            interval_sum = interval_sum + batch["imu"][:, offset:offset + length, :24]
            interval_valid = interval_valid & usable[:, offset:offset + length]
        interval_mean = interval_sum / interval
        imu_target = interval_mean - batch["imu"][:, :length, :24]
        imu_prediction = output["intent_imu_delta"][:, :length, index]
        imu_total = imu_total + weight * masked_mean(F.smooth_l1_loss(
            imu_prediction, imu_target, reduction="none"), interval_valid[..., None])

        if output["intent_state_logits"] is not None and extra.long_state_weight > 0:
            state_valid = (source_usable & target_usable
                           & batch["gripper_state_valid"][:, horizon:])
            state = F.cross_entropy(
                output["intent_state_logits"][:, :-horizon, index].transpose(1, 2),
                batch["gripper_state"][:, horizon:], weight=class_weight,
                reduction="none")
            state_total = state_total + weight * masked_mean(state, state_valid)
        weight_total = weight_total + weight
    divisor = weight_total.clamp_min(1e-8)
    return position_total / divisor, imu_total / divisor, state_total / divisor


def task_loss(model, output, batch, class_weight, args):
    motion = model.motion if isinstance(model, NeuroClassifierConditionedAttention) else model
    usable = batch["emg_usable"] & batch["imu_usable"]
    labels = batch["gripper_state"]
    state_valid = usable & batch["gripper_state_valid"]
    state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                            labels, weight=class_weight, reduction="none")
    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = F.smooth_l1_loss(output["position"], batch["pose"], reduction="none")
    state_weight = (0. if isinstance(model, NeuroClassifierConditionedAttention)
                    else extra.state_weight)
    total = (state_weight * masked_mean(state, state_valid)
             + args.position_weight * masked_mean(position, pose_valid[..., None]))

    if output["click"] is not None:
        click_valid = usable & batch["click_valid"]
        canvas = batch["canvas_px"][:, None, :] / 100.
        pixel_error = (output["click"] - batch["click_target"]) * canvas
        pixel_frame = F.smooth_l1_loss(
            pixel_error, torch.zeros_like(pixel_error), reduction="none").mean(-1)
        late = (.25 + 3.75 * batch["trial_progress"].pow(4)) * click_valid.float()
        pixel = (pixel_frame * late).sum() / late.sum().clamp_min(1.)

        target_xy = (batch["click_target"] * 3).floor().long().clamp(0, 2)
        target_grid = target_xy[..., 1] * 3 + target_xy[..., 0]
        grid = F.cross_entropy(output["grid_logits"].transpose(1, 2),
                               target_grid, reduction="none")
        grid = masked_mean(grid, click_valid)
        selected_offset = output["grid_offsets"].gather(
            -2, target_grid[..., None, None].expand(*target_grid.shape, 1, 2)).squeeze(-2)
        centers = motion.grid_centers[target_grid]
        target_offset = batch["click_target"] - centers
        residual_error = (selected_offset - target_offset) * canvas
        residual = masked_mean(F.smooth_l1_loss(
            residual_error, torch.zeros_like(residual_error), reduction="none"),
            click_valid[..., None])
        total = total + args.pixel_weight * pixel + extra.grid_weight * grid
        total = total + extra.grid_residual_weight * residual

    future, consistency = future_position_losses(output, batch, usable)
    total = total + args.future_pose_weight * future
    total = total + extra.future_consistency_weight * consistency
    total = total + extra.future_imu_weight * future_imu_loss(output, batch, usable)
    long_position, long_imu, long_state = long_intent_loss(
        motion, output, batch, usable, class_weight)
    total = total + extra.long_position_weight * long_position
    total = total + extra.long_imu_weight * long_imu
    total = total + extra.long_state_weight * long_state

    hidden = base.span_mask(labels.shape, labels.device) & batch["emg_usable"]
    masked_emg = batch["emg"].clone()
    masked_emg[..., :8] = torch.where(
        hidden[..., None], torch.zeros_like(masked_emg[..., :8]), masked_emg[..., :8])
    reconstructed = model(masked_emg, batch["imu"])["emg_reconstruction"]
    reconstruction_valid = hidden[..., None] & batch["emg"][..., 8:].bool()
    reconstruction = masked_mean(
        (reconstructed - batch["emg"][..., :8]).square(), reconstruction_valid)
    correction_penalty = masked_mean(output["emg_correction"].square(), usable[..., None])
    return (total + extra.masked_emg_weight * reconstruction
            + extra.correction_weight * correction_penalty)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-weight", type=float, default=1.)
    parser.add_argument("--grid-weight", type=float, default=.15)
    parser.add_argument("--grid-residual-weight", type=float, default=.1)
    parser.add_argument("--masked-emg-weight", type=float, default=.05)
    parser.add_argument("--future-imu-weight", type=float, default=.05)
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    parser.add_argument("--correction-weight", type=float, default=.01)
    parser.add_argument("--reconstruction-horizon-ms", type=int, default=0)
    parser.add_argument("--reconstruction-step-ms", type=int, default=100)
    parser.add_argument("--reconstruction-decay-ms", type=float, default=500.)
    parser.add_argument("--long-position-weight", type=float, default=.05)
    parser.add_argument("--long-imu-weight", type=float, default=.03)
    parser.add_argument("--long-state-weight", type=float, default=.03)
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        help="Freeze this proven classifier and use its detached probabilities")
    global extra
    extra, remaining = parser.parse_known_args()
    numeric = [value for name, value in vars(extra).items()
               if name != "classifier_checkpoint"]
    if min(numeric) < 0 or extra.reconstruction_decay_ms <= 0:
        parser.error("all auxiliary weights must be nonnegative")
    if extra.reconstruction_horizon_ms % 10 or extra.reconstruction_step_ms % 10:
        parser.error("reconstruction horizon and step must be multiples of 10 ms")
    if (extra.reconstruction_horizon_ms and
            (extra.reconstruction_step_ms <= 0
             or extra.reconstruction_horizon_ms % extra.reconstruction_step_ms)):
        parser.error("reconstruction horizon must be divisible by its positive step")
    global intent_horizons_steps
    intent_horizons_steps = tuple(range(
        extra.reconstruction_step_ms // 10,
        extra.reconstruction_horizon_ms // 10 + 1,
        extra.reconstruction_step_ms // 10)) if extra.reconstruction_horizon_ms else ()
    global classifier_state
    classifier_state = None
    if extra.classifier_checkpoint is not None:
        if not extra.classifier_checkpoint.is_file():
            parser.error(f"classifier checkpoint not found: {extra.classifier_checkpoint}")
        classifier_state = torch.load(
            extra.classifier_checkpoint, map_location="cpu", weights_only=False)
        supported = classifier_state.get("format") in {
            "gripper_neuromuscular_future_v1",
            "reach_grasp_architecture_baseline_v1",
        }
        if not supported:
            parser.error("classifier must be a neuromuscular-future or architecture-baseline checkpoint")
        if classifier_state.get("format") == "reach_grasp_architecture_baseline_v1" \
                and classifier_state.get("architecture") != "gru":
            parser.error("architecture-baseline classifier must be the causal GRU")
        if classifier_state["model_args"].get("modality") != "emg+imu":
            parser.error("classifier checkpoint must use modality emg+imu")

    sys.argv = [sys.argv[0], *remaining]
    defaults = {"--models": "emg+imu", "--pixel-architecture": "direct",
                "--pixel-weight": ".35", "--position-weight": "1.0",
                "--orientation-weight": "0", "--final-pose-weight": "0",
                "--future-pose-ms": "200", "--future-pose-weight": ".5"}
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for option, value in defaults.items():
        if option not in present:
            sys.argv.extend((option, value))

    old_model, old_loss = base.GripperStatePoseModel, base.loss
    old_evaluate, old_train_one = base.evaluate, base.train_one

    def position_only(*args, **kwargs):
        report = old_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"], value["valid_orientation_frames"] = None, 0
        return report

    def train_counted(*args, **kwargs):
        global trained_parameter_count, active_normalization
        active_normalization = args[4]
        trained = old_train_one(*args, **kwargs)
        trained_parameter_count = sum(
            parameter.numel() for parameter in trained[0].parameters()
            if parameter.requires_grad)
        return trained

    def model_factory(**model_args):
        motion = NeuromuscularResidualGRU(
            **model_args, intent_horizons_steps=intent_horizons_steps,
            predict_intent_state=classifier_state is None)
        if classifier_state is None:
            return motion
        # The authoritative frozen classifier supplies the state. Do not count
        # or optimize the motion model's unused private state estimator.
        for parameter in motion.state_head.parameters():
            parameter.requires_grad_(False)
        motion.imu_state_prior.requires_grad_(False)
        if active_normalization is None:
            raise RuntimeError("motion normalization was not initialized")
        classifier_args = classifier_state["model_args"]
        if classifier_state["format"] == "gripper_neuromuscular_future_v1":
            classifier = NeuromuscularFutureGripperPoseModel(**classifier_args)
        else:
            classifier = ArchitectureBaseline(
                classifier_state["architecture"], **classifier_args)
        classifier.load_state_dict(classifier_state["state_dict"])
        return NeuroClassifierConditionedAttention(
            classifier, motion, classifier_state["normalization"],
            active_normalization)

    base.GripperStatePoseModel, base.loss = model_factory, task_loss
    base.evaluate, base.train_one = position_only, train_counted
    try:
        base.main()
    finally:
        base.GripperStatePoseModel, base.loss = old_model, old_loss
        base.evaluate, base.train_one = old_evaluate, old_train_one

    output_arg = next((value.split("=", 1)[1] for value in remaining
                       if value.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/neuromuscular_residual_gru")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    auxiliary_settings = {
        name: value for name, value in vars(extra).items()
        if name != "classifier_checkpoint"}
    results["protocol"].update({
        "architecture": ("frozen-classifier-conditioned-neuromuscular-residual-gru"
                         if classifier_state is not None else
                         "state-conditioned-neuromuscular-residual-gru"),
        "comparison_family": "causal-capacity-controlled-v1",
        "orientation_disabled": True,
        "pixel_axes": "x/width and y/height independently",
        "auxiliary_weights": auxiliary_settings,
        "classifier_checkpoint": (str(extra.classifier_checkpoint)
                                  if extra.classifier_checkpoint else None),
        "classifier_frozen": classifier_state is not None,
        "heads": ["EMG-only gripper state", "current XYZ",
                  "3x3 grid + within-cell pixel residual", "future XYZ",
                  "training-only masked EMG", "training-only future IMU delta"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "format": ("frozen_classifier_residual_gru_v1"
                       if classifier_state is not None else
                       "neuromuscular_residual_gru_v1"),
            "architecture": results["protocol"]["architecture"],
            "parameter_count": int(trained_parameter_count),
            "auxiliary_weights": auxiliary_settings,
        })
        checkpoint["model_args"]["intent_horizons_steps"] = intent_horizons_steps
        checkpoint["model_args"]["predict_intent_state"] = classifier_state is None
        if classifier_state is not None:
            checkpoint.update({
                "classifier_source": str(extra.classifier_checkpoint),
                "classifier_format": classifier_state["format"],
                "classifier_model_args": classifier_state["model_args"],
                "classifier_normalization": classifier_state["normalization"],
                "classifier_frozen": True,
            })
        torch.save(checkpoint, path)
    print(f"residual GRU protocol written to {results_path}")


if __name__ == "__main__":
    main()
