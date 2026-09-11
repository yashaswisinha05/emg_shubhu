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
from emg_touch.models.state_attention_loss import future_position_losses, masked_mean
from scripts import train_gripper_state_pose as base


extra = None
trained_parameter_count = None


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


def task_loss(model, output, batch, class_weight, args):
    usable = batch["emg_usable"] & batch["imu_usable"]
    labels = batch["gripper_state"]
    state_valid = usable & batch["gripper_state_valid"]
    state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                            labels, weight=class_weight, reduction="none")
    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = F.smooth_l1_loss(output["position"], batch["pose"], reduction="none")
    total = (extra.state_weight * masked_mean(state, state_valid)
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
        centers = model.grid_centers[target_grid]
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
    global extra
    extra, remaining = parser.parse_known_args()
    if min(vars(extra).values()) < 0:
        parser.error("all auxiliary weights must be nonnegative")

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
        global trained_parameter_count
        trained = old_train_one(*args, **kwargs)
        trained_parameter_count = sum(
            parameter.numel() for parameter in trained[0].parameters()
            if parameter.requires_grad)
        return trained

    base.GripperStatePoseModel, base.loss = NeuromuscularResidualGRU, task_loss
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
    results["protocol"].update({
        "architecture": "state-conditioned-neuromuscular-residual-gru",
        "comparison_family": "causal-capacity-controlled-v1",
        "orientation_disabled": True,
        "pixel_axes": "x/width and y/height independently",
        "auxiliary_weights": vars(extra),
        "heads": ["EMG-only gripper state", "current XYZ",
                  "3x3 grid + within-cell pixel residual", "future XYZ",
                  "training-only masked EMG", "training-only future IMU delta"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "format": "neuromuscular_residual_gru_v1",
            "architecture": results["protocol"]["architecture"],
            "parameter_count": int(trained_parameter_count),
            "auxiliary_weights": vars(extra),
        })
        torch.save(checkpoint, path)
    print(f"residual GRU protocol written to {results_path}")


if __name__ == "__main__":
    main()
