#!/usr/bin/env python3
"""Train only the final causal EMG+IMU tasks and write one checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.state_attention_loss import future_position_losses
from minimal_emg_imu.model import MinimalEMGIMUModel
from scripts import train_gripper_state_pose as base


def minimal_loss(model, output, batch, class_weight, args):
    """Exactly four objectives: state, XYZ, pixels, and future XYZ."""
    usable = batch["emg_usable"] & batch["imu_usable"]
    state_valid = usable & batch["gripper_state_valid"]
    state = base.masked_mean(F.cross_entropy(
        output["gripper_state_logits"].transpose(1, 2),
        batch["gripper_state"], weight=class_weight, reduction="none"),
        state_valid)

    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = base.masked_mean(F.smooth_l1_loss(
        output["position"], batch["pose"], reduction="none"),
        pose_valid[..., None])

    click_valid = usable & batch["click_valid"]
    # Optimize physical pixels with independent width/height scaling.
    scaled_delta = ((output["click"] - batch["click_target"])
                    * batch["canvas_px"][:, None, :] / 100.)
    pixel_frame = F.smooth_l1_loss(
        scaled_delta, torch.zeros_like(scaled_delta), reduction="none").mean(-1)
    late_weight = .25 + 3.75 * batch["trial_progress"].pow(4)
    weighted_valid = late_weight * click_valid.float()
    pixel = ((pixel_frame * weighted_valid).sum()
             / weighted_valid.sum().clamp_min(1.))

    future, _ = future_position_losses(
        output, batch, usable, include_consistency=False)
    return (args.state_weight * state
            + args.position_weight * position
            + args.pixel_weight * pixel
            + args.future_pose_weight * future)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--state-weight", type=float, default=1.)
    parser.add_argument("--position-weight", type=float, default=1.)
    parser.add_argument("--pixel-weight", type=float, default=.35)
    parser.add_argument("--future-pose-weight", type=float, default=.5)
    args = parser.parse_args()
    if min(args.state_weight, args.position_weight, args.pixel_weight,
           args.future_pose_weight) < 0:
        parser.error("loss weights must be nonnegative")
    return args


def main():
    args = parse_args()
    original_model, original_loss = base.GripperStatePoseModel, base.loss
    original_evaluate = base.evaluate

    def model_factory(**kwargs):
        if kwargs.get("modality") != "emg+imu":
            raise ValueError("this codebase trains EMG+IMU only")
        return MinimalEMGIMUModel(
            width=kwargs.get("width", 128), patch=kwargs.get("patch", 16),
            stride=kwargs.get("stride", 4), layers=kwargs.get("layers", 4),
            heads=kwargs.get("heads", 4), dropout=kwargs.get("dropout", .1),
            react_context=kwargs.get("react_context", 100),
            future_steps=20)

    def loss_wrapper(model, output, batch, class_weight, _base_args):
        return minimal_loss(model, output, batch, class_weight, args)

    def position_only_evaluate(*evaluate_args, **evaluate_kwargs):
        report = original_evaluate(*evaluate_args, **evaluate_kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"] = None
            value["valid_orientation_frames"] = 0
        return report

    base.GripperStatePoseModel = model_factory
    base.loss = loss_wrapper
    base.evaluate = position_only_evaluate
    old_argv = sys.argv
    sys.argv = [old_argv[0],
        "--root", *args.root,
        "--output-dir", str(args.output_dir),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
        "--raw-rate-hz", str(args.raw_rate_hz),
        "--models", "emg+imu",
        "--react-weight", "0", "--masked-weight", "0",
        "--stability-weight", "0", "--orientation-weight", "0",
        "--final-pose-weight", "0", "--pixel-architecture", "direct",
        "--position-weight", str(args.position_weight),
        "--pixel-weight", str(args.pixel_weight),
        "--future-pose-ms", "200",
        "--future-pose-weight", str(args.future_pose_weight)]
    try:
        base.main()
    finally:
        sys.argv = old_argv
        base.GripperStatePoseModel, base.loss = original_model, original_loss
        base.evaluate = original_evaluate

    checkpoint_path = args.output_dir / "emg_imu_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MinimalEMGIMUModel()
    checkpoint.update({
        "format": "minimal_emg_imu_v1",
        "model_args": model.model_args,
        "losses": {
            "state_cross_entropy": args.state_weight,
            "current_xyz_smooth_l1": args.position_weight,
            "late_weighted_physical_pixel_smooth_l1": args.pixel_weight,
            "future_xyz_10_to_200ms_smooth_l1": args.future_pose_weight,
        },
    })
    torch.save(checkpoint, checkpoint_path)
    results_path = args.output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results.pop("fusion_zero_emg", None)
    results.pop("fusion_zero_imu", None)
    results["protocol"].update({
        "architecture": "minimal-causal-emg-imu-v1",
        "modality": "emg+imu only",
        "heads": ["open/close", "current XYZ", "pixel XY", "future XYZ 10--200 ms"],
        "losses": checkpoint["losses"],
        "removed": ["orientation", "endpoint", "grid", "offset", "state conditioning",
                    "masked reconstruction", "future IMU", "one-second intent",
                    "future consistency", "correction regularization"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
