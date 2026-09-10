#!/usr/bin/env python3
"""Train EMG-state-conditioned causal attention for XYZ and pixel intent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_state_attention import StateConditionedAttentionModel
from emg_touch.models.state_attention_loss import future_position_losses, masked_mean
from scripts import train_gripper_state_pose as base


def task_loss(model, output, batch, class_weight, args):
    if model.modality == "emg":
        usable = batch["emg_usable"]
    elif model.modality == "imu":
        usable = batch["imu_usable"]
    else:
        usable = batch["emg_usable"] & batch["imu_usable"]

    total = output["position"].sum() * 0
    if model.modality != "imu":
        state_valid = usable & batch["gripper_state_valid"]
        state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                                batch["gripper_state"], weight=class_weight,
                                reduction="none")
        total = total + extra.state_condition_weight * masked_mean(state, state_valid)
        same = (state_valid[:, 1:] & state_valid[:, :-1]
                & (batch["gripper_state"][:, 1:] == batch["gripper_state"][:, :-1]))
        probability = output["state_probability"][..., 1]
        total = total + extra.state_stability_weight * masked_mean(
            (probability[:, 1:] - probability[:, :-1]).square(), same)

    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = F.smooth_l1_loss(output["position"], batch["pose"], reduction="none")
    total = total + args.position_weight * masked_mean(position, pose_valid[..., None])

    if output["click"] is not None:
        click_valid = usable & batch["click_valid"]
        # Optimize in physical pixels (scaled by 100 for balanced magnitude).
        error = ((output["click"] - batch["click_target"])
                 * batch["canvas_px"][:, None, :] / 100.)
        click = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
        total = total + args.pixel_weight * masked_mean(click, click_valid[..., None])

    future, consistency = future_position_losses(output, batch, usable)
    return (total + args.future_pose_weight * future
            + extra.future_consistency_weight * consistency)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-condition-weight", type=float, default=.5)
    parser.add_argument("--state-stability-weight", type=float, default=.03)
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    global extra
    extra, remaining = parser.parse_known_args()
    if min(extra.state_condition_weight, extra.state_stability_weight,
           extra.future_consistency_weight) < 0:
        parser.error("state and consistency weights must be nonnegative")

    sys.argv = [sys.argv[0], *remaining]
    defaults = {
        "--pixel-architecture": "direct",
        "--pixel-weight": ".35",
        "--position-weight": "1.0",
        "--orientation-weight": "0",
        "--final-pose-weight": "0",
        "--future-pose-ms": "200",
        "--future-pose-weight": ".5",
    }
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for option, value in defaults.items():
        if option not in present:
            sys.argv.extend((option, value))

    original_model, original_loss = base.GripperStatePoseModel, base.loss
    original_evaluate = base.evaluate

    def evaluate_position_only(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"] = None
            value["valid_orientation_frames"] = 0
        return report

    base.GripperStatePoseModel = StateConditionedAttentionModel
    base.loss = task_loss
    base.evaluate = evaluate_position_only
    try:
        base.main()
    finally:
        base.GripperStatePoseModel = original_model
        base.loss = original_loss
        base.evaluate = original_evaluate

    output_arg = next((x.split("=", 1)[1] for x in remaining
                       if x.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/gripper_state_pose_seed42")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "architecture": "emg-state-conditioned-causal-cross-attention",
        "state_condition_weight": extra.state_condition_weight,
        "state_stability_weight": extra.state_stability_weight,
        "future_consistency_weight": extra.future_consistency_weight,
        "deployed_heads": ["emg-only open/close", "current XYZ", "pixel XY"],
        "training_only_head": "200 ms future XYZ",
        "orientation_and_endpoint_heads": False,
        "heads": ["EMG-only gripper state", "current XYZ", "pixel XY",
                  "future XYZ (training regularizer)"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for checkpoint_path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["format"] = "gripper_state_attention_v1"
        checkpoint["architecture"] = results["protocol"]["architecture"]
        checkpoint["loss_weights"] = {
            "position": results["protocol"]["position_weight"]
                if "position_weight" in results["protocol"] else 1.,
            "pixel": results["protocol"]["pixel_weight"],
            "state": extra.state_condition_weight,
            "future_position": results["protocol"]["future_pose_weight"],
            "future_consistency": extra.future_consistency_weight,
        }
        torch.save(checkpoint, checkpoint_path)
    print(f"state-conditioned attention protocol written to {results_path}")


if __name__ == "__main__":
    main()
