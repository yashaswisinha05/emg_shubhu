#!/usr/bin/env python3
"""Train stronger EMG state decoding and late geometric pixel attention."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_state_attention_v2 import StateConditionedAttentionV2
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
        labels = batch["gripper_state"]
        valid = usable & batch["gripper_state_valid"]
        state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                                labels, weight=class_weight, reduction="none")
        react = F.cross_entropy(output["react_gripper_state_logits"].transpose(1, 2),
                                labels, weight=class_weight, reduction="none")
        hidden = base.span_mask(labels.shape, labels.device)
        actions = torch.where(hidden & valid, torch.full_like(labels, 2), labels)
        actions = torch.where(valid, actions, torch.full_like(labels, 2))
        emg_hidden = base.span_mask(labels.shape, labels.device)
        auxiliary = model.react(batch["emg"], actions=actions, hidden=emg_hidden)
        reconstruction_valid = emg_hidden[..., None] & batch["emg"][..., 8:].bool()
        reconstruction = masked_mean(
            (auxiliary["reconstruction"] - batch["emg"][..., :8]).square(),
            reconstruction_valid)
        masked_state = F.cross_entropy(auxiliary["holding_logits"].transpose(1, 2),
                                       labels, weight=class_weight, reduction="none")
        same = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] == labels[:, :-1])
        probability = output["state_probability"][..., 1]
        stability = masked_mean(
            (probability[:, 1:] - probability[:, :-1]).square(), same)
        total = total + (extra.state_condition_weight * masked_mean(state, valid)
                         + extra.react_weight * masked_mean(react, valid)
                         + extra.masked_weight * (
                             masked_mean(masked_state, hidden & valid) + reconstruction)
                         + extra.state_stability_weight * stability)

    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = F.smooth_l1_loss(output["position"], batch["pose"], reduction="none")
    total = total + args.position_weight * masked_mean(position, pose_valid[..., None])

    if output["click"] is not None:
        click_valid = usable & batch["click_valid"]

        def pixel_loss(prediction, weights):
            error = ((prediction - batch["click_target"])
                     * batch["canvas_px"][:, None, :] / 100.)
            per_frame = F.smooth_l1_loss(
                error, torch.zeros_like(error), reduction="none").mean(-1)
            weights = weights * click_valid.float()
            return (per_frame * weights).sum() / weights.sum().clamp_min(1.)

        late = .25 + 3.75 * batch["trial_progress"].pow(4)
        uniform = torch.ones_like(late)
        pixel = (pixel_loss(output["click"], late)
                 + .15 * pixel_loss(output["click_direct"], uniform)
                 + .15 * pixel_loss(output["click_from_position"], late))
        total = total + args.pixel_weight * pixel

    future, consistency = future_position_losses(output, batch, usable)
    return (total + args.future_pose_weight * future
            + extra.future_consistency_weight * consistency)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-condition-weight", type=float, default=.5)
    parser.add_argument("--react-weight-v2", dest="react_weight", type=float, default=.2)
    parser.add_argument("--masked-weight-v2", dest="masked_weight", type=float, default=.15)
    parser.add_argument("--state-stability-weight", type=float, default=.03)
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    global extra
    extra, remaining = parser.parse_known_args()
    if min(vars(extra).values()) < 0:
        parser.error("all V2 auxiliary weights must be nonnegative")
    sys.argv = [sys.argv[0], *remaining]
    defaults = {"--pixel-architecture": "direct", "--pixel-weight": ".35",
                "--position-weight": "1.0", "--orientation-weight": "0",
                "--final-pose-weight": "0", "--future-pose-ms": "200",
                "--future-pose-weight": ".5"}
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for option, value in defaults.items():
        if option not in present:
            sys.argv.extend((option, value))

    old_model, old_loss, old_evaluate = base.GripperStatePoseModel, base.loss, base.evaluate

    def position_only(*args, **kwargs):
        report = old_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"], value["valid_orientation_frames"] = None, 0
        return report

    base.GripperStatePoseModel, base.loss, base.evaluate = (
        StateConditionedAttentionV2, task_loss, position_only)
    try:
        base.main()
    finally:
        base.GripperStatePoseModel, base.loss, base.evaluate = old_model, old_loss, old_evaluate

    output_arg = next((x.split("=", 1)[1] for x in remaining
                       if x.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/gripper_state_pose_seed42")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "architecture": "state-conditioned-attention-v2",
        "gripper_improvement": "causal local/context + masked EMG state decoder",
        "pixel_improvement": "late-weighted direct/XYZ geometric blend",
        "future_consistency_weight": extra.future_consistency_weight,
        "heads": ["EMG-only gripper state", "current XYZ", "pixel XY",
                  "future XYZ (training regularizer)"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["format"] = "gripper_state_attention_v2"
        checkpoint["architecture"] = results["protocol"]["architecture"]
        torch.save(checkpoint, path)
    print(f"V2 protocol written to {results_path}")


if __name__ == "__main__":
    main()
