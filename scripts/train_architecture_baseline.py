#!/usr/bin/env python3
"""Train one causal, capacity-controlled architecture comparison baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.models.state_attention_loss import future_position_losses, masked_mean
from scripts import train_gripper_state_pose as base


selected_architecture = None
selected_width = 128
predict_state = True
constant_initialization = None
future_consistency_weight = .1
trained_parameter_count = None


def model_factory(**kwargs):
    kwargs["width"] = selected_width
    kwargs["predict_state"] = predict_state
    return ArchitectureBaseline(
        selected_architecture, constant_initialization=constant_initialization,
        **kwargs)


def comparison_loss(model, output, batch, class_weight, args):
    if model.modality == "emg":
        usable = batch["emg_usable"]
    elif model.modality == "imu":
        usable = batch["imu_usable"]
    else:
        usable = batch["emg_usable"] & batch["imu_usable"]
    labels = batch["gripper_state"]
    state_valid = usable & batch["gripper_state_valid"]
    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = F.smooth_l1_loss(output["position"], batch["pose"], reduction="none")
    total = args.position_weight * masked_mean(position, pose_valid[..., None])
    if predict_state:
        state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                                labels, weight=class_weight, reduction="none")
        total = total + masked_mean(state, state_valid)

    if output["click"] is not None:
        click_valid = usable & batch["click_valid"]
        error = ((output["click"] - batch["click_target"])
                 * batch["canvas_px"][:, None, :] / 100.)
        per_frame = F.smooth_l1_loss(
            error, torch.zeros_like(error), reduction="none").mean(-1)
        # Identical late-evidence weighting to the proposed V2 pixel objective.
        weight = (.25 + 3.75 * batch["trial_progress"].pow(4)) * click_valid.float()
        total = total + args.pixel_weight * (
            per_frame * weight).sum() / weight.sum().clamp_min(1.)

    future, consistency = future_position_losses(output, batch, usable)
    return (total + args.future_pose_weight * future
            + future_consistency_weight * consistency)


def make_constant_initialization(train, stats, steps):
    labels = np.concatenate([
        trial["gripper_state"][trial["gripper_state_valid"]] for trial in train])
    counts = np.bincount(labels, minlength=2).astype(float) + 1e-6
    positions = np.concatenate([
        trial["position"][trial["pose_valid"]] for trial in train])
    position = ((positions.mean(0) - np.asarray(stats["position"]["mean"]))
                / np.asarray(stats["position"]["std"]))
    clicks = [trial["click_target"] for trial in train if "click_target" in trial]
    click = np.mean(clicks, axis=0) if clicks else np.asarray([.5, .5])
    standardized = [
        (trial["position"] - np.asarray(stats["position"]["mean"]))
        / np.asarray(stats["position"]["std"]) for trial in train]
    future = []
    for horizon in range(1, steps + 1):
        values = [value[horizon:] for value in standardized if len(value) > horizon]
        future.append(np.concatenate(values).mean(0) if values else position)
    return {"state_logits": np.log(counts / counts.sum()), "position": position,
            "click": click, "future_position": future}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--architecture", required=True,
                        choices=sorted(ArchitectureBaseline.NAMES))
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--no-gripper-head", action="store_true")
    option, remaining = parser.parse_known_args()
    if option.future_consistency_weight < 0:
        parser.error("future consistency weight must be nonnegative")
    global selected_architecture, selected_width, predict_state, future_consistency_weight
    selected_architecture = option.architecture
    selected_width = option.width
    predict_state = not option.no_gripper_head
    future_consistency_weight = option.future_consistency_weight

    sys.argv = [sys.argv[0], *remaining]
    defaults = {"--models": "emg+imu", "--pixel-architecture": "direct",
                "--pixel-weight": ".35", "--position-weight": "1.0",
                "--orientation-weight": "0", "--final-pose-weight": "0",
                "--future-pose-ms": "200", "--future-pose-weight": ".5"}
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for name, value in defaults.items():
        if name not in present:
            sys.argv.extend((name, value))
    if "--models" in sys.argv:
        index = sys.argv.index("--models") + 1
        models = []
        while index < len(sys.argv) and not sys.argv[index].startswith("--"):
            models.append(sys.argv[index]); index += 1
        if len(models) != 1 or models[0] not in {"imu", "emg+imu"}:
            parser.error("motion baselines require one of --models imu or emg+imu")

    old_model, old_loss = base.GripperStatePoseModel, base.loss
    old_evaluate, old_train_one = base.evaluate, base.train_one

    def position_only(*args, **kwargs):
        report = old_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        if not predict_state:
            report["gripper_accuracy"] = None
            report["gripper_macro_f1"] = None
            report["confusion_open_close"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"], value["valid_orientation_frames"] = None, 0
        return report

    def train_baseline(modality, args, train, validation, stats, class_weight,
                       settings, augmenter=None):
        global constant_initialization, trained_parameter_count
        constant_initialization = (make_constant_initialization(
            train, stats, args.future_pose_ms // 10)
            if selected_architecture == "constant" else None)
        trained = old_train_one(modality, args, train, validation, stats,
                                class_weight, settings, augmenter)
        trained_parameter_count = sum(
            parameter.numel() for parameter in trained[0].parameters()
            if parameter.requires_grad)
        return trained

    base.GripperStatePoseModel, base.loss = model_factory, comparison_loss
    base.evaluate, base.train_one = position_only, train_baseline
    try:
        base.main()
    finally:
        base.GripperStatePoseModel, base.loss = old_model, old_loss
        base.evaluate, base.train_one = old_evaluate, old_train_one

    output_arg = next((value.split("=", 1)[1] for value in remaining
                       if value.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or f"runs/comparison_{selected_architecture}")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "architecture": selected_architecture,
        "comparison_family": "causal-capacity-controlled-v1",
        "future_consistency_weight": future_consistency_weight,
        "orientation_disabled": True,
        "heads": ["gripper state", "current XYZ", "pixel XY", "future XYZ"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["format"] = "reach_grasp_architecture_baseline_v1"
        checkpoint["architecture"] = selected_architecture
        checkpoint["parameter_count"] = int(trained_parameter_count)
        checkpoint["model_args"]["width"] = selected_width
        checkpoint["model_args"]["predict_state"] = predict_state
        torch.save(checkpoint, path)
    print(f"comparison protocol written to {results_path}")


if __name__ == "__main__":
    main()
