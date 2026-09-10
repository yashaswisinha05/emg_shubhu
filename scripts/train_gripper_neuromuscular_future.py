#!/usr/bin/env python3
"""Train the project-specific IMU-rollout + EMG-innovation future model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.neuromuscular_future_loss import future_imu_delta_loss
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from scripts import train_gripper_state_pose as base


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion-reconstruction-weight", type=float, default=.05)
    extra, remaining = parser.parse_known_args()
    if extra.motion_reconstruction_weight < 0:
        parser.error("--motion-reconstruction-weight cannot be negative")

    sys.argv = [sys.argv[0], *remaining]
    defaults = {
        "--pixel-architecture": "goal-consistent",
        "--pixel-weight": ".35",
        "--final-pose-weight": ".2",
        "--endpoint-loss-space": "physical",
        "--final-position-multiplier": "2.5",
        "--endpoint-progress-weight": "1.0",
        "--future-pose-ms": "200",
        "--future-pose-weight": ".1",
    }
    present = {item.split("=", 1)[0] for item in remaining if item.startswith("--")}
    for option, value in defaults.items():
        if option not in present:
            sys.argv.extend((option, value))

    original_model = base.GoalConsistentGripperPoseModel
    original_loss = base.loss

    def combined_loss(model, output, batch, class_weight, args):
        task = original_loss(model, output, batch, class_weight, args)
        return task + extra.motion_reconstruction_weight * future_imu_delta_loss(output, batch)

    base.GoalConsistentGripperPoseModel = NeuromuscularFutureGripperPoseModel
    base.loss = combined_loss
    try:
        base.main()
    finally:
        base.GoalConsistentGripperPoseModel = original_model
        base.loss = original_loss

    output_arg = next((x.split("=", 1)[1] for x in remaining
                       if x.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output = Path(output_arg or "runs/gripper_state_pose_seed42")
    results_path = output / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"]["future_architecture"] = (
        "causal IMU dynamics rollout + gated EMG neuromuscular innovation")
    results["protocol"]["motion_reconstruction_weight"] = extra.motion_reconstruction_weight
    results_path.write_text(json.dumps(results, indent=2))
    for checkpoint_path in output.glob("*_best.pt"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["format"] = "gripper_neuromuscular_future_v1"
        checkpoint["future_architecture"] = results["protocol"]["future_architecture"]
        checkpoint["motion_reconstruction_weight"] = extra.motion_reconstruction_weight
        torch.save(checkpoint, checkpoint_path)
    print(f"Neuromuscular future protocol written to {results_path}")


if __name__ == "__main__":
    main()
