#!/usr/bin/env python3
"""Train the shared Stage-I encoder using only open/close supervision."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import train_gripper_state_pose as base


def classification_loss(model, output, batch, class_weight, args):
    """The sole Stage-I objective: class-balanced open/close cross-entropy."""
    del model, args
    valid = (batch["emg_usable"] & batch["imu_usable"]
             & batch["gripper_state_valid"])
    frame_loss = F.cross_entropy(
        output["gripper_state_logits"].transpose(1, 2),
        batch["gripper_state"], weight=class_weight, reduction="none")
    return base.masked_mean(frame_loss, valid)


def main():
    remaining = sys.argv[1:]
    sys.argv = [sys.argv[0], *remaining]
    defaults = {
        "--models": "emg+imu",
        "--pixel-architecture": "direct",
        "--react-weight": "0",
        "--masked-weight": "0",
        "--stability-weight": "0",
        "--position-weight": "0",
        "--orientation-weight": "0",
        "--pixel-weight": "0",
        "--final-pose-weight": "0",
        "--future-pose-weight": "0",
    }
    present = {value.split("=", 1)[0] for value in remaining
               if value.startswith("--")}
    for flag, value in defaults.items():
        if flag not in present:
            sys.argv.extend((flag, value))

    original_loss = base.loss
    base.loss = classification_loss
    try:
        base.main()
    finally:
        base.loss = original_loss

    output_arg = next((value.split("=", 1)[1] for value in remaining
                       if value.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/gripper_classifier_minimal")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "stage": "Stage I",
        "objective": "class-balanced open/close cross-entropy only",
        "heads_trained": ["open/close state"],
        "auxiliary_losses": [],
    })
    results_path.write_text(json.dumps(results, indent=2))

    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "format": "gripper_classifier_minimal_v1",
            "stage": "Stage I",
            "objective": "class-balanced open/close cross-entropy only",
        })
        torch.save(checkpoint, path)
    print(f"Minimal Stage-I protocol written to {results_path}")


if __name__ == "__main__":
    main()
