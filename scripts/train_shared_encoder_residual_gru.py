#!/usr/bin/env python3
"""Train shared frozen patch encoders with causal residual-GRU adapters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.models.reach_grasp_shared_encoder_adapter import (
    SharedEncoderResidualGRU,
)
from scripts import train_gripper_state_pose as base
from scripts import train_neuromuscular_residual_gru as residual


def load_classifier(state):
    if state.get("format") == "gripper_neuromuscular_future_v1":
        model = NeuromuscularFutureGripperPoseModel(**state["model_args"])
    elif (state.get("format") == "reach_grasp_architecture_baseline_v1"
          and state.get("architecture") == "gru"):
        model = ArchitectureBaseline("gru", **state["model_args"])
        raise ValueError(
            "the causal-GRU classifier does not expose separate shared patch features; "
            "use a gripper_neuromuscular_future_v1 checkpoint")
    else:
        raise ValueError(
            "classifier checkpoint must have format gripper_neuromuscular_future_v1")
    model.load_state_dict(state["state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-layers", type=int, default=2)
    parser.add_argument("--reconstruction-horizon-ms", type=int, default=1000)
    parser.add_argument("--reconstruction-step-ms", type=int, default=100)
    parser.add_argument("--reconstruction-decay-ms", type=float, default=500.)
    parser.add_argument("--grid-weight", type=float, default=.15)
    parser.add_argument("--grid-residual-weight", type=float, default=.1)
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    parser.add_argument("--long-position-weight", type=float, default=.05)
    parser.add_argument("--emg-to-future-imu-weight", type=float, default=.05)
    parser.add_argument("--correction-weight", type=float, default=.01)
    option, remaining = parser.parse_known_args()
    if option.adapter_layers <= 0:
        parser.error("adapter-layers must be positive")
    numeric = [value for name, value in vars(option).items()
               if name != "classifier_checkpoint" and name != "adapter_layers"]
    if min(numeric) < 0 or option.reconstruction_decay_ms <= 0:
        parser.error("weights and horizons must be nonnegative; decay must be positive")
    if (option.reconstruction_horizon_ms <= 0
            or option.reconstruction_step_ms <= 0
            or option.reconstruction_horizon_ms % option.reconstruction_step_ms
            or option.reconstruction_horizon_ms % 10
            or option.reconstruction_step_ms % 10):
        parser.error("reconstruction horizon must be a positive multiple of its 10-ms step")
    if not option.classifier_checkpoint.is_file():
        parser.error(f"classifier checkpoint not found: {option.classifier_checkpoint}")

    classifier_state = torch.load(
        option.classifier_checkpoint, map_location="cpu", weights_only=False)
    if classifier_state.get("format") != "gripper_neuromuscular_future_v1":
        parser.error("shared encoders require a gripper_neuromuscular_future_v1 checkpoint")
    if classifier_state["model_args"].get("modality") != "emg+imu":
        parser.error("classifier checkpoint must use fused emg+imu input")

    horizons = tuple(range(option.reconstruction_step_ms // 10,
                           option.reconstruction_horizon_ms // 10 + 1,
                           option.reconstruction_step_ms // 10))
    residual.extra = SimpleNamespace(
        state_weight=0.,
        grid_weight=option.grid_weight,
        grid_residual_weight=option.grid_residual_weight,
        masked_emg_weight=0.,
        future_imu_weight=0.,
        future_consistency_weight=option.future_consistency_weight,
        correction_weight=option.correction_weight,
        reconstruction_horizon_ms=option.reconstruction_horizon_ms,
        reconstruction_step_ms=option.reconstruction_step_ms,
        reconstruction_decay_ms=option.reconstruction_decay_ms,
        long_position_weight=option.long_position_weight,
        long_imu_weight=option.emg_to_future_imu_weight,
        long_state_weight=0.,
    )

    sys.argv = [sys.argv[0], *remaining]
    defaults = {
        "--models": "emg+imu", "--pixel-architecture": "direct",
        "--pixel-weight": ".35", "--position-weight": "1.0",
        "--orientation-weight": "0", "--final-pose-weight": "0",
        "--future-pose-ms": "200", "--future-pose-weight": ".5",
    }
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for flag, value in defaults.items():
        if flag not in present:
            sys.argv.extend((flag, value))

    active_normalization = None
    trained_parameter_count = None
    original_model, original_loss = base.GripperStatePoseModel, base.loss
    original_evaluate, original_train = base.evaluate, base.train_one

    def model_factory(**model_args):
        if active_normalization is None:
            raise RuntimeError("motion normalization was not initialized")
        classifier = load_classifier(classifier_state)
        return SharedEncoderResidualGRU(
            classifier,
            classifier_state["normalization"],
            active_normalization,
            width=model_args.get("width", 128),
            adapter_layers=option.adapter_layers,
            dropout=model_args.get("dropout", .1),
            future_steps=model_args.get("future_steps", 20),
            intent_horizons_steps=horizons,
        )

    def train_wrapper(*args, **kwargs):
        nonlocal active_normalization, trained_parameter_count
        active_normalization = args[4]
        trained = original_train(*args, **kwargs)
        trained_parameter_count = sum(
            p.numel() for p in trained[0].parameters() if p.requires_grad)
        return trained

    def position_only(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"] = None
            value["valid_orientation_frames"] = 0
        return report

    base.GripperStatePoseModel = model_factory
    base.loss = residual.task_loss
    base.evaluate = position_only
    base.train_one = train_wrapper
    try:
        base.main()
    finally:
        base.GripperStatePoseModel = original_model
        base.loss = original_loss
        base.evaluate = original_evaluate
        base.train_one = original_train

    output_arg = next((x.split("=", 1)[1] for x in remaining
                       if x.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/shared_encoder_residual_gru")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "architecture": "shared-frozen-patch-encoder-residual-gru-adapter",
        "classifier_checkpoint": str(option.classifier_checkpoint),
        "classifier_and_encoders_frozen": True,
        "adapter_layers": option.adapter_layers,
        "orientation_disabled": True,
        "endpoint_disabled": True,
        "masked_emg_reconstruction": False,
        "emg_to_future_imu": True,
        "intent_horizons_ms": [step * 10 for step in horizons],
        "heads": ["frozen open/close state", "current XYZ",
                  "3x3 grid + pixel residual", "future XYZ through 200 ms",
                  "intent XYZ through 1000 ms",
                  "training-only EMG-to-future-IMU summary"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "format": "shared_encoder_residual_gru_v1",
            "architecture": results["protocol"]["architecture"],
            "parameter_count": int(trained_parameter_count),
            "classifier_source": str(option.classifier_checkpoint),
            "classifier_format": classifier_state["format"],
            "classifier_model_args": classifier_state["model_args"],
            "classifier_normalization": classifier_state["normalization"],
            "classifier_frozen": True,
            "shared_model_args": {
                "width": 128, "adapter_layers": option.adapter_layers,
                "dropout": .1, "future_steps": 20,
                "intent_horizons_steps": horizons,
            },
        })
        torch.save(checkpoint, path)
    print(f"shared-encoder protocol written to {results_path}")


if __name__ == "__main__":
    main()
