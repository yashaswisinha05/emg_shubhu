#!/usr/bin/env python3
"""Train the frozen Chronos-2 encoder ablation with the same four losses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from minimal_emg_imu.chronos2_model import Chronos2EMGIMUModel
from minimal_emg_imu.train import minimal_loss
from scripts import train_gripper_state_pose as base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--foundation", default="amazon/chronos-2")
    parser.add_argument("--chronos-dtype", choices=["float32", "bfloat16"],
                        default="bfloat16")
    parser.add_argument("--context-frames", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--state-weight", type=float, default=1.)
    parser.add_argument("--position-weight", type=float, default=1.)
    parser.add_argument("--pixel-weight", type=float, default=.35)
    parser.add_argument("--future-pose-weight", type=float, default=.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.state_weight, args.position_weight, args.pixel_weight,
           args.future_pose_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    try:
        from chronos import BaseChronosPipeline
    except ImportError as error:
        raise SystemExit(
            "Install the optional dependency first: "
            "pip install 'chronos-forecasting>=2.1,<3'") from error

    dtype = getattr(torch, args.chronos_dtype)
    pipeline = BaseChronosPipeline.from_pretrained(
        args.foundation, device_map=args.device, torch_dtype=dtype)
    chronos_model = pipeline.model
    chronos_model.eval()
    for parameter in chronos_model.parameters():
        parameter.requires_grad_(False)

    original_model, original_loss = base.GripperStatePoseModel, base.loss
    original_evaluate = base.evaluate

    def model_factory(**kwargs):
        if kwargs.get("modality") != "emg+imu":
            raise ValueError("the Chronos-2 ablation trains EMG+IMU only")
        return Chronos2EMGIMUModel(
            chronos_model=chronos_model,
            width=kwargs.get("width", 128), dropout=kwargs.get("dropout", .1),
            react_context=kwargs.get("react_context", 100), future_steps=20,
            context_frames=args.context_frames, chunk_frames=args.chunk_frames,
            foundation_id=args.foundation)

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
        "--root", *args.root, "--output-dir", str(args.output_dir),
        "--device", args.device, "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--patience", str(args.patience),
        "--seed", str(args.seed), "--split-seed", str(args.split_seed),
        "--raw-rate-hz", str(args.raw_rate_hz), "--models", "emg+imu",
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
    checkpoint.update({
        "format": "minimal_emg_imu_chronos2_v1",
        "model_args": {
            "width": 128, "future_steps": 20,
            "context_frames": args.context_frames,
            "chunk_frames": args.chunk_frames,
            "foundation_id": args.foundation,
        },
        "foundation": {
            "model_id": args.foundation, "frozen": True,
            "dtype": args.chronos_dtype, "causal_time_attention": True,
            "weights_in_checkpoint": False,
            "patch_endpoint_hold_frames": 16,
        },
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
        "architecture": "frozen-pretrained-chronos2-encoder-ablation-v1",
        "foundation": checkpoint["foundation"],
        "modality": "emg+imu only",
        "heads": ["open/close", "current XYZ", "pixel XY",
                  "future XYZ 10--200 ms"],
        "losses": checkpoint["losses"],
        "causal_context_frames": args.context_frames,
    })
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Chronos-2 checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
