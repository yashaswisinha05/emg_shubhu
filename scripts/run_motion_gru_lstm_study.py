#!/usr/bin/env python3
"""Train parameter-matched motion-only GRU/LSTM baselines."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_gripper_pose_architecture_study import matched_width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--parameter-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-pose-ms", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    args = parser.parse_args()

    reference = torch.load(
        args.parameter_reference_checkpoint, map_location="cpu", weights_only=False)
    target = reference.get("parameter_count")
    if not target:
        parser.error("reference checkpoint has no trainable parameter_count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"target_trainable_parameters": int(target), "models": {}}

    for architecture in ("gru", "lstm"):
        width, count = matched_width(
            architecture, int(target), args.future_pose_ms // 10,
            predict_state=False)
        for modality in ("imu", "emg+imu"):
            key = f"{modality.replace('+', '_')}_{architecture}"
            summary["models"][key] = {
                "architecture": architecture, "modality": modality,
                "width": width, "parameters": count,
                "parameter_difference": count - int(target),
            }
            for seed in args.seeds:
                directory = args.output_dir / key / f"seed{seed}"
                command = [
                    sys.executable, "scripts/train_architecture_baseline.py",
                    "--architecture", architecture, "--no-gripper-head",
                    "--width", str(width), "--root", *args.root,
                    "--models", modality, "--device", args.device,
                    "--epochs", str(args.epochs), "--seed", str(seed),
                    "--split-seed", str(args.split_seed),
                    "--raw-rate-hz", str(args.raw_rate_hz),
                    "--pixel-weight", ".35", "--position-weight", "1.0",
                    "--orientation-weight", "0", "--final-pose-weight", "0",
                    "--future-pose-weight", ".5",
                    "--future-pose-ms", str(args.future_pose_ms),
                    "--output-dir", str(directory),
                ]
                print(" ".join(command), flush=True)
                subprocess.run(command, check=True, cwd=ROOT)

    (args.output_dir / "motion_baseline_protocol.json").write_text(
        json.dumps(summary, indent=2))
    print(f"Protocol: {args.output_dir / 'motion_baseline_protocol.json'}")


if __name__ == "__main__":
    main()
