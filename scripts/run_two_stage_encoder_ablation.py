#!/usr/bin/env python3
"""Compare frozen two-stage training with controlled joint alternatives."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metric(result, *keys):
    value = result
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return value if isinstance(value, (int, float)) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("frozen", "joint", "single-stage"),
                        default=("frozen", "joint", "single-stage"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for seed in args.seeds:
            destination = args.output_dir / mode / f"seed{seed}"
            results = destination / "results.json"
            if args.resume and results.is_file():
                print(f"[{mode} seed={seed}] reusing {results}", flush=True)
                continue
            if destination.exists() and any(destination.iterdir()):
                parser.error(f"{destination} is not empty; remove it or pass --resume")
            command = [
                sys.executable, "scripts/train_shared_encoder_residual_gru.py",
                "--classifier-checkpoint", args.classifier_checkpoint,
                "--classifier-mode", mode,
                "--root", *args.root,
                "--models", "emg+imu",
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--seed", str(seed),
                "--split-seed", str(args.split_seed),
                "--raw-rate-hz", str(args.raw_rate_hz),
                "--output-dir", str(destination),
            ]
            print(f"[{mode} seed={seed}] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    summary = {
        "question": "Does freezing a separately trained state encoder improve generalization?",
        "controlled_variables": [
            "architecture", "inputs", "trial split", "motion heads",
            "motion losses", "optimizer settings",
        ],
        "interpretation": {
            "frozen_beats_joint": "Stage separation prevents motion losses from erasing state features.",
            "joint_beats_frozen": "Freezing is unnecessary; end-to-end adaptation is beneficial.",
            "single_stage_matches_frozen": "Separate classifier pretraining is unnecessary.",
        },
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "modes": {},
    }
    for mode in args.modes:
        runs = []
        for seed in args.seeds:
            path = args.output_dir / mode / f"seed{seed}" / "results.json"
            result = json.loads(path.read_text())
            test = result["emg+imu"]
            future = test.get("future_pose_by_ms", {}).get("200", {})
            runs.append({
                "seed": seed,
                "gripper_macro_f1": metric(test, "gripper_macro_f1"),
                "current_position_cm": metric(test, "position_cm"),
                "pixel_error_px": metric(test, "click_pixel_error"),
                "late_pixel_error_px": test.get("click_pixel_error_by_quarter", [None])[-1],
                "future_200ms_cm": metric(future, "position_cm"),
                "trainable_parameters": result.get("protocol", {}).get(
                    "trainable_parameters"),
                "results": str(path),
            })
        aggregates = {}
        for key in ("gripper_macro_f1", "current_position_cm", "pixel_error_px",
                    "late_pixel_error_px", "future_200ms_cm"):
            values = [run[key] for run in runs if run[key] is not None]
            aggregates[key] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.,
            }
        summary["modes"][mode] = {"runs": runs, "aggregate": aggregates}
    output = args.output_dir / "two_stage_ablation.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
