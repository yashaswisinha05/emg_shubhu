#!/usr/bin/env python3
"""Run single-seed, retrained component and loss ablations for the final model."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "full": [],
    "emg_only": ["--modality", "emg"],
    "imu_only": ["--modality", "imu"],
    "no_state_conditioning": ["--no-state-conditioning"],
    "concat_fusion": ["--fusion-mode", "concat"],
    "no_residual_gru": ["--no-residual-gru"],
    "no_local_branch": ["--encoder-feature-mode", "context-only"],
    "no_context_branch": ["--encoder-feature-mode", "local-only"],
    "no_state_loss": ["--state-weight", "0"],
    "no_current_position_loss": ["--position-weight", "0"],
    "no_pixel_loss": ["--pixel-weight", "0"],
    "no_future_imu_loss": [
        "--reconstruction-horizon-ms", "0", "--emg-to-future-imu-weight", "0"],
    "no_future_200ms_loss": ["--future-pose-weight", "0"],
    "uniform_pixel_weight": ["--uniform-pixel-weighting"],
    "future_imu_200ms": ["--reconstruction-horizon-ms", "200"],
    "future_imu_500ms": ["--reconstruction-horizon-ms", "500"],
    "future_imu_with_imu_input": ["--auxiliary-imu-input"],
    "joint_pretrained": ["--classifier-mode", "joint"],
    "frozen_pretrained": ["--classifier-mode", "frozen"],
}


def value(source, *keys):
    for key in keys:
        if not isinstance(source, dict):
            return None
        source = source.get(key)
    return source if isinstance(source, (int, float)) else None


def metrics(result):
    test = result["emg+imu"] if "emg+imu" in result else (
        result["emg"] if "emg" in result else result["imu"])
    future = test.get("future_pose_by_ms", {}).get("200", {})
    quarters = test.get("click_pixel_error_by_quarter", [])
    return {
        "gripper_macro_f1": value(test, "gripper_macro_f1"),
        "current_position_cm": value(test, "position_cm"),
        "pixel_error_px": value(test, "click_pixel_error"),
        "pixel_50_75_px": quarters[2] if len(quarters) >= 3 else None,
        "pixel_75_100_px": quarters[3] if len(quarters) >= 4 else None,
        "future_200ms_cm": value(future, "position_cm"),
        "future_200ms_hold_cm": value(future, "hold_current_prediction_cm"),
        "trainable_parameters": value(result, "protocol", "trainable_parameters"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", required=True,
                        help="Stage-I checkpoint used as architecture template; full "
                             "single-stage variants reinitialize it")
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=tuple(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not Path(args.classifier_checkpoint).is_file():
        parser.error(f"classifier checkpoint not found: {args.classifier_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.variants:
        destination = args.output_dir / name
        result_path = destination / "results.json"
        if args.resume and result_path.is_file():
            print(f"[{name}] reusing {result_path}", flush=True)
            continue
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"{destination} is not empty; remove it or pass --resume")
        mode = "single-stage"
        overrides = list(VARIANTS[name])
        if "--classifier-mode" in overrides:
            index = overrides.index("--classifier-mode")
            mode = overrides[index + 1]
            del overrides[index:index + 2]
        command = [
            sys.executable, "scripts/train_shared_encoder_residual_gru.py",
            "--classifier-checkpoint", args.classifier_checkpoint,
            "--classifier-mode", mode,
            "--root", *args.root,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--split-seed", str(args.split_seed),
            "--raw-rate-hz", str(args.raw_rate_hz),
            "--output-dir", str(destination),
            *overrides,
        ]
        print(f"[{name}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    rows = {}
    for name in args.variants:
        path = args.output_dir / name / "results.json"
        rows[name] = metrics(json.loads(path.read_text()))
        rows[name]["results"] = str(path)

    full = rows.get("full")
    if full is not None:
        lower_is_better = {
            "current_position_cm", "pixel_error_px", "pixel_50_75_px",
            "pixel_75_100_px", "future_200ms_cm"}
        for name, row in rows.items():
            delta = {}
            for key, baseline in full.items():
                candidate = row.get(key)
                if key == "results" or not isinstance(baseline, (int, float)) \
                        or not isinstance(candidate, (int, float)):
                    continue
                change = candidate - baseline
                delta[key] = -change if key in lower_is_better else change
            row["improvement_over_full"] = delta

    summary = {
        "protocol": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "retrained_each_variant": True,
            "same_roots_split_optimizer_and_epochs": True,
            "positive_improvement_over_full_means_ablation_is_better": True,
            "future_imu_1000ms_is_the_full_model": True,
        },
        "variants": rows,
    }
    output = args.output_dir / "ablation_results.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
