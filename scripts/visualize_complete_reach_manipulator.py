#!/usr/bin/env python3
"""Animate a 3R arm along the complete-reach EMG+IMU prediction.

The checkpoint is queried at ``--observation-lead-ms`` (zero by default, so
the complete wearable trial is available).  Its complete onset-to-touch 3D
path is anchored at the manipulator's initial hand pose and passed directly to
analytical inverse kinematics.  VIVE is rendered only as a black comparison
path and never enters the student model.

Example:

    python scripts/visualize_complete_reach_manipulator.py \
      --root "/media/.../emg_imu_vive" \
      --checkpoint runs/complete_reach/final.pt \
      --cache-dir artifacts/tracked_cache_posture \
      --device cuda --observation-lead-ms 0 --num-trials 4 --save-gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_complete_reach_model as complete_training  # noqa: E402
from scripts import train_latent_distillation_model as training  # noqa: E402
from scripts.visualize_wearable_manipulator import (  # noqa: E402
    one_row,
    run_viewer,
    transform_axes,
)
from emg_touch.live_distillation import LiveDistillationModel  # noqa: E402
from emg_touch.physics.manipulator_ik import ThreeRManipulator  # noqa: E402


@torch.inference_mode()
def collect_complete_reach_trials(
    runner: LiveDistillationModel,
    loader: torch.utils.data.DataLoader,
    manipulator: ThreeRManipulator,
    count: int,
    observation_lead_samples: int,
    initial_angles: np.ndarray,
    base_world: np.ndarray | None,
    axis_order: str,
    axis_signs: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Run the wearable model and turn its complete path into IK requests."""
    supported_kinds = {
        "complete_reach",
        "direction_aware_complete_reach",
        "monotonic_complete_reach",
        "task_separated_complete_reach",
    }
    if runner.kind not in supported_kinds:
        raise ValueError(
            "this viewer requires a complete-reach, direction-aware, hard "
            "monotonic, or task-separated best.pt/final.pt checkpoint; "
            f"received {runner.kind}"
        )
    config = runner.config
    model = runner.model
    device = runner.device
    steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    rate = training.effective_rate(config)
    trained_window = sorted(map(
        float,
        config.get("distillation", {}).get("lead_window_ms", [0.0, 400.0]),
    ))
    generator = np.random.default_rng(0)
    trials: list[dict[str, Any]] = []

    for raw_batch in loader:
        if raw_batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        for row in range(len(batch["lengths"])):
            length = int(batch["lengths"][row])
            touch = length - 1
            onset = int(np.clip(int(batch["onset"][row]), 0, max(0, touch - 1)))
            cut = touch - int(observation_lead_samples)
            if cut < 0:
                continue
            single = one_row(batch, row)
            window = complete_training.make_complete_reach_window(
                single,
                runner.context_samples,
                runner.patch_length,
                steps,
                generator,
                trajectory_limit,
                velocity_scale,
                fallback_canvas=None,
                fixed_lead=int(observation_lead_samples),
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"],
                window["imu"],
                window["time_mask"],
                sample=False,
            )
            predicted_relative = transform_axes(
                outputs["complete_trajectory"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            endpoint_relative = transform_axes(
                outputs["endpoint_3d"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            true_relative = transform_axes(
                window["complete_trajectory_target"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            true_endpoint_relative = transform_axes(
                window["endpoint_3d_target"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )

            vive_world = batch["position"][row].detach().cpu().numpy()
            if base_world is None:
                start_hand = manipulator.forward(initial_angles)[-1]
                starting_angles = initial_angles
                calibration = "synthetic initial pose"
                initial_was_projected = False
            else:
                start_hand = transform_axes(
                    vive_world[onset] - base_world, axis_order, axis_signs
                )
                initial_solution = manipulator.inverse(
                    start_hand, previous=initial_angles
                )
                start_hand = initial_solution.projected
                starting_angles = initial_solution.angles
                calibration = "measured shoulder/base"
                initial_was_projected = initial_solution.was_projected

            # The first point is a shared onset anchor. Every following target
            # is the complete-trajectory head's actual wearable-only output.
            predicted_path = np.concatenate([
                start_hand[None, :],
                start_hand[None, :] + predicted_relative,
            ])
            vive_path = np.concatenate([
                start_hand[None, :],
                start_hand[None, :] + true_relative,
            ])
            explicit_endpoint = start_hand + endpoint_relative
            true_endpoint = start_hand + true_endpoint_relative
            followed = manipulator.follow(
                predicted_path, initial_angles=starting_angles
            )
            model_error_cm = 100.0 * np.linalg.norm(
                predicted_path - vive_path, axis=-1
            )
            ik_error_cm = 100.0 * np.linalg.norm(
                followed["chain"][:, -1] - predicted_path, axis=-1
            )
            endpoint_agreement_cm = 100.0 * float(np.linalg.norm(
                explicit_endpoint - predicted_path[-1]
            ))
            endpoint_error_cm = 100.0 * float(np.linalg.norm(
                explicit_endpoint - true_endpoint
            ))
            duration_ms = 1000.0 * max(1, touch - onset) / rate
            # The prepended onset anchor and first predicted path sample share
            # time zero. This makes the plotted duration physically meaningful.
            time_ms = np.concatenate([
                [0.0], np.linspace(0.0, duration_ms, steps)
            ])
            lead_ms = 1000.0 * int(observation_lead_samples) / rate
            source = window["source_paths"][0]
            trials.append({
                "path": source,
                "label": Path(source).stem,
                "calibration": calibration,
                "initial_was_projected": initial_was_projected,
                "vive_path": vive_path,
                "predicted_path": predicted_path,
                "forecast_true_path": vive_path,
                "followed": followed,
                "model_followed": followed,
                "model_error_cm": model_error_cm,
                "ik_error_cm": ik_error_cm,
                "time_ms": time_ms,
                "forecast_time_ms": time_ms,
                "forecast_start_ms": 0.0,
                "lead_ms": lead_ms,
                "trained_max_ms": trained_window[1],
                "out_of_range": not (
                    trained_window[0] - 1e-6
                    <= lead_ms
                    <= trained_window[1] + 1e-6
                ),
                "trajectory_window": "complete-reach",
                "cut_samples_past_onset": cut - onset,
                "explicit_endpoint": explicit_endpoint,
                "true_endpoint": true_endpoint,
                "endpoint_agreement_cm": endpoint_agreement_cm,
                "endpoint_error_cm": endpoint_error_cm,
                "prediction_label": "EMG+IMU complete predicted path",
                "model_error_label": "complete model path vs VIVE",
                "time_axis_label": "time from movement onset (ms)",
            })
            if len(trials) >= count:
                return trials
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument(
        "--observation-lead-ms",
        type=float,
        default=0.0,
        help="When to query the model; 0 uses the complete wearable trial",
    )
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--link-lengths", type=float, nargs=2, default=(0.50, 0.60))
    parser.add_argument(
        "--initial-joint-deg",
        type=float,
        nargs=3,
        default=(0.0, 20.0, 90.0),
        metavar=("YAW", "SHOULDER", "ELBOW"),
    )
    parser.add_argument(
        "--base-world",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Measured shoulder/base position in VIVE-world metres",
    )
    parser.add_argument("--axis-order", default="xyz")
    parser.add_argument(
        "--axis-signs",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--output-dir", default="runs/complete_reach_manipulator")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.num_trials < 1:
        raise SystemExit("--num-trials must be positive")
    if args.observation_lead_ms < 0.0:
        raise SystemExit("--observation-lead-ms cannot be negative")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")

    runner = LiveDistillationModel(
        "complete-reach manipulator", args.checkpoint, args.device
    )
    if runner.kind not in {
        "complete_reach",
        "direction_aware_complete_reach",
        "monotonic_complete_reach",
        "task_separated_complete_reach",
    }:
        raise SystemExit(
            "checkpoint must come from a complete-reach trainer; "
            f"detected {runner.kind}"
        )
    loaders = training.build_experiment_loaders(
        runner.config, args.root, Path(args.cache_dir)
    )
    loader = loaders[{"train": 0, "validation": 1, "test": 2}[args.split]]
    rate = training.effective_rate(runner.config)
    lead_samples = complete_training.milliseconds_to_samples(
        args.observation_lead_ms, rate
    )
    manipulator = ThreeRManipulator(tuple(args.link_lengths))
    initial_angles = np.deg2rad(
        np.asarray(args.initial_joint_deg, dtype=np.float64)
    )
    base_world = (
        None
        if args.base_world is None
        else np.asarray(args.base_world, dtype=np.float64)
    )
    trials = collect_complete_reach_trials(
        runner,
        loader,
        manipulator,
        args.num_trials,
        lead_samples,
        initial_angles,
        base_world,
        args.axis_order.lower(),
        tuple(args.axis_signs),
    )
    if not trials:
        raise SystemExit("no usable trials were long enough for this observation lead")

    print(
        f"loaded {runner.kind} | {len(trials)} {args.split} trial(s) | "
        f"observation lead={1000.0 * lead_samples / rate:.1f} ms"
    )
    print("deployment check: complete path receives EMG+IMU only")
    print(
        "visualisation check: orange arm follows the cyan complete-reach model "
        "path; black VIVE is comparison only; purple star is endpoint head"
    )
    print("base=P1=[0,0,0] in manipulator coordinates")
    for index, trial in enumerate(trials):
        followed = trial["followed"]
        print(
            f"  {index}: {trial['label']} | model↔VIVE path="
            f"{trial['model_error_cm'][1:].mean():.2f}cm | explicit endpoint↔VIVE="
            f"{trial['endpoint_error_cm']:.2f}cm | endpoint↔path="
            f"{trial['endpoint_agreement_cm']:.2f}cm | IK projection="
            f"{int(followed['was_projected'].sum())}/{len(followed['was_projected'])} "
            f"| IK/FK residual={trial['ik_error_cm'].mean():.4f}cm"
        )
    run_viewer(
        trials,
        manipulator,
        Path(args.output_dir),
        show=not args.no_show,
        save_gif=args.save_gif,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
