#!/usr/bin/env python3
"""Transfer live EMG+IMU model pose and interaction predictions to Franka."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.live_reach_grasp_orientation import (EMG_NAMES, IMU_NAMES,
                                                   delsys_loop, emit, stdin_loop)
from emg_touch.live_reach_grasp import LiveReachGraspPredictor
from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController,
                                                PoseMapper)
from emg_touch.physics.confidence_se3 import ConfidenceAwareSE3Controller


def choose_trial(root, seed):
    paths = sorted(Path(root).expanduser().resolve().rglob("trial_*.csv"))
    if not paths:
        raise ValueError(f"no trial_*.csv found below {root}")
    return random.Random(seed).choice(paths)


def replay_csv(predictor, controller, trial_csv, interval_s, speed=1.,
               settle_steps=240, emit_fn=emit, sleep_fn=time.sleep):
    """Causally replay one recorded trial using only its wearable columns."""
    path = Path(trial_csv).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"trial CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [name for name in EMG_NAMES + IMU_NAMES if name not in frame]
    if missing:
        raise ValueError("trial CSV is missing trained channels: " + ", ".join(missing))
    time_name = next((name for name in ("time_perf_counter", "time_s")
                      if name in frame), None)
    if time_name is None:
        raise ValueError("trial CSV needs time_perf_counter or time_s")
    stamps = pd.to_numeric(frame[time_name], errors="coerce").to_numpy(dtype=float)
    order = np.flatnonzero(np.isfinite(stamps))
    order = order[np.argsort(stamps[order], kind="stable")]
    if not len(order):
        raise ValueError("trial CSV has no finite timestamps")
    # Keep the first row for a duplicate stamp. Live preprocessing requires
    # strictly increasing time, matching the training preprocessor.
    keep = np.r_[True, np.diff(stamps[order]) > 0]
    order = order[keep]
    emg = frame[EMG_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    imu = frame[IMU_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    predictor.reset()
    controller.reset()
    vive_names = [f"VIVE_T0_pos_{axis}_m" for axis in "xyz"]
    reference = None
    if all(name in frame for name in vive_names):
        reference = frame[vive_names].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=float)[order]
        reference = reference[np.isfinite(reference).all(axis=1)]
        if len(reference):
            # A few hundred segments are visually continuous without flooding
            # the PyBullet debug renderer with every raw 1 kHz sample.
            stride = max(1, int(np.ceil(len(reference) / 300)))
            reference = reference[::stride]
            controller.set_reference_trajectory(reference)
    emit_fn({"event": "replay_started", "trial": str(path),
             "rows": int(len(order)), "input": "4 EMG + 24 IMU only",
             "vive_is_model_input": False,
             "vive_comparison_frames": 0 if reference is None else int(len(reference))})
    last_prediction = -np.inf
    wall_start, data_start = time.monotonic(), float(stamps[order[0]])
    predictions = 0
    for row in order:
        stamp = float(stamps[row])
        if speed > 0:
            delay = (stamp - data_start) / speed - (time.monotonic() - wall_start)
            if delay > 0:
                sleep_fn(delay)
        predictor.add_sample(stamp, emg[row], imu[row])
        if stamp - last_prediction < interval_s:
            continue
        try:
            result = controller.attach(predictor.predict())
        except RuntimeError:
            continue
        emit_fn(result)
        predictions += 1
        last_prediction = stamp
    settled = controller.settle(settle_steps) if hasattr(controller, "settle") else None
    emit_fn({"event": "replay_complete", "trial": str(path),
             "predictions": predictions, "final_settle": settled})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="orientation run's emg_imu_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval-ms", type=float, default=40.)
    parser.add_argument("--warmup-ms", type=float, default=200.)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--delsys-sdk-path")
    source.add_argument("--trial-csv",
                        help="Replay this unseen recorded trial causally")
    source.add_argument("--trial-root",
                        help="Choose one random trial_*.csv recursively")
    parser.add_argument("--trial-seed", type=int, default=42,
                        help="Random choice seed used with --trial-root")
    parser.add_argument("--speed", type=float, default=1.,
                        help="Recorded replay speed; 0 runs as fast as possible")
    parser.add_argument("--poll-ms", type=float, default=10.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--robot-base", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--robot-from-vive-translation", type=float, nargs=3)
    parser.add_argument("--robot-from-vive-quaternion-wxyz", type=float, nargs=4)
    parser.add_argument("--home-position", type=float, nargs=3, default=(.45, 0., .50))
    parser.add_argument("--home-orientation-wxyz", type=float, nargs=4,
                        default=(0., 1., 0., 0.))
    parser.add_argument("--trajectory-z-rotation-deg", type=float, default=90.,
                        help="Rotate mapped trajectory anticlockwise about robot Z")
    parser.add_argument("--simulation-steps", type=int, default=24,
                        help="Physics settling steps after each model prediction")
    parser.add_argument("--final-settle-steps", type=int, default=240,
                        help="Simulation steps after recorded data ends")
    parser.add_argument("--grasp-probability-threshold", type=float, default=.9)
    parser.add_argument("--release-probability-threshold", type=float, default=.9)
    parser.add_argument("--disable-confidence-aware-control", action="store_true")
    parser.add_argument("--workspace-lower", type=float, nargs=3,
                        default=(.20, -.45, .15))
    parser.add_argument("--workspace-upper", type=float, nargs=3,
                        default=(.75, .45, .85))
    parser.add_argument("--max-cartesian-velocity-mps", type=float, default=.35)
    parser.add_argument("--max-cartesian-acceleration-mps2", type=float, default=1.2)
    parser.add_argument("--max-angular-velocity-degps", type=float, default=90.)
    parser.add_argument("--max-angular-acceleration-degps2", type=float, default=360.)
    parser.add_argument("--position-hold-uncertainty-cm", type=float, default=20.)
    parser.add_argument("--orientation-hold-uncertainty-deg", type=float, default=60.)
    args = parser.parse_args()
    if (min(args.interval_ms, args.warmup_ms, args.poll_ms) <= 0
            or min(args.speed, args.final_settle_steps) < 0):
        parser.error("interval, warmup and poll must be positive")
    if args.trial_root:
        try:
            args.trial_csv = choose_trial(args.trial_root, args.trial_seed)
        except ValueError as error:
            parser.error(str(error))
    try:
        mapper = PoseMapper(args.robot_from_vive_translation,
            args.robot_from_vive_quaternion_wxyz,
            args.home_position, args.home_orientation_wxyz,
            args.trajectory_z_rotation_deg)
    except ValueError as error:
        parser.error(str(error))
    predictor = LiveReachGraspPredictor(args.checkpoint, args.device, args.warmup_ms)
    motion_filter = None
    if not args.disable_confidence_aware_control:
        try:
            motion_filter = ConfidenceAwareSE3Controller(
                args.workspace_lower, args.workspace_upper,
                args.max_cartesian_velocity_mps,
                args.max_cartesian_acceleration_mps2,
                args.max_angular_velocity_degps,
                args.max_angular_acceleration_degps2,
                position_hold_uncertainty_cm=args.position_hold_uncertainty_cm,
                orientation_hold_uncertainty_deg=args.orientation_hold_uncertainty_deg)
        except ValueError as error:
            parser.error(str(error))
    try:
        controller = LiveFrankaPyBulletController(
            gui=not args.headless, mapper=mapper, base_position=args.robot_base,
            simulation_steps=args.simulation_steps,
            grasp_probability_threshold=args.grasp_probability_threshold,
            release_probability_threshold=args.release_probability_threshold,
            motion_filter=motion_filter)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    emit({"event": "franka_ready", "input": "4 EMG + 24 IMU only",
          "ik_target": "constrained model XYZ + model quaternion",
          "confidence_aware_control": motion_filter is not None,
          "gui": not args.headless})
    try:
        if args.trial_csv:
            replay_csv(predictor, controller, args.trial_csv,
                       args.interval_ms / 1000, args.speed,
                       args.final_settle_steps)
        elif args.delsys_sdk_path:
            delsys_loop(predictor, controller, args.delsys_sdk_path,
                        args.interval_ms / 1000, args.poll_ms / 1000)
        else:
            stdin_loop(predictor, controller, args.interval_ms / 1000)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
