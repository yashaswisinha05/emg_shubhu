#!/usr/bin/env python3
"""Transfer live EMG+IMU model pose and interaction predictions to Franka."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.live_reach_grasp_orientation import (delsys_loop, emit, stdin_loop)
from emg_touch.live_reach_grasp import LiveReachGraspPredictor
from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController,
                                                PoseMapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="orientation run's emg_imu_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval-ms", type=float, default=40.)
    parser.add_argument("--warmup-ms", type=float, default=200.)
    parser.add_argument("--delsys-sdk-path")
    parser.add_argument("--poll-ms", type=float, default=10.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--robot-base", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--robot-from-vive-translation", type=float, nargs=3)
    parser.add_argument("--robot-from-vive-quaternion-wxyz", type=float, nargs=4)
    parser.add_argument("--home-position", type=float, nargs=3, default=(.45, 0., .50))
    parser.add_argument("--home-orientation-wxyz", type=float, nargs=4,
                        default=(0., 1., 0., 0.))
    parser.add_argument("--simulation-steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.interval_ms, args.warmup_ms, args.poll_ms) <= 0:
        parser.error("interval, warmup and poll must be positive")
    try:
        mapper = PoseMapper(args.robot_from_vive_translation,
            args.robot_from_vive_quaternion_wxyz,
            args.home_position, args.home_orientation_wxyz)
    except ValueError as error:
        parser.error(str(error))
    predictor = LiveReachGraspPredictor(args.checkpoint, args.device, args.warmup_ms)
    try:
        controller = LiveFrankaPyBulletController(
            gui=not args.headless, mapper=mapper, base_position=args.robot_base,
            simulation_steps=args.simulation_steps)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    emit({"event": "franka_ready", "input": "4 EMG + 24 IMU only",
          "ik_target": "model XYZ + model quaternion", "gui": not args.headless})
    try:
        if args.delsys_sdk_path:
            delsys_loop(predictor, controller, args.delsys_sdk_path,
                        args.interval_ms / 1000, args.poll_ms / 1000)
        else:
            stdin_loop(predictor, controller, args.interval_ms / 1000)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
