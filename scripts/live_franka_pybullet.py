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


def choose_trial(root, seed):
    paths = sorted(Path(root).expanduser().resolve().rglob("trial_*.csv"))
    if not paths:
        raise ValueError(f"no trial_*.csv found below {root}")
    return random.Random(seed).choice(paths)


def replay_csv(predictor, controller, trial_csv, interval_s, speed=1.,
               emit_fn=emit, sleep_fn=time.sleep):
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
    emit_fn({"event": "replay_started", "trial": str(path),
             "rows": int(len(order)), "input": "4 EMG + 24 IMU only",
             "vive_columns_ignored": True})
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
    emit_fn({"event": "replay_complete", "trial": str(path),
             "predictions": predictions})
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
    parser.add_argument("--simulation-steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.interval_ms, args.warmup_ms, args.poll_ms) <= 0 or args.speed < 0:
        parser.error("interval, warmup and poll must be positive")
    if args.trial_root:
        try:
            args.trial_csv = choose_trial(args.trial_root, args.trial_seed)
        except ValueError as error:
            parser.error(str(error))
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
        if args.trial_csv:
            replay_csv(predictor, controller, args.trial_csv,
                       args.interval_ms / 1000, args.speed)
        elif args.delsys_sdk_path:
            delsys_loop(predictor, controller, args.delsys_sdk_path,
                        args.interval_ms / 1000, args.poll_ms / 1000)
        else:
            stdin_loop(predictor, controller, args.interval_ms / 1000)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
