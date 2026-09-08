#!/usr/bin/env python3
"""Live EMG+IMU inference for interaction, XYZ position, and orientation.

Input is either newline-delimited JSON on stdin or a directly connected Delsys
collector. No VIVE value is accepted by the inference protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emg_touch.live_reach_grasp import (LiveReachGraspPredictor,
                                        LiveThreeRGripperController)

SENSORS = ["S0", "S4", "S8", "S12"]
EMG_NAMES = [f"EMG 1_{sensor}" for sensor in SENSORS]
IMU_NAMES = [f"{kind} {axis}_{sensor}" for sensor in SENSORS
             for kind in ["ACC", "GYRO"] for axis in "XYZ"]
PROTOCOL = {
    "channel_order": {"emg": EMG_NAMES, "imu": IMU_NAMES},
    "events": [
        {"event": "start"},
        {"event": "sample", "time_s": 100.0,
         "emg": [0.0] * 4, "imu": [0.0] * 24},
        {"event": "samples", "time_s": [100.001, 100.002],
         "emg": [[0.0] * 4, [0.0] * 4],
         "imu": [[0.0] * 24, [0.0] * 24]},
        {"event": "stop"},
    ],
}


def emit(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def safe_predict(predictor, controller):
    try:
        emit(controller.attach(predictor.predict()))
    except RuntimeError as error:
        emit({"event": "not_ready", "reason": str(error),
              "frames": predictor.pipeline.frames})


def add_batch(predictor, message):
    times = message["time_s"]
    emg, imu = message["emg"], message["imu"]
    if not (len(times) == len(emg) == len(imu)):
        raise ValueError("samples time_s/emg/imu lengths must agree")
    for stamp, e, i in zip(times, emg, imu):
        predictor.add_sample(stamp, e, i)


def stdin_loop(predictor, controller, interval_s):
    last_prediction = -np.inf
    for line_number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            event = message.get("event")
            if event in {"start", "reset"}:
                predictor.reset()
                controller.reset()
                last_prediction = -np.inf
                emit({"event": "ready"})
            elif event == "sample":
                predictor.add_sample(message["time_s"], message["emg"], message["imu"])
            elif event == "samples":
                add_batch(predictor, message)
            elif event == "predict":
                safe_predict(predictor, controller)
                continue
            elif event == "stop":
                return
            else:
                raise ValueError(f"unknown event {event!r}")
            latest = predictor.pipeline.last_time
            if latest is not None and latest - last_prediction >= interval_s:
                safe_predict(predictor, controller)
                last_prediction = latest
        except Exception as error:
            emit({"event": "error", "line": line_number, "reason": str(error)})


def channel_indices(names):
    position = {}
    for index, name in enumerate(names):
        if name in position:
            raise ValueError(f"duplicate live channel {name!r}")
        position[name] = index
    missing = [name for name in EMG_NAMES + IMU_NAMES if name not in position]
    if missing:
        raise ValueError(f"live scan is missing trained channels: {missing}")
    return ([position[name] for name in EMG_NAMES],
            [position[name] for name in IMU_NAMES])


def delsys_loop(predictor, controller, sdk_path, interval_s, poll_s):
    directory = Path(sdk_path).resolve()
    if not directory.is_dir():
        raise SystemExit(f"Delsys SDK directory does not exist: {directory}")
    sys.path.insert(0, str(directory))
    try:
        from EMGCollector import EMGCollector
    except ImportError as error:
        raise SystemExit(f"could not import EMGCollector from {directory}: {error}") from error
    collector = EMGCollector()
    collector.connect()
    collector.scan()
    collector.configure()
    emg_index, imu_index = channel_indices(collector.channel_names)
    collector.start_streaming()
    print("Delsys live stream started: 4 EMG + 24 IMU; Ctrl+C to stop", file=sys.stderr)
    last_sample = last_prediction = None
    try:
        while True:
            time.sleep(poll_s)
            if collector.ring_buffer is None:
                continue
            if last_sample is None:
                stamps, _ = collector.ring_buffer.get_latest(1)
                if not len(stamps):
                    continue
                last_sample = float(stamps[0]) - 1e-6
            stamps, values = collector.ring_buffer.get_since(last_sample)
            for stamp, row in zip(stamps, values):
                predictor.add_sample(float(stamp), row[emg_index], row[imu_index])
            if len(stamps):
                last_sample = float(stamps[-1])
                if last_prediction is None or last_sample - last_prediction >= interval_s:
                    safe_predict(predictor, controller)
                    last_prediction = last_sample
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop_streaming()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="emg_imu_best.pt from the orientation-hybrid run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval-ms", type=float, default=40.)
    parser.add_argument("--warmup-ms", type=float, default=200.)
    parser.add_argument("--delsys-sdk-path",
                        help="If supplied, connect directly instead of reading stdin")
    parser.add_argument("--poll-ms", type=float, default=10.)
    parser.add_argument("--link-lengths", type=float, nargs=2, default=(.50, .60),
                        metavar=("L1", "L2"))
    parser.add_argument("--initial-joint-deg", type=float, nargs=3,
                        default=(0., 20., 90.), metavar=("YAW", "SHOULDER", "ELBOW"))
    parser.add_argument("--base-world", type=float, nargs=3,
                        metavar=("X", "Y", "Z"),
                        help="Measured shoulder=P1 in VIVE world metres; omission uses synthetic anchoring")
    parser.add_argument("--axis-order", default="xyz")
    parser.add_argument("--axis-signs", type=float, nargs=3, default=(1., 1., 1.))
    parser.add_argument("--gripper-open-m", type=float, default=.08)
    parser.add_argument("--gripper-closed-m", type=float, default=.015)
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()
    if args.print_protocol:
        print(json.dumps(PROTOCOL, indent=2))
        return
    if min(args.interval_ms, args.warmup_ms, args.poll_ms) <= 0:
        parser.error("interval, warmup and poll must be positive")
    predictor = LiveReachGraspPredictor(args.checkpoint, args.device, args.warmup_ms)
    controller = LiveThreeRGripperController(
        args.link_lengths, args.initial_joint_deg, args.base_world,
        args.axis_order, args.axis_signs, args.gripper_open_m, args.gripper_closed_m)
    emit({"event": "model_loaded", "checkpoint": str(args.checkpoint),
          "device": str(predictor.device), "input": "4 EMG + 24 IMU only",
          "outputs": ["holding", "grasp", "release", "position_xyz",
                      "orientation_quaternion", "yaw_pitch_roll", "3r_joint_angles",
                      "gripper_open_close"]})
    if args.delsys_sdk_path:
        delsys_loop(predictor, controller, args.delsys_sdk_path,
                    args.interval_ms / 1000, args.poll_ms / 1000)
    else:
        stdin_loop(predictor, controller, args.interval_ms / 1000)


if __name__ == "__main__":
    main()
