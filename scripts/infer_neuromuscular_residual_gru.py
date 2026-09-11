#!/usr/bin/env python3
"""Live JSON-lines or unseen-CSV inference for the residual GRU paper model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.reach_grasp import SENSORS, constant
from emg_touch.residual_gru_inference import ResidualGRUStream


EMG_COLUMNS = [f"EMG 1_{sensor}" for sensor in SENSORS]
IMU_COLUMNS = [f"{kind} {axis}_{sensor}" for sensor in SENSORS
               for kind in ("ACC", "GYRO") for axis in "XYZ"]


def emit(result):
    if result is not None:
        print(json.dumps(result, separators=(",", ":")), flush=True)


def make_stream(args, raw_rate=None):
    return ResidualGRUStream(
        args.checkpoint, args.device, args.warmup_ms, args.context_ms,
        args.close_threshold, args.open_threshold, raw_rate)


def replay(args):
    frame = pd.read_csv(args.trial_csv)
    required = ["time_perf_counter", *EMG_COLUMNS, *IMU_COLUMNS]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError("missing wearable columns: " + ", ".join(missing))
    raw_rate = args.raw_rate_hz or constant(frame, "sample_rate_hz_declared")
    stream = make_stream(args, raw_rate)
    previous_stamp = previous_wall = None
    for _, row in frame.iterrows():
        stamp = pd.to_numeric(row["time_perf_counter"], errors="coerce")
        if not np.isfinite(stamp):
            continue
        if args.speed > 0 and previous_stamp is not None:
            wait = (float(stamp) - previous_stamp) / args.speed
            elapsed = time.perf_counter() - previous_wall
            if wait > elapsed:
                time.sleep(wait - elapsed)
        previous_stamp, previous_wall = float(stamp), time.perf_counter()
        emg = pd.to_numeric(row[EMG_COLUMNS], errors="coerce").to_numpy(float)
        imu = pd.to_numeric(row[IMU_COLUMNS], errors="coerce").to_numpy(float)
        emit(stream.update(stamp, emg, imu, args.canvas_px))


def stdin_stream(args):
    stream = make_stream(args, args.raw_rate_hz)
    for line in sys.stdin:
        if line.strip():
            sample = json.loads(line)
            emit(stream.update(sample["time_s"], sample["emg"], sample["imu"],
                               args.canvas_px))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trial-csv", type=Path,
                        help="omit to consume {time_s, emg[4], imu[24]} JSON lines")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canvas-px", nargs=2, type=int, default=(1920, 1080),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--warmup-ms", type=float, default=300.)
    parser.add_argument("--context-ms", type=float, default=2000.)
    parser.add_argument("--close-threshold", type=float, default=.7)
    parser.add_argument("--open-threshold", type=float, default=.3)
    parser.add_argument("--raw-rate-hz", type=float)
    parser.add_argument("--speed", type=float, default=0.)
    args = parser.parse_args()
    replay(args) if args.trial_csv else stdin_stream(args)


if __name__ == "__main__":
    main()
