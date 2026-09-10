#!/usr/bin/env python3
"""Run V2 inference from an unseen CSV or newline-delimited live samples."""
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
from emg_touch.state_attention_inference import StateAttentionStream


EMG_COLUMNS = [f"EMG 1_{sensor}" for sensor in SENSORS]
IMU_COLUMNS = [f"{kind} {axis}_{sensor}" for sensor in SENSORS
               for kind in ("ACC", "GYRO") for axis in "XYZ"]


def emit(value):
    if value is not None:
        print(json.dumps(value, separators=(",", ":")), flush=True)


def replay(args):
    frame = pd.read_csv(args.trial_csv)
    missing = [name for name in ["time_perf_counter", *EMG_COLUMNS, *IMU_COLUMNS]
               if name not in frame]
    if missing:
        raise ValueError("missing required wearable columns: " + ", ".join(missing))
    declared = constant(frame, "sample_rate_hz_declared")
    raw_rate = args.raw_rate_hz if args.raw_rate_hz is not None else declared
    stream = StateAttentionStream(
        args.checkpoint, args.device, args.warmup_ms, args.context_ms,
        args.close_threshold, args.open_threshold, raw_rate)
    last_wall = time.perf_counter()
    last_stamp = None
    for _, row in frame.iterrows():
        stamp = pd.to_numeric(row["time_perf_counter"], errors="coerce")
        if not np.isfinite(stamp):
            continue
        if args.speed > 0 and last_stamp is not None:
            target = max(0., (float(stamp) - last_stamp) / args.speed)
            elapsed = time.perf_counter() - last_wall
            if target > elapsed:
                time.sleep(target - elapsed)
            last_wall = time.perf_counter()
        last_stamp = float(stamp)
        emg = pd.to_numeric(row[EMG_COLUMNS], errors="coerce").to_numpy(float)
        imu = pd.to_numeric(row[IMU_COLUMNS], errors="coerce").to_numpy(float)
        emit(stream.update(stamp, emg, imu, args.canvas_px))


def stdin_stream(args):
    stream = StateAttentionStream(
        args.checkpoint, args.device, args.warmup_ms, args.context_ms,
        args.close_threshold, args.open_threshold, args.raw_rate_hz)
    for line in sys.stdin:
        if not line.strip():
            continue
        sample = json.loads(line)
        emit(stream.update(sample["time_s"], sample["emg"], sample["imu"],
                           args.canvas_px))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trial-csv", type=Path,
                        help="optional unseen CSV; omit to consume JSON lines on stdin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canvas-px", type=int, nargs=2, default=(1920, 1080),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--warmup-ms", type=float, default=200.)
    parser.add_argument("--context-ms", type=float, default=2000.)
    parser.add_argument("--close-threshold", type=float, default=.7)
    parser.add_argument("--open-threshold", type=float, default=.3)
    parser.add_argument("--raw-rate-hz", type=float,
                        help="live acquisition rate; CSV mode auto-detects it")
    parser.add_argument("--speed", type=float, default=0.,
                        help="CSV replay speed; 0 runs without sleeping")
    args = parser.parse_args()
    if args.speed < 0:
        parser.error("--speed cannot be negative")
    replay(args) if args.trial_csv else stdin_stream(args)


if __name__ == "__main__":
    main()
