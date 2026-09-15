#!/usr/bin/env python3
"""Stream raw EMG+IMU or replay one CSV through the minimal model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.reach_grasp import SENSORS, constant
from emg_touch.live_reach_grasp import LiveReachGraspPreprocessor
from minimal_emg_imu.model import MinimalEMGIMUModel

EMG_COLUMNS = [f"EMG 1_{sensor}" for sensor in SENSORS]
IMU_COLUMNS = [f"{kind} {axis}_{sensor}" for sensor in SENSORS
               for kind in ("ACC", "GYRO") for axis in "XYZ"]


class MinimalEMGIMUStream:
    """Raw-sample API; VIVE and labels are never accepted as inputs."""

    def __init__(self, checkpoint, device="cuda", raw_rate_hz=None,
                 context_ms=2000., warmup_ms=300.,
                 close_threshold=.7, open_threshold=.3):
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= open < close <= 1")
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if state.get("format") != "minimal_emg_imu_v1":
            raise ValueError("checkpoint must come from minimal_emg_imu/train.py")
        self.model = MinimalEMGIMUModel(**state["model_args"]).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().requires_grad_(False)
        self.stats = state["normalization"]
        settings = dict(state["preprocessing"])
        if raw_rate_hz is not None:
            settings["raw_rate_hz"] = float(raw_rate_hz)
        self.pipeline = LiveReachGraspPreprocessor(settings)
        self.context_frames = max(1, round(context_ms * self.pipeline.rate / 1000.))
        self.warmup_s = warmup_ms / 1000.
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.reset()

    def reset(self):
        self.pipeline.reset()
        self.gripper_state = "open"
        self.last_prediction_frame = 0

    @torch.no_grad()
    def predict(self, canvas_px=(1920, 1080)):
        arrays = self.pipeline.arrays()
        time_s, emg, imu, emg_valid, imu_valid = [
            value[-self.context_frames:] for value in arrays]
        if not len(time_s) or time_s[-1] - time_s[0] < self.warmup_s:
            return None
        if emg_valid[-1].mean() < .75 or imu_valid[-1].mean() < .75:
            return {"valid": False, "time_s": float(time_s[-1]),
                    "gripper_state": self.gripper_state,
                    "reason": "insufficient recent EMG or IMU channels"}

        def pack(values, valid, key):
            z = ((values - np.asarray(self.stats[key]["mean"]))
                 / np.asarray(self.stats[key]["std"]))
            return np.concatenate((
                np.where(valid, np.clip(z, -20, 20), 0), valid), -1).astype("float32")

        output = self.model(
            torch.from_numpy(pack(emg, emg_valid, "emg"))[None].to(self.device),
            torch.from_numpy(pack(imu, imu_valid, "imu"))[None].to(self.device))
        probability = output["gripper_state_logits"][0, -1].softmax(-1)
        close_probability = float(probability[1])
        if close_probability >= self.close_threshold:
            self.gripper_state = "close"
        elif close_probability <= self.open_threshold:
            self.gripper_state = "open"
        scale = np.asarray(self.stats["position"]["std"])
        mean = np.asarray(self.stats["position"]["mean"])
        position = output["position"][0, -1].cpu().numpy() * scale + mean
        future = output["future_position"][0, -1].cpu().numpy() * scale + mean
        pixel_norm = output["click"][0, -1].cpu().numpy()
        canvas = np.asarray(canvas_px, dtype=float)
        return {
            "valid": True,
            "time_s": float(time_s[-1]),
            "gripper_state": self.gripper_state,
            "open_probability": float(probability[0]),
            "close_probability": close_probability,
            "position_m": position.tolist(),
            "pixel_normalized_xy": pixel_norm.tolist(),
            "pixel_xy": (pixel_norm * canvas).tolist(),
            "future_horizons_ms": list(range(10, 201, 10)),
            "future_positions_m": future.tolist(),
        }

    def update(self, time_s, emg_4, imu_24, canvas_px=(1920, 1080)):
        self.pipeline.add_sample(time_s, emg_4, imu_24)
        if self.pipeline.frames == self.last_prediction_frame:
            return None
        self.last_prediction_frame = self.pipeline.frames
        return self.predict(canvas_px)


def replay(args):
    frame = pd.read_csv(args.trial_csv)
    required = ["time_perf_counter", *EMG_COLUMNS, *IMU_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("missing wearable columns: " + ", ".join(missing))
    frame["_time"] = pd.to_numeric(frame["time_perf_counter"], errors="coerce")
    frame = frame[np.isfinite(frame["_time"])].sort_values(
        "_time", kind="stable").drop_duplicates("_time", keep="last")
    raw_rate = args.raw_rate_hz or constant(frame, "sample_rate_hz_declared")
    stream = MinimalEMGIMUStream(
        args.checkpoint, args.device, raw_rate, args.context_ms, args.warmup_ms,
        args.close_threshold, args.open_threshold)
    previous = None
    for _, row in frame.iterrows():
        stamp = float(row["_time"])
        if args.speed > 0 and previous is not None:
            time.sleep(max(0., (stamp - previous) / args.speed))
        previous = stamp
        result = stream.update(
            stamp,
            pd.to_numeric(row[EMG_COLUMNS], errors="coerce").to_numpy(float),
            pd.to_numeric(row[IMU_COLUMNS], errors="coerce").to_numpy(float),
            args.canvas_px)
        if result is not None:
            print(json.dumps(result, separators=(",", ":")), flush=True)


def stdin_stream(args):
    stream = MinimalEMGIMUStream(
        args.checkpoint, args.device, args.raw_rate_hz,
        args.context_ms, args.warmup_ms,
        args.close_threshold, args.open_threshold)
    for line in sys.stdin:
        if line.strip():
            sample = json.loads(line)
            result = stream.update(
                sample["time_s"], sample["emg"], sample["imu"], args.canvas_px)
            if result is not None:
                print(json.dumps(result, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trial-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canvas-px", nargs=2, type=int, default=(1920, 1080))
    parser.add_argument("--raw-rate-hz", type=float)
    parser.add_argument("--context-ms", type=float, default=2000.)
    parser.add_argument("--warmup-ms", type=float, default=300.)
    parser.add_argument("--close-threshold", type=float, default=.7)
    parser.add_argument("--open-threshold", type=float, default=.3)
    parser.add_argument("--speed", type=float, default=0.)
    args = parser.parse_args()
    replay(args) if args.trial_csv else stdin_stream(args)


if __name__ == "__main__":
    main()
