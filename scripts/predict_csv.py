#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from emg_touch.checkpointing import load_model_state
from emg_touch.config import load_config
from emg_touch.data.preprocessing import (
    RobustScaler,
    causal_median_filter,
    csv_to_signal_arrays,
    previous_sample_resample,
)
from emg_touch.models.factory import build_model
from emg_touch.training import forward_model
from emg_touch.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a touch location from one trial CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=["teacher", "student", "tcn", "patchtst"], required=True)
    parser.add_argument("--scaler")
    parser.add_argument("--cutoff", type=float, default=-1.0, help="Cue-relative seconds; -1 uses full trial")
    parser.add_argument("--canvas-width", type=float, default=1536.0)
    parser.add_argument("--canvas-height", type=float, default=774.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model = build_model(args.kind, config).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()
    scaler = RobustScaler.load(args.scaler or config["paths"]["scaler"])
    arrays = csv_to_signal_arrays(args.csv_path)
    cutoff = float(arrays["time_s"][-1]) if args.cutoff < 0 else min(
        args.cutoff, float(arrays["time_s"][-1])
    )
    rate = float(config["data"]["sample_rate_hz"])
    length = int(config["data"]["context_length"])
    grid = cutoff - np.arange(length - 1, -1, -1) / rate
    emg, emg_mask = previous_sample_resample(
        arrays["time_s"], arrays["emg"], arrays["emg_mask"], grid
    )
    imu, imu_mask = previous_sample_resample(
        arrays["time_s"], arrays["imu"], arrays["imu_mask"], grid
    )
    emg = causal_median_filter(emg, int(config["data"]["median_kernel"]))
    emg = scaler.transform_emg(emg).astype(np.float32)
    imu = scaler.transform_imu(imu).astype(np.float32)
    emg[~emg_mask] = 0.0
    imu[~imu_mask] = 0.0
    batch = {
        "emg": torch.from_numpy(emg).unsqueeze(0).to(device),
        "emg_mask": torch.from_numpy(emg_mask).unsqueeze(0).to(device),
        "imu": torch.from_numpy(imu).unsqueeze(0).to(device),
        "imu_mask": torch.from_numpy(imu_mask).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        outputs = forward_model(model, batch, args.kind)
        normalized = outputs["prediction"][0].cpu().numpy()
        result = {
            "cutoff_s": cutoff,
            "x_norm": float(normalized[0]),
            "y_norm": float(normalized[1]),
            "x_px": float(normalized[0] * args.canvas_width),
            "y_px": float(normalized[1] * args.canvas_height),
        }
        if args.kind in {"teacher", "student"} and args.samples > 0:
            samples = model.sample(outputs, args.samples)[0].cpu().numpy()
            result["sampled_coordinates_norm"] = samples.tolist()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

