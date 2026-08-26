#!/usr/bin/env python3
"""Run fixed-weight continual touch inference from live JSON or a recorded CSV.

Live mode reads one JSON object per line from stdin. This keeps hardware I/O
separate: a BLE, serial, LSL, or socket collector only needs to emit the protocol
documented by ``--print-protocol``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from emg_touch.data.preprocessing import csv_to_signal_arrays
from emg_touch.deployment import ContinualTouchPredictor, DEPLOYABLE_KINDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALER = ROOT / "artifacts/continual_attention/mix7/fold-0/scaler.npz"
DEFAULT_CHECKPOINTS = {
    "grid_emg": ROOT / "runs/continual_attention/mix7/fold-0/grid_emg/best.pt",
    "grid_fusion": ROOT
    / "runs/continual_attention/mix7/fold-0/grid_fusion/best.pt",
}

PROTOCOL = {
    "channel_order": {
        "emg": ["EMG RMS 1_S0", "EMG RMS 1_S4", "EMG RMS 1_S8", "EMG RMS 1_S12"],
        "imu": [
            f"{quantity} {axis}_{sensor}"
            for sensor in ("S0", "S4", "S8", "S12")
            for quantity in ("ACC", "GYRO")
            for axis in ("X", "Y", "Z")
        ],
    },
    "events": [
        {"event": "start"},
        {
            "event": "sample",
            "time_s": 100.000,
            "emg": [0.0, 0.0, 0.0, 0.0],
            "imu": [0.0] * 24,
        },
        {
            "event": "movement_start",
            "time_s": 100.300,
            "note": "Send only after at least 0.3 s of rest for fusion.",
        },
        {"event": "touch", "time_s": 101.250},
    ],
}


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def build_predictor(args: argparse.Namespace) -> ContinualTouchPredictor:
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS[args.kind])
    return ContinualTouchPredictor(
        checkpoint,
        args.scaler,
        device=args.device,
        screen_width_px=args.screen_width,
        screen_height_px=args.screen_height,
    )


def add_arrays(
    predictor: ContinualTouchPredictor,
    arrays: dict[str, np.ndarray],
) -> None:
    for index, timestamp in enumerate(arrays["time_s"]):
        predictor.add_sample(
            float(timestamp),
            arrays["emg"][index],
            arrays["imu"][index] if predictor.requires_imu else None,
            emg_mask=arrays["emg_mask"][index],
            imu_mask=arrays["imu_mask"][index] if predictor.requires_imu else None,
        )


def replay_csv(args: argparse.Namespace, predictor: ContinualTouchPredictor) -> None:
    if args.movement_start_s is None:
        raise SystemExit("--movement-start-s is required with --input-csv")
    arrays = csv_to_signal_arrays(args.input_csv)
    add_arrays(predictor, arrays)
    predictor.set_movement_start(args.movement_start_s)
    final_time = float(arrays["time_s"][-1] if args.touch_s is None else args.touch_s)
    elapsed_final = final_time - float(args.movement_start_s)
    if elapsed_final < 0:
        raise SystemExit("--touch-s must not precede --movement-start-s")
    cutoff = 0.0
    while cutoff < elapsed_final - 1e-9:
        try:
            emit(
                predictor.predict(
                    float(args.movement_start_s) + cutoff,
                    label=f"{cutoff:.1f}s",
                )
            )
        except RuntimeError as error:
            emit({"event": "not_ready", "label": f"{cutoff:.1f}s", "reason": str(error)})
        cutoff += predictor.interval_s
    emit(predictor.predict(final_time, label="touch"))


def run_live(args: argparse.Namespace, predictor: ContinualTouchPredictor) -> None:
    next_elapsed = 0.0
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            event = message.get("event")
            if event == "start":
                predictor.reset()
                next_elapsed = 0.0
                emit({"event": "ready", "model_kind": predictor.kind})
            elif event == "sample":
                predictor.add_sample(
                    message["time_s"],
                    message["emg"],
                    message.get("imu"),
                    emg_mask=message.get("emg_mask"),
                    imu_mask=message.get("imu_mask"),
                )
                if predictor.movement_start_s is not None:
                    latest = float(message["time_s"])
                    available = latest - predictor.movement_start_s
                    while available + 1e-9 >= next_elapsed:
                        cutoff = predictor.movement_start_s + next_elapsed
                        try:
                            emit(predictor.predict(cutoff, label=f"{next_elapsed:.1f}s"))
                        except RuntimeError as error:
                            emit(
                                {
                                    "event": "not_ready",
                                    "label": f"{next_elapsed:.1f}s",
                                    "reason": str(error),
                                }
                            )
                        next_elapsed += predictor.interval_s
            elif event == "movement_start":
                predictor.set_movement_start(message["time_s"])
                next_elapsed = 0.0
                emit({"event": "movement_started", "time_s": float(message["time_s"])})
            elif event == "touch":
                emit(predictor.predict(message.get("time_s"), label="touch"))
            elif event == "reset":
                predictor.reset()
                next_elapsed = 0.0
                emit({"event": "ready", "model_kind": predictor.kind})
            elif event == "stop":
                return
            else:
                raise ValueError(f"Unknown event {event!r}")
        except Exception as error:
            emit({"event": "error", "line": line_number, "reason": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=DEPLOYABLE_KINDS, default="grid_fusion")
    parser.add_argument("--checkpoint", help="Defaults to the saved mix7 fold-0 checkpoint")
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--device", help="cpu, mps, cuda, or cuda:0")
    parser.add_argument("--screen-width", type=float, default=1536.0)
    parser.add_argument("--screen-height", type=float, default=774.0)
    parser.add_argument("--input-csv", help="Replay one recorded hardware CSV instead of stdin")
    parser.add_argument("--movement-start-s", type=float, help="Required for CSV replay")
    parser.add_argument("--touch-s", type=float, help="CSV replay endpoint; defaults to final sample")
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()

    if args.print_protocol:
        print(json.dumps(PROTOCOL, indent=2))
        return
    predictor = build_predictor(args)
    emit(
        {
            "event": "model_loaded",
            "model_kind": predictor.kind,
            "device": str(predictor.device),
            "checkpoint": str(predictor.checkpoint_path),
            "scaler": str(predictor.scaler_path),
            "interval_s": predictor.interval_s,
        }
    )
    if args.input_csv:
        replay_csv(args, predictor)
    else:
        run_live(args, predictor)


if __name__ == "__main__":
    main()
