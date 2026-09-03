#!/usr/bin/env python3
"""Serve true streaming inference for multiple wearable checkpoints.

The server accepts newly acquired raw EMG+IMU samples over HTTP or newline-
delimited JSON on stdin. It performs a fixed-weight causal forward pass at a
configurable interval and publishes the latest predictions to a browser UI.
It does not read or replay dataset trials.

Ground truth is an optional, separate UI event. It is used to calculate and
display pixel error but is never passed to ``LiveDistillationModel.predict``.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from emg_touch.data.schema import emg_columns, imu_columns  # noqa: E402
from emg_touch.live_distillation import (  # noqa: E402
    LiveDistillationModel,
    LiveFeaturePipeline,
    preprocessing_signature,
)


HTML_PATH = (
    REPOSITORY_ROOT / "src/emg_touch/static/live_distillation.html"
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "checkpoint must be NAME=/path/to/final.pt"
        )
    name, raw_path = value.split("=", 1)
    name, raw_path = name.strip(), raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "checkpoint must have a non-empty name and path"
        )
    return name, Path(raw_path)


class LiveApplication:
    """Thread-safe session state shared by ingestion and the browser."""

    def __init__(
        self,
        models: list[LiveDistillationModel],
        pipeline: LiveFeaturePipeline,
        interval_ms: float,
        canvas: tuple[float, float],
        maximum_predictions: int = 600,
    ) -> None:
        if not models:
            raise ValueError("at least one model checkpoint is required")
        self.models = models
        self.pipeline = pipeline
        self.interval_s = float(interval_ms) / 1000.0
        self.maximum_predictions = int(maximum_predictions)
        self.lock = threading.RLock()
        self.sequence = 0
        self.canvas = self._valid_canvas(canvas)
        self.target: dict[str, float] | None = None
        self.predictions: list[dict[str, Any]] = []
        self.status = "waiting"
        self.started_time_s: float | None = None
        self.last_prediction_time_s: float | None = None
        self.effective_rate_hz: float | None = None

    @staticmethod
    def _valid_canvas(canvas: Any) -> tuple[float, float]:
        values = tuple(map(float, canvas))
        if len(values) != 2 or min(values) <= 0 or not np.isfinite(values).all():
            raise ValueError("canvas must contain two positive finite values")
        return values

    def configuration(self) -> dict[str, Any]:
        return {
            "models": [
                {"name": model.name, "kind": model.kind}
                for model in self.models
            ],
            "interval_ms": 1000.0 * self.interval_s,
            "raw_emg_channels": list(emg_columns(self.models[0].config["data"])),
            "raw_imu_channels": list(imu_columns(self.models[0].config["data"])),
        }

    def reset(self, canvas: Any | None = None) -> None:
        with self.lock:
            if canvas is not None:
                self.canvas = self._valid_canvas(canvas)
            self.pipeline.reset()
            self.target = None
            self.predictions = []
            self.status = "waiting"
            self.started_time_s = None
            self.last_prediction_time_s = None
            self.effective_rate_hz = None
            self.sequence += 1

    def set_target(self, payload: dict[str, Any]) -> None:
        with self.lock:
            if "canvas" in payload:
                self.canvas = self._valid_canvas(payload["canvas"])
            target = {
                "x_px": float(payload["x_px"]),
                "y_px": float(payload["y_px"]),
            }
            if not np.isfinite(list(target.values())).all():
                raise ValueError("target coordinates must be finite")
            self.target = target
            self._attach_errors()
            self.sequence += 1

    def mark_movement_start(self, time_s: float | None = None) -> None:
        """Keep the resting buffer but begin a clean prediction trajectory."""
        with self.lock:
            latest = self.pipeline.latest_time_s
            timestamp = latest if time_s is None else float(time_s)
            if timestamp is None or not np.isfinite(timestamp):
                raise ValueError("movement_start needs a finite time_s")
            first = self.pipeline.first_time_s
            if first is not None and timestamp < first - 1e-9:
                raise ValueError("movement_start cannot precede the first sample")
            if latest is not None and timestamp > latest + 1e-9:
                raise ValueError("movement_start cannot be later than the newest sample")
            self.started_time_s = timestamp
            self.predictions = []
            self.last_prediction_time_s = None
            self.status = "live"
            if latest is not None:
                self._infer(latest)
            self.sequence += 1

    def _attach_errors(self) -> None:
        if self.target is None:
            return
        target = np.asarray(
            [self.target["x_px"], self.target["y_px"]], dtype=np.float64
        )
        for prediction in self.predictions:
            for model in prediction["models"]:
                point = np.asarray([model["x_px"], model["y_px"]])
                model["error_px"] = float(np.linalg.norm(point - target))

    def _ingest_samples(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "sample":
            self.pipeline.add_sample(
                payload["time_s"], payload["emg"], payload["imu"]
            )
        else:
            self.pipeline.add_samples(
                payload["time_s"], payload["emg"], payload["imu"]
            )
        latest = self.pipeline.latest_time_s
        if latest is None:
            return
        if self.started_time_s is None:
            self.started_time_s = self.pipeline.first_time_s
        self.status = "live"
        due = (
            self.last_prediction_time_s is None
            or latest - self.last_prediction_time_s + 1e-9 >= self.interval_s
        )
        if due:
            self._infer(latest)

    def _infer(self, timestamp: float) -> None:
        try:
            emg, imu, rate = self.pipeline.processed()
        except RuntimeError:
            return
        # A fresh stream is expected to need enough samples for one complete
        # transformer patch. Other runtime failures must remain visible.
        if any(len(emg) < model.patch_length for model in self.models):
            return
        outputs = [
            model.predict(emg, imu, self.canvas, rate)
            for model in self.models
        ]
        entry = {
            "sequence": self.sequence + 1,
            "time_s": timestamp,
            "elapsed_s": timestamp - float(self.started_time_s),
            "models": outputs,
        }
        self.predictions.append(entry)
        self.predictions = self.predictions[-self.maximum_predictions :]
        self.effective_rate_hz = rate
        self.last_prediction_time_s = timestamp
        self.sequence += 1
        self._attach_errors()

    def handle_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = payload.get("event")
        with self.lock:
            if event == "start" or event == "reset":
                self.reset(payload.get("canvas"))
            elif event in {"sample", "samples"}:
                self._ingest_samples(payload)
            elif event == "movement_start":
                self.mark_movement_start(payload.get("time_s"))
            elif event == "target":
                self.set_target(payload)
            elif event == "touch":
                latest = self.pipeline.latest_time_s
                if latest is not None and self.last_prediction_time_s != latest:
                    self._infer(latest)
                if "x_px" in payload and "y_px" in payload:
                    self.set_target(payload)
                self.status = "touched"
                self.sequence += 1
            else:
                raise ValueError(f"unknown event {event!r}")
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            latest = self.pipeline.latest_time_s
            return {
                "sequence": self.sequence,
                "status": self.status,
                "sample_count": self.pipeline.sample_count,
                "raw_rate_hz": self.pipeline.raw_rate_hz,
                "effective_rate_hz": self.effective_rate_hz,
                "latest_time_s": latest,
                "elapsed_s": (
                    latest - self.started_time_s
                    if latest is not None and self.started_time_s is not None
                    else 0.0
                ),
                "canvas": list(self.canvas),
                "target": self.target,
                "predictions": self.predictions,
            }


def make_handler(application: LiveApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "WearableIntentLive/1.0"

        def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/config":
                self._json(application.configuration())
            elif path == "/api/status":
                self._json(application.snapshot())
            elif path in {"/", "/index.html"}:
                body = HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/event":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 16 MiB")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("event body must be a JSON object")
                self._json(application.handle_event(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # noqa: BLE001
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            if self.command != "GET" or self.path != "/api/status":
                super().log_message(format, *args)

    return Handler


def read_stdin(application: LiveApplication) -> None:
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            application.handle_event(payload)
        except Exception as error:  # noqa: BLE001
            print(f"stdin event {line_number}: {error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        metavar="NAME=PATH",
        help="Repeat once per model to compare",
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval-ms", type=float, default=40.0)
    parser.add_argument("--screen-width", type=float, default=1920.0)
    parser.add_argument("--screen-height", type=float, default=1080.0)
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Also consume newline-delimited live events from stdin",
    )
    args = parser.parse_args()

    models = [
        LiveDistillationModel(name, path, args.device)
        for name, path in args.checkpoint
    ]
    reference = preprocessing_signature(models[0].config)
    incompatible = [
        model.name
        for model in models[1:]
        if preprocessing_signature(model.config) != reference
    ]
    if incompatible:
        raise SystemExit(
            "all live-comparison checkpoints must use identical preprocessing; "
            "incompatible: " + ", ".join(incompatible)
        )
    pipeline = LiveFeaturePipeline(models[0].config, args.calibration)
    application = LiveApplication(
        models,
        pipeline,
        args.interval_ms,
        (args.screen_width, args.screen_height),
    )
    if args.stdin:
        threading.Thread(
            target=read_stdin, args=(application,), daemon=True
        ).start()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(application)
    )
    print(
        f"live inference UI: http://{args.host}:{args.port} | "
        f"models: {', '.join(model.name for model in models)} | "
        f"device: {models[0].device}"
    )
    print(
        f"raw input: {pipeline.raw_emg_dim} EMG + {pipeline.raw_imu_dim} IMU; "
        f"prediction every {args.interval_ms:.0f} ms"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
