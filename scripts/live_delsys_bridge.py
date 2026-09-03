#!/usr/bin/env python3
"""Bridge: real Delsys Trigno hardware -> the already-running live-inference server.

scripts/run_live_distillation_ui.py already implements a complete, data-
-source-agnostic live-inference backend (LiveFeaturePipeline + LiveDistillation
Model, an HTTP /api/event ingestion endpoint, a browser UI) - it does not
care where samples come from, only that something POSTs
{"event": "sample"/"samples", "time_s", "emg", "imu"} to it. That server is
NOT modified here, per "change a different script" - this is a new,
separate process that feeds it.

This is the missing half: a client that connects to the SAME Delsys
hardware /Users/.../Example-Applications/Python/run_simple.py drives for
data collection (EMGCollector, unmodified - connect/scan/configure/
start_streaming reused verbatim, since that is proven, already-working code
for this hardware), reads new samples off its ring buffer, and forwards
them to the live-inference server over HTTP.

    # terminal 1 - the inference server (already exists, unmodified)
    python scripts/run_live_distillation_ui.py \\
        --checkpoint "Best=runs/channel_horizon_distillation/final.pt" \\
        --calibration runs/channel_horizon_distillation/calibration.npz \\
        --device cuda
    # open http://127.0.0.1:8765 in a browser

    # terminal 2 - this bridge, run where the Delsys base station is attached
    python scripts/live_delsys_bridge.py \\
        --delsys-sdk-path "/Users/yashaswi/Downloads/Example-Applications/Python" \\
        --config configs/tracked_channel_horizon_distillation.yaml \\
        --server http://127.0.0.1:8765

WHAT COULD NOT BE VERIFIED HERE, stated plainly rather than glossed over:
the actual hardware calls (Connect_Callback, scan_callback, PollDataByString)
need the Delsys AeroPy SDK and a physically attached Trigno base station -
neither exists in this environment. Those calls are reused verbatim from
EMGCollector.py, unmodified, rather than reimplemented, which is the most
this environment can do to reduce risk on that half - it cannot test it.

WHAT WAS VERIFIED: the channel remapping (Delsys's scan-order channel names
-> the exact column order emg_columns()/imu_columns() expect, the same
order every training script in this project reads) is checked against
synthetic channel-name lists shaped exactly like collector.channel_names
produces (confirmed against TrialSaver.py: the real dataset's "EMG 1_S0"
column names come from this exact property, not its Slot-N fallback). And
the HTTP-forwarding half - build a batch, POST it, confirm the server
ingests it and starts producing predictions - was verified against a real,
locally-run instance of run_live_distillation_ui.py, not assumed to work
because the endpoint exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.schema import emg_columns, imu_columns  # noqa: E402


def channel_index_map(
    live_channel_names: list[str], data_config: dict,
) -> tuple[list[int], list[int]]:
    """Delsys scan-order channel names -> training column order indices.

    Raises with the exact missing/duplicate names rather than silently
    mis-assigning a column - a sensor that failed to pair or was scanned in
    a different slot must fail loudly here, not feed the model a
    quietly-wrong channel.
    """
    position = {}
    for index, name in enumerate(live_channel_names):
        if name in position:
            raise ValueError(
                f"duplicate live channel name {name!r} at indices "
                f"{position[name]} and {index} - cannot map unambiguously"
            )
        position[name] = index

    def resolve(expected: tuple[str, ...], label: str) -> list[int]:
        missing = [name for name in expected if name not in position]
        if missing:
            raise ValueError(
                f"{label}: {len(missing)} channel(s) trained on are not in the "
                f"live scan: {missing}. Live scan has: {live_channel_names}. "
                "Check that every trained sensor is connected and paired."
            )
        return [position[name] for name in expected]

    return (
        resolve(emg_columns(data_config), "EMG"),
        resolve(imu_columns(data_config), "IMU"),
    )


def post_event(server: str, payload: dict, timeout: float = 2.0) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{server.rstrip('/')}/api/event", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delsys-sdk-path", required=True,
        help="Directory containing EMGCollector.py, RingBuffer.py, and the "
        "AeroPy/ SDK package (the Example-Applications/Python folder).",
    )
    parser.add_argument("--config", required=True,
                        help="The checkpoint's own training config - used only "
                        "to get the trained channel order (emg_columns/"
                        "imu_columns), not to load a model here.")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--poll-interval-ms", type=float, default=20.0,
                        help="How often to check the ring buffer for new samples "
                        "and forward them.")
    parser.add_argument("--canvas-width", type=float, default=1920.0)
    parser.add_argument("--canvas-height", type=float, default=1080.0)
    args = parser.parse_args()

    sdk_path = Path(args.delsys_sdk_path).resolve()
    if not sdk_path.is_dir():
        raise SystemExit(f"--delsys-sdk-path does not exist: {sdk_path}")
    sys.path.insert(0, str(sdk_path))
    try:
        from EMGCollector import EMGCollector
    except ImportError as error:
        raise SystemExit(
            f"could not import EMGCollector from {sdk_path}: {error}\n"
            "This needs to run on a machine with the Delsys AeroPy SDK "
            "installed and a Trigno base station attached - the same "
            "requirements as run_simple.py in that same folder."
        ) from error

    config = load_config(args.config)
    data_config = config["data"]

    try:
        urllib.request.urlopen(f"{args.server.rstrip('/')}/api/config", timeout=2.0)
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(
            f"cannot reach the live-inference server at {args.server}: {error}\n"
            "Start it first: python scripts/run_live_distillation_ui.py "
            "--checkpoint NAME=path/to/final.pt --calibration path/to/calibration.npz"
        ) from error
    print(f"[BRIDGE] live-inference server reachable at {args.server}")

    print("[BRIDGE] connecting to Delsys hardware...")
    collector = EMGCollector()
    collector.connect()
    collector.scan()
    collector.configure()

    emg_indices, imu_indices = channel_index_map(collector.channel_names, data_config)
    print(f"[BRIDGE] channel map resolved: {len(emg_indices)} EMG + "
          f"{len(imu_indices)} IMU channels matched against the trained order")

    post_event(args.server, {
        "event": "start", "canvas": [args.canvas_width, args.canvas_height],
    })
    collector.start_streaming()
    print(f"[BRIDGE] streaming started - forwarding every {args.poll_interval_ms:.0f} ms. "
          f"Ctrl+C to stop.")

    last_sent_time: float | None = None
    poll_interval_s = args.poll_interval_ms / 1000.0
    forwarded_total = 0
    try:
        while True:
            time.sleep(poll_interval_s)
            if collector.ring_buffer is None:
                continue
            if last_sent_time is None:
                timestamps, data = collector.ring_buffer.get_latest(1)
                if len(timestamps) == 0:
                    continue
                last_sent_time = float(timestamps[0]) - 1e-6
            timestamps, data = collector.ring_buffer.get_since(last_sent_time)
            if len(timestamps) == 0:
                continue
            emg_batch = data[:, emg_indices].tolist()
            imu_batch = data[:, imu_indices].tolist()
            try:
                post_event(args.server, {
                    "event": "samples",
                    "time_s": timestamps.tolist(),
                    "emg": emg_batch,
                    "imu": imu_batch,
                })
            except (urllib.error.URLError, OSError) as error:
                print(f"[BRIDGE] forward failed, will retry next poll: {error}",
                      file=sys.stderr)
                continue
            last_sent_time = float(timestamps[-1])
            forwarded_total += len(timestamps)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[BRIDGE] stopping - forwarded {forwarded_total} samples total")
        collector.stop_streaming()


if __name__ == "__main__":
    main()
