"""Delsys Trigno recorder -- UNVERIFIED, verify before you rely on it.

This is the one file in the package that could not be tested: it needs the
Delsys AeroPy SDK and a physically attached Trigno base station. It reuses your
existing, proven ``EMGCollector`` rather than reimplementing the hardware
calls, which is the most that can be done to reduce risk without the device.

Before the session, run:

    python -m grasp_confirm.recorders.delsys --sdk-path /path/to/Example-Applications/Python

That connects, streams for five seconds and prints per-channel RMS. If it
prints four sensible non-zero numbers you are good. If it does not, fix it
BEFORE the subject is wired up -- do not debug hardware with electrodes on
someone's arm at 1am.

If the API in your SDK build differs, the only thing that has to change is
``_read_new_samples``: it must return an (n, C) array of new samples since the
previous call, oldest first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .base import Recorder


class DelsysRecorder(Recorder):
    clock_note = (
        "sample index / sample_rate offset from host perf_counter at stream start "
        "(estimated, drifts over a session)"
    )

    def __init__(
        self,
        sdk_path: str | Path,
        sample_rate: float = 2000.0,
        channels: int = 4,
        host: str = "localhost",
    ):
        self.sdk_path = Path(sdk_path).expanduser()
        self.sample_rate = float(sample_rate)
        self.n_channels = int(channels)
        self.host = host
        self.channel_names = [f"EMG{i + 1}" for i in range(channels)]
        self._collector = None
        self._t0 = 0.0
        self._n_emitted = 0

    def start(self) -> None:
        if not self.sdk_path.exists():
            raise SystemExit(f"Delsys SDK path does not exist: {self.sdk_path}")
        sys.path.insert(0, str(self.sdk_path))
        try:
            from EMGCollector import EMGCollector  # type: ignore
        except Exception as exc:  # pragma: no cover - hardware only
            raise SystemExit(
                f"could not import EMGCollector from {self.sdk_path}: {exc}\n"
                "Point --sdk-path at the directory containing EMGCollector.py "
                "(the same one your live_delsys_bridge.py uses)."
            ) from exc

        self._collector = EMGCollector(host=self.host)
        self._collector.Connect_Callback()
        self._collector.scan_callback()
        self._collector.configure_callback()
        self._collector.start_streaming()

        names = getattr(self._collector, "channel_names", None)
        if names:
            emg = [n for n in names if "EMG" in n.upper()][: self.n_channels]
            if emg:
                self.channel_names = emg

        self._t0 = time.perf_counter()
        self._n_emitted = 0

    def _read_new_samples(self) -> np.ndarray:
        """Return (n, C) of samples acquired since the last call, oldest first.

        Adapt this method, and only this method, if your SDK build differs.
        """
        collector = self._collector
        data = collector.PollDataByString()  # type: ignore[union-attr]
        if not data:
            return np.zeros((0, self.n_channels))
        columns = [
            np.asarray(data[name], dtype=float)
            for name in self.channel_names
            if name in data
        ]
        if len(columns) != self.n_channels:
            raise RuntimeError(
                f"expected {self.n_channels} EMG channels, polled "
                f"{len(columns)} of {sorted(data)}"
            )
        n = min(col.size for col in columns)
        return np.stack([col[:n] for col in columns], axis=1)

    def poll(self) -> tuple[np.ndarray, np.ndarray]:
        x = self._read_new_samples()
        if x.size == 0:
            return np.zeros(0), np.zeros((0, self.n_channels))
        index = np.arange(self._n_emitted, self._n_emitted + x.shape[0])
        self._n_emitted += x.shape[0]
        return self._t0 + index / self.sample_rate, x

    def stop(self) -> None:
        if self._collector is not None:
            for method in ("stop_streaming", "Quit_Callback"):
                fn = getattr(self._collector, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            self._collector = None


def _selftest() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Five-second Delsys stream check.")
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--sample-rate", type=float, default=2000.0)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    rec = DelsysRecorder(args.sdk_path, args.sample_rate, args.channels)
    chunks = []
    with rec:
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            _, x = rec.poll()
            if x.size:
                chunks.append(x)
            time.sleep(0.02)

    if not chunks:
        raise SystemExit("no samples received -- check sensors are paired and streaming")
    x = np.concatenate(chunks, axis=0)
    print(f"{x.shape[0]} samples ({x.shape[0] / args.sample_rate:.1f} s), "
          f"{x.shape[1]} channels")
    for name, rms in zip(rec.channel_names, np.sqrt((x**2).mean(axis=0))):
        print(f"  {name:10s} RMS {rms:.6g}")


if __name__ == "__main__":
    _selftest()
