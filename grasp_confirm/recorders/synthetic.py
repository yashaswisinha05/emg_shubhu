"""Synthetic recorder -- for testing the pipeline, never for results.

This generates fake 4-channel EMG whose statistics depend on the cue state, so
you can run a whole session, segment it, and execute qc.py end to end without
touching hardware. The data is invented. ``synthetic=True`` is written into
session.json and qc.py refuses to print a pass/fail verdict when it sees that
flag, precisely so a dry run can never be mistaken for a result.

The fake signal does encode a crude version of the real physics (loaded holds
plateau and carry 0.5-5 Hz grip-force ripple; air clenches decay and are
smooth), which is enough to exercise the feature code -- and is exactly why any
"accuracy" measured on it is meaningless.
"""

from __future__ import annotations

import time

import numpy as np

from .base import Recorder

CHANNEL_NAMES = ["FDS", "FCR", "EDC", "ECU"]


class SyntheticRecorder(Recorder):
    synthetic = True
    clock_note = "synthetic: host perf_counter at generation time"

    def __init__(self, sample_rate: float = 2000.0, n_channels: int = 4, seed: int = 0):
        self.sample_rate = float(sample_rate)
        self.channel_names = CHANNEL_NAMES[:n_channels] or [
            f"ch{i}" for i in range(n_channels)
        ]
        self._rng = np.random.default_rng(seed)
        self._t0 = 0.0
        self._last = 0.0
        self._state = "REST"
        self._state_t0 = 0.0
        self._effort = 1.0

    # -- cue hook -------------------------------------------------------
    def set_state(self, state: str, effort: float = 1.0) -> None:
        """Called by record.py so the fake signal follows the cue. Fake only."""
        self._state = state
        self._effort = float(effort)
        self._state_t0 = time.perf_counter()

    # -- Recorder -------------------------------------------------------
    def start(self) -> None:
        self._t0 = self._last = time.perf_counter()

    def stop(self) -> None:
        pass

    def poll(self) -> tuple[np.ndarray, np.ndarray]:
        now = time.perf_counter()
        n = int((now - self._last) * self.sample_rate)
        if n <= 0:
            empty = np.zeros(0)
            return empty, np.zeros((0, len(self.channel_names)))

        t = self._last + np.arange(1, n + 1) / self.sample_rate
        self._last = t[-1]

        since = t - self._state_t0
        c = len(self.channel_names)

        if self._state == "REST":
            amp = np.full(n, 0.01)
            flex_gain, ext_gain, ripple = 1.0, 1.0, 0.0
        elif self._state == "AIR":
            # Phasic burst that decays: nothing to hold against.
            amp = 0.02 + self._effort * 0.30 * np.exp(-np.maximum(since, 0.0) / 0.45)
            flex_gain, ext_gain, ripple = 1.0, 0.45, 0.0
        else:  # GRASP -- isometric plateau with closed-loop grip ripple
            amp = 0.02 + self._effort * 0.16 * (1.0 - np.exp(-np.maximum(since, 0.0) / 0.12))
            flex_gain, ext_gain, ripple = 1.0, 0.85, 0.22

        if ripple:
            amp = amp * (1.0 + ripple * np.sin(2 * np.pi * 2.3 * since + 0.7))

        gains = np.array(
            [flex_gain, flex_gain * 0.9, ext_gain, ext_gain * 0.8][:c]
        )
        if gains.size < c:
            gains = np.resize(gains, c)

        noise = self._rng.standard_normal((n, c))
        x = noise * (amp[:, None] * gains[None, :]) + 0.002 * self._rng.standard_normal(
            (n, c)
        )
        return t, x
