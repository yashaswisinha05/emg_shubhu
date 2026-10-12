"""Training-matched causal preprocessing for live EMG+IMU samples."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class LiveReachGraspPreprocessor:
    """Convert raw 4-channel EMG and 24-channel IMU samples to a 100-Hz grid."""

    def __init__(self, preprocessing):
        self.raw_rate = float(preprocessing["raw_rate_hz"])
        self.rate = float(preprocessing["rate_hz"])
        self.gap = float(preprocessing["gap_s"])
        high = min(450., self.raw_rate * .4)
        self.emg_sos = butter(4, [20., high], btype="bandpass",
                              fs=self.raw_rate, output="sos")
        self.imu_sos = butter(2, 15., fs=self.raw_rate, output="sos")
        self.short = max(1, round(.02 * self.raw_rate))
        self.long = max(1, round(.05 * self.raw_rate))
        self.reset()

    def reset(self):
        self.last_time = None
        self.next_grid = None
        self.last_raw_emg = np.zeros(4)
        self.last_raw_imu = np.zeros(24)
        self.last_seen_emg = np.full(4, -np.inf)
        self.last_seen_imu = np.full(24, -np.inf)
        self.emg_state = [None] * 4
        self.imu_state = [None] * 24
        self.emg_square = [deque(maxlen=self.long) for _ in range(4)]
        self.imu_median = [deque(maxlen=3) for _ in range(24)]
        self.latest_emg = np.zeros(8, dtype="float32")
        self.latest_imu = np.zeros(24, dtype="float32")
        self.latest_ev = np.zeros(8, dtype=bool)
        self.latest_iv = np.zeros(24, dtype=bool)
        self.grid_time, self.grid_emg, self.grid_imu = [], [], []
        self.grid_ev, self.grid_iv = [], []

    @property
    def frames(self):
        return len(self.grid_time)

    def _append_grid(self, stamp):
        recent = self.last_time is not None and stamp - self.last_time <= self.gap + 1e-9
        self.grid_time.append(float(stamp))
        self.grid_emg.append(self.latest_emg.copy())
        self.grid_imu.append(self.latest_imu.copy())
        self.grid_ev.append(self.latest_ev.copy() & recent)
        self.grid_iv.append(self.latest_iv.copy() & recent)

    @staticmethod
    def _values(values, width, name):
        result = np.asarray(values, dtype=float)
        if result.shape != (width,):
            raise ValueError(f"{name} must contain exactly {width} values")
        return result

    def add_sample(self, time_s, emg, imu):
        stamp = float(time_s)
        if not np.isfinite(stamp) or (self.last_time is not None and stamp <= self.last_time):
            raise ValueError("time_s must be finite and strictly increasing")
        raw_emg = self._values(emg, 4, "emg")
        raw_imu = self._values(imu, 24, "imu")
        if self.next_grid is None:
            self.next_grid = stamp
        while self.last_time is not None and self.next_grid < stamp - 1e-9:
            self._append_grid(self.next_grid)
            self.next_grid += 1 / self.rate

        row_gap = self.last_time is not None and stamp - self.last_time > self.gap
        finite = np.isfinite(raw_emg)
        self.last_raw_emg[finite] = raw_emg[finite]
        self.last_seen_emg[finite] = stamp
        emg_valid = stamp - self.last_seen_emg <= self.gap
        finite = np.isfinite(raw_imu)
        self.last_raw_imu[finite] = raw_imu[finite]
        self.last_seen_imu[finite] = stamp
        imu_valid = stamp - self.last_seen_imu <= self.gap
        if row_gap:
            self.emg_state, self.imu_state = [None] * 4, [None] * 24
            self.emg_square = [deque(maxlen=self.long) for _ in range(4)]
            self.imu_median = [deque(maxlen=3) for _ in range(24)]

        short_rms, long_rms = np.zeros(4), np.zeros(4)
        for channel in range(4):
            if not emg_valid[channel]:
                self.emg_state[channel] = None
                self.emg_square[channel].clear()
                continue
            if self.emg_state[channel] is None:
                self.emg_state[channel] = np.zeros((self.emg_sos.shape[0], 2))
            filtered, self.emg_state[channel] = sosfilt(
                self.emg_sos, [self.last_raw_emg[channel]], zi=self.emg_state[channel])
            self.emg_square[channel].append(float(filtered[0] ** 2))
            values = list(self.emg_square[channel])
            short_rms[channel] = np.sqrt(max(sum(values[-self.short:]) / self.short, 0))
            long_rms[channel] = np.sqrt(max(sum(values) / self.long, 0))

        for channel in range(24):
            if not imu_valid[channel]:
                self.imu_state[channel] = None
                self.imu_median[channel].clear()
                continue
            self.imu_median[channel].append(float(self.last_raw_imu[channel]))
            median = float(np.median(self.imu_median[channel]))
            if self.imu_state[channel] is None:
                self.imu_state[channel] = sosfilt_zi(self.imu_sos) * median
            filtered, self.imu_state[channel] = sosfilt(
                self.imu_sos, [median], zi=self.imu_state[channel])
            self.latest_imu[channel] = filtered[0]

        self.latest_emg = np.concatenate((short_rms, long_rms)).astype("float32")
        self.latest_ev = np.concatenate((emg_valid, emg_valid))
        self.latest_iv = imu_valid
        self.last_time = stamp
        while self.next_grid <= stamp + 1e-9:
            self._append_grid(self.next_grid)
            self.next_grid += 1 / self.rate

    def arrays(self):
        return (np.asarray(self.grid_time), np.asarray(self.grid_emg),
                np.asarray(self.grid_imu), np.asarray(self.grid_ev),
                np.asarray(self.grid_iv))
