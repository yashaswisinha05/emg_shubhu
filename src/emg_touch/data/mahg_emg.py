"""Causal loader for the Multi-Angle Hand Gesture EMG dataset (MAHG-EMG)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, sosfilt


CHANNELS = [f"EMG_Channel_{index}" for index in range(1, 5)]
FILE_PATTERN = re.compile(r"EMGData(\d+)\.csv$", re.IGNORECASE)


def discover(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.csv"):
        match = FILE_PATTERN.search(path.name)
        if match:
            files.append(path)
    return sorted(files, key=lambda path: int(FILE_PATTERN.search(path.name).group(1)))


def subject_for(path: Path, files_per_subject: int = 10) -> int:
    """Infer the documented 10-recordings-per-subject grouping from file number."""
    match = FILE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"not an EMGData<number>.csv file: {path}")
    return (int(match.group(1)) - 1) // files_per_subject + 1


def _bounded_fill(values: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    filled = np.zeros_like(values, dtype=float)
    valid = np.zeros_like(values, dtype=bool)
    last = np.zeros(values.shape[1], dtype=float)
    age = np.full(values.shape[1], limit + 1, dtype=int)
    for row in range(len(values)):
        finite = np.isfinite(values[row])
        last[finite] = values[row, finite]
        age[finite] = 0
        age[~finite] += 1
        valid[row] = age <= limit
        filled[row] = np.where(valid[row], last, 0.0)
    return filled, valid


def _segments(mask: np.ndarray):
    start = None
    for index, good in enumerate(mask):
        if start is not None and not good:
            yield start, index
            start = None
        if good and start is None:
            start = index
    if start is not None:
        yield start, len(mask)


def preprocess(path: Path, raw_rate_hz: float = 1259.0,
               output_rate_hz: float = 100.0, gap_ms: float = 20.0):
    """Return the same two-scale, four-channel causal EMG representation used here."""
    frame = pd.read_csv(path)
    missing = [name for name in [*CHANNELS, "State"] if name not in frame]
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    raw = frame[CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    labels = frame["State"].astype("string").str.strip().str.upper().to_numpy(object)
    if len(raw) < max(64, round(raw_rate_hz * .3)):
        raise ValueError(f"{path}: recording is too short")
    gap_samples = max(1, round(raw_rate_hz * gap_ms / 1000.0))
    raw, valid = _bounded_fill(raw, gap_samples)
    high = min(450.0, raw_rate_hz * .4)
    if high <= 20:
        raise ValueError("raw rate is too low for the 20 Hz EMG high-pass")
    bandpass = butter(4, [20.0, high], btype="bandpass", fs=raw_rate_hz,
                      output="sos")
    short, long = np.zeros_like(raw), np.zeros_like(raw)
    short_n, long_n = max(1, round(.02 * raw_rate_hz)), max(1, round(.05 * raw_rate_hz))
    for channel in range(4):
        for start, stop in _segments(valid[:, channel]):
            filtered = sosfilt(bandpass, raw[start:stop, channel])
            short[start:stop, channel] = np.sqrt(np.maximum(
                lfilter(np.ones(short_n) / short_n, [1], filtered ** 2), 0))
            long[start:stop, channel] = np.sqrt(np.maximum(
                lfilter(np.ones(long_n) / long_n, [1], filtered ** 2), 0))
    step = raw_rate_hz / output_rate_hz
    indices = np.minimum((np.arange(int(np.floor((len(raw) - 1) / step)) + 1)
                          * step).astype(int), len(raw) - 1)
    features = np.concatenate((short[indices], long[indices]), axis=1).astype("float32")
    flags = np.concatenate((valid[indices], valid[indices]), axis=1).astype("float32")
    labels = labels[indices]
    label_valid = np.array([isinstance(value, str) and value.startswith("GESTURE")
                            for value in labels])
    return features, flags, labels, label_valid


def make_windows(features, flags, labels, label_valid, window_steps=30,
                 stride_steps=5, minimum_valid=.9):
    windows, targets = [], []
    for stop in range(window_steps, len(features) + 1, stride_steps):
        start = stop - window_steps
        local_labels = labels[start:stop]
        if (label_valid[start:stop].all()
                and len(set(local_labels.tolist())) == 1
                and flags[start:stop].mean() >= minimum_valid):
            windows.append(np.concatenate((features[start:stop], flags[start:stop]), -1))
            targets.append(str(local_labels[-1]))
    if not windows:
        return np.empty((0, window_steps, 16), dtype="float32"), np.empty(0, dtype=object)
    return np.stack(windows).astype("float32"), np.asarray(targets, dtype=object)
