"""Causal wearable preprocessing for independently annotated grasp/release trials."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter
from emg_touch.physics.rotation_6d import (
    matrix_to_rotation_6d_numpy, quaternion_to_matrix_numpy)

SENSORS = ["S0", "S4", "S8", "S12"]


def constant(frame, name):
    if name not in frame:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy()
    if not len(values):
        return None
    if not np.isfinite(values).all() or np.ptp(values) > 1e-3:
        raise ValueError(f"inconsistent trial metadata: {name}")
    return float(values[0])


def event_times(frame, origin="auto"):
    absolute = [constant(frame, k) for k in ["t_grasp_perf", "t_release_perf"]]
    if all(v is not None for v in absolute):
        if absolute[1] <= absolute[0]:
            raise ValueError("release must follow grasp")
        return np.array(absolute), "absolute_perf"
    if any(v is not None for v in absolute):
        raise ValueError("only one absolute event timestamp is present")
    if origin != "start":
        raise ValueError("missing absolute events: confirm relative origin, then use --event-origin start")
    start = constant(frame, "t_start_perf")
    offsets = [constant(frame, k) for k in ["grasp_onset_s", "grasp_offset_s"]]
    if start is None or any(v is None for v in offsets) or offsets[1] <= offsets[0]:
        raise ValueError("invalid start-relative event metadata")
    return start + np.array(offsets), "confirmed_start_relative"


def numeric(frame, names):
    missing = [n for n in names if n not in frame]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    return frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def past_fill(t, x, max_gap):
    """Bounded forward fill using elapsed seconds, not number of CSV rows."""
    filled, valid = np.zeros_like(x), np.zeros_like(x, dtype=bool)
    last = np.zeros(x.shape[1])
    seen = np.full(x.shape[1], -np.inf)
    for i, stamp in enumerate(t):
        finite = np.isfinite(x[i])
        last[finite], seen[finite] = x[i, finite], stamp
        valid[i] = stamp - seen <= max_gap
        filled[i] = np.where(valid[i], last, 0.)
    return filled, valid


def segments(t, mask, max_gap):
    start = None
    for i, good in enumerate(mask):
        if start is not None and (not good or t[i] - t[i - 1] > max_gap):
            yield start, i
            start = None
        if good and start is None:
            start = i
    if start is not None:
        yield start, len(t)


def preprocess(path, settings):
    frame = pd.read_csv(path)
    times = numeric(frame, ["time_perf_counter"])[:, 0]
    frame = frame.loc[np.isfinite(times)].copy()
    frame["time_perf_counter"] = times[np.isfinite(times)]
    frame = frame.sort_values("time_perf_counter", kind="stable").drop_duplicates("time_perf_counter")
    t = frame["time_perf_counter"].to_numpy()
    if len(t) < 64:
        raise ValueError("fewer than 64 timestamped rows")
    events, label_source = event_times(frame, settings["event_origin"])
    if events[0] < t[0] or events[1] > t[-1]:
        raise ValueError("event timestamps outside recording; check clock/origin")
    rate = settings["raw_rate_hz"]
    declared = constant(frame, "sample_rate_hz_declared")
    if declared is not None and abs(declared / rate - 1) > .25:
        raise ValueError(f"declared rate {declared} disagrees with --raw-rate-hz {rate}")
    emg_names = [f"EMG 1_{s}" for s in SENSORS]
    imu_names = [f"{kind} {axis}_{s}" for s in SENSORS for kind in ["ACC", "GYRO"] for axis in "XYZ"]
    emg, ev = past_fill(t, numeric(frame, emg_names), settings["gap_s"])
    imu, iv = past_fill(t, numeric(frame, imu_names), settings["gap_s"])
    # Filter raw EMG before forming trailing activation features.
    high = min(450., rate * .4)
    if high <= 20:
        raise ValueError("raw sampling rate too low for the default EMG filter")
    sos = butter(4, [20., high], btype="bandpass", fs=rate, output="sos")
    rms = np.zeros_like(emg)
    long_rms = np.zeros_like(emg)
    for channel in range(4):
        for a, b in segments(t, ev[:, channel], settings["gap_s"]):
            signal = sosfilt(sos, emg[a:b, channel])
            size = max(1, round(.02 * rate))
            rms[a:b, channel] = np.sqrt(np.maximum(lfilter(np.ones(size) / size, [1], signal**2), 0))
            size = max(1, round(.05 * rate))
            long_rms[a:b, channel] = np.sqrt(np.maximum(lfilter(np.ones(size) / size, [1], signal**2), 0))
    # No centered median/filtfilt. IMU retains DC (including gravity).
    lowpass = butter(2, 15., fs=rate, output="sos")
    for channel in range(24):
        for a, b in segments(t, iv[:, channel], settings["gap_s"]):
            segment = imu[a:b, channel]
            median = pd.Series(segment).rolling(3, min_periods=1).median().to_numpy()
            imu[a:b, channel] = sosfilt(lowpass, median,
                zi=sosfilt_zi(lowpass) * median[0])[0]
    grid = np.arange(t[0], t[-1] + 1e-9, 1 / settings["rate_hz"])
    index = np.searchsorted(t, grid, side="right") - 1
    recent = (grid - t[index]) <= settings["gap_s"]
    ev = ev[index] & recent[:, None]
    iv = iv[index] & recent[:, None]
    # Two trailing RMS scales, not decimated high-frequency raw EMG.
    features = np.concatenate([rms[index], long_rms[index]], 1)
    emg_valid = np.concatenate([ev, ev], 1)
    position = numeric(frame, [f"VIVE_T0_pos_{a}_m" for a in "xyz"])
    pv = np.isfinite(position).all(1)
    for name, limit in [("VIVE_T0_sync_error_ms", 20.), ("VIVE_T0_tracking_age_us", 50000.)]:
        if name in frame:
            value = pd.to_numeric(frame[name], errors="coerce").to_numpy()
            pv &= np.isfinite(value) & (np.abs(value) <= limit)
    position = np.nan_to_num(position[index]).astype("float32")
    pose_valid = pv[index] & recent
    quaternion_names = [f"VIVE_T0_quat_{axis}" for axis in "wxyz"]
    if all(name in frame for name in quaternion_names):
        quaternion = numeric(frame, quaternion_names)
        quaternion_norm = np.linalg.norm(quaternion, axis=1)
        qv = np.isfinite(quaternion).all(1) & (quaternion_norm > .5) & (quaternion_norm < 1.5)
        qv &= pv
        rotation = quaternion_to_matrix_numpy(np.nan_to_num(quaternion[index], nan=0.))
        orientation = matrix_to_rotation_6d_numpy(rotation).astype("float32")
        orientation_valid = qv[index] & recent
    else:
        orientation = np.zeros((len(grid), 6), dtype="float32")
        orientation_valid = np.zeros(len(grid), dtype=bool)
    holding = ((grid >= events[0]) & (grid < events[1])).astype("float32")
    event_labels = ((grid[:, None] >= events) &
                    (grid[:, None] < events + settings["event_pulse_s"])).astype("float32")
    return {"path": str(path), "time": grid - t[0], "events": events - t[0],
            "emg": features.astype("float32"), "emg_valid": emg_valid,
            "imu": imu[index].astype("float32"), "imu_valid": iv,
            "position": position, "pose_valid": pose_valid,
            "orientation": orientation, "orientation_valid": orientation_valid,
            "holding": holding, "event_labels": event_labels,
            "audit": {"label_source": label_source, "frames": len(grid),
                      "emg_valid_fraction": float(ev.mean()), "imu_valid_fraction": float(iv.mean()),
                      "pose_valid_fraction": float(pose_valid.mean()),
                      "orientation_valid_fraction": float(orientation_valid.mean())}}
