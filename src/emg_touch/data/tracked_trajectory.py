"""CSV contract and loader for a trial recorded with a tracked end-effector.

Extends the existing recording format (preprocessing.csv_to_signal_arrays)
rather than inventing a new one, so a session already exportable in this
project's format needs only the tracker columns added, not a rewrite of the
capture pipeline.

Required per-trial CSV columns, one row per sample:

    time_perf_counter   monotonic hardware clock, seconds - required for
                         ordering and deduplication, same as the existing
                         format
    time_s              relative time against the pre-buffer start, same
                         meaning as the existing format
    <EMG columns>        per data.emg_column_template / data.sensors, exactly
                         as today
    <IMU columns>         per data.acc_column_template / data.gyro_column_template
                         / data.sensors, exactly as today
    pos_x, pos_y, pos_z  tracked end-effector position, in metres, in a
                         consistent world frame across the whole session -
                         not required to be gravity-aligned or origin-zeroed,
                         the loader does neither

Optional columns:

    pos_valid            0/1 or blank; if absent, a position sample is valid
                         whenever pos_x/y/z are all finite
    quat_w, quat_x, quat_y, quat_z
                         end-effector orientation, if the tracker reports
                         pose rather than position alone - not consumed by
                         the current model, carried through for later use

Position is assumed to already be on the same clock as EMG/IMU (both derived
from time_perf_counter in the same file) - this loader does not align two
separate recording clocks. If the tracker is logged by a separate process on
its own clock, that alignment has to happen before the file reaches this
contract, and validate_tracked_recording.py will not catch a silent
misalignment - only a missing or malformed one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import emg_columns, imu_columns

POSITION_COLUMNS = ("pos_x", "pos_y", "pos_z")
QUATERNION_COLUMNS = ("quat_w", "quat_x", "quat_y", "quat_z")
REQUIRED_TIMING_COLUMNS = ("time_perf_counter", "time_s")


def _extract_channels(
    frame: pd.DataFrame, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    arrays, masks = [], []
    for name in names:
        if name in frame:
            column = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
            valid = np.isfinite(column)
            column = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            column = np.zeros(len(frame), dtype=np.float32)
            valid = np.zeros(len(frame), dtype=bool)
        arrays.append(column)
        masks.append(valid)
    return np.stack(arrays, axis=1), np.stack(masks, axis=1)


def tracked_csv_to_signal_arrays(
    csv_path: str | Path, data_config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Same timing reconstruction as preprocessing.csv_to_signal_arrays,
    with tracked position added on the identical clock.
    """
    frame = pd.read_csv(csv_path)
    missing_timing = set(REQUIRED_TIMING_COLUMNS) - set(frame.columns)
    if missing_timing:
        raise ValueError(f"{csv_path} is missing timing columns: {sorted(missing_timing)}")
    missing_position = set(POSITION_COLUMNS) - set(frame.columns)
    if missing_position:
        raise ValueError(f"{csv_path} is missing tracker columns: {sorted(missing_position)}")

    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy(dtype=np.float64)
    relative = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(perf) & np.isfinite(relative)
    frame = frame.loc[finite].copy()
    perf = perf[finite]
    relative = relative[finite]
    order = np.argsort(perf, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    perf = perf[order]
    relative = relative[order]

    _, unique_reverse = np.unique(perf[::-1], return_index=True)
    keep = np.sort(len(perf) - 1 - unique_reverse)
    frame = frame.iloc[keep].reset_index(drop=True)
    perf = perf[keep]
    relative = relative[keep]

    zero_index = int(np.argmin(np.abs(relative)))
    cue_perf = perf[zero_index] - relative[zero_index]
    time_s = perf - cue_perf

    emg, emg_mask = _extract_channels(frame, emg_columns(data_config))
    imu, imu_mask = _extract_channels(frame, imu_columns(data_config))
    position, position_mask = _extract_channels(frame, POSITION_COLUMNS)
    if "pos_valid" in frame:
        explicit = pd.to_numeric(frame["pos_valid"], errors="coerce").fillna(0).to_numpy() != 0
        position_mask = position_mask & explicit[:, None]

    result = {
        "time_s": time_s.astype(np.float64),
        "emg": emg,
        "emg_mask": emg_mask,
        "imu": imu,
        "imu_mask": imu_mask,
        "position": position,
        "position_mask": position_mask,
    }
    if all(column in frame for column in QUATERNION_COLUMNS):
        quaternion, quaternion_mask = _extract_channels(frame, QUATERNION_COLUMNS)
        result["quaternion"] = quaternion
        result["quaternion_mask"] = quaternion_mask
    return result
