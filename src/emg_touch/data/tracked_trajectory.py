"""CSV contract and loader for a trial recorded with a Vive tracker.

Extends the existing recording format (preprocessing.csv_to_signal_arrays)
rather than inventing a new one, so a session already exportable in this
project's format needs only the tracker columns added, not a rewrite of the
capture pipeline.

Column layout matches a real SteamVR/OpenVR export, one row per sample:

    time_perf_counter          monotonic hardware clock, seconds - required
                                for ordering and deduplication, same as the
                                existing EMG/IMU format
    time_s                     relative time against the pre-buffer start,
                                same meaning as the existing format
    <EMG columns>               per data.emg_column_template / data.sensors,
                                exactly as today
    <IMU columns>                per data.acc_column_template /
                                data.gyro_column_template / data.sensors,
                                exactly as today
    {PREFIX}_{TRACKER}_pos_x_m / _pos_y_m / _pos_z_m
                                tracked position, metres, in a consistent
                                world frame across the session - not required
                                to be gravity-aligned or origin-zeroed
    {PREFIX}_{TRACKER}_quat_w / _quat_x / _quat_y / _quat_z
                                tracker orientation
    {PREFIX}_{TRACKER}_vel_x_mps / _vel_y_mps / _vel_z_mps
                                linear velocity, ALREADY MEASURED by the
                                tracker rather than differenced from
                                position. Used directly - one differencing
                                pass (velocity -> acceleration) instead of
                                two, so acceleration carries less noise than
                                differencing position twice would.
    {PREFIX}_{TRACKER}_angvel_x_radps / _angvel_y_radps / _angvel_z_radps
                                angular velocity - not consumed by the
                                current model, carried through for later use
                                (e.g. a wrist-worn tracker's rotation during
                                grasp)
    {PREFIX}_{TRACKER}_tracking_age_us
                                staleness of the reported pose, microseconds
    {PREFIX}_{TRACKER}_vive_timestamp_us
                                the tracker's own hardware timestamp,
                                independent of time_perf_counter
    {PREFIX}_{TRACKER}_sync_error_ms
                                the capture pipeline's own estimate of
                                residual clock misalignment between the
                                tracker and time_perf_counter, per sample

PREFIX defaults to "VIVE" and TRACKER to "T0" (data.tracker_column_prefix /
data.tracker_id), so a second tracker on the same rig - a forearm or upper
arm mount, say - is `data.tracker_id: "T1"` with no code change.

tracking_age_us and sync_error_ms are surfaced in the returned dict rather
than silently filtered. Turning them into a validity threshold means picking
a cutoff in microseconds/milliseconds, and there is no principled value to
pick without having seen this rig's actual jitter - the same reasoning
behind every other threshold in this project being set from a real
measurement, not a guess. Set data.tracker_max_sync_error_ms and/or
data.tracker_max_tracking_age_us once real recordings show what a
reasonable cutoff is, and they will be ANDed into position_mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import emg_columns, imu_columns

REQUIRED_TIMING_COLUMNS = ("time_perf_counter", "time_s")

_POSITION_SUFFIXES = ("pos_x_m", "pos_y_m", "pos_z_m")
_QUATERNION_SUFFIXES = ("quat_w", "quat_x", "quat_y", "quat_z")
_VELOCITY_SUFFIXES = ("vel_x_mps", "vel_y_mps", "vel_z_mps")
_ANGULAR_VELOCITY_SUFFIXES = ("angvel_x_radps", "angvel_y_radps", "angvel_z_radps")


def tracker_prefix(data_config: dict[str, Any] | None = None) -> str:
    config = data_config or {}
    return f"{config.get('tracker_column_prefix', 'VIVE')}_{config.get('tracker_id', 'T0')}"


def _tracker_columns(
    data_config: dict[str, Any] | None, suffixes: tuple[str, ...]
) -> tuple[str, ...]:
    prefix = tracker_prefix(data_config)
    return tuple(f"{prefix}_{suffix}" for suffix in suffixes)


def position_columns(data_config: dict[str, Any] | None = None) -> tuple[str, ...]:
    return _tracker_columns(data_config, _POSITION_SUFFIXES)


def quaternion_columns(data_config: dict[str, Any] | None = None) -> tuple[str, ...]:
    return _tracker_columns(data_config, _QUATERNION_SUFFIXES)


def velocity_columns(data_config: dict[str, Any] | None = None) -> tuple[str, ...]:
    return _tracker_columns(data_config, _VELOCITY_SUFFIXES)


def angular_velocity_columns(data_config: dict[str, Any] | None = None) -> tuple[str, ...]:
    return _tracker_columns(data_config, _ANGULAR_VELOCITY_SUFFIXES)


def tracking_age_column(data_config: dict[str, Any] | None = None) -> str:
    return f"{tracker_prefix(data_config)}_tracking_age_us"


def vive_timestamp_column(data_config: dict[str, Any] | None = None) -> str:
    return f"{tracker_prefix(data_config)}_vive_timestamp_us"


def sync_error_column(data_config: dict[str, Any] | None = None) -> str:
    return f"{tracker_prefix(data_config)}_sync_error_ms"


# Defaults (prefix VIVE, tracker T0), for callers that only need the common case.
POSITION_COLUMNS = position_columns()
QUATERNION_COLUMNS = quaternion_columns()
VELOCITY_COLUMNS = velocity_columns()
ANGULAR_VELOCITY_COLUMNS = angular_velocity_columns()


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


def _extract_scalar(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A single quality/timing column as float64, NaN where absent or non-finite."""
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    column = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return column


def tracked_csv_to_signal_arrays(
    csv_path: str | Path, data_config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Same timing reconstruction as preprocessing.csv_to_signal_arrays,
    with the tracker channels added on the identical clock.
    """
    frame = pd.read_csv(csv_path)
    missing_timing = set(REQUIRED_TIMING_COLUMNS) - set(frame.columns)
    if missing_timing:
        raise ValueError(f"{csv_path} is missing timing columns: {sorted(missing_timing)}")
    position_names = position_columns(data_config)
    missing_position = set(position_names) - set(frame.columns)
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
    position, position_mask = _extract_channels(frame, position_names)

    tracking_age = _extract_scalar(frame, tracking_age_column(data_config))
    sync_error = _extract_scalar(frame, sync_error_column(data_config))

    max_sync_error_ms = data_config.get("tracker_max_sync_error_ms")
    if max_sync_error_ms is not None:
        within_sync = np.isfinite(sync_error) & (np.abs(sync_error) <= float(max_sync_error_ms))
        position_mask = position_mask & within_sync[:, None]
    max_tracking_age_us = data_config.get("tracker_max_tracking_age_us")
    if max_tracking_age_us is not None:
        fresh = np.isfinite(tracking_age) & (tracking_age <= float(max_tracking_age_us))
        position_mask = position_mask & fresh[:, None]

    result = {
        "time_s": time_s.astype(np.float64),
        "emg": emg,
        "emg_mask": emg_mask,
        "imu": imu,
        "imu_mask": imu_mask,
        "position": position,
        "position_mask": position_mask,
        "tracking_age_us": tracking_age,
        "sync_error_ms": sync_error,
    }

    if all(column in frame for column in quaternion_columns(data_config)):
        quaternion, quaternion_mask = _extract_channels(frame, quaternion_columns(data_config))
        result["quaternion"] = quaternion
        result["quaternion_mask"] = quaternion_mask

    if all(column in frame for column in velocity_columns(data_config)):
        # Measured, not differenced - see module docstring. Downstream code
        # (tracked_virtual_leader.py) uses this directly rather than
        # differencing position, and only differences this once more for
        # acceleration.
        velocity, velocity_mask = _extract_channels(frame, velocity_columns(data_config))
        result["velocity"] = velocity
        result["velocity_mask"] = velocity_mask

    if all(column in frame for column in angular_velocity_columns(data_config)):
        angular_velocity, angular_velocity_mask = _extract_channels(
            frame, angular_velocity_columns(data_config)
        )
        result["angular_velocity"] = angular_velocity
        result["angular_velocity_mask"] = angular_velocity_mask

    return result
