from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .full_trajectory import LengthBucketBatchSampler, trajectory_analysis_interval
from .manifest import load_manifest
from .preprocessing import RobustScaler, causal_median_filter, previous_sample_resample
from .schema import IMU_COLUMNS, SENSORS
from .splits import subset_from_trial_ids
from ..utils import load_json


SENSOR_COUNT = 4
RAW_CHANNELS_PER_SENSOR = 6
RAW_IMU_DIM = SENSOR_COUNT * RAW_CHANNELS_PER_SENSOR
CALIBRATED_FEATURES_PER_SENSOR = 16
CALIBRATED_IMU_DIM = SENSOR_COUNT * CALIBRATED_FEATURES_PER_SENSOR
CALIBRATED_FEATURE_NAMES_PER_SENSOR = (
    "acc_cal_x",
    "acc_cal_y",
    "acc_cal_z",
    "gyro_cal_x",
    "gyro_cal_y",
    "gyro_cal_z",
    "orientation_rel_x",
    "orientation_rel_y",
    "orientation_rel_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "acc_magnitude_change",
    "gyro_magnitude",
    "jerk_magnitude",
    "angular_acceleration_magnitude",
)


def grid_imu_feature_dim(data_config: dict[str, Any]) -> int:
    return CALIBRATED_IMU_DIM + (
        RAW_IMU_DIM if bool(data_config.get("include_raw_imu", False)) else 0
    )


def grid_imu_sensor_indices(data_config: dict[str, Any]) -> tuple[int, ...]:
    calibrated = tuple(
        sensor
        for sensor in range(SENSOR_COUNT)
        for _ in range(CALIBRATED_FEATURES_PER_SENSOR)
    )
    if not bool(data_config.get("include_raw_imu", False)):
        return calibrated
    raw = tuple(
        sensor
        for sensor in range(SENSOR_COUNT)
        for _ in range(RAW_CHANNELS_PER_SENSOR)
    )
    return raw + calibrated


def grid_imu_feature_names(data_config: dict[str, Any]) -> tuple[str, ...]:
    calibrated = tuple(
        f"{sensor}_{feature}"
        for sensor in SENSORS
        for feature in CALIBRATED_FEATURE_NAMES_PER_SENSOR
    )
    if not bool(data_config.get("include_raw_imu", False)):
        return calibrated
    return tuple(f"raw_{name}" for name in IMU_COLUMNS) + calibrated


def _masked_channel_median(
    values: np.ndarray, mask: np.ndarray, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(values.shape[1], dtype=np.float32)
    available = np.zeros(values.shape[1], dtype=bool)
    for channel in range(values.shape[1]):
        valid = mask[:sample_count, channel]
        if valid.any():
            centers[channel] = np.median(values[:sample_count, channel][valid])
            available[channel] = True
    return centers, available


def _masked_derivative(
    values: np.ndarray, mask: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.zeros_like(values, dtype=np.float32)
    derivative_mask = np.zeros_like(mask, dtype=bool)
    if len(values) > 1:
        derivative[1:] = np.diff(values, axis=0) * float(sample_rate_hz)
        derivative_mask[1:] = mask[1:] & mask[:-1]
    derivative[~derivative_mask] = 0.0
    return derivative, derivative_mask


def calibrate_imu_features(
    imu: np.ndarray,
    imu_mask: np.ndarray,
    sample_rate_hz: float,
    calibration_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create per-trajectory, pre-cue calibrated IMU features.

    Each sensor contributes calibrated acceleration (3), bias-corrected gyro (3),
    integrated relative orientation (3), initial gravity direction (3), acceleration
    magnitude change (1), gyro magnitude (1), jerk magnitude (1), and
    angular-acceleration magnitude (1).
    """

    if imu.ndim != 2 or imu.shape[1] != SENSOR_COUNT * RAW_CHANNELS_PER_SENSOR:
        raise ValueError(f"Expected IMU shape [time, 24], received {imu.shape}")
    if imu_mask.shape != imu.shape:
        raise ValueError("IMU mask shape does not match values")
    calibration_samples = max(
        1, min(len(imu), int(round(calibration_window_s * sample_rate_hz)))
    )
    values = imu.reshape(len(imu), SENSOR_COUNT, RAW_CHANNELS_PER_SENSOR)
    masks = imu_mask.reshape(len(imu), SENSOR_COUNT, RAW_CHANNELS_PER_SENSOR)
    sensor_features = []
    sensor_masks = []
    dt = 1.0 / float(sample_rate_hz)

    for sensor in range(SENSOR_COUNT):
        sensor_values = values[:, sensor]
        sensor_valid = masks[:, sensor]
        baseline, baseline_valid = _masked_channel_median(
            sensor_values, sensor_valid, calibration_samples
        )
        acc = sensor_values[:, :3]
        gyro = sensor_values[:, 3:]
        acc_mask = sensor_valid[:, :3]
        gyro_mask = sensor_valid[:, 3:]
        acc_baseline = baseline[:3]
        gyro_baseline = baseline[3:]

        gravity_norm = float(np.linalg.norm(acc_baseline))
        gravity_valid = bool(baseline_valid[:3].all() and gravity_norm > 1e-6)
        if gravity_valid:
            gravity_direction_vector = (acc_baseline / gravity_norm).astype(np.float32)
        else:
            gravity_direction_vector = np.zeros(3, dtype=np.float32)
        gravity_direction = np.broadcast_to(
            gravity_direction_vector, (len(imu), 3)
        ).copy()
        gravity_direction_mask = np.full(
            (len(imu), 3), gravity_valid, dtype=bool
        )

        calibrated_acc = acc - acc_baseline
        calibrated_gyro = gyro - gyro_baseline
        calibrated_acc_mask = acc_mask & baseline_valid[:3]
        calibrated_gyro_mask = gyro_mask & baseline_valid[3:]
        calibrated_acc[~calibrated_acc_mask] = 0.0
        calibrated_gyro[~calibrated_gyro_mask] = 0.0

        relative_orientation = np.cumsum(
            np.where(calibrated_gyro_mask, calibrated_gyro, 0.0) * dt,
            axis=0,
        ).astype(np.float32)
        orientation_mask = calibrated_gyro_mask.copy()

        acc_vector_valid = calibrated_acc_mask.all(axis=1)
        gyro_vector_valid = calibrated_gyro_mask.all(axis=1)
        acc_magnitude_change = (
            np.linalg.norm(acc, axis=1)
            - float(np.linalg.norm(acc_baseline))
        ).astype(np.float32)
        gyro_magnitude = np.linalg.norm(calibrated_gyro, axis=1).astype(
            np.float32
        )
        acc_magnitude_change[~acc_vector_valid] = 0.0
        gyro_magnitude[~gyro_vector_valid] = 0.0

        jerk, jerk_mask = _masked_derivative(
            calibrated_acc, calibrated_acc_mask, sample_rate_hz
        )
        angular_acceleration, angular_acceleration_mask = _masked_derivative(
            calibrated_gyro, calibrated_gyro_mask, sample_rate_hz
        )
        jerk_vector_valid = jerk_mask.all(axis=1)
        angular_vector_valid = angular_acceleration_mask.all(axis=1)
        jerk_magnitude = np.linalg.norm(jerk, axis=1).astype(np.float32)
        angular_magnitude = np.linalg.norm(
            angular_acceleration, axis=1
        ).astype(np.float32)
        jerk_magnitude[~jerk_vector_valid] = 0.0
        angular_magnitude[~angular_vector_valid] = 0.0

        features = np.concatenate(
            [
                calibrated_acc,
                calibrated_gyro,
                relative_orientation,
                gravity_direction,
                acc_magnitude_change[:, None],
                gyro_magnitude[:, None],
                jerk_magnitude[:, None],
                angular_magnitude[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        feature_mask = np.concatenate(
            [
                calibrated_acc_mask,
                calibrated_gyro_mask,
                orientation_mask,
                gravity_direction_mask,
                acc_vector_valid[:, None],
                gyro_vector_valid[:, None],
                jerk_vector_valid[:, None],
                angular_vector_valid[:, None],
            ],
            axis=1,
        )
        features[~feature_mask] = 0.0
        sensor_features.append(features)
        sensor_masks.append(feature_mask)

    return (
        np.concatenate(sensor_features, axis=1).astype(np.float32),
        np.concatenate(sensor_masks, axis=1),
    )


def preprocess_grid_signals(
    row: pd.Series | Any,
    data_config: dict[str, Any],
    scaler: RobustScaler | None,
) -> dict[str, np.ndarray | float | int]:
    cache_path = Path(row.cache_path if hasattr(row, "cache_path") else row["cache_path"])
    with np.load(cache_path) as cached:
        time_s = cached["time_s"].copy()
        raw_emg = cached["emg"].copy()
        raw_emg_mask = cached["emg_mask"].copy()
        raw_imu = cached["imu"].copy()
        raw_imu_mask = cached["imu_mask"].copy()

    reaction_time = float(
        row.reaction_time_s if hasattr(row, "reaction_time_s") else row["reaction_time_s"]
    )
    touch_time = float(
        row.touch_time_s if hasattr(row, "touch_time_s") else row["touch_time_s"]
    )
    start, end = trajectory_analysis_interval(
        time_s, reaction_time, data_config, touch_time
    )
    sample_rate = float(data_config["sample_rate_hz"])
    duration = end - start
    length = max(2, int(math.floor(duration * sample_rate)) + 1)
    grid = start + np.arange(length, dtype=np.float64) / sample_rate
    emg, emg_mask = previous_sample_resample(
        time_s, raw_emg, raw_emg_mask, grid
    )
    imu, imu_mask = previous_sample_resample(
        time_s, raw_imu, raw_imu_mask, grid
    )
    emg = causal_median_filter(emg, int(data_config.get("median_kernel", 1)))
    imu_features, imu_feature_mask = calibrate_imu_features(
        imu,
        imu_mask,
        sample_rate,
        float(data_config.get("imu_calibration_window_s", 0.3)),
    )
    if bool(data_config.get("include_raw_imu", False)):
        # Preserve the measured signal alongside invariant calibrated features.
        # Robust scaling is fitted on training trajectories only.
        imu_features = np.concatenate(
            [imu.astype(np.float32), imu_features], axis=1
        )
        imu_feature_mask = np.concatenate([imu_mask, imu_feature_mask], axis=1)
    if scaler is None:
        if bool(data_config.get("emg_log1p", True)):
            emg = np.log1p(np.maximum(emg, 0.0))
        emg = emg.astype(np.float32)
    else:
        emg = scaler.transform_emg(emg).astype(np.float32)
        imu_features = scaler.transform_imu(imu_features).astype(np.float32)
    emg[~emg_mask] = 0.0
    imu_features[~imu_feature_mask] = 0.0
    return {
        "emg": emg,
        "emg_mask": emg_mask,
        "imu": imu_features,
        "imu_mask": imu_feature_mask,
        "length": length,
        "duration_s": float(duration),
        "touch_time_s": touch_time,
        "reaction_time_s": reaction_time,
        "start_time_s": float(start),
        "cue_offset_s": float(
            np.clip(reaction_time - start, 0.0, duration)
        ),
        "movement_duration_s": float(
            np.clip(end - reaction_time, 0.0, duration)
        ),
    }


class GridTrajectoryDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_config: dict[str, Any],
        scaler: RobustScaler,
    ) -> None:
        if "touch_time_s" not in frame or not np.isfinite(
            frame["touch_time_s"].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Grid models require touch_time_s for every trial; rebuild the manifest"
            )
        self.config = data_config
        self.scaler = scaler
        self.frame = frame.reset_index(drop=True)
        self.lengths: list[int] = []
        self.durations: list[float] = []
        self.movement_durations: list[float] = []
        self.excluded: list[dict[str, Any]] = []
        minimum = float(data_config.get("min_duration_s", 0.0))
        maximum = float(data_config.get("max_duration_s", float("inf")))
        accepted = []
        for row in self.frame.itertuples(index=False):
            with np.load(row.cache_path) as cached:
                start, end = trajectory_analysis_interval(
                    cached["time_s"],
                    float(row.reaction_time_s),
                    data_config,
                    float(row.touch_time_s),
                )
            duration = end - start
            if not np.isfinite(duration) or duration < minimum or duration > maximum:
                self.excluded.append(
                    {"trial_id": str(row.trial_id), "duration_s": float(duration)}
                )
                continue
            accepted.append(row._asdict())
            self.durations.append(float(duration))
            self.movement_durations.append(
                float(np.clip(end - float(row.reaction_time_s), 0.0, duration))
            )
            self.lengths.append(
                max(2, int(math.floor(duration * float(data_config["sample_rate_hz"]))) + 1)
            )
        if not accepted:
            raise ValueError("No valid touch-aligned trajectories remain")
        self.frame = pd.DataFrame(accepted).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        arrays = preprocess_grid_signals(row, self.config, self.scaler)
        target_prefix = "click" if self.config.get("target", "click") == "click" else "target"
        target = np.asarray(
            [row[f"{target_prefix}_x_norm"], row[f"{target_prefix}_y_norm"]],
            dtype=np.float32,
        )
        return {
            "emg": torch.from_numpy(arrays["emg"]),
            "emg_mask": torch.from_numpy(arrays["emg_mask"]),
            "imu": torch.from_numpy(arrays["imu"]),
            "imu_mask": torch.from_numpy(arrays["imu_mask"]),
            "target": torch.from_numpy(target),
            "canvas_size": torch.tensor(
                [row["canvas_width_px"], row["canvas_height_px"]], dtype=torch.float32
            ),
            "button_size": torch.tensor(
                [row["button_width_px"], row["button_height_px"]], dtype=torch.float32
            ),
            "duration_s": torch.tensor(arrays["duration_s"], dtype=torch.float32),
            "reaction_time_s": torch.tensor(
                arrays["reaction_time_s"], dtype=torch.float32
            ),
            "touch_time_s": torch.tensor(arrays["touch_time_s"], dtype=torch.float32),
            "start_time_s": torch.tensor(arrays["start_time_s"], dtype=torch.float32),
            "cue_offset_s": torch.tensor(arrays["cue_offset_s"], dtype=torch.float32),
            "movement_duration_s": torch.tensor(
                arrays["movement_duration_s"], dtype=torch.float32
            ),
            "trial_id": str(row["trial_id"]),
            "subject": str(row["subject"]),
            "configuration": str(row["configuration"]),
        }


def pad_grid_trajectories(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(samples)
    maximum = max(sample["emg"].size(0) for sample in samples)
    imu_channels = samples[0]["imu"].size(1)
    if any(sample["imu"].size(1) != imu_channels for sample in samples):
        raise ValueError("All trajectories in a batch must use the same IMU features")
    emg = torch.zeros(batch, maximum, 4, dtype=torch.float32)
    emg_mask = torch.zeros(batch, maximum, 4, dtype=torch.bool)
    imu = torch.zeros(batch, maximum, imu_channels, dtype=torch.float32)
    imu_mask = torch.zeros(batch, maximum, imu_channels, dtype=torch.bool)
    lengths = torch.empty(batch, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = sample["emg"].size(0)
        lengths[index] = length
        emg[index, :length] = sample["emg"]
        emg_mask[index, :length] = sample["emg_mask"]
        imu[index, :length] = sample["imu"]
        imu_mask[index, :length] = sample["imu_mask"]
    return {
        "emg": emg,
        "emg_mask": emg_mask,
        "imu": imu,
        "imu_mask": imu_mask,
        "lengths": lengths,
        "target": torch.stack([sample["target"] for sample in samples]),
        "canvas_size": torch.stack([sample["canvas_size"] for sample in samples]),
        "button_size": torch.stack([sample["button_size"] for sample in samples]),
        "duration_s": torch.stack([sample["duration_s"] for sample in samples]),
        "reaction_time_s": torch.stack(
            [sample["reaction_time_s"] for sample in samples]
        ),
        "touch_time_s": torch.stack([sample["touch_time_s"] for sample in samples]),
        "start_time_s": torch.stack([sample["start_time_s"] for sample in samples]),
        "cue_offset_s": torch.stack([sample["cue_offset_s"] for sample in samples]),
        "movement_duration_s": torch.stack(
            [sample["movement_duration_s"] for sample in samples]
        ),
        "trial_id": [sample["trial_id"] for sample in samples],
        "subject": [sample["subject"] for sample in samples],
        "configuration": [sample["configuration"] for sample in samples],
    }


def build_grid_trajectory_loaders(
    config: dict[str, Any],
    split_path: str | None = None,
    scaler_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    manifest = load_manifest(config["paths"]["manifest"])
    split = load_json(split_path or config["paths"]["split_file"])
    scaler = RobustScaler.load(scaler_path or config["paths"]["scaler"])
    datasets = {
        name: GridTrajectoryDataset(
            subset_from_trial_ids(manifest, split[name]), config["data"], scaler
        )
        for name in ("train", "val", "test")
    }
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])
    bucket_multiplier = int(config["training"].get("bucket_size_multiplier", 10))
    loaders = []
    for name in ("train", "val", "test"):
        dataset = datasets[name]
        sampler = LengthBucketBatchSampler(
            dataset.lengths,
            batch_size=batch_size,
            bucket_size_multiplier=bucket_multiplier,
            shuffle=name == "train",
            drop_last=False,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
                collate_fn=pad_grid_trajectories,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]
