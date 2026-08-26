from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import load_manifest
from .preprocessing import RobustScaler, causal_median_filter, previous_sample_resample
from .splits import subset_from_trial_ids
from ..utils import load_json


def emg_window_bounds(data_config: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return the configured EMG window in seconds relative to touch.

    ``None`` at either edge means unbounded.  The default preserves the complete
    trajectory.  A window such as ``[-0.3, 0.0]`` retains only the final 300 ms.
    """

    configured = data_config.get("emg_window_s")
    if configured is None:
        return None, None
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        raise ValueError("data.emg_window_s must be [start_s, end_s]")
    start = None if configured[0] is None else float(configured[0])
    end = None if configured[1] is None else float(configured[1])
    if start is not None and end is not None and start > end:
        raise ValueError(
            f"data.emg_window_s start must not exceed end; received {configured}"
        )
    if start is not None and start > 0.0:
        raise ValueError("EMG study windows must not start after touch")
    if end is not None and end > 0.0:
        raise ValueError("EMG study windows must not include post-touch samples")
    return start, end


def trajectory_analysis_interval(
    time_s: np.ndarray,
    reaction_time_s: float,
    data_config: dict[str, Any],
    touch_time_s: float = float("nan"),
) -> tuple[float, float]:
    """Resolve the recorded interval used by a trajectory model."""

    if len(time_s) == 0 or not np.isfinite(time_s).all():
        return float("nan"), float("nan")
    start = float(time_s[0])
    mode = str(data_config.get("trajectory_end", "recording_end")).lower()
    if mode == "recording_end":
        end = float(time_s[-1])
    elif mode == "reaction_time":
        if not np.isfinite(reaction_time_s):
            raise ValueError("reaction_time_s is required for trajectory_end=reaction_time")
        # Never extrapolate beyond the samples that were actually recorded.
        end = min(float(time_s[-1]), float(reaction_time_s))
    elif mode == "touch_time":
        if not np.isfinite(touch_time_s):
            raise ValueError(
                "touch_time_s is required for trajectory_end=touch_time; rebuild "
                "the manifest with scripts/build_manifest.py"
            )
        end = min(float(time_s[-1]), float(touch_time_s))
    else:
        raise ValueError(
            "data.trajectory_end must be 'recording_end', 'reaction_time', or "
            "'touch_time'; "
            f"received {mode!r}"
        )
    return start, end


def temporal_anchor_time(
    reaction_time_s: float,
    touch_time_s: float,
    data_config: dict[str, Any],
) -> float:
    """Return the signal-time anchor used for relative EMG windows."""

    anchor = str(data_config.get("temporal_anchor", "reaction_time")).lower()
    if anchor == "reaction_time":
        value = reaction_time_s
    elif anchor == "touch_time":
        value = touch_time_s
    else:
        raise ValueError(
            "data.temporal_anchor must be 'reaction_time' or 'touch_time'; "
            f"received {anchor!r}"
        )
    if not np.isfinite(value):
        raise ValueError(
            f"A finite timestamp is required for temporal_anchor={anchor}"
        )
    return float(value)


def touch_relative_window_mask(
    times_s: np.ndarray,
    reaction_time_s: float,
    window: tuple[float | None, float | None],
) -> np.ndarray:
    """Select timestamps inside a closed, touch-relative EMG window."""

    start, end = window
    relative = times_s - float(reaction_time_s)
    selected = np.ones(len(times_s), dtype=bool)
    if start is not None:
        selected &= relative >= start
    if end is not None:
        selected &= relative <= end
    return selected


class FullTrajectoryDataset(Dataset[dict[str, Any]]):
    """Variable-length trajectories with optional causal and EMG-window controls."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_config: dict[str, Any],
        scaler: RobustScaler,
    ) -> None:
        self.config = data_config
        self.scaler = scaler
        self.sample_rate = float(data_config["sample_rate_hz"])
        self.median_kernel = int(data_config.get("median_kernel", 1))
        self.emg_window = emg_window_bounds(data_config)
        self.temporal_label = str(data_config.get("temporal_label", "full"))
        self.excluded: list[dict[str, Any]] = []
        minimum = float(data_config.get("min_duration_s", 0.0))
        maximum = float(data_config.get("max_duration_s", float("inf")))
        policy = str(data_config.get("outlier_policy", "exclude")).lower()
        if policy not in {"exclude", "error", "include"}:
            raise ValueError(f"Unknown outlier_policy={policy!r}")

        accepted_rows = []
        lengths = []
        durations = []
        analysis_starts = []
        analysis_ends = []
        reaction_times = []
        touch_times = []
        temporal_anchors = []
        recording_durations = []
        emg_window_sample_counts = []
        for row in frame.itertuples(index=False):
            cache_path = Path(row.cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing cache {cache_path}; run scripts/prepare_cache.py first"
                )
            with np.load(cache_path) as cached:
                time_s = cached["time_s"]
                recording_duration = (
                    float(time_s[-1] - time_s[0]) if len(time_s) >= 2 else 0.0
                )
                reaction_time = float(row.reaction_time_s)
                touch_time = float(getattr(row, "touch_time_s", float("nan")))
                analysis_start, analysis_end = trajectory_analysis_interval(
                    time_s, reaction_time, data_config, touch_time
                )
                temporal_anchor = temporal_anchor_time(
                    reaction_time, touch_time, data_config
                )
                duration = analysis_end - analysis_start
            invalid = not np.isfinite(duration) or duration < minimum or duration > maximum
            if invalid:
                detail = {
                    "trial_id": str(row.trial_id),
                    "duration_s": duration,
                    "minimum_s": minimum,
                    "maximum_s": maximum,
                }
                if policy == "error":
                    raise ValueError(f"Invalid trajectory duration: {detail}")
                if policy == "exclude":
                    self.excluded.append(detail)
                    continue
            length = max(2, int(math.floor(duration * self.sample_rate)) + 1)
            analysis_grid = analysis_start + np.arange(length, dtype=np.float64) / self.sample_rate
            window_sample_count = int(
                touch_relative_window_mask(
                    analysis_grid, temporal_anchor, self.emg_window
                ).sum()
            )
            accepted_rows.append(row._asdict())
            lengths.append(length)
            durations.append(duration)
            analysis_starts.append(analysis_start)
            analysis_ends.append(analysis_end)
            reaction_times.append(reaction_time)
            touch_times.append(touch_time)
            temporal_anchors.append(temporal_anchor)
            recording_durations.append(recording_duration)
            emg_window_sample_counts.append(window_sample_count)

        if not accepted_rows:
            raise ValueError("No valid full trajectories remain after duration checks")
        self.frame = pd.DataFrame(accepted_rows).reset_index(drop=True)
        self.lengths = lengths
        self.durations = durations
        self.analysis_starts = analysis_starts
        self.analysis_ends = analysis_ends
        self.reaction_times = reaction_times
        self.touch_times = touch_times
        self.temporal_anchors = temporal_anchors
        self.recording_durations = recording_durations
        self.emg_window_sample_counts = emg_window_sample_counts

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with np.load(row["cache_path"]) as cached:
            time_s = cached["time_s"].copy()
            raw_emg = cached["emg"].copy()
            raw_emg_mask = cached["emg_mask"].copy()
            raw_imu = cached["imu"].copy()
            raw_imu_mask = cached["imu_mask"].copy()

        step = 1.0 / self.sample_rate
        grid = self.analysis_starts[index] + np.arange(
            self.lengths[index], dtype=np.float64
        ) * step
        emg, emg_mask = previous_sample_resample(
            time_s, raw_emg, raw_emg_mask, grid
        )
        imu, imu_mask = previous_sample_resample(
            time_s, raw_imu, raw_imu_mask, grid
        )
        emg = causal_median_filter(emg, self.median_kernel)
        emg = self.scaler.transform_emg(emg).astype(np.float32)
        imu = self.scaler.transform_imu(imu).astype(np.float32)
        emg_time_mask = touch_relative_window_mask(
            grid, self.temporal_anchors[index], self.emg_window
        )
        emg_mask &= emg_time_mask[:, None]
        emg[~emg_mask] = 0.0
        imu[~imu_mask] = 0.0

        target_prefix = "click" if self.config.get("target", "click") == "click" else "target"
        target = np.asarray(
            [row[f"{target_prefix}_x_norm"], row[f"{target_prefix}_y_norm"]],
            dtype=np.float32,
        )
        if not np.isfinite(target).all():
            raise ValueError(f"Invalid target for {row['trial_id']}: {target}")
        return {
            "emg": torch.from_numpy(emg),
            "emg_mask": torch.from_numpy(emg_mask),
            "imu": torch.from_numpy(imu),
            "imu_mask": torch.from_numpy(imu_mask),
            "target": torch.from_numpy(target),
            "canvas_size": torch.tensor(
                [float(row["canvas_width_px"]), float(row["canvas_height_px"])],
                dtype=torch.float32,
            ),
            "button_size": torch.tensor(
                [float(row["button_width_px"]), float(row["button_height_px"])],
                dtype=torch.float32,
            ),
            "duration_s": torch.tensor(self.durations[index], dtype=torch.float32),
            "recording_duration_s": torch.tensor(
                self.recording_durations[index], dtype=torch.float32
            ),
            "reaction_time_s": torch.tensor(
                self.reaction_times[index], dtype=torch.float32
            ),
            "touch_time_s": torch.tensor(
                self.touch_times[index], dtype=torch.float32
            ),
            "emg_window_samples": torch.tensor(
                self.emg_window_sample_counts[index], dtype=torch.long
            ),
            "temporal_label": self.temporal_label,
            "trial_id": str(row["trial_id"]),
            "subject": str(row["subject"]),
            "configuration": str(row["configuration"]),
        }


def pad_full_trajectories(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(samples)
    maximum = max(sample["emg"].size(0) for sample in samples)
    emg = torch.zeros(batch_size, maximum, 4, dtype=torch.float32)
    emg_mask = torch.zeros(batch_size, maximum, 4, dtype=torch.bool)
    imu = torch.zeros(batch_size, maximum, 24, dtype=torch.float32)
    imu_mask = torch.zeros(batch_size, maximum, 24, dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)
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
        "recording_duration_s": torch.stack(
            [sample["recording_duration_s"] for sample in samples]
        ),
        "reaction_time_s": torch.stack(
            [sample["reaction_time_s"] for sample in samples]
        ),
        "touch_time_s": torch.stack(
            [sample["touch_time_s"] for sample in samples]
        ),
        "emg_window_samples": torch.stack(
            [sample["emg_window_samples"] for sample in samples]
        ),
        "temporal_label": [sample["temporal_label"] for sample in samples],
        "trial_id": [sample["trial_id"] for sample in samples],
        "subject": [sample["subject"] for sample in samples],
        "configuration": [sample["configuration"] for sample in samples],
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size_multiplier: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [ordered[i : i + self.bucket_size] for i in range(0, len(ordered), self.bucket_size)]
        batches: list[list[int]] = []
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for index in range(0, len(bucket), self.batch_size):
                batch = bucket[index : index + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches


def build_full_trajectory_loaders(
    config: dict[str, Any],
    split_path: str | None = None,
    scaler_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    manifest = load_manifest(config["paths"]["manifest"])
    split = load_json(split_path or config["paths"]["split_file"])
    scaler = RobustScaler.load(scaler_path or config["paths"]["scaler"])
    datasets = {
        name: FullTrajectoryDataset(
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
            # Retain every valid trajectory, including the final partial batch.
            drop_last=False,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
                collate_fn=pad_full_trajectories,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]
