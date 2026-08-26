from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import RobustScaler, causal_median_filter, previous_sample_resample


class TouchTrialDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_config: dict[str, Any],
        scaler: RobustScaler,
        training: bool,
        fixed_cutoff_s: float | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = data_config
        self.scaler = scaler
        self.training = training
        self.fixed_cutoff_s = fixed_cutoff_s
        self.sample_rate = float(data_config["sample_rate_hz"])
        self.context_length = int(data_config["context_length"])
        self.future_samples = int(data_config.get("future_imu_samples", 0))
        self.median_kernel = int(data_config.get("median_kernel", 1))

    def __len__(self) -> int:
        return len(self.frame)

    def _choose_cutoff(self, maximum: float) -> float:
        if self.fixed_cutoff_s is not None:
            return maximum if self.fixed_cutoff_s < 0 else min(self.fixed_cutoff_s, maximum)
        if not self.training:
            return maximum
        if random.random() < float(self.config.get("full_trial_probability", 0.25)):
            return maximum
        choices = [float(x) for x in self.config.get("train_cutoffs_s", []) if float(x) <= maximum]
        return random.choice(choices) if choices else maximum

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cache {cache_path}; run scripts/prepare_cache.py")
        with np.load(cache_path) as cached:
            time_s = cached["time_s"].copy()
            cached_emg = cached["emg"].copy()
            cached_emg_mask = cached["emg_mask"].copy()
            cached_imu = cached["imu"].copy()
            cached_imu_mask = cached["imu_mask"].copy()
        cutoff = self._choose_cutoff(float(time_s[-1]))
        step = 1.0 / self.sample_rate
        context_grid = cutoff - np.arange(self.context_length - 1, -1, -1) * step

        emg, emg_mask = previous_sample_resample(
            time_s, cached_emg, cached_emg_mask, context_grid
        )
        imu, imu_mask = previous_sample_resample(
            time_s, cached_imu, cached_imu_mask, context_grid
        )
        emg = causal_median_filter(emg, self.median_kernel)
        emg = self.scaler.transform_emg(emg).astype(np.float32)
        imu = self.scaler.transform_imu(imu).astype(np.float32)
        emg[~emg_mask] = 0.0
        imu[~imu_mask] = 0.0
        time_mask = emg_mask.any(axis=1) | imu_mask.any(axis=1)

        future_grid = cutoff + np.arange(1, self.future_samples + 1) * step
        future_imu, future_imu_mask = previous_sample_resample(
            time_s, cached_imu, cached_imu_mask, future_grid
        )
        future_imu = self.scaler.transform_imu(future_imu).astype(np.float32)
        future_imu[~future_imu_mask] = 0.0

        target_prefix = "click" if self.config.get("target", "click") == "click" else "target"
        target = np.asarray(
            [row[f"{target_prefix}_x_norm"], row[f"{target_prefix}_y_norm"]],
            dtype=np.float32,
        )
        return {
            "emg": torch.from_numpy(emg),
            "emg_mask": torch.from_numpy(emg_mask),
            "imu": torch.from_numpy(imu),
            "imu_mask": torch.from_numpy(imu_mask),
            "time_mask": torch.from_numpy(time_mask),
            "future_imu": torch.from_numpy(future_imu),
            "future_imu_mask": torch.from_numpy(future_imu_mask),
            "target": torch.from_numpy(target),
            "canvas_size": torch.tensor(
                [float(row["canvas_width_px"]), float(row["canvas_height_px"])],
                dtype=torch.float32,
            ),
            "button_size": torch.tensor(
                [float(row["button_width_px"]), float(row["button_height_px"])],
                dtype=torch.float32,
            ),
            "cutoff_s": torch.tensor(cutoff, dtype=torch.float32),
            "trial_id": str(row["trial_id"]),
            "subject": str(row["subject"]),
            "configuration": str(row["configuration"]),
        }
