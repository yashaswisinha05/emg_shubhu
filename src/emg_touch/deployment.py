from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .checkpointing import load_model_state
from .data.grid_trajectory import calibrate_imu_features
from .data.preprocessing import (
    RobustScaler,
    causal_median_filter,
    previous_sample_resample,
)
from .models.grid_point import build_grid_model
from .utils import choose_device


DEPLOYABLE_KINDS = ("grid_emg", "grid_fusion")
RAW_EMG_DIM = 4
RAW_IMU_DIM = 24


def _clean_sample(
    values: Sequence[float] | np.ndarray,
    expected: int,
    supplied_mask: Sequence[bool] | np.ndarray | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape [{expected}], received {array.shape}")
    finite = np.isfinite(array)
    if supplied_mask is None:
        mask = finite
    else:
        mask = np.asarray(supplied_mask, dtype=bool)
        if mask.shape != (expected,):
            raise ValueError(
                f"{name}_mask must have shape [{expected}], received {mask.shape}"
            )
        mask &= finite
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), mask


class ContinualTouchPredictor:
    """Stateful, fixed-weight continual inference for one movement trajectory.

    Raw samples may arrive at an irregular rate. Every prediction causally
    resamples the samples observed so far onto the rate used during training,
    applies the saved fold scaler, and evaluates one trajectory prefix.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        scaler_path: str | Path,
        *,
        device: str | None = None,
        screen_width_px: float = 1536.0,
        screen_height_px: float = 774.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if not self.scaler_path.is_file():
            raise FileNotFoundError(self.scaler_path)

        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.kind = str(checkpoint.get("model_kind", ""))
        if self.kind not in DEPLOYABLE_KINDS:
            raise ValueError(
                f"Expected one of {DEPLOYABLE_KINDS}, found {self.kind!r}"
            )
        self.config: dict[str, Any] = checkpoint["config"]
        self.sample_rate_hz = float(self.config["data"]["sample_rate_hz"])
        self.interval_s = float(
            self.config.get("continual", {}).get("interval_s", 0.2)
        )
        self.median_kernel = int(self.config["data"].get("median_kernel", 1))
        self.calibration_window_s = float(
            self.config["data"].get("imu_calibration_window_s", 0.3)
        )
        self.include_raw_imu = bool(
            self.config["data"].get("include_raw_imu", False)
        )
        self.maximum_duration_s = float(
            self.config["data"].get("max_duration_s", 10.0)
        )
        self.screen_width_px = float(screen_width_px)
        self.screen_height_px = float(screen_height_px)
        if self.screen_width_px <= 0 or self.screen_height_px <= 0:
            raise ValueError("Screen dimensions must be positive")

        self.scaler = RobustScaler.load(self.scaler_path)
        self.device = choose_device(device)
        self.model = build_grid_model(self.kind, self.config).to(self.device)
        load_model_state(self.model, self.checkpoint_path)
        self.model.eval()
        self.reset()

    @property
    def requires_imu(self) -> bool:
        return self.kind == "grid_fusion"

    def reset(self) -> None:
        self._times: list[float] = []
        self._emg: list[np.ndarray] = []
        self._emg_mask: list[np.ndarray] = []
        self._imu: list[np.ndarray] = []
        self._imu_mask: list[np.ndarray] = []
        self.movement_start_s: float | None = None
        self._duration_warning_emitted = False

    def set_movement_start(self, time_s: float) -> None:
        time_s = float(time_s)
        if not math.isfinite(time_s):
            raise ValueError("movement_start time must be finite")
        if self._times and time_s < self._times[0]:
            raise ValueError("movement_start cannot precede the first buffered sample")
        self.movement_start_s = time_s

    def add_sample(
        self,
        time_s: float,
        emg: Sequence[float] | np.ndarray,
        imu: Sequence[float] | np.ndarray | None = None,
        *,
        emg_mask: Sequence[bool] | np.ndarray | None = None,
        imu_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
        timestamp = float(time_s)
        if not math.isfinite(timestamp):
            raise ValueError("sample time must be finite")
        emg_array, emg_valid = _clean_sample(
            emg, RAW_EMG_DIM, emg_mask, "emg"
        )
        if imu is None:
            if self.requires_imu:
                raise ValueError("grid_fusion requires 24 raw IMU values per sample")
            imu_array = np.zeros(RAW_IMU_DIM, dtype=np.float32)
            imu_valid = np.zeros(RAW_IMU_DIM, dtype=bool)
        else:
            imu_array, imu_valid = _clean_sample(
                imu, RAW_IMU_DIM, imu_mask, "imu"
            )

        if self._times and timestamp < self._times[-1]:
            raise ValueError("Samples must arrive in monotonically increasing time order")
        if self._times and timestamp == self._times[-1]:
            # Match the training cache policy: retain the final sample for a
            # duplicate hardware timestamp.
            self._emg[-1] = emg_array
            self._emg_mask[-1] = emg_valid
            self._imu[-1] = imu_array
            self._imu_mask[-1] = imu_valid
            return
        self._times.append(timestamp)
        self._emg.append(emg_array)
        self._emg_mask.append(emg_valid)
        self._imu.append(imu_array)
        self._imu_mask.append(imu_valid)

    def _processed_prefix(
        self, cutoff_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if len(self._times) < 2:
            raise RuntimeError("At least two raw samples are required")
        raw_times = np.asarray(self._times, dtype=np.float64)
        selected = raw_times <= cutoff_s + 1e-9
        if selected.sum() < 2:
            raise RuntimeError("The requested cutoff has fewer than two samples")
        raw_times = raw_times[selected]
        raw_emg = np.stack(self._emg, axis=0)[selected]
        raw_emg_mask = np.stack(self._emg_mask, axis=0)[selected]
        duration = float(cutoff_s - raw_times[0])
        length = max(2, int(math.floor(duration * self.sample_rate_hz)) + 1)
        grid = raw_times[0] + np.arange(length, dtype=np.float64) / self.sample_rate_hz
        emg, emg_mask = previous_sample_resample(
            raw_times, raw_emg, raw_emg_mask, grid
        )
        emg = causal_median_filter(emg, self.median_kernel)
        emg = self.scaler.transform_emg(emg).astype(np.float32)
        emg[~emg_mask] = 0.0

        if not self.requires_imu:
            return emg, emg_mask, None, None

        raw_imu = np.stack(self._imu, axis=0)[selected]
        raw_imu_mask = np.stack(self._imu_mask, axis=0)[selected]
        imu, imu_mask = previous_sample_resample(
            raw_times, raw_imu, raw_imu_mask, grid
        )
        imu_features, imu_feature_mask = calibrate_imu_features(
            imu,
            imu_mask,
            self.sample_rate_hz,
            self.calibration_window_s,
        )
        if self.include_raw_imu:
            imu_features = np.concatenate(
                [imu.astype(np.float32), imu_features], axis=1
            )
            imu_feature_mask = np.concatenate(
                [imu_mask, imu_feature_mask], axis=1
            )
        imu_features = self.scaler.transform_imu(imu_features).astype(np.float32)
        imu_features[~imu_feature_mask] = 0.0
        return emg, emg_mask, imu_features, imu_feature_mask

    def predict(
        self,
        cutoff_s: float | None = None,
        *,
        label: str | None = None,
    ) -> dict[str, Any]:
        if not self._times:
            raise RuntimeError("No samples have been buffered")
        if self.movement_start_s is None:
            raise RuntimeError("Call set_movement_start() before prediction")
        cutoff = self._times[-1] if cutoff_s is None else float(cutoff_s)
        if cutoff > self._times[-1] + 1e-9:
            raise ValueError("Cannot predict beyond the latest observed sample")
        if cutoff < self.movement_start_s:
            raise ValueError("Prediction cutoff precedes movement_start")
        pre_movement_s = self.movement_start_s - self._times[0]
        if self.requires_imu and pre_movement_s + 1e-9 < self.calibration_window_s:
            raise RuntimeError(
                f"Fusion inference needs at least {self.calibration_window_s:.3f}s "
                "of resting samples before movement_start"
            )

        emg, emg_mask, imu, imu_mask = self._processed_prefix(cutoff)
        minimum_samples = int(self.config["model"].get("patch_length", 16))
        if len(emg) < minimum_samples:
            raise RuntimeError(
                f"Need at least {minimum_samples} resampled samples, found {len(emg)}"
            )
        elapsed = max(0.0, cutoff - self.movement_start_s)
        recording_duration = cutoff - self._times[0]
        outside_training_duration = recording_duration > self.maximum_duration_s

        batch: dict[str, torch.Tensor] = {
            "emg": torch.from_numpy(emg).unsqueeze(0).to(self.device),
            "emg_mask": torch.from_numpy(emg_mask).unsqueeze(0).to(self.device),
            "lengths": torch.tensor([len(emg)], dtype=torch.long, device=self.device),
            "prefix_elapsed_s": torch.tensor(
                [elapsed], dtype=torch.float32, device=self.device
            ),
            "movement_duration_s": torch.tensor(
                [elapsed], dtype=torch.float32, device=self.device
            ),
        }
        if self.requires_imu:
            assert imu is not None and imu_mask is not None
            batch["imu"] = torch.from_numpy(imu).unsqueeze(0).to(self.device)
            batch["imu_mask"] = torch.from_numpy(imu_mask).unsqueeze(0).to(
                self.device
            )

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(batch)
        inference_ms = (time.perf_counter() - started) * 1000.0
        prediction = outputs["prediction"][0].detach().cpu().numpy()
        result: dict[str, Any] = {
            "event": "prediction",
            "label": label or f"{elapsed:.1f}s",
            "model_kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "cutoff_time_s": cutoff,
            "elapsed_movement_s": elapsed,
            "resampled_samples": int(len(emg)),
            "x_norm": float(prediction[0]),
            "y_norm": float(prediction[1]),
            "x_px": float(prediction[0] * self.screen_width_px),
            "y_px": float(prediction[1] * self.screen_height_px),
            "screen_width_px": self.screen_width_px,
            "screen_height_px": self.screen_height_px,
            "inference_ms": inference_ms,
            "outside_training_duration": outside_training_duration,
        }
        for output_name, result_name in (
            ("heatmap_confidence", "aux_grid_confidence"),
            ("heatmap_entropy", "aux_grid_entropy"),
            ("emg_reliability", "emg_reliability"),
        ):
            value = outputs.get(output_name)
            if value is not None:
                result[result_name] = float(value.reshape(-1)[0].detach().cpu())
        predicted_cell = outputs.get("predicted_cell")
        if predicted_cell is not None:
            cell = int(predicted_cell[0].detach().cpu())
            width = int(self.config["model"].get("grid_size", [8, 5])[0])
            result["aux_grid_cell"] = {
                "index": cell,
                "x": cell % width,
                "y": cell // width,
            }
        lookback = outputs.get("emg_lookback_weights")
        if lookback is not None:
            weights = lookback[0].detach().cpu().tolist()
            result["emg_lookback_weights"] = {
                "final_500ms": float(weights[0]),
                "final_300ms": float(weights[1]),
            }
        return result

