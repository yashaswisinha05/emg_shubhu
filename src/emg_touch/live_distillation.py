"""Stateful live inference for tracked-dataset distillation checkpoints.

Unlike the offline evaluation scripts, this module never reads a recorded
future trajectory. Raw EMG and IMU samples are appended as they arrive, the
training feature bank is rebuilt causally up to the newest timestamp, and
each model receives only its rolling EMG+IMU history.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data.schema import sensor_names
from .data.tracked_dataset import (
    apply_sensor_local_pca,
    emg_feature_bank,
    emg_feature_count,
    imu_feature_count,
    imu_posture_bank,
    raw_emg_feature_count,
)
from .models.channel_horizon_distillation import (
    ChannelHorizonLatentDistillationModel,
)
from .models.complete_reach_distillation import (
    CompleteReachDistillationModel,
)
from .models.direction_aware_complete_reach import (
    DirectionAwareCompleteReachModel,
)
from .models.monotonic_complete_reach import MonotonicCompleteReachModel
from .models.task_separated_complete_reach import (
    TaskSeparatedCompleteReachModel,
)
from .models.deterministic_complete_reach import (
    DeterministicCompleteReachModel,
)
from .models.asymmetric_intent_motion import AsymmetricIntentMotionModel
from .models.latent_distillation import WearableLatentDistillationModel
from .models.rolling_dual_head_distillation import (
    RollingDualHeadDistillationModel,
)
from .models.semantic_residual_distillation import (
    SemanticResidualDistillationModel,
)
from .models.temporal_cross_attention_distillation import (
    TemporalCrossAttentionDistillationModel,
)
from .models.teacher_bridge_distillation import TeacherBridgeDistillationModel
from .utils import choose_device


RAW_IMU_AXES_PER_SENSOR = 6


def checkpoint_kind(state: dict[str, torch.Tensor]) -> str:
    keys = tuple(state)
    if any(
        key.startswith("student.asymmetric_motion_heads.correction_adapter.")
        for key in keys
    ):
        return "asymmetric_intent_motion"
    if any(
        key.startswith("student.deterministic_heads.correction_adapter.")
        for key in keys
    ):
        return "deterministic_complete_reach"
    if any(
        key.startswith("student.endpoint_decoder.path_correction_head.")
        for key in keys
    ):
        return "task_separated_complete_reach"
    if any(
        key.startswith("student.endpoint_decoder.progress_increment_head.")
        for key in keys
    ):
        return "monotonic_complete_reach"
    if any(
        key.startswith("student.endpoint_decoder.axis_direction_head.")
        for key in keys
    ):
        return "direction_aware_complete_reach"
    if any(
        key.startswith("student.endpoint_decoder.endpoint_3d_head.")
        for key in keys
    ):
        return "complete_reach"
    if any(
        key.startswith("student.endpoint_decoder.screen_semantic.")
        for key in keys
    ) and any(
        key.startswith("student.endpoint_decoder.motion_semantic.")
        for key in keys
    ):
        return "rolling_dual_head"
    if any(key.startswith("student.teacher_latent_bridge.") for key in keys) and any(
        key.startswith("student.endpoint_decoder.") for key in keys
    ):
        return "teacher_bridge"
    if any(key.startswith("student.lag_attention.") for key in keys) and any(
        key.startswith("student.emg_from_imu.") for key in keys
    ):
        return "temporal_cross_attention"
    if any(key.startswith("student.fused_endpoint_residual.") for key in keys):
        return "semantic_residual"
    if any(key.startswith("student.channel_gate.") for key in keys):
        return "channel_horizon"
    if any(key.startswith("teacher.") for key in keys) and any(
        key.startswith("student.") for key in keys
    ):
        return "latent_distillation"
    raise ValueError(
        "unsupported checkpoint architecture; expected an asymmetric intent-motion, "
        "deterministic, task-separated, "
        "monotonic, direction-aware complete-reach, complete-reach, "
        "rolling-dual-head, "
        "latent-distillation, "
        "channel+horizon, semantic-residual, temporal-cross-attention, or "
        "teacher-bridge "
        "final.pt/best.pt"
    )


def _load_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a dictionary")
    state = payload.get("model_state")
    config = payload.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError(
            f"checkpoint {path} must contain model_state and config; use final.pt"
        )
    return config, state


def preprocessing_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    data = config["data"]
    keys = (
        "sensors",
        "sample_rate_hz",
        "decimation",
        "emg_feature_windows_ms",
        "emg_feature_kinds",
        "emg_envelope_ms",
        "emg_bandpass_hz",
        "emg_filter_order",
        "emg_notch_hz",
        "emg_notch_quality",
        "emg_pca_components_per_sensor",
        "imu_posture_features",
        "posture_lowpass_ms",
        "posture_baseline_ms",
    )
    return tuple(
        (key, tuple(data[key]) if isinstance(data.get(key), list) else data.get(key))
        for key in keys
    )


class LiveFeaturePipeline:
    """Causal raw-sample buffer using the exact tracked training features."""

    def __init__(
        self,
        config: dict[str, Any],
        calibration_path: str | Path,
        maximum_buffer_s: float = 8.0,
    ) -> None:
        self.config = config
        self.data_config = config["data"]
        self.sensors = sensor_names(self.data_config)
        self.raw_emg_dim = len(self.sensors)
        self.raw_imu_dim = RAW_IMU_AXES_PER_SENSOR * len(self.sensors)
        self.emg_dim = emg_feature_count(self.data_config)
        self.raw_emg_feature_dim = raw_emg_feature_count(self.data_config)
        self.imu_dim = imu_feature_count(self.data_config)
        self.decimation = max(1, int(self.data_config.get("decimation", 10)))
        self.maximum_buffer_s = float(maximum_buffer_s)
        self.calibration_path = Path(calibration_path).resolve()
        if not self.calibration_path.is_file():
            raise FileNotFoundError(self.calibration_path)
        with np.load(self.calibration_path, allow_pickle=False) as calibration:
            self.emg_scale = calibration["emg_scale"].astype(np.float32)
            self.imu_center = calibration["imu_center"].astype(np.float32)
            self.imu_scale = calibration["imu_scale"].astype(np.float32)
        expected = {
            "emg_scale": (self.raw_emg_feature_dim,),
            "imu_center": (self.imu_dim,),
            "imu_scale": (self.imu_dim,),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(
                    f"calibration {name} has shape {actual}; expected {shape}"
                )
        self.reset()

    def reset(self) -> None:
        self._times: list[float] = []
        self._emg: list[np.ndarray] = []
        self._imu: list[np.ndarray] = []

    @property
    def sample_count(self) -> int:
        return len(self._times)

    @property
    def latest_time_s(self) -> float | None:
        return self._times[-1] if self._times else None

    @property
    def first_time_s(self) -> float | None:
        return self._times[0] if self._times else None

    @property
    def raw_rate_hz(self) -> float | None:
        if len(self._times) < 2:
            return None
        duration = self._times[-1] - self._times[0]
        return len(self._times) / duration if duration > 0.0 else None

    def _clean(self, values: Sequence[float], expected: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (expected,):
            raise ValueError(f"{name} must have shape [{expected}], got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return array

    def add_sample(
        self, time_s: float, emg: Sequence[float], imu: Sequence[float]
    ) -> None:
        timestamp = float(time_s)
        if not math.isfinite(timestamp):
            raise ValueError("time_s must be finite")
        emg_array = self._clean(emg, self.raw_emg_dim, "emg")
        imu_array = self._clean(imu, self.raw_imu_dim, "imu")
        if self._times and timestamp < self._times[-1]:
            raise ValueError("live samples must have monotonically increasing time_s")
        if self._times and timestamp == self._times[-1]:
            self._emg[-1] = emg_array
            self._imu[-1] = imu_array
            return
        self._times.append(timestamp)
        self._emg.append(emg_array)
        self._imu.append(imu_array)
        self._trim()

    def add_samples(
        self,
        times_s: Sequence[float],
        emg: Sequence[Sequence[float]],
        imu: Sequence[Sequence[float]],
    ) -> None:
        if not (len(times_s) == len(emg) == len(imu)):
            raise ValueError("time_s, emg, and imu batches must have equal length")
        for timestamp, emg_row, imu_row in zip(times_s, emg, imu):
            self.add_sample(timestamp, emg_row, imu_row)

    def _trim(self) -> None:
        if len(self._times) < 2:
            return
        cutoff = self._times[-1] - self.maximum_buffer_s
        first = int(np.searchsorted(np.asarray(self._times), cutoff, side="left"))
        if first > 0:
            # Keep one preceding sample so causal derivatives at the retained
            # boundary have continuity instead of inventing a zero derivative.
            first -= 1
            self._times = self._times[first:]
            self._emg = self._emg[first:]
            self._imu = self._imu[first:]

    def processed(self) -> tuple[np.ndarray, np.ndarray, float]:
        if len(self._times) < 2:
            raise RuntimeError("at least two live samples are required")
        times = np.asarray(self._times, dtype=np.float64)
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            raise RuntimeError("live sample timestamps have zero duration")
        raw_rate = len(times) / duration
        raw_emg = np.stack(self._emg)
        raw_imu = np.stack(self._imu)
        emg = emg_feature_bank(raw_emg, raw_rate, self.data_config)
        imu = raw_imu
        if self.data_config.get("imu_posture_features", False):
            imu = np.concatenate(
                [imu, imu_posture_bank(raw_imu, raw_rate, self.data_config)],
                axis=1,
            )
        indices = np.arange(0, len(times), self.decimation)
        emg = emg[indices] / self.emg_scale
        emg = apply_sensor_local_pca(emg, self.data_config)
        imu = (imu[indices] - self.imu_center) / self.imu_scale
        effective_rate = raw_rate / self.decimation
        return emg.astype(np.float32), imu.astype(np.float32), effective_rate


class LiveDistillationModel:
    """One fixed-weight wearable model evaluated on a rolling live buffer."""

    def __init__(
        self,
        name: str,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.name = str(name)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.config, state = _load_payload(self.checkpoint_path)
        self.kind = checkpoint_kind(state)
        self.device = choose_device(device)
        emg_dim = emg_feature_count(self.config["data"])
        imu_dim = imu_feature_count(self.config["data"])
        if self.kind == "asymmetric_intent_motion":
            self.model = AsymmetricIntentMotionModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "deterministic_complete_reach":
            self.model = DeterministicCompleteReachModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "task_separated_complete_reach":
            self.model = TaskSeparatedCompleteReachModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "monotonic_complete_reach":
            self.model = MonotonicCompleteReachModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "direction_aware_complete_reach":
            self.model = DirectionAwareCompleteReachModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "complete_reach":
            self.model = CompleteReachDistillationModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "rolling_dual_head":
            self.model = RollingDualHeadDistillationModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "teacher_bridge":
            self.model = TeacherBridgeDistillationModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "temporal_cross_attention":
            self.model = TemporalCrossAttentionDistillationModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "semantic_residual":
            self.model = SemanticResidualDistillationModel(
                self.config, emg_dim, imu_dim
            )
        elif self.kind == "channel_horizon":
            self.model = ChannelHorizonLatentDistillationModel(
                self.config, emg_dim, imu_dim
            )
        else:
            self.model = WearableLatentDistillationModel(
                self.config, emg_dim, imu_dim
            )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        model_config = self.config["model"]
        configured_rate = float(self.config["data"]["sample_rate_hz"]) / max(
            1, int(self.config["data"].get("decimation", 10))
        )
        self.context_samples = max(
            int(model_config["patch_length"]),
            int(round(float(model_config.get("context_ms", 2000.0))
                      * configured_rate / 1000.0)),
        )
        self.patch_length = int(model_config["patch_length"])

    def _window(
        self, values: np.ndarray, channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = values[-self.context_samples :]
        padded = np.zeros((self.context_samples, channels), dtype=np.float32)
        padded[-len(available) :] = available
        mask = np.zeros(self.context_samples, dtype=bool)
        mask[-len(available) :] = True
        return (
            torch.from_numpy(padded).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )

    @torch.inference_mode()
    def predict(
        self,
        emg: np.ndarray,
        imu: np.ndarray,
        canvas: tuple[float, float],
        effective_rate_hz: float,
    ) -> dict[str, Any]:
        if len(emg) < self.patch_length:
            raise RuntimeError(
                f"{self.name} needs {self.patch_length} processed samples; "
                f"currently {len(emg)}"
            )
        emg_tensor, mask = self._window(emg, emg.shape[1])
        imu_tensor, _ = self._window(imu, imu.shape[1])
        started = time.perf_counter()
        outputs = self.model.student_forward(
            emg_tensor, imu_tensor, mask, sample=False
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = 1000.0 * (time.perf_counter() - started)
        prediction = outputs["prediction"][0].detach().cpu().numpy()
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "x_norm": float(prediction[0]),
            "y_norm": float(prediction[1]),
            "x_px": float(prediction[0] * canvas[0]),
            "y_px": float(prediction[1] * canvas[1]),
            "inference_ms": inference_ms,
        }
        trajectory = outputs.get("trajectory")
        if trajectory is not None:
            relative = trajectory[0].detach().cpu().numpy()
            result["trajectory_relative_m"] = relative.tolist()
            result["trajectory_endpoint_relative_m"] = relative[-1].tolist()
        endpoint_3d = outputs.get("endpoint_3d")
        if endpoint_3d is not None:
            result["endpoint_3d_relative_m"] = (
                endpoint_3d[0].detach().cpu().tolist()
            )
            result["complete_trajectory_relative_m"] = result.get(
                "trajectory_relative_m", []
            )
        direction_logits = outputs.get("axis_direction_logits")
        if direction_logits is not None:
            direction_names = ("negative", "stationary", "positive")
            classes = direction_logits[0].argmax(dim=-1).detach().cpu().tolist()
            result["axis_directions"] = {
                axis: direction_names[int(direction)]
                for axis, direction in zip(("x", "y", "z"), classes)
            }
        guidance = outputs.get("guidance", {})
        if "horizon_expected_ms" in guidance:
            result["horizon_ms"] = float(
                guidance["horizon_expected_ms"][0].detach().cpu()
            )
        attention = outputs.get("channel_attention")
        if attention is not None:
            final_samples = max(1, int(round(0.1 * effective_rate_hz)))
            valid_attention = attention[0, -min(final_samples, len(emg)) :]
            mean_attention = valid_attention.mean(dim=0).detach().cpu().tolist()
            result["channel_attention"] = {
                sensor: float(value)
                for sensor, value in zip(
                    sensor_names(self.config["data"]), mean_attention
                )
            }
        return result
