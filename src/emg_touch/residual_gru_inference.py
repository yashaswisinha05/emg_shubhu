"""Streaming inference for the latest state-conditioned residual GRU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .live_reach_grasp import LiveReachGraspPreprocessor
from .models.reach_grasp_architecture_baselines import ArchitectureBaseline
from .models.reach_grasp_neuro_classifier_attention import (
    NeuroClassifierConditionedAttention,
)
from .models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from .models.reach_grasp_residual_gru import NeuromuscularResidualGRU
from .models.reach_grasp_shared_encoder_adapter import SharedEncoderResidualGRU


def load_residual_gru(checkpoint, device="cuda"):
    device = torch.device(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    supported = {"neuromuscular_residual_gru_v1", "frozen_classifier_residual_gru_v1",
                 "shared_encoder_residual_gru_v1"}
    if state.get("format") not in supported:
        raise ValueError(
            "checkpoint must come from scripts/train_neuromuscular_residual_gru.py")
    if state["model_args"].get("modality") != "emg+imu":
        raise ValueError("live inference requires an emg+imu checkpoint")
    if state["format"] == "shared_encoder_residual_gru_v1":
        classifier_args = state["classifier_model_args"]
        classifier = NeuromuscularFutureGripperPoseModel(**classifier_args)
        model = SharedEncoderResidualGRU(
            classifier, state["classifier_normalization"], state["normalization"],
            **state["shared_model_args"])
    else:
        motion = NeuromuscularResidualGRU(**state["model_args"])
    if state["format"] == "frozen_classifier_residual_gru_v1":
        classifier_format = state["classifier_format"]
        classifier_args = state["classifier_model_args"]
        if classifier_format == "gripper_neuromuscular_future_v1":
            classifier = NeuromuscularFutureGripperPoseModel(**classifier_args)
        elif classifier_format == "reach_grasp_architecture_baseline_v1":
            architecture = state.get("classifier_architecture", "gru")
            classifier = ArchitectureBaseline(architecture, **classifier_args)
        else:
            raise ValueError(f"unsupported frozen classifier: {classifier_format}")
        model = NeuroClassifierConditionedAttention(
            classifier, motion, state["classifier_normalization"],
            state["normalization"])
    elif state["format"] == "neuromuscular_residual_gru_v1":
        model = motion
    model = model.to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False), state


class ResidualGRUStream:
    """Consume raw 4-channel EMG and 24-channel IMU samples causally."""

    def __init__(self, checkpoint, device="cuda", warmup_ms=300., context_ms=2000.,
                 close_threshold=.7, open_threshold=.3, raw_rate_hz=None):
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= open < close <= 1")
        self.model, self.state = load_residual_gru(checkpoint, device)
        self.motion = (self.model.motion if isinstance(
            self.model, NeuroClassifierConditionedAttention) else self.model)
        self.device = torch.device(device)
        self.stats = self.state["normalization"]
        preprocessing = dict(self.state["preprocessing"])
        if raw_rate_hz is not None:
            if not np.isfinite(raw_rate_hz) or raw_rate_hz <= 50:
                raise ValueError("raw_rate_hz must be finite and greater than 50")
            preprocessing["raw_rate_hz"] = float(raw_rate_hz)
        self.pipeline = LiveReachGraspPreprocessor(preprocessing)
        self.warmup_s = warmup_ms / 1000
        self.context_frames = max(1, round(context_ms / 1000 * self.pipeline.rate))
        self.close_threshold, self.open_threshold = close_threshold, open_threshold
        self.reset()

    def reset(self):
        self.pipeline.reset()
        self.gripper_state = "open"
        self.last_prediction_frame = 0

    @torch.no_grad()
    def predict(self, canvas_px=(1920, 1080)):
        time, emg, imu, emg_valid, imu_valid = self.pipeline.arrays()
        if not len(time) or time[-1] - time[0] < self.warmup_s:
            return None
        time, emg, imu, emg_valid, imu_valid = [value[-self.context_frames:]
            for value in (time, emg, imu, emg_valid, imu_valid)]
        if emg_valid[-1].mean() < .75 or imu_valid[-1].mean() < .75:
            return {"valid": False, "time_s": float(time[-1]),
                    "gripper_state": self.gripper_state,
                    "reason": "fewer than 75% of EMG or IMU channels are recent"}

        emg_z = (emg - self.stats["emg"]["mean"]) / self.stats["emg"]["std"]
        imu_z = (imu - self.stats["imu"]["mean"]) / self.stats["imu"]["std"]
        diagnostics = {
            "emg_abs_z_max": float(np.nanmax(np.abs(emg_z))),
            "imu_abs_z_max": float(np.nanmax(np.abs(imu_z))),
            "emg_fraction_abs_z_gt_10": float(np.mean(np.abs(emg_z) > 10)),
            "imu_fraction_abs_z_gt_10": float(np.mean(np.abs(imu_z) > 10)),
        }
        emg_input = np.concatenate((np.where(emg_valid, np.clip(emg_z, -20, 20), 0),
                                    emg_valid), -1).astype("float32")
        imu_input = np.concatenate((np.where(imu_valid, np.clip(imu_z, -20, 20), 0),
                                    imu_valid), -1).astype("float32")
        output = self.model(torch.from_numpy(emg_input)[None].to(self.device),
                            torch.from_numpy(imu_input)[None].to(self.device))
        probability = output["gripper_state_logits"][0, -1].softmax(-1)
        close_probability = float(probability[1])
        if close_probability >= self.close_threshold:
            self.gripper_state = "close"
        elif close_probability <= self.open_threshold:
            self.gripper_state = "open"

        position_std = np.asarray(self.stats["position"]["std"])
        position_mean = np.asarray(self.stats["position"]["mean"])
        position = output["position"][0, -1].cpu().numpy() * position_std + position_mean
        future = output["future_position"][0, -1].cpu().numpy() * position_std + position_mean
        click = output["click"][0, -1].clamp(0, 1).cpu().numpy()
        canvas = np.asarray(canvas_px, dtype=float)
        if canvas.shape != (2,) or not np.isfinite(canvas).all() or np.any(canvas <= 0):
            raise ValueError("canvas_px must be a finite (width, height) pair")
        result = {
            "valid": True,
            "time_s": float(time[-1]),
            "gripper_state": self.gripper_state,
            "open_probability": float(probability[0]),
            "close_probability": close_probability,
            "position_m": position.tolist(),
            "pixel_normalized_xy": click.tolist(),
            "pixel_xy": (click * canvas).tolist(),
            "future_horizons_ms": list(range(10, 10 * len(future) + 1, 10)),
            "future_positions_m": future.tolist(),
            "emg_correction_gate": float(output["emg_correction_gate"][0, -1, 0]),
            "normalization_diagnostics": diagnostics,
        }
        intent_delta = output.get("intent_position_delta")
        if intent_delta is not None:
            delta_m = intent_delta[0, -1].cpu().numpy() * position_std
            result["intent_horizons_ms"] = [step * 10
                for step in self.motion.intent_horizons_steps]
            result["intent_positions_m"] = (position[None] + delta_m).tolist()
        return result

    def update(self, time_s, emg_4, imu_24, canvas_px=(1920, 1080)):
        """Add one raw sample and return at most one prediction per 100 Hz frame."""
        self.pipeline.add_sample(time_s, emg_4, imu_24)
        if self.pipeline.frames == self.last_prediction_frame:
            return None
        self.last_prediction_frame = self.pipeline.frames
        return self.predict(canvas_px)
