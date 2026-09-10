"""Streaming inference for the state-conditioned attention V2 model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .live_reach_grasp import LiveReachGraspPreprocessor
from .models.reach_grasp_state_attention_v2 import StateConditionedAttentionV2


def load_state_attention_model(checkpoint, device="cuda"):
    device = torch.device(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if state.get("format") != "gripper_state_attention_v2":
        raise ValueError("checkpoint must come from train_gripper_state_attention_v2.py")
    if state["model_args"].get("modality") != "emg+imu":
        raise ValueError("streaming deployment requires the fused emg+imu checkpoint")
    model = StateConditionedAttentionV2(**state["model_args"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False), state


class StateAttentionStream:
    """Consume raw four-channel EMG and 24-channel IMU observations."""

    def __init__(self, checkpoint, device="cuda", warmup_ms=200., context_ms=2000.,
                 close_threshold=.7, open_threshold=.3, raw_rate_hz=None):
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= open < close <= 1")
        self.model, self.state = load_state_attention_model(checkpoint, device)
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

    def add_sample(self, time_s, emg_4, imu_24):
        self.pipeline.add_sample(time_s, emg_4, imu_24)

    @torch.no_grad()
    def predict(self, canvas_px=(1440, 900)):
        time, emg, imu, ev, iv = self.pipeline.arrays()
        if not len(time) or time[-1] - time[0] < self.warmup_s:
            return None
        time, emg, imu, ev, iv = [value[-self.context_frames:]
                                  for value in (time, emg, imu, ev, iv)]
        valid = bool(ev[-1].mean() >= .75 and iv[-1].mean() >= .75)
        if not valid:
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
        emg_z, imu_z = np.clip(emg_z, -20, 20), np.clip(imu_z, -20, 20)
        emg_input = np.concatenate((np.where(ev, emg_z, 0), ev), 1).astype("float32")
        imu_input = np.concatenate((np.where(iv, imu_z, 0), iv), 1).astype("float32")
        output = self.model(torch.from_numpy(emg_input)[None].to(self.device),
                            torch.from_numpy(imu_input)[None].to(self.device))

        probability = output["gripper_state_logits"][0, -1].softmax(-1)
        close_probability = float(probability[1])
        if close_probability >= self.close_threshold:
            self.gripper_state = "close"
        elif close_probability <= self.open_threshold:
            self.gripper_state = "open"

        scale = np.asarray(self.stats["position"]["std"])
        mean = np.asarray(self.stats["position"]["mean"])
        position = output["position"][0, -1].cpu().numpy() * scale + mean
        future = output["future_position"][0, -1].cpu().numpy() * scale + mean
        click = output["click"][0, -1].clamp(0, 1).cpu().numpy()
        canvas = np.asarray(canvas_px, dtype=float)
        if canvas.shape != (2,) or not np.isfinite(canvas).all() or np.any(canvas <= 0):
            raise ValueError("canvas_px must be a finite (width, height) pair")
        channel_attention = output["emg_channel_attention"][0, -1].cpu().numpy()
        pixel_blend = float(output["click_position_blend"][0, -1, 0])
        return {
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
            "emg_channel_attention": {
                sensor: float(weight) for sensor, weight in
                zip(("S0", "S4", "S8", "S12"), channel_attention)},
            "pixel_xyz_blend": pixel_blend,
            "emg_correction_gate": float(output["emg_correction_gate"]),
            "normalization_diagnostics": diagnostics,
        }

    def update(self, time_s, emg_4, imu_24, canvas_px=(1440, 900)):
        """Add one raw sample; return at most one prediction per 100 Hz frame."""
        self.add_sample(time_s, emg_4, imu_24)
        if self.pipeline.frames == self.last_prediction_frame:
            return None
        self.last_prediction_frame = self.pipeline.frames
        return self.predict(canvas_px)
