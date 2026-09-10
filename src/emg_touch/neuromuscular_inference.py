"""Minimal streaming API for the selected EMG+IMU future model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .live_reach_grasp import LiveReachGraspPreprocessor
from .models.pixel_gripper_film import PixelGripperFiLM
from .models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from .physics.rotation_6d import (matrix_to_quaternion_numpy,
                                  rotation_6d_to_matrix)


def load_neuromuscular_model(checkpoint, device="cuda"):
    """Load either the original or participant-calibrated checkpoint."""
    device = torch.device(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if state.get("format") == "gripper_pixel_film_calibration_v1":
        base_state = state["base_state_dict"]
        model = NeuromuscularFutureGripperPoseModel(
            **state["model_args"]).to(device)
        model.load_state_dict(base_state)
        calibrated = PixelGripperFiLM(model, state["film_groups"]).to(device)
        calibrated.load_calibration_state_dict(state["calibration_state_dict"])
        model = calibrated
    elif state.get("format") == "gripper_neuromuscular_future_v1":
        model = NeuromuscularFutureGripperPoseModel(
            **state["model_args"]).to(device)
        model.load_state_dict(state["state_dict"])
    else:
        raise ValueError("checkpoint is not a neuromuscular future model or its FiLM calibration")
    return model.eval().requires_grad_(False), state


class NeuromuscularStream:
    """Call ``update`` for each raw 4-EMG/24-IMU observation."""

    def __init__(self, checkpoint, device="cuda", warmup_ms=200., context_ms=2000.):
        self.model, self.state = load_neuromuscular_model(checkpoint, device)
        self.device = torch.device(device)
        self.stats = self.state["normalization"]
        self.pipeline = LiveReachGraspPreprocessor(self.state["preprocessing"])
        self.warmup_s = warmup_ms / 1000
        self.context_frames = max(1, round(context_ms / 1000 * self.pipeline.rate))

    def reset(self):
        self.pipeline.reset()

    @torch.no_grad()
    def update(self, time_s, emg, imu, canvas_px=None):
        """Add one raw sample and return None until warm, then one prediction.

        ``emg`` must contain the four physical channels S0/S4/S8/S12.
        ``imu`` must contain 24 values in the training channel order.
        """
        self.pipeline.add_sample(time_s, emg, imu)
        time, emg, imu, ev, iv = self.pipeline.arrays()
        if not len(time) or time[-1] - time[0] < self.warmup_s:
            return None
        time, emg, imu, ev, iv = [value[-self.context_frames:]
                                  for value in (time, emg, imu, ev, iv)]
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
        close_probability = output["gripper_state_logits"][0, -1].softmax(-1)[1]
        click = output["click"][0, -1].clamp(0, 1).cpu().numpy()
        position = (output["position"][0, -1].cpu().numpy()
                    * self.stats["position"]["std"] + self.stats["position"]["mean"])
        final_position = (output["final_position"][0, -1].cpu().numpy()
                          * self.stats["position"]["std"] + self.stats["position"]["mean"])
        future = (output["future_position"][0, -1].cpu().numpy()
                  * self.stats["position"]["std"] + self.stats["position"]["mean"])
        rotation = rotation_6d_to_matrix(output["orientation_6d"][0, -1]).cpu().numpy()
        final_rotation = rotation_6d_to_matrix(
            output["final_orientation_6d"][0, -1]).cpu().numpy()
        future_rotation = rotation_6d_to_matrix(
            output["future_orientation_6d"][0, -1]).cpu().numpy()
        result = {
            "time_s": float(time[-1]),
            "gripper_state": "close" if close_probability >= .5 else "open",
            "close_probability": float(close_probability),
            "click_normalized_xy": click.tolist(),
            "position_m": position.tolist(),
            "orientation_wxyz": matrix_to_quaternion_numpy(rotation).tolist(),
            "final_position_m": final_position.tolist(),
            "final_orientation_wxyz": matrix_to_quaternion_numpy(final_rotation).tolist(),
            "future_horizons_ms": list(range(10, 10 * len(future) + 1, 10)),
            "future_positions_m": future.tolist(),
            "future_orientations_wxyz": [
                matrix_to_quaternion_numpy(value).tolist() for value in future_rotation],
            "normalization_diagnostics": diagnostics,
        }
        if canvas_px is not None:
            canvas = np.asarray(canvas_px, dtype=float)
            if canvas.shape != (2,) or np.any(canvas <= 0):
                raise ValueError("canvas_px must be (width, height)")
            result["click_pixel_xy"] = (click * canvas).tolist()
        return result
