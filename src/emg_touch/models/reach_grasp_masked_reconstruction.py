"""Robust reach-grasp model with a training-only EMG reconstruction head."""
from __future__ import annotations

from torch import nn

from .reach_grasp_robust import RobustReachGraspModel


class MaskedReconstructionReachGraspModel(RobustReachGraspModel):
    """Decode hidden normalized EMG features from the causal EMG context.

    The decoder is used only for the auxiliary training loss. Deployment keeps
    the same four raw EMG and 24 raw IMU inputs and the same task outputs.
    """

    def __init__(self, *args, **kwargs):
        width = kwargs.get("width", 128)
        super().__init__(*args, **kwargs)
        self.emg_reconstruction = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, 8))

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        result["emg_reconstruction"] = self.emg_reconstruction(
            result["emg_context_features"])
        return result
