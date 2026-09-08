"""Hybrid reach/grasp model with an additional continuous orientation head."""
from __future__ import annotations

from torch import nn

from .reach_grasp_hybrid import ReachGraspHybrid


class ReachGraspOrientationHybrid(ReachGraspHybrid):
    """Predict translation and 6D rotation from the shared global context."""
    def __init__(self, *args, **kwargs):
        width = kwargs.get("width", 128)
        super().__init__(*args, **kwargs)
        self.orientation = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 6))

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        result["orientation_6d"] = self.orientation(result["context_features"])
        return result
