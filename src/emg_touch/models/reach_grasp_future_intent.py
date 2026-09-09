"""Causal wearable model for current state and one-second future intent."""
from __future__ import annotations

import torch
from torch import nn

from .reach_grasp_robust import RobustReachGraspModel


class ReachGraspFutureIntentModel(RobustReachGraspModel):
    """Predict current interaction/SE(3) plus fixed future SE(3) waypoints.

    The encoder receives only EMG and IMU. VIVE quantities predicted here are
    supervision targets during training, never encoder inputs.
    """

    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, event_time_bins=6,
                 future_horizons_ms=(100, 250, 500, 750, 1000)):
        super().__init__(modality, width, patch, stride, layers, heads, dropout,
                         event_time_bins)
        horizons = torch.as_tensor(future_horizons_ms, dtype=torch.float32)
        if horizons.ndim != 1 or len(horizons) < 2 or not torch.all(horizons > 0):
            raise ValueError("future horizons must contain at least two positive values")
        if not torch.all(horizons[1:] > horizons[:-1]):
            raise ValueError("future horizons must be strictly increasing")
        self.future_steps = len(horizons)
        self.register_buffer("future_horizons_ms", horizons)
        context_width = width * 2
        self.future_position = nn.Sequential(
            nn.Linear(context_width, context_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(context_width, self.future_steps * 3))
        self.future_orientation = nn.Sequential(
            nn.Linear(context_width, context_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(context_width, self.future_steps * 6))
        self.future_event = nn.Sequential(
            nn.Linear(context_width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 2))
        # Time-to-event centers are the trajectory horizons plus "now"; the
        # final categorical output is no event within one second.
        self.intent_time_bins = self.future_steps + 2
        self.intent_time = nn.Sequential(
            nn.Linear(context_width, width), nn.GELU(),
            nn.Linear(width, 2 * self.intent_time_bins))

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        context = result["context_features"]
        shape = context.shape[:2]
        result["future_position"] = self.future_position(context).reshape(
            *shape, self.future_steps, 3)
        result["future_orientation_6d"] = self.future_orientation(context).reshape(
            *shape, self.future_steps, 6)
        result["future_event_logits"] = self.future_event(context)
        result["intent_time_logits"] = self.intent_time(context).reshape(
            *shape, 2, self.intent_time_bins)
        result["future_endpoint"] = result["future_position"][..., -1, :]
        return result

