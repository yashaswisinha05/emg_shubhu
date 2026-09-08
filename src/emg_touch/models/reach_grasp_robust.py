"""Causal multimodal model with event horizons and calibrated pose uncertainty."""
from __future__ import annotations

import torch
from torch import nn

from .reach_grasp_hybrid import ReachGraspHybrid


class RobustReachGraspModel(ReachGraspHybrid):
    """Specialized EMG/IMU model used by the robust reach-grasp experiment.

    The frame-resolution local path predicts both event pulses and time until
    each event.  The longer patch context predicts translation, orientation,
    and heteroscedastic uncertainty.  Every output remains causal.
    """
    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, event_time_bins=6):
        super().__init__(modality, width, patch, stride, layers, heads, dropout)
        if event_time_bins < 3:
            raise ValueError("event_time_bins must include at least two horizons and no-event")
        self.event_time_bins = int(event_time_bins)
        self.orientation = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 6))
        self.event_time = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=7, padding=0, groups=width),
            nn.GELU(), nn.Conv1d(width, width, 1), nn.GELU(),
            nn.Conv1d(width, 2 * self.event_time_bins, 1))
        # Start near the proven local event head and let training decide how
        # much the time-to-event estimate should amend its current-event logit.
        self.horizon_blend_logit = nn.Parameter(torch.full((2,), -2.))
        self.position_log_variance = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 3))
        self.orientation_log_variance = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        local = result["local_features"]
        horizon = self.event_time(torch.nn.functional.pad(
            local.transpose(1, 2), (6, 0))).transpose(1, 2)
        horizon = horizon.reshape(*horizon.shape[:2], 2, self.event_time_bins)
        # Bin zero is "event now" and the last bin is "no event in horizon".
        horizon_log_odds = horizon[..., 0] - horizon[..., -1]
        event_logits = (result["logits"][..., 1:] +
                        self.horizon_blend_logit.sigmoid() * horizon_log_odds)
        result["logits"] = torch.cat([result["logits"][..., :1], event_logits], -1)
        result["event_time_logits"] = horizon
        result["orientation_6d"] = self.orientation(result["context_features"])
        result["position_log_variance"] = self.position_log_variance(
            result["context_features"]).clamp(-7., 5.)
        result["orientation_log_variance"] = self.orientation_log_variance(
            result["context_features"]).clamp(-7., 5.)
        return result
