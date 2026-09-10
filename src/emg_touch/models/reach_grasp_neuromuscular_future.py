"""Causal neuromuscular dynamics decoder for future pose intent.

The decoder is specific to this project: IMU supplies a motion-state rollout,
while EMG supplies a gated innovation that may alter that rollout before the
mechanical response appears. It is not a Siamese or masked-token architecture.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .reach_grasp_gripper_pixel import GoalConsistentGripperPoseModel


class NeuromuscularFutureGripperPoseModel(GoalConsistentGripperPoseModel):
    """Predict future SE(3) pose as IMU dynamics plus EMG innovation."""

    def __init__(self, future_steps=20, **kwargs):
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        super().__init__(future_steps=0, **kwargs)
        width = kwargs.get("width", 128)
        self.future_steps = future_steps
        self.future_pose = None

        self.imu_state = nn.Sequential(
            nn.Linear(24, width), nn.LayerNorm(width), nn.GELU())
        self.emg_intent = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU())
        self.fused_state = nn.Sequential(
            nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

        self.inertial_dynamics = nn.Sequential(
            nn.Linear(width * 2 + 4, width * 2), nn.GELU(),
            nn.Linear(width * 2, width))
        self.emg_innovation = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, width))
        self.innovation_gate = nn.Sequential(
            nn.Linear(width * 2 + 4, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.innovation_gate[-1].weight)
        nn.init.constant_(self.innovation_gate[-1].bias, -2.)

        self.future_pose_projector = nn.Linear(width, 9)
        self.future_imu_delta_projector = nn.Linear(width, 24)

        tau = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        horizon = torch.stack((tau, tau.square(), torch.sin(math.pi * tau),
                               torch.cos(math.pi * tau)), -1)
        self.register_buffer("horizon_basis", horizon, persistent=True)

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        fused = self.fused_state(result["context_features"])
        inertial = self.imu_state(imu[..., :24])
        intent = self.emg_intent(result["emg_context_features"])
        if self.modality == "emg":
            inertial = torch.zeros_like(inertial)
        elif self.modality == "imu":
            intent = torch.zeros_like(intent)

        horizon = self.horizon_basis.view(1, 1, self.future_steps, 4)
        fused_h = fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1)
        inertial_h = inertial.unsqueeze(2).expand_as(fused_h)
        intent_h = intent.unsqueeze(2).expand_as(fused_h)
        horizon_h = horizon.expand(*fused_h.shape[:-1], 4)

        base = self.inertial_dynamics(torch.cat((fused_h, inertial_h, horizon_h), -1))
        innovation = self.emg_innovation(torch.cat((intent_h, horizon_h), -1))
        gate = self.innovation_gate(torch.cat((fused_h, intent_h, horizon_h), -1)).sigmoid()
        if self.modality == "imu":
            gate = torch.zeros_like(gate)
        future = base + gate * innovation
        pose = self.future_pose_projector(future)
        result["future_position"] = pose[..., :3]
        result["future_orientation_6d"] = pose[..., 3:]
        result["future_imu_delta"] = self.future_imu_delta_projector(future)
        result["emg_innovation_gate"] = gate
        return result
