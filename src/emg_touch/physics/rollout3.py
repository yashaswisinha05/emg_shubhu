"""Differentiable forward rollout of the 3-DOF arm, torque predicted directly
from grid_fusion's learned features rather than a Hill muscle chain.

The Hill model in rollout.py structured EMG through activation dynamics,
force-length/velocity curves, and moment arms - physiologically motivated, but
its per-muscle parameters (peak force, moment arm) had to be learned from
scratch with no direct supervision, which is what made the joint dynamics
fragile enough to pin the elbow at 0 for an entire trial. This branch instead
predicts total joint torque with a plain MLP reading EMG, the current joint
state, and grid_fusion's pooled context - the "known physics you already know"
(M(q)qdd + C(q,qd)qd + Gv(q)) is still handled exactly, by arm3.ThreeDofArm;
what's learned is only the torque that drives it.

There is still no ground truth for torque or joint angle anywhere in this
dataset - only the final screen touch location - so this remains exactly the
same weak, indirect supervision as the Hill branch: forward-integrate under
the predicted torque and backprop from the endpoint loss alone.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .arm3 import EndpointToScreen3, ThreeDofArm


class TorqueHead(nn.Module):
    """EMG + joint state + pooled context -> joint torque, per decimated step."""

    def __init__(self, d_model: int, context_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.context_project = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, context_dim), nn.GELU()
        )
        input_dim = 4 + 3 + 3 + context_dim  # emg, angles, velocity, context
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        # Zero-initialised so the rollout starts torque-free (gravity/damping
        # only) rather than at some arbitrary scale - same rationale as the
        # Hill branch's residual_torque zero-init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        emg_amplitude: torch.Tensor,
        angles: torch.Tensor,
        velocity: torch.Tensor,
        context_embed: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([emg_amplitude, angles, velocity, context_embed], dim=-1)
        return self.net(features)


class PhysicsBranch3(nn.Module):
    """EMG + context -> torque -> integrated 3-DOF joint angles -> screen point."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        physics = config.get("physics", {})
        d_model = int(model["d_model"])
        self.sample_rate_hz = float(data["sample_rate_hz"])
        self.decimation = int(physics.get("decimation", 4))
        self.substeps = int(physics.get("substeps", 8))

        self.arm = ThreeDofArm()
        self.to_screen = EndpointToScreen3()
        self.torque_head = TorqueHead(d_model)

        self.initial_state = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 6)
        )

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = emg.shape
        amplitude = emg[:, :, :4]
        valid = emg_mask[:, :, :4].to(amplitude.dtype)
        amplitude = amplitude * valid

        context_embed = self.torque_head.context_project(context)

        state = self.initial_state(context)
        angles = state[:, 0:3] * 0.5
        angles = angles + angles.new_tensor([0.0, 0.0, 1.4])  # elbow centred mid-range
        velocity = state[:, 3:6] * 0.5

        dt = self.decimation / self.sample_rate_hz / self.substeps
        indices = list(range(0, steps, self.decimation))
        lower = torch.tensor([-1.8, -1.8, 0.0], device=angles.device, dtype=angles.dtype)
        upper = torch.tensor([1.8, 1.8, 2.8], device=angles.device, dtype=angles.dtype)
        trajectory = []
        torques = []
        for step in indices:
            active = (step < lengths).to(angles.dtype).unsqueeze(-1)
            torque = self.torque_head(amplitude[:, step], angles, velocity, context_embed)
            torques.append(torque)
            for _ in range(self.substeps):
                acceleration = self.arm.acceleration(angles, velocity, torque)
                velocity = velocity + active * dt * acceleration
                velocity = velocity.clamp(-25.0, 25.0)
                unclamped = angles + active * dt * velocity
                angles = unclamped.clamp(lower, upper)
                # Same inelastic-joint-stop fix as the 2-link model: zero the
                # velocity component that caused the clamp, or the joint pins
                # at the limit forever regardless of later torque changes.
                at_limit = unclamped != angles
                velocity = torch.where(at_limit, torch.zeros_like(velocity), velocity)
            trajectory.append(angles)

        final = angles
        endpoint = self.arm.endpoint(final)
        return {
            "physics_prediction": self.to_screen(endpoint),
            "physics_angles": final,
            "physics_velocity": velocity,
            "physics_torque": torch.stack(torques, dim=1),
            "physics_trajectory": torch.stack(trajectory, dim=1),
        }
