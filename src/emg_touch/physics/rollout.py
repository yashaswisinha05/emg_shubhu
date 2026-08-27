"""Differentiable forward rollout of the Hill-driven arm, as a model branch.

The rollout is the paper's Eq. 6 integrated forward in time, driven by muscle
activations derived from EMG. It is attached alongside the existing fusion
coordinate head rather than replacing it: an analytic kinematic chain needs
arm orientation accurate to about 3 degrees to reach the accuracy the learned
model already achieves, and IMU orientation on this data is 12-19 degrees, so
physics cannot carry the prediction by itself. As an auxiliary pathway it
still gives EMG a structured route - activation, force, torque, joint angle -
instead of competing with IMU to regress the same coordinate.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .arm import EndpointToScreen, TwoLinkArm
from .hill import ActivationDynamics, HillMuscle


class PhysicsBranch(nn.Module):
    """EMG -> activation -> Hill force -> torque -> integrated joint angles."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        physics = config.get("physics", {})
        d_model = int(model["d_model"])
        self.sample_rate_hz = float(data["sample_rate_hz"])
        # Integrating every sample is wasteful and memory-hungry; the arm's
        # dynamics are far slower than 148 Hz.
        self.decimation = int(physics.get("decimation", 4))
        self.residual_scale = float(physics.get("residual_torque_scale", 5.0))
        # Muscle torque divided by this arm's rotational inertia gives
        # accelerations of order 10^2 rad/s^2 (measured: mass matrix diagonal
        # ~0.03-0.5 kg*m^2 against ~15 N*m of elbow torque). A single
        # semi-implicit Euler step at the decimated interval (~27 ms) can then
        # change velocity by several rad/s in one step - unconditionally
        # unstable, not merely inaccurate. Sub-stepping the integration within
        # each recorded interval keeps the physics stable without changing
        # what gets recorded in physics_trajectory.
        self.substeps = int(physics.get("substeps", 8))

        self.activation = ActivationDynamics(4, self.sample_rate_hz)
        self.muscle = HillMuscle(4)
        self.arm = TwoLinkArm()
        self.to_screen = EndpointToScreen()

        # EMG arrives robust-scaled, not in [0, 1]; learn the mapping to a
        # neural-excitation range per muscle.
        self.excitation_gain = nn.Parameter(torch.ones(4))
        self.excitation_bias = nn.Parameter(torch.zeros(4))
        # Initial joint state is not observed, so predict it from the encoder
        # context that already sees the pre-movement window.
        self.initial_state = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 4)
        )
        # Residual torque absorbs what a planar 2-DOF Hill model cannot
        # represent: trunk motion, wrist, scapular rhythm, co-contraction.
        self.residual_torque = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2)
        )
        nn.init.zeros_(self.residual_torque[-1].weight)
        nn.init.zeros_(self.residual_torque[-1].bias)

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
        excitation = torch.sigmoid(
            amplitude * self.excitation_gain + self.excitation_bias
        ) * valid
        activation = self.activation(excitation)

        state = self.initial_state(context)
        angles = torch.stack([state[:, 0] * 0.5, 0.5 + state[:, 1] * 0.5], dim=-1)
        velocity = state[:, 2:] * 0.5
        residual = torch.tanh(self.residual_torque(context)) * self.residual_scale

        dt = self.decimation / self.sample_rate_hz / self.substeps
        indices = list(range(0, steps, self.decimation))
        lower = torch.tensor([-1.8, 0.0], device=angles.device, dtype=angles.dtype)
        upper = torch.tensor([1.8, 2.8], device=angles.device, dtype=angles.dtype)
        # Only integrate while a sample is inside the trial's causal prefix;
        # padded steps must not advance the state.
        trajectory = []
        for step in indices:
            active = (step < lengths).to(angles.dtype).unsqueeze(-1)
            for _ in range(self.substeps):
                torque = (
                    self.muscle.torque(activation[:, step], angles, velocity) + residual
                )
                acceleration = self.arm.acceleration(angles, velocity, torque)
                # Semi-implicit Euler: symplectic, and stable at this step
                # size where explicit Euler drifts.
                velocity = velocity + active * dt * acceleration
                velocity = velocity.clamp(-25.0, 25.0)
                unclamped = angles + active * dt * velocity
                angles = unclamped.clamp(lower, upper)
                # An inelastic joint stop: clamping position alone leaves
                # velocity driving into the wall untouched, so the next step
                # reclamps to the same limit and the joint is pinned there
                # permanently, independent of any later change in torque.
                # Zeroing the velocity component that caused the clamp lets a
                # torque reversal - a change in EMG - lift the joint off the
                # stop on a later step.
                at_limit = unclamped != angles
                velocity = torch.where(at_limit, torch.zeros_like(velocity), velocity)
            trajectory.append(angles)

        final = angles
        endpoint = self.arm.endpoint(final)
        return {
            "physics_prediction": self.to_screen(endpoint),
            "physics_angles": final,
            "physics_velocity": velocity,
            "physics_activation": activation,
            "physics_residual_torque": residual,
            "physics_trajectory": torch.stack(trajectory, dim=1),
        }
