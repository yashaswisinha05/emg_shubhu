"""Virtual-leader branch for datasets where hand kinematics are MEASURED.

The branch in virtual_leader.py infers (position, velocity, acceleration) with
a GRU because this project's original dataset has no hand trajectory - only
the final touch location. With an end-effector tracker that inference
disappears: position AND velocity are measured directly (a real Vive export
reports vel_x/y/z_mps alongside position, already computed by the tracking
pipeline), so only acceleration needs differencing - one pass instead of two,
which matters because each differencing pass amplifies sensor noise. The
attractor readout

    r = x + (xddot + rho xdot) / eta

then computes a destination estimate from measurements at every timestep,
rather than from a latent state the network had to invent. Measured
kinematics with intent inferred from them is the setting the attractor
model is actually meant for, so this is the faithful version rather than
the approximation the untracked dataset forced.

What EMG contributes once kinematics are measured is a sharper question than
before, and this module is arranged so it can be answered rather than
assumed:

  - eta and rho are predicted per timestep from EMG, not held constant. The
    attractor's stiffness is how hard the arm is being pulled toward the
    goal, which is what muscle activity encodes; a fixed eta cannot express
    a reach that starts gently and commits late.
  - EMG leads motion by the electromechanical delay (~40-80 ms), so during
    the earliest samples - before the tracker registers movement at all -
    EMG is the only channel carrying intent. Setting emg_only_warmup marks
    that window so the contribution can be measured instead of argued about.

Deliberately causal throughout: backward differences only, and a
unidirectional GRU for the EMG pathway. Early prediction is the point.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

ETA_RANGE = (0.5, 20.0)
DRAG_RANGE = (0.0, 5.0)


def causal_difference(
    values: torch.Tensor, dt: float, valid: torch.Tensor
) -> torch.Tensor:
    """Backward difference along time, zero at t=0 and wherever invalid.

    Backward rather than central: a central difference at time t reads
    t+1, which would leak future motion into an estimate the whole model is
    built to make from the past only.
    """
    difference = torch.zeros_like(values)
    difference[:, 1:] = (values[:, 1:] - values[:, :-1]) / dt
    both_valid = valid[:, 1:] * valid[:, :-1]
    difference[:, 1:] = difference[:, 1:] * both_valid.unsqueeze(-1)
    return difference


class TrackedVirtualLeaderBranch(nn.Module):
    """Measured hand kinematics + EMG -> per-timestep destination posterior."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        settings = config.get("virtual_leader", {})
        self.decimation = int(settings.get("decimation", 4))
        self.sample_rate_hz = float(data["sample_rate_hz"])
        hidden = int(settings.get("hidden", 96))
        # 3 for a 3-D tracker; set 2 if the task is genuinely planar.
        self.position_dim = int(settings.get("position_dim", 3))
        # Samples at the start of a trial where the tracker has not moved yet
        # but EMG is already active. Reported separately so the "EMG predicts
        # intent before motion is visible" claim is measured, not asserted.
        self.emg_only_warmup = int(settings.get("emg_only_warmup", 0))

        emg_channels = int(settings.get("emg_channels", 0)) or None
        if emg_channels is None:
            from ..data.grid_trajectory import emg_channel_count

            emg_channels = emg_channel_count(data)
        self.emg_channels = emg_channels

        context_dim = 2 * int(model["d_model"])
        self.context_project = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 32), nn.GELU()
        )
        # EMG plus the measured kinematics it is being asked to explain.
        self.encoder = nn.GRU(
            input_size=emg_channels + 3 * self.position_dim + 32,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.normalise = nn.LayerNorm(hidden)
        # Per-timestep attractor parameters and a reliability weight.
        self.dynamics = nn.Linear(hidden, 2)
        self.confidence = nn.Linear(hidden, 1)
        nn.init.zeros_(self.dynamics.weight)
        nn.init.zeros_(self.dynamics.bias)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        position: torch.Tensor,
        position_mask: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
        velocity: torch.Tensor | None = None,
        velocity_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """position: (B, T, position_dim) measured end-effector track.

        velocity, if given, is used directly (this rig's tracker measures it)
        and only acceleration is differenced. Without it, both are derived
        from position by two differencing passes, as before - kept as a
        fallback for a tracker that reports position only.
        """
        steps = emg.size(1)
        device = emg.device
        valid_time = position_mask.all(dim=-1).to(position.dtype)
        dt = 1.0 / self.sample_rate_hz

        if velocity is None:
            velocity = causal_difference(position, dt, valid_time)
            velocity_valid = valid_time
        else:
            velocity_valid = (
                velocity_mask.all(dim=-1).to(position.dtype)
                if velocity_mask is not None
                else valid_time
            )
        acceleration = causal_difference(velocity, dt, velocity_valid)

        indices = torch.arange(0, steps, self.decimation, device=device, dtype=torch.long)
        amplitude = emg * emg_mask.to(emg.dtype)
        features = torch.cat(
            [amplitude, position, velocity, acceleration], dim=-1
        ).index_select(1, indices)
        conditioning = self.context_project(context)
        features = torch.cat(
            [features, conditioning.unsqueeze(1).expand(-1, features.size(1), -1)],
            dim=-1,
        )

        encoded = self.normalise(self.encoder(features)[0])
        dynamics = self.dynamics(encoded)
        # Zero-initialised, so both start at the midpoint of their range and
        # EMG has to earn any per-timestep modulation of the attractor.
        eta = ETA_RANGE[0] + (ETA_RANGE[1] - ETA_RANGE[0]) * torch.sigmoid(
            dynamics[..., 0]
        )
        drag = DRAG_RANGE[0] + (DRAG_RANGE[1] - DRAG_RANGE[0]) * torch.sigmoid(
            dynamics[..., 1]
        )

        sampled_position = position.index_select(1, indices)
        sampled_velocity = velocity.index_select(1, indices)
        sampled_acceleration = acceleration.index_select(1, indices)
        # The attractor equation, now entirely from measurements except for
        # eta and rho.
        destination = sampled_position + (
            sampled_acceleration + drag.unsqueeze(-1) * sampled_velocity
        ) / eta.unsqueeze(-1)

        active = (indices.unsqueeze(0) < lengths.unsqueeze(1)).to(emg.dtype)
        active = active * valid_time.index_select(1, indices)
        weight = torch.sigmoid(self.confidence(encoded)).squeeze(-1) * active
        normalised = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        mean = (normalised.unsqueeze(-1) * destination).sum(dim=1)
        deviation = destination - mean.unsqueeze(1)
        sigma = (
            (normalised.unsqueeze(-1) * deviation.square()).sum(dim=1) + 1e-6
        ).sqrt()

        outputs = {
            "vl_prediction": mean,
            "vl_sigma": sigma,
            "vl_destinations": destination,
            "vl_weights": normalised,
            "vl_valid": active,
            "vl_eta": eta,
            "vl_drag": drag,
        }
        if self.emg_only_warmup > 0:
            # Destination estimated using only the window before the tracker
            # has meaningfully moved. If this is accurate, EMG carries intent
            # ahead of visible motion - the electromechanical-delay claim,
            # stated as a number.
            warmup = (indices < self.emg_only_warmup).to(emg.dtype).unsqueeze(0)
            warmup_weight = weight * warmup
            warmup_normalised = warmup_weight / warmup_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            outputs["vl_prediction_premotion"] = (
                warmup_normalised.unsqueeze(-1) * destination
            ).sum(dim=1)
        return outputs
