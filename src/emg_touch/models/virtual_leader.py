"""Destination-as-latent-state branch, after the virtual-leader formulation.

Follows Liang, Ahmad & Godsill, "Joint Object Tracking and Intent
Recognition" (IEEE T-AES 2026), which models a target's hidden goal as a
virtual leader with its own dynamics that pulls the kinematic state toward
it:

    d(xdot_t) = eta (r_t - x_t) dt - rho xdot_t dt + sigma_x dB_t
    d(r_t)    = sigma_r dB_t

with r the destination, eta the mean-reversion (attractor) strength and rho
a drag term. The destination is not a regression output there - it is a
latent state inferred jointly with the kinematics, and it *causes* the
observed motion.

Why that structure is worth borrowing here. Every coordinate model in this
project is supervised by exactly one signal per trial: the final touch
location. That is a very thin gradient for anything with temporal structure,
and it is the common thread behind the physics branches failing to earn any
blend weight. Rearranged, the attractor equation reads

    r  =  x + (xddot + rho xdot) / eta

i.e. the destination is recoverable from *local* kinematics at every
instant. Estimating it per timestep turns one supervision signal per trial
into one per decimated step, on exactly the quantity being predicted.

The other reason to prefer this over the physics branches already tried:
those built a long generative chain (EMG -> activation -> torque ->
integrate -> forward kinematics -> learned affine) with an endpoint loss at
the far end, and the middle of that chain was unidentifiable. Here the
accelerometers measure xddot directly, so the link from data to latent is
one algebraic step.

Deliberately kept causal (unidirectional GRU): the whole point is early
prediction, and the evaluation protocol scores truncated prefixes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..data.grid_trajectory import grid_imu_acceleration_indices

# The per-timestep destination readout divides by eta, so a small eta sends
# the estimate to infinity. Bounded to a range wide enough to cover slow and
# snappy reaches without letting the division blow up.
ETA_RANGE = (0.5, 20.0)
DRAG_RANGE = (0.0, 5.0)
# Destination estimates live in normalised screen units. Reaches can aim
# slightly outside the panel, so this is deliberately wider than [0, 1] -
# tight enough to stop a diverging estimate poisoning the batch mean.
DESTINATION_LIMIT = (-0.5, 1.5)


def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


class VirtualLeaderBranch(nn.Module):
    """Per-timestep kinematic state -> per-timestep destination -> posterior."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        settings = config.get("virtual_leader", {})
        self.decimation = int(settings.get("decimation", 4))
        hidden = int(settings.get("hidden", 96))

        acceleration = grid_imu_acceleration_indices(data)
        self.register_buffer(
            "acceleration_indices",
            torch.tensor(acceleration, dtype=torch.long),
            persistent=False,
        )
        # Conditioned on the fusion encoder's pooled representation, not just
        # the raw channels. Measured the hard way: a first version read only
        # EMG + accelerometers through this GRU and *replaced* the fusion
        # coordinate, which discarded a pretrained IMU backbone and
        # transformer and asked a 96-unit GRU to relearn the task from 719
        # trials - it reached 253 px against fusion's 184 px and was still
        # descending at the epoch limit. The attractor readout was working
        # (its per-step loss fell steadily); the branch was simply starved of
        # everything the rest of the model already knows.
        context_dim = 2 * int(model["d_model"])
        self.context_project = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 32), nn.GELU()
        )
        input_dim = 4 + len(acceleration) + 32
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.normalise = nn.LayerNorm(hidden)
        # Position, velocity and acceleration of the hand in normalised screen
        # units. None of these are directly supervised - there is no
        # ground-truth hand trajectory in this dataset - they are the latent
        # kinematic state the attractor equation is written in terms of, which
        # is exactly the "joint tracking and intent recognition" split.
        self.kinematics = nn.Linear(hidden, 6)
        # Per-timestep reliability of that step's destination estimate. Early
        # in a reach, before the arm has committed, the readout is genuinely
        # uninformative and should be allowed to say so.
        self.confidence = nn.Linear(hidden, 1)
        self.raw_eta = nn.Parameter(torch.zeros(()))
        self.raw_drag = nn.Parameter(torch.zeros(()))
        # Bias starts the position near screen centre so the first
        # destination estimates are sane. The weight is small but NOT zero,
        # which matters more than it looks: with the usual zero-init this
        # project uses for new heads, velocity and acceleration are exactly 0
        # at step one, so d(destination)/d(eta) = -(a + rho v)/eta^2 is
        # exactly 0 and the attractor strength gets no gradient at all -
        # measured, eta sat at its initial 10.25 for every step of a real run.
        # A zero weight also severs the GRU's only path to the loss (measured
        # gradient norm 7e-7, i.e. dead). Small random weights give both a
        # gradient from the first step.
        nn.init.normal_(self.kinematics.weight, std=0.01)
        with torch.no_grad():
            self.kinematics.bias.copy_(
                torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
            )

    @property
    def eta(self) -> torch.Tensor:
        return _bounded(self.raw_eta, *ETA_RANGE)

    @property
    def drag(self) -> torch.Tensor:
        return _bounded(self.raw_drag, *DRAG_RANGE)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        steps = emg.size(1)
        indices = torch.arange(
            0, steps, self.decimation, device=emg.device, dtype=torch.long
        )
        amplitude = emg[:, :, :4] * emg_mask[:, :, :4].to(emg.dtype)
        acceleration = imu.index_select(-1, self.acceleration_indices)
        acceleration = acceleration * imu_mask.index_select(
            -1, self.acceleration_indices
        ).to(imu.dtype)
        features = torch.cat([amplitude, acceleration], dim=-1)
        features = features.index_select(1, indices)
        conditioning = self.context_project(context)
        features = torch.cat(
            [features, conditioning.unsqueeze(1).expand(-1, features.size(1), -1)],
            dim=-1,
        )

        encoded, _ = self.encoder(features)
        encoded = self.normalise(encoded)
        kinematics = self.kinematics(encoded)
        position = kinematics[..., 0:2]
        velocity = kinematics[..., 2:4]
        acceleration_state = kinematics[..., 4:6]

        # The attractor equation solved for the destination. This is the whole
        # point of the branch: one estimate of where the reach is going per
        # decimated step, from that step's kinematics alone.
        destination = position + (
            acceleration_state + self.drag * velocity
        ) / self.eta
        destination = destination.clamp(*DESTINATION_LIMIT)

        valid = (indices.unsqueeze(0) < lengths.unsqueeze(1)).to(emg.dtype)
        weight = torch.sigmoid(self.confidence(encoded)).squeeze(-1) * valid
        total = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        normalised = weight / total

        mean = (normalised.unsqueeze(-1) * destination).sum(dim=1)
        # Spread of the per-step estimates around their own mean, rather than
        # a separately learned head: when successive steps agree on where the
        # reach is going the destination is genuinely well determined, and
        # when they disagree it is not. This is what should shrink as a trial
        # accumulates evidence.
        deviation = destination - mean.unsqueeze(1)
        variance = (normalised.unsqueeze(-1) * deviation.square()).sum(dim=1)
        sigma = (variance + 1e-6).sqrt()

        return {
            "vl_prediction": mean,
            "vl_sigma": sigma,
            "vl_destinations": destination,
            "vl_weights": normalised,
            "vl_valid": valid,
            "vl_position": position,
            "vl_velocity": velocity,
            "vl_acceleration": acceleration_state,
            "vl_eta": self.eta.detach(),
            "vl_drag": self.drag.detach(),
        }
