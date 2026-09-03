"""Differentiable virtual-leader regularization for trajectory predictions.

The state-space model in Liang et al. relates instantaneous acceleration to
the displacement from a latent destination and to velocity-dependent drag:

    acceleration = eta * (destination - position) - rho * velocity + noise

This module turns that relation into training-only losses. It does not run a
Kalman or particle filter and does not add tracker measurements to inference.
The true VIVE trajectory is used only as a supervision target, just like the
existing dense trajectory loss.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def trajectory_kinematics(
    trajectory: torch.Tensor,
    lead_samples: torch.Tensor,
    sample_rate_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interval velocity, interior acceleration, and per-trial dt."""
    if trajectory.ndim != 3 or trajectory.size(-1) != 3:
        raise ValueError("trajectory must have shape [batch, steps, 3]")
    if trajectory.size(1) < 3:
        raise ValueError("at least three trajectory steps are required")
    intervals = trajectory.size(1) - 1
    duration = lead_samples.to(trajectory.dtype).clamp_min(1.0) / float(
        sample_rate_hz
    )
    dt = (duration / intervals).view(-1, 1, 1).clamp_min(1e-5)
    velocity = torch.diff(trajectory, dim=1) / dt
    acceleration = torch.diff(velocity, dim=1) / dt
    return velocity, acceleration, dt


def _scaled_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: float,
    beta: float,
    residual_clip: float,
) -> torch.Tensor:
    safe_scale = max(float(scale), 1e-6)
    clip = max(float(residual_clip), 1e-3)
    normalized_residual = (prediction - target) / safe_scale
    # At short lead times, finite-difference acceleration from an untrained
    # trajectory can be enormous. Smooth saturation keeps this auxiliary loss
    # bounded until the ordinary trajectory objective has made the prediction
    # plausible; unlike hard clipping it remains differentiable everywhere.
    bounded_residual = clip * torch.tanh(normalized_residual / clip)
    return F.smooth_l1_loss(
        bounded_residual,
        torch.zeros_like(bounded_residual),
        beta=float(beta),
    )


def virtual_leader_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lead_samples: torch.Tensor,
    sample_rate_hz: float,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute tracking and destination-driven dynamics losses.

    ``destination`` is the true final relative VIVE position during training.
    This anchors the dynamics prior without introducing a deployment input.
    """
    predicted_velocity, predicted_acceleration, _ = trajectory_kinematics(
        prediction, lead_samples, sample_rate_hz
    )
    target_velocity, target_acceleration, _ = trajectory_kinematics(
        target, lead_samples, sample_rate_hz
    )
    velocity_scale = float(settings.get("velocity_scale_mps", 1.0))
    acceleration_scale = float(settings.get("acceleration_scale_mps2", 10.0))
    beta = float(settings.get("huber_beta", 0.1))
    residual_clip = float(settings.get("normalized_residual_clip", 5.0))

    endpoint = torch.linalg.vector_norm(
        prediction[:, -1] - target[:, -1], dim=-1
    ).mean()
    velocity = _scaled_huber(
        predicted_velocity, target_velocity, velocity_scale, beta, residual_clip
    )
    acceleration = _scaled_huber(
        predicted_acceleration, target_acceleration, acceleration_scale, beta,
        residual_clip,
    )

    # Acceleration is centered at the interior positions. Averaging adjacent
    # interval velocities gives a velocity at the same approximate instant.
    position_mid = prediction[:, 1:-1]
    velocity_mid = 0.5 * (
        predicted_velocity[:, :-1] + predicted_velocity[:, 1:]
    )
    destination = target[:, -1:].detach()
    eta = float(settings.get("mean_reversion_per_s2", 25.0))
    rho = float(settings.get("drag_per_s", 10.0))
    expected_acceleration = (
        eta * (destination - position_mid) - rho * velocity_mid
    )
    dynamics = _scaled_huber(
        predicted_acceleration,
        expected_acceleration,
        acceleration_scale,
        beta,
        residual_clip,
    )
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "endpoint": endpoint,
        "dynamics": dynamics,
    }


def weighted_virtual_leader_loss(
    losses: dict[str, torch.Tensor],
    settings: dict[str, Any],
    prefix: str,
) -> torch.Tensor:
    weights = {
        "velocity": float(settings.get(f"{prefix}_velocity_weight", 0.0)),
        "acceleration": float(
            settings.get(f"{prefix}_acceleration_weight", 0.0)
        ),
        "endpoint": float(settings.get(f"{prefix}_endpoint_weight", 0.0)),
        "dynamics": float(settings.get(f"{prefix}_dynamics_weight", 0.0)),
    }
    total = next(iter(losses.values())).new_zeros(())
    for name, value in losses.items():
        total = total + weights[name] * value
    return total
