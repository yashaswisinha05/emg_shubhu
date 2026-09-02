"""Goal-conditioned trajectory forecasting: destination latent, attractor decoder.

For a dataset with a measured end-effector track and the full trajectory as
the target, the virtual-leader model stops being a readout and becomes the
generative decoder. Sample a destination z from the posterior, integrate

    xddot = eta (z - x) - rho xdot

forward from the measured current state, and the result is a predicted future
path. That is the attractor's stochastic differential equation used the way
it was written, rather than the algebraically-inverted form the untracked
dataset required.

The pieces:

  latent z      the reach destination, in tracker coordinates
  posterior     q(z | history) from the fused EMG/IMU/tracker encoder
  prior         p(z | instantaneous kinematics), the attractor solved for the
                goal at the current timestep - the same data-derived prior
                that replaced N(0, I) in the screen-coordinate model
  decoder       forward integration of the attractor, producing a trajectory
  loss          trajectory reconstruction against the measured track, plus KL

Two things carry over from this project's physics work rather than being
rediscovered. The integration is semi-implicit Euler with sub-stepping, because
plain Euler at the recorded interval was measured to be unconditionally
unstable on a comparable rollout here. And eta is bounded away from zero, since
the destination readout divides by it.

Expected batch contents (the loader's contract):

    emg           (B, T, C)   EMG amplitude, any channel count
    emg_mask      (B, T, C)
    imu           (B, T, F)   per-sensor IMU features, any sensor count
    imu_mask      (B, T, F)
    position      (B, T, D)   measured end-effector track, D = 2 or 3
    position_mask (B, T, D)
    lengths       (B,)        valid samples per trial
    context       (B, d)      pooled encoder representation

Position must already be on the EMG/IMU clock; this module does not resample.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

ETA_RANGE = (0.5, 20.0)
DRAG_RANGE = (0.0, 5.0)


class VirtualLeaderTrajectoryVAE(nn.Module):
    """Encode history -> destination posterior -> integrate -> future path."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        settings = config.get("virtual_leader", {})
        data = config["data"]
        self.sample_rate_hz = float(data["sample_rate_hz"])
        self.decimation = int(settings.get("decimation", 4))
        # Sub-stepping inside each decimated interval. Same reason as the arm
        # rollout in physics/rollout3.py: a single Euler step at the recorded
        # interval was measured unstable there, and this integrates the same
        # class of second-order system.
        self.substeps = int(settings.get("substeps", 4))
        self.horizon = int(settings.get("horizon", 32))
        self.position_dim = int(settings.get("position_dim", 3))
        self.prior_displacement_limit = float(
            settings.get("prior_displacement_limit_m", 0.8)
        )
        context_dim = int(settings.get("context_dim", 2 * int(config["model"]["d_model"])))

        self.posterior_mean = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(),
            nn.Linear(128, self.position_dim),
        )
        self.posterior_log_variance = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(),
            nn.Linear(128, self.position_dim),
        )
        # Posterior mean is a residual on the current measured position, so an
        # untrained model predicts approximately "the hand stays put" rather
        # than an arbitrary point in space - a sane, physically meaningful
        # start. The BIAS carries that; the weight is small but deliberately
        # NOT zero.
        #
        # Zeroing the final weight of every head instead - the obvious way to
        # get that same starting behaviour - silently severs the encoder.
        # d(head output)/d(context) is W^T, so a zero W means the context, and
        # therefore the whole EMG/IMU encoder feeding it, receives exactly no
        # gradient. Measured before this fix: encoder GRU gradient norm 0.0,
        # and validation error frozen to four decimals across six epochs while
        # the training loss moved. This is the same trap as the kinematics
        # head in virtual_leader.py, and it is worth the second explicit note.
        nn.init.normal_(self.posterior_mean[-1].weight, std=0.01)
        nn.init.zeros_(self.posterior_mean[-1].bias)
        nn.init.normal_(self.posterior_log_variance[-1].weight, std=0.01)
        # sigma ~0.05 m at init: small against a reach, large enough that the
        # sampling noise is not degenerate.
        nn.init.constant_(self.posterior_log_variance[-1].bias, -6.0)

        self.dynamics = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 64), nn.GELU(),
            nn.Linear(64, 2),
        )
        # Same reasoning: small random rather than zero, so eta and rho are
        # a real path from the context rather than a dead one. Bias stays at
        # zero, putting both at their range midpoint to begin with.
        nn.init.normal_(self.dynamics[-1].weight, std=0.01)
        nn.init.zeros_(self.dynamics[-1].bias)

    def attractor_parameters(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.dynamics(context)
        eta = ETA_RANGE[0] + (ETA_RANGE[1] - ETA_RANGE[0]) * torch.sigmoid(raw[..., 0])
        drag = DRAG_RANGE[0] + (DRAG_RANGE[1] - DRAG_RANGE[0]) * torch.sigmoid(raw[..., 1])
        return eta, drag

    def rollout(
        self,
        destination: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        eta: torch.Tensor,
        drag: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Integrate the attractor forward. Returns (B, horizon, D)."""
        step = self.decimation / self.sample_rate_hz
        dt = step / self.substeps
        eta = eta.unsqueeze(-1)
        drag = drag.unsqueeze(-1)
        trajectory = []
        for _ in range(horizon):
            for _ in range(self.substeps):
                acceleration = eta * (destination - position) - drag * velocity
                # Semi-implicit: velocity updated first, then position with
                # the new velocity. Symplectic, and stable where explicit
                # Euler drifts.
                velocity = velocity + dt * acceleration
                position = position + dt * velocity
            trajectory.append(position)
        return torch.stack(trajectory, dim=1)

    def forward(
        self,
        context: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """position/velocity/acceleration: measured state at the cutoff, (B, D)."""
        horizon = int(horizon or self.horizon)
        eta, drag = self.attractor_parameters(context)

        mean = position + self.posterior_mean(context)
        log_variance = self.posterior_log_variance(context).clamp(-10.0, 2.0)
        if self.training:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mean + standard_deviation * torch.randn_like(standard_deviation)
        else:
            latent = mean

        predicted = self.rollout(latent, position, velocity, eta, drag, horizon)

        # The same attractor, solved for the goal instead of integrated, gives
        # a prior from the instantaneous measured kinematics. Detached so the
        # KL can only move the posterior toward the dynamics, never widen the
        # prior to escape its own penalty.
        #
        # The displacement is bounded, and that bound is load-bearing rather
        # than defensive. eta is learned and shared with the rollout, so
        # reconstruction is free to shrink it; dividing by a shrinking eta
        # sends the prior mean off to infinity, and since the KL penalises
        # (mu - prior_mu)^2 / sigma_prior^2 the loss follows it. Measured on
        # the real dataset without this: KL ran 25 -> 316 -> 752 -> ~1260 and
        # saturated, at which point it was 99.7% of the objective and
        # validation error had gone from beating the linear baseline at epoch
        # one to losing to it for the rest of the run. The limit is a human
        # arm's reach - a destination further away than that is not a
        # hypothesis worth representing.
        displacement = (
            acceleration + drag.unsqueeze(-1) * velocity
        ) / eta.unsqueeze(-1)
        limit = float(self.prior_displacement_limit)
        magnitude = displacement.norm(dim=-1, keepdim=True)
        displacement = displacement * (limit / magnitude.clamp_min(limit))
        prior_mean = position + displacement
        return {
            "trajectory": predicted,
            "destination_mu": mean,
            "destination_log_variance": log_variance,
            "destination_sigma": torch.exp(0.5 * log_variance),
            "prior_mu": prior_mean.detach(),
            "eta": eta,
            "drag": drag,
        }


def trajectory_loss(
    outputs: dict[str, torch.Tensor],
    future_position: torch.Tensor,
    future_mask: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Reconstruction against the measured future track, plus KL.

    future_position: (B, horizon, D) measured track after the cutoff.
    future_mask:     (B, horizon)    which of those steps exist.
    """
    settings = config.get("loss", {})
    epsilon = float(settings.get("trajectory_epsilon_m", 0.002))
    predicted = outputs["trajectory"]
    steps = min(predicted.size(1), future_position.size(1))
    predicted = predicted[:, :steps]
    target = future_position[:, :steps]
    mask = future_mask[:, :steps].to(predicted.dtype)

    # Charbonnier rather than plain L2: a tracker dropout or a single
    # mislabelled frame should not dominate the whole trial's gradient.
    distance = torch.sqrt(
        (predicted - target).square().sum(dim=-1) + epsilon * epsilon
    )
    denominator = mask.sum(dim=1).clamp_min(1.0)
    reconstruction = ((distance * mask).sum(dim=1) / denominator).mean()

    prior_sigma = float(settings.get("trajectory_prior_sigma", 0.05))
    mu = outputs["destination_mu"]
    log_variance = outputs["destination_log_variance"]
    prior_log_variance = torch.full_like(log_variance, 2.0 * torch.log(
        torch.tensor(prior_sigma)
    ).item())
    kl = 0.5 * (
        prior_log_variance
        - log_variance
        + (log_variance.exp() + (mu - outputs["prior_mu"]).square())
        / prior_log_variance.exp()
        - 1.0
    ).sum(dim=-1).mean()

    total = reconstruction + float(settings.get("kl_weight", 0.01)) * kl
    return {"loss": total, "reconstruction": reconstruction, "kl": kl}
