"""Let the network pick its own forecast horizon, per example.

Every trajectory number so far used a FIXED horizon, chosen by us (254 ms,
then 1000 ms). This asks the model to choose instead: given the current
EMG+IMU context, how far ahead does it trust its own forecast right now?

NOT a discriminator/GRL setup, on purpose - the same reasoning as this
project's EMG-importance critic (vae_discriminator.py). A gradient-reversal
discriminator needs an adversarial target (fool a classifier); "predict
further ahead" has no natural adversary, it is a direct trade-off between
two THINGS THE SAME NETWORK WANTS: more reach vs. more accuracy. Expressing
that as one differentiable loss - reward larger tau, penalised by whatever
accuracy it costs - has a stable, single-direction gradient. A discriminator
would need a fabricated adversarial framing for no benefit.

tau is continuous (it needs a gradient), but the trajectory decoder emits
a rollout at fixed integer sample steps. interpolate_at bridges the two
with a differentiable linear interpolation between the two nearest
timesteps, the 1-D analogue of bilinear sampling - the same reason
decode_grid_outputs's soft-argmax exists in this project's grid+offset head
(grid_point.py): a discrete structure made continuous by interpolating
between its neighbours, not by rounding.
"""
from __future__ import annotations

import torch
from torch import nn


class AdaptiveHorizonHead(nn.Module):
    """context -> tau (predicted confident horizon), bounded to a feasible range.

    Bounds matter concretely, not just numerically: scripts/diagnose_
    horizon_feasibility.py measured this dataset directly and found >=95%
    of trials support a horizon up to ~1000 ms but only 18.5% support
    1500 ms. tau_max_samples should be set from that number, not guessed -
    letting the network learn to WANT a horizon most trials cannot supply
    would only be teaching it to want something the data structurally
    cannot give it.
    """

    def __init__(
        self, context_dim: int, tau_min_samples: int, tau_max_samples: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if tau_max_samples <= tau_min_samples:
            raise ValueError("tau_max_samples must exceed tau_min_samples")
        self.tau_min = float(tau_min_samples)
        self.tau_max = float(tau_max_samples)
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero init -> sigmoid(0)=0.5 -> tau starts at the window's midpoint,
        # not pinned to either extreme before any signal has been learned.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.net(context).squeeze(-1)
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(raw)


def interpolate_at(
    sequence: torch.Tensor, tau: torch.Tensor, mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation of sequence[:, t, ...] at continuous index tau.

    sequence: (B, T, ...). tau: (B,), continuous, expected in [0, T-1].
    Returns (interpolated value (B, ...), validity (B,) - 1.0 only where
    BOTH neighbours used for interpolation are valid per `mask`, else 0.0).
    Gradient flows into tau through the (1-frac)/frac interpolation weights,
    which is the entire point: this is what lets tau receive a gradient from
    an error computed on a discrete-timestep sequence.
    """
    steps = sequence.size(1)
    tau = tau.clamp(0.0, float(steps - 1))
    floor_index = tau.floor().long().clamp(0, steps - 2)
    ceil_index = floor_index + 1
    fraction = (tau - floor_index.float()).clamp(0.0, 1.0)

    extra_dims = (1,) * (sequence.dim() - 2)
    floor_gather = floor_index.view(-1, 1, *extra_dims).expand(
        -1, 1, *sequence.shape[2:]
    )
    ceil_gather = ceil_index.view(-1, 1, *extra_dims).expand(
        -1, 1, *sequence.shape[2:]
    )
    floor_value = sequence.gather(1, floor_gather).squeeze(1)
    ceil_value = sequence.gather(1, ceil_gather).squeeze(1)
    weight_shape = (-1,) + (1,) * (sequence.dim() - 2)
    value = (
        floor_value * (1.0 - fraction).view(*weight_shape)
        + ceil_value * fraction.view(*weight_shape)
    )

    if mask is None:
        validity = torch.ones_like(fraction)
    else:
        floor_valid = mask.gather(1, floor_index.unsqueeze(1)).squeeze(1).float()
        ceil_valid = mask.gather(1, ceil_index.unsqueeze(1)).squeeze(1).float()
        validity = floor_valid * ceil_valid
    return value, validity


def adaptive_horizon_loss(
    trajectory: torch.Tensor, future: torch.Tensor, future_mask: torch.Tensor,
    tau: torch.Tensor, tau_max_samples: int, reach_weight: float,
) -> dict[str, torch.Tensor]:
    """error_at(tau) + reach_weight * (1 - tau/tau_max), masked to valid rows.

    Minimising this trades off directly: shrinking tau always helps the
    first term (closer-in forecasts are easier - measured throughout this
    project) and always hurts the second, so the optimum sits wherever the
    marginal accuracy cost of reaching further starts to exceed reach_weight
    - a real equilibrium, not a knob that saturates at one end unless badly
    mis-set. Reported tau statistics (see the training script) are how a
    degenerate choice of reach_weight would actually be caught: collapsed to
    tau_min or tau_max regardless of the input means the trade-off has
    stopped mattering, not that it was solved.
    """
    predicted_at_tau, validity = interpolate_at(trajectory, tau, mask=None)
    true_at_tau, target_validity = interpolate_at(future, tau, mask=future_mask)
    validity = validity * target_validity
    weight = validity / validity.sum().clamp_min(1.0)

    error_at_tau = (predicted_at_tau - true_at_tau).norm(dim=-1)
    accuracy_term = (error_at_tau * weight).sum()

    reach_penalty = (1.0 - tau / float(tau_max_samples))
    reach_term = (reach_penalty * weight).sum()

    total = accuracy_term + reach_weight * reach_term
    return {
        "loss": total,
        "error_at_tau": accuracy_term.detach(),
        "reach_penalty": reach_term.detach(),
        "tau": tau.detach(),
        "valid_fraction": validity.mean().detach(),
    }
