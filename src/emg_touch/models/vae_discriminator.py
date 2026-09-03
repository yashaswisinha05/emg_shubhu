"""VAE + IMU-only critic, kept entirely separate from the proven pipeline.

Explicit request: a VAE, and a discriminator network to disambiguate the
latent and make EMG matter. Built as its own file and its own model class -
NOTHING here is imported by grid_reach.py or any proven training script, so
nothing about the 176 px GridReachModel / UncertaintyHead pipeline changes
by this file existing.

THE DISCRIMINATOR DIRECTION, worked through explicitly because the naive
version does the opposite of what was asked. This project's own
session_discriminator (removed along with PointingIntentVAE) used gradient
reversal on "which session produced this latent" - correctly, because
session-INVARIANCE is what generalisation needs there: GRL pushes the
encoder to make the latent look the same regardless of session.

Copying that pattern onto "which modality produced this latent" would train
the encoder to make EMG's presence in z UNDETECTABLE - i.e. to make z look
the same whether EMG was there or not. That is EMG-invariance, which
suppresses exactly the information being asked for. Same mechanism,
opposite goal, and it would fail silently: training would converge, the
discriminator would look "fooled", and the actual effect would be an
encoder actively discarding EMG.

What is built instead: an IMU-only CRITIC (its own SpatialPointHead, own
parameters, reading imu_context.detach() so its own training never touches
the shared encoder) measures the ceiling of what IMU alone can solve - a
linear-probe-style methodology, not adversarial. A one-directional hinge
loss then requires the FULL (EMG+IMU) prediction to beat that ceiling by a
margin, with gradient flowing only through the full path. This directly
optimises the exact quantity this project has measured at or below noise
three separate times (ablation sweep, architecture swap, anticipatory-gain
reading) - EMG's marginal contribution - rather than an indirect proxy for
it, and it has no two-player minimax dynamics to destabilise: the critic's
own objective (be the best it can from IMU alone) and the full path's
objective (beat the critic) do not fight each other's parameters, because
the critic never touches the shared encoder and the encoder is never
pushed to make the critic WORSE, only to make the full path better than it.

THE VAE PART - sampling only at train time. Reparameterises during
training; forward() in eval mode uses mu_z directly, no sampling. This
project's earlier VAE (PointingIntentVAE) sampled unconditionally and
measured sigma drifting from a 0.135 init to 0.5-0.9 over training - every
epoch's eval score was therefore partly noise, which is a second, avoidable
source of instability on top of KL's own dynamics. Removing sampling from
eval does not fix KL's tendency to push sigma toward 1; it only stops that
drift from also corrupting the numbers used to judge whether the run is any
good, which is worth having independent of whether KL itself needs
retuning.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .grid_point import SpatialPointHead, decode_grid_outputs, finalize_point_prediction
from .grid_reach import ModalityEncoder


class EMGImportanceVAE(nn.Module):
    """EMG+IMU -> sampled latent -> screen point, with an IMU-only critic."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int,
    ) -> None:
        super().__init__()
        model_config = config["model"]
        width = int(model_config["d_model"])
        common = dict(
            d_model=width,
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            dropout=float(model_config["dropout"]),
            patch_length=int(model_config["patch_length"]),
            patch_stride=int(model_config["patch_stride"]),
            kernel_sizes=list(model_config["tcn_kernel_sizes"]),
        )
        self.emg_encoder = ModalityEncoder(emg_channels, **common)
        self.imu_encoder = ModalityEncoder(imu_channels, **common)
        context_dim = width * 2

        self.latent_dim = int(model_config.get("vae_latent_dim", 64))
        self.to_latent = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(),
        )
        self.to_mu = nn.Linear(128, self.latent_dim)
        self.to_log_var = nn.Linear(128, self.latent_dim)
        # Small, not zero: a zero-init final layer on BOTH mu and log_var
        # would make every early z identical and every early KL gradient
        # zero at once, giving the optimiser nothing to differentiate
        # between examples with. log_var biased toward a small starting
        # variance (~0.01) so early samples stay close to mu rather than
        # swamping the decoder with noise before either head has learned
        # anything - the specific failure this project has already measured
        # once (sigma drifting up over training, not starting there, made
        # it worse, not better).
        nn.init.normal_(self.to_mu.weight, std=0.01)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_log_var.weight)
        nn.init.constant_(self.to_log_var.bias, -4.6)  # log(0.01)

        grid_width, grid_height = map(int, model_config.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.point_head = SpatialPointHead(
            self.latent_dim, grid_width, grid_height, float(model_config["dropout"]),
            direct_prediction=True, zero_initialize=False,
        )
        # Separate parameters from point_head; reads IMU alone, never EMG.
        self.critic_head = SpatialPointHead(
            width, grid_width, grid_height, float(model_config["dropout"]),
            direct_prediction=True, zero_initialize=False,
        )

    def encode(self, emg: torch.Tensor, imu: torch.Tensor, time_mask: torch.Tensor):
        emg_context = self.emg_encoder(emg, time_mask)
        imu_context = self.imu_encoder(imu, time_mask)
        context = torch.cat([emg_context, imu_context], dim=-1)
        hidden = self.to_latent(context)
        return self.to_mu(hidden), self.to_log_var(hidden), imu_context

    def forward(
        self, emg: torch.Tensor, imu: torch.Tensor, time_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mu_z, log_var_z, imu_context = self.encode(emg, imu, time_mask)
        if self.training:
            std = torch.exp(0.5 * log_var_z)
            z = mu_z + std * torch.randn_like(std)
        else:
            # Deterministic at eval - see module docstring. Sampling here
            # would make every reported score partly a function of which
            # random draw happened to land, on top of whatever the model
            # actually learned.
            z = mu_z

        raw = self.point_head(z)
        decoded = decode_grid_outputs(
            raw["heatmap_logits"], raw["offset_logits"], self.grid_width, self.grid_height
        )
        outputs = {**raw, **decoded}
        finalize_point_prediction(outputs)
        outputs["mu_z"] = mu_z
        outputs["log_var_z"] = log_var_z

        # imu_context.detach(): the critic's OWN loss must never reach the
        # shared imu_encoder - see module docstring for why that keeps this
        # a stable probe-and-beat-it setup rather than an adversarial one.
        critic_raw = self.critic_head(imu_context.detach())
        critic_decoded = decode_grid_outputs(
            critic_raw["heatmap_logits"], critic_raw["offset_logits"],
            self.grid_width, self.grid_height,
        )
        critic_outputs = {**critic_raw, **critic_decoded}
        finalize_point_prediction(critic_outputs)
        # Kept as a SELF-CONTAINED sub-dict (candidates/probabilities/offsets
        # included, not just prediction) so grid_point_loss can be called on
        # it directly and score the critic against its OWN decode - a first
        # version that copied only a few renamed fields across silently fed
        # the critic's loss the main head's candidates/probabilities instead
        # of its own, which would have scored the wrong thing without ever
        # raising an error.
        outputs["critic"] = critic_outputs
        outputs["critic_prediction"] = critic_outputs["prediction"]
        return outputs


def vae_kl_loss(mu_z: torch.Tensor, log_var_z: torch.Tensor) -> torch.Tensor:
    """Standard closed-form KL(N(mu, var) || N(0, I)), mean over the batch."""
    per_dim = -0.5 * (1.0 + log_var_z - mu_z.pow(2) - log_var_z.exp())
    return per_dim.sum(-1).mean()


def emg_importance_margin_loss(
    main_prediction: torch.Tensor, critic_prediction: torch.Tensor,
    target: torch.Tensor, margin: float,
) -> torch.Tensor:
    """Hinge: penalised whenever main does not beat the critic by `margin`.

    Gradient flows only through main_prediction - critic_prediction is a
    reference point here, not something this term trains (its own point
    loss, computed separately by the caller, is what trains critic_head).
    Normalised-coordinate units throughout, matching every other loss term
    in grid_point_loss, so `margin` composes with the existing loss weights
    without a unit conversion.
    """
    main_error = (main_prediction - target).norm(dim=-1)
    critic_error = (critic_prediction.detach() - target).norm(dim=-1)
    return torch.relu(margin + main_error - critic_error).mean()
