"""Privileged trajectory teacher and wearable-only latent student.

This module is intentionally isolated from ``vae_discriminator.py`` and the
proven deterministic pointing models.  It adapts the useful part of MUSIC's
latent-distillation recipe to an inference problem rather than copying its
muscle-control direction:

* a training-only teacher compresses the true future VIVE trajectory;
* an EMG+IMU student predicts the same latent from causal wearable history;
* one shared decoder predicts both the screen destination and a dense future
  relative trajectory from either latent;
* an EMG-only path and IMU-only motion head give each modality an explicit,
  testable job.

Only ``student_forward(emg, imu, time_mask)`` is needed at deployment.  The
teacher accepts privileged kinematics, but those values cannot enter the
student because the two encoders have disjoint parameters and APIs.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .grid_point import SpatialPointHead, decode_grid_outputs, finalize_point_prediction
from .grid_reach import ModalityEncoder


def _initialise_gaussian_heads(mu: nn.Linear, log_variance: nn.Linear) -> None:
    nn.init.normal_(mu.weight, std=0.01)
    nn.init.zeros_(mu.bias)
    nn.init.zeros_(log_variance.weight)
    # sigma ~= 0.1: sampled latents start near their means rather than
    # overwhelming a new decoder with unit Gaussian noise.
    nn.init.constant_(log_variance.bias, -4.6)


def reparameterize(
    mu: torch.Tensor,
    log_variance: torch.Tensor,
    sample: bool,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    if not sample or noise_scale <= 0.0:
        return mu
    standard_deviation = torch.exp(0.5 * log_variance)
    return mu + float(noise_scale) * standard_deviation * torch.randn_like(mu)


def standard_normal_kl(mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Mean KL per latent dimension, keeping its scale independent of width."""
    return (-0.5 * (1.0 + log_variance - mu.square() - log_variance.exp())).mean()


def diagonal_gaussian_kl(
    student_mu: torch.Tensor,
    student_log_variance: torch.Tensor,
    teacher_mu: torch.Tensor,
    teacher_log_variance: torch.Tensor,
    teacher_sigma_floor: float = 0.05,
) -> torch.Tensor:
    """KL(q_student || stop-gradient q_teacher), averaged over all dimensions."""
    teacher_mu = teacher_mu.detach()
    teacher_variance = teacher_log_variance.detach().exp().clamp_min(
        float(teacher_sigma_floor) ** 2
    )
    student_variance = student_log_variance.exp()
    return 0.5 * (
        teacher_variance.log()
        - student_log_variance
        + (student_variance + (student_mu - teacher_mu).square()) / teacher_variance
        - 1.0
    ).mean()


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and sign reversal in backpropagation."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(values, float(scale))


class FactorGuidanceHeads(nn.Module):
    """Paper-inspired factor guidance for target, motion, and nuisance.

    The target predictor makes the first latent block explicitly useful for
    pointing. A target discriminator reads all non-intent dimensions through
    gradient reversal, preventing destination information from spreading
    throughout the latent. Session prediction is assigned to the residual
    block and adversarially removed from the intent block. The middle block is
    directly supervised to represent the future motion chunk.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["factor_latent"]
        latent_dim = int(model["latent_dim"])
        self.intent_dim = int(settings["intent_dim"])
        self.motion_dim = int(settings["motion_dim"])
        self.residual_dim = latent_dim - self.intent_dim - self.motion_dim
        if min(self.intent_dim, self.motion_dim, self.residual_dim) <= 0:
            raise ValueError(
                "factor_latent intent_dim and motion_dim must leave a positive "
                "residual subspace"
            )
        self.grl_scale = float(settings.get("gradient_reversal_scale", 1.0))
        hidden = int(settings.get("head_width", 64))
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        target_classes = grid_width * grid_height
        session_classes = int(config.get("virtual_leader", {}).get("session_count", 0))
        trajectory_steps = int(model["teacher_trajectory_steps"])
        trajectory_limit = float(model.get("trajectory_limit_m", 0.8))
        self.trajectory_steps = trajectory_steps
        self.trajectory_limit = trajectory_limit

        def classifier(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
                nn.Linear(hidden, output_dim),
            )

        self.target_from_intent = classifier(self.intent_dim, target_classes)
        self.target_from_other = classifier(
            self.motion_dim + self.residual_dim, target_classes
        )
        self.motion_from_motion = nn.Sequential(
            nn.LayerNorm(self.motion_dim),
            nn.Linear(self.motion_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, trajectory_steps * 3),
        )
        self.session_from_residual = (
            classifier(self.residual_dim, session_classes)
            if session_classes > 1 else None
        )
        self.session_from_intent = (
            classifier(self.intent_dim, session_classes)
            if session_classes > 1 else None
        )

    def forward(self, latent_mean: torch.Tensor) -> dict[str, torch.Tensor]:
        intent = latent_mean[:, : self.intent_dim]
        motion_end = self.intent_dim + self.motion_dim
        motion = latent_mean[:, self.intent_dim : motion_end]
        residual = latent_mean[:, motion_end:]
        other = latent_mean[:, self.intent_dim :]
        outputs = {
            "target_intent_logits": self.target_from_intent(intent),
            "target_other_logits": self.target_from_other(
                gradient_reverse(other, self.grl_scale)
            ),
            "motion_trajectory": self.trajectory_limit * torch.tanh(
                self.motion_from_motion(motion).reshape(
                    -1, self.trajectory_steps, 3
                )
            ),
        }
        if self.session_from_residual is not None:
            outputs["session_residual_logits"] = self.session_from_residual(residual)
            outputs["session_intent_logits"] = self.session_from_intent(
                gradient_reverse(intent, self.grl_scale)
            )
        return outputs


class PrivilegedTrajectoryEncoder(nn.Module):
    """Bidirectional encoder for the complete training-only future trajectory."""

    def __init__(self, input_dim: int, width: int, layers: int, latent_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()
        )
        self.encoder = nn.GRU(
            width,
            width,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if layers == 1 else 0.1,
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(4 * width), nn.Linear(4 * width, 2 * width), nn.GELU()
        )
        self.to_mu = nn.Linear(2 * width, latent_dim)
        self.to_log_variance = nn.Linear(2 * width, latent_dim)
        _initialise_gaussian_heads(self.to_mu, self.to_log_variance)

    def forward(self, trajectory_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.encoder(self.input_projection(trajectory_features))
        summary = torch.cat([hidden[:, -1], hidden.mean(dim=1)], dim=-1)
        summary = self.summary(summary)
        return self.to_mu(summary), self.to_log_variance(summary).clamp(-8.0, 2.0)


class WearableStudentEncoder(nn.Module):
    """Balanced EMG/IMU branches with EMG-only and IMU-motion auxiliaries."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int,
        trajectory_steps: int,
    ) -> None:
        super().__init__()
        model = config["model"]
        width = int(model["d_model"])
        latent_dim = int(model["latent_dim"])
        common = dict(
            d_model=width,
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            ffn_dim=int(model["ffn_dim"]),
            dropout=float(model["dropout"]),
            patch_length=int(model["patch_length"]),
            patch_stride=int(model["patch_stride"]),
            kernel_sizes=list(model["tcn_kernel_sizes"]),
        )
        self.emg_encoder = ModalityEncoder(emg_channels, **common)
        self.imu_encoder = ModalityEncoder(imu_channels, **common)
        self.imu_dropout_probability = float(model.get("imu_modality_dropout", 0.0))

        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, 2 * width), nn.GELU(),
            nn.Dropout(float(model["dropout"])), nn.Linear(2 * width, width),
            nn.LayerNorm(width), nn.GELU(),
        )
        self.emg_adapter = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.to_mu = nn.Linear(width, latent_dim)
        self.to_log_variance = nn.Linear(width, latent_dim)
        self.emg_to_mu = nn.Linear(width, latent_dim)
        self.emg_to_log_variance = nn.Linear(width, latent_dim)
        _initialise_gaussian_heads(self.to_mu, self.to_log_variance)
        _initialise_gaussian_heads(self.emg_to_mu, self.emg_to_log_variance)

        # This head makes the answer to "is IMU learning motion?" measurable:
        # it is directly supervised against the relative VIVE trajectory in
        # training, while the endpoint still has to be solved by fusion.
        self.imu_motion_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, trajectory_steps * 3),
        )
        self.trajectory_steps = trajectory_steps

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        apply_imu_dropout: bool = False,
    ) -> dict[str, torch.Tensor]:
        emg_context = self.emg_encoder(emg, time_mask)
        imu_context = self.imu_encoder(imu, time_mask)
        fused_imu = imu_context
        if apply_imu_dropout and self.training and self.imu_dropout_probability > 0.0:
            keep = (
                torch.rand(imu_context.size(0), 1, device=imu_context.device)
                >= self.imu_dropout_probability
            ).to(imu_context.dtype)
            fused_imu = fused_imu * keep

        context = self.fusion(torch.cat([emg_context, fused_imu], dim=-1))
        emg_only_context = self.emg_adapter(emg_context)
        return {
            "mu": self.to_mu(context),
            "log_variance": self.to_log_variance(context).clamp(-8.0, 2.0),
            "emg_mu": self.emg_to_mu(emg_only_context),
            "emg_log_variance": self.emg_to_log_variance(emg_only_context).clamp(-8.0, 2.0),
            "imu_trajectory": self.imu_motion_head(imu_context).reshape(
                -1, self.trajectory_steps, 3
            ),
        }


class SharedIntentDecoder(nn.Module):
    """Decode one slow latent into a destination and a multi-step motion chunk."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        latent_dim = int(model["latent_dim"])
        hidden = int(model.get("decoder_width", 128))
        dropout = float(model["dropout"])
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.trajectory_steps = int(model["teacher_trajectory_steps"])
        self.trajectory_limit_m = float(model.get("trajectory_limit_m", 0.8))
        self.trunk = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.point_head = SpatialPointHead(
            hidden, grid_width, grid_height, dropout,
            direct_prediction=True, zero_initialize=False,
        )
        self.trajectory_head = nn.Linear(hidden, self.trajectory_steps * 3)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(latent)
        raw = self.point_head(hidden)
        decoded = decode_grid_outputs(
            raw["heatmap_logits"], raw["offset_logits"],
            self.grid_width, self.grid_height,
        )
        outputs = {**raw, **decoded}
        finalize_point_prediction(outputs)
        outputs["trajectory"] = self.trajectory_limit_m * torch.tanh(
            self.trajectory_head(hidden).reshape(-1, self.trajectory_steps, 3)
        )
        return outputs

class WearableLatentDistillationModel(nn.Module):
    """Container for the training-only VIVE teacher and deployable student."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int,
    ) -> None:
        super().__init__()
        model = config["model"]
        width = int(model["d_model"])
        latent_dim = int(model["latent_dim"])
        steps = int(model["teacher_trajectory_steps"])
        self.teacher = PrivilegedTrajectoryEncoder(
            input_dim=6,
            width=width,
            layers=int(model.get("teacher_layers", 2)),
            latent_dim=latent_dim,
        )
        self.student = WearableStudentEncoder(
            config, emg_channels, imu_channels, trajectory_steps=steps
        )
        self.decoder = SharedIntentDecoder(config)
        factor_settings = model.get("factor_latent", {})
        self.guidance = (
            FactorGuidanceHeads(config)
            if bool(factor_settings.get("enabled", False)) else None
        )

    def teacher_forward(
        self,
        trajectory_features: torch.Tensor,
        sample: bool = False,
        noise_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mu, log_variance = self.teacher(trajectory_features)
        latent = reparameterize(mu, log_variance, sample, noise_scale)
        outputs = {
            **self.decoder(latent),
            "latent": latent,
            "mu": mu,
            "log_variance": log_variance,
        }
        if self.guidance is not None:
            outputs["guidance"] = self.guidance(mu)
        return outputs

    def student_forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        sample: bool = False,
        noise_scale: float = 1.0,
        include_emg_only: bool = False,
        apply_imu_dropout: bool = False,
    ) -> dict[str, torch.Tensor]:
        encoded = self.student(
            emg, imu, time_mask, apply_imu_dropout=apply_imu_dropout
        )
        latent = reparameterize(
            encoded["mu"], encoded["log_variance"], sample, noise_scale
        )
        outputs = {
            **self.decoder(latent),
            "latent": latent,
            "mu": encoded["mu"],
            "log_variance": encoded["log_variance"],
            "imu_trajectory": encoded["imu_trajectory"],
        }
        if self.guidance is not None:
            outputs["guidance"] = self.guidance(encoded["mu"])
        if include_emg_only:
            emg_latent = reparameterize(
                encoded["emg_mu"], encoded["emg_log_variance"],
                sample, noise_scale,
            )
            outputs["emg_only"] = {
                **self.decoder(emg_latent),
                "latent": emg_latent,
                "mu": encoded["emg_mu"],
                "log_variance": encoded["emg_log_variance"],
            }
            if self.guidance is not None:
                outputs["emg_only"]["guidance"] = self.guidance(
                    encoded["emg_mu"]
                )
        return outputs
