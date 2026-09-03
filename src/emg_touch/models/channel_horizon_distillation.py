"""Wearable latent distillation with interpretable EMG gating and time-to-go.

This module deliberately starts from the factor-guided (non virtual-leader)
model.  It adds two quantities that are available from wearables at inference:

* a gate over physical EMG sensors at every causal timestep;
* a named latent subspace that predicts the remaining time to touch.

VIVE is still confined to the teacher and supervision assembled by the
training script.  ``student_forward`` accepts only EMG, IMU, and a time mask.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .latent_distillation import (
    FactorGuidanceHeads,
    WearableLatentDistillationModel,
    WearableStudentEncoder,
    gradient_reverse,
    reparameterize,
)


def physical_sensor_feature_indices(
    emg_channels: int, sensor_count: int, sensor_index: int
) -> list[int]:
    """Indices for one sensor in the [window, kind, sensor] feature layout."""
    if sensor_count <= 0 or emg_channels % sensor_count:
        raise ValueError("EMG feature count must be divisible by sensor count")
    if not 0 <= sensor_index < sensor_count:
        raise IndexError(sensor_index)
    return list(range(sensor_index, emg_channels, sensor_count))


class EMGChannelTimeGate(nn.Module):
    """Score each physical sensor separately at every causal sample.

    Enhanced EMG features are ordered [window, kind, sensor].  The shared
    scorer therefore sees all features belonging to one sensor, while a
    learned sensor bias can retain stable anatomical differences.  Softmax
    weights are multiplied by the sensor count so uniform attention is an
    exact identity transform at initialization.
    """

    def __init__(
        self,
        emg_channels: int,
        sensor_count: int,
        hidden: int = 32,
        temperature: float = 1.0,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if emg_channels % sensor_count:
            raise ValueError(
                f"{emg_channels} EMG features cannot be grouped into "
                f"{sensor_count} physical sensors"
            )
        self.emg_channels = int(emg_channels)
        self.sensor_count = int(sensor_count)
        self.features_per_sensor = emg_channels // sensor_count
        self.temperature = max(float(temperature), 1e-3)
        self.dropout_probability = float(dropout_probability)
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.features_per_sensor),
            nn.Linear(self.features_per_sensor, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.sensor_bias = nn.Parameter(torch.zeros(sensor_count))
        # Begin as an identity gate. Learning must earn departures from equal
        # weighting rather than randomly suppressing a muscle on step one.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        emg: torch.Tensor,
        time_mask: torch.Tensor,
        apply_channel_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = emg.shape
        if channels != self.emg_channels:
            raise ValueError(
                f"expected {self.emg_channels} EMG features, got {channels}"
            )
        grouped = emg.reshape(
            batch, steps, self.features_per_sensor, self.sensor_count
        )
        per_sensor = grouped.permute(0, 1, 3, 2)
        logits = self.scorer(per_sensor).squeeze(-1) + self.sensor_bias
        attention = torch.softmax(logits / self.temperature, dim=-1)
        gain = attention * float(self.sensor_count)
        gated = grouped * gain.unsqueeze(2)

        if (
            apply_channel_dropout
            and self.training
            and self.dropout_probability > 0.0
        ):
            keep = torch.rand(
                batch, self.sensor_count, device=emg.device
            ) >= self.dropout_probability
            empty = ~keep.any(dim=-1)
            if empty.any():
                replacement = torch.randint(
                    self.sensor_count, (int(empty.sum()),), device=emg.device
                )
                keep[empty] = F.one_hot(
                    replacement, num_classes=self.sensor_count
                ).bool()
            gated = gated * keep[:, None, None, :].to(gated.dtype)

        gated = gated.reshape(batch, steps, channels)
        gated = gated * time_mask.unsqueeze(-1).to(gated.dtype)
        return gated, attention


class ChannelAwareStudentEncoder(WearableStudentEncoder):
    """Original balanced student preceded by physical-sensor EMG gating."""

    def __init__(
        self,
        config: dict[str, Any],
        emg_channels: int,
        imu_channels: int,
        trajectory_steps: int,
    ) -> None:
        super().__init__(config, emg_channels, imu_channels, trajectory_steps)
        settings = config["model"]["channel_time_attention"]
        sensor_count = len(config["data"].get("sensors", []))
        if sensor_count < 2:
            raise ValueError("channel-time attention requires at least two sensors")
        self.channel_gate = EMGChannelTimeGate(
            emg_channels,
            sensor_count,
            hidden=int(settings.get("hidden", 32)),
            temperature=float(settings.get("temperature", 1.0)),
            dropout_probability=float(
                settings.get("sensor_dropout_probability", 0.0)
            ),
        )

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        apply_imu_dropout: bool = False,
        apply_channel_dropout: bool = False,
    ) -> dict[str, torch.Tensor]:
        gated_emg, attention = self.channel_gate(
            emg, time_mask, apply_channel_dropout=apply_channel_dropout
        )
        outputs = super().forward(
            gated_emg, imu, time_mask, apply_imu_dropout=apply_imu_dropout
        )
        outputs["channel_attention"] = attention
        return outputs


class HorizonLatentHeads(nn.Module):
    """Read time-to-go from one named latent block and suppress it elsewhere."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["horizon_latent"]
        latent_dim = int(model["latent_dim"])
        self.start = int(settings["start"])
        self.dim = int(settings["dim"])
        self.end = self.start + self.dim
        if self.start < 0 or self.end > latent_dim or self.dim <= 0:
            raise ValueError("horizon latent slice is outside the full latent")
        centers = torch.tensor(
            settings.get("bins_ms", [50.0, 100.0, 200.0, 300.0, 400.0]),
            dtype=torch.float32,
        )
        if centers.ndim != 1 or len(centers) < 2:
            raise ValueError("horizon_latent.bins_ms needs at least two centers")
        self.register_buffer("centers_ms", centers)
        self.grl_scale = float(settings.get("gradient_reversal_scale", 0.25))
        hidden = int(settings.get("head_width", 64))
        other_dim = latent_dim - self.dim

        def classifier(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, len(centers)),
            )

        self.from_horizon = classifier(self.dim)
        self.from_other = classifier(other_dim)

    def forward(self, latent_mean: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon = latent_mean[:, self.start : self.end]
        other = torch.cat(
            [latent_mean[:, : self.start], latent_mean[:, self.end :]], dim=-1
        )
        logits = self.from_horizon(horizon)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "horizon_logits": logits,
            "horizon_other_logits": self.from_other(
                gradient_reverse(other, self.grl_scale)
            ),
            "horizon_probabilities": probabilities,
            "horizon_expected_ms": (
                probabilities * self.centers_ms.to(probabilities)
            ).sum(dim=-1),
        }


class DisambiguatedGuidanceHeads(nn.Module):
    """Keep the target/motion/session probes and add a horizon subspace."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.factor = FactorGuidanceHeads(config)
        self.horizon = HorizonLatentHeads(config)
        # Compatibility with the existing trainer's informative console line.
        self.intent_dim = self.factor.intent_dim
        self.motion_dim = self.factor.motion_dim
        self.residual_dim = self.factor.residual_dim

    def forward(self, latent_mean: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            **self.factor(latent_mean),
            **self.horizon(latent_mean),
        }


class ChannelHorizonLatentDistillationModel(WearableLatentDistillationModel):
    """Factor-guided baseline plus channel/time and horizon disambiguation."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        steps = int(config["model"]["teacher_trajectory_steps"])
        self.student = ChannelAwareStudentEncoder(
            config, emg_channels, imu_channels, trajectory_steps=steps
        )
        if not bool(config["model"].get("factor_latent", {}).get("enabled", False)):
            raise ValueError("this experiment requires factor_latent.enabled=true")
        self.guidance = DisambiguatedGuidanceHeads(config)

    def student_forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        sample: bool = False,
        noise_scale: float = 1.0,
        include_emg_only: bool = False,
        apply_imu_dropout: bool = False,
        apply_channel_dropout: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if apply_channel_dropout is None:
            apply_channel_dropout = apply_imu_dropout
        encoded = self.student(
            emg,
            imu,
            time_mask,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
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
            "channel_attention": encoded["channel_attention"],
            "guidance": self.guidance(encoded["mu"]),
        }
        if include_emg_only:
            emg_latent = reparameterize(
                encoded["emg_mu"],
                encoded["emg_log_variance"],
                sample,
                noise_scale,
            )
            outputs["emg_only"] = {
                **self.decoder(emg_latent),
                "latent": emg_latent,
                "mu": encoded["emg_mu"],
                "log_variance": encoded["emg_log_variance"],
                "guidance": self.guidance(encoded["emg_mu"]),
            }
        return outputs


def soft_horizon_targets(
    lead_ms: torch.Tensor, centers_ms: torch.Tensor, sigma_ms: float
) -> torch.Tensor:
    distance = (lead_ms.unsqueeze(-1) - centers_ms.unsqueeze(0)) / max(
        float(sigma_ms), 1e-3
    )
    return torch.softmax(-0.5 * distance.square(), dim=-1)


def horizon_guidance_losses(
    outputs: dict[str, Any],
    lead_samples: torch.Tensor,
    sample_rate_hz: float,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Supervise time-to-touch as a label, never as a student input."""
    guidance = outputs["guidance"]
    lead_ms = 1000.0 * lead_samples.to(torch.float32) / float(sample_rate_hz)
    centers = torch.as_tensor(
        settings.get("bins_ms", [50.0, 100.0, 200.0, 300.0, 400.0]),
        dtype=lead_ms.dtype,
        device=lead_ms.device,
    )
    target = soft_horizon_targets(
        lead_ms, centers, float(settings.get("target_sigma_ms", 45.0))
    )
    log_probabilities = F.log_softmax(guidance["horizon_logits"], dim=-1)
    other_log_probabilities = F.log_softmax(
        guidance["horizon_other_logits"], dim=-1
    )
    maximum = float(centers.max().item())
    return {
        "horizon_classification": -(target * log_probabilities).sum(-1).mean(),
        "horizon_regression": F.smooth_l1_loss(
            guidance["horizon_expected_ms"] / maximum,
            lead_ms / maximum,
            beta=float(settings.get("regression_huber_beta", 0.1)),
        ),
        # The classifier learns normally; GRL removes its information from
        # every latent coordinate outside the named horizon slice.
        "horizon_adversarial": -(
            target * other_log_probabilities
        ).sum(-1).mean(),
    }


def channel_attention_regularizers(
    attention: torch.Tensor, time_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return scale-stable selectivity and temporal-smoothness penalties."""
    probabilities = attention.clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    entropy = entropy / math.log(attention.size(-1))
    valid = time_mask.to(attention.dtype)
    entropy_loss = (entropy * valid).sum() / valid.sum().clamp_min(1.0)

    pair_mask = (time_mask[:, 1:] & time_mask[:, :-1]).to(attention.dtype)
    differences = (attention[:, 1:] - attention[:, :-1]).abs().mean(dim=-1)
    smoothness = (
        (differences * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    )
    return {"channel_entropy": entropy_loss, "channel_smoothness": smoothness}
