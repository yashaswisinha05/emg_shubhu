"""Deterministic wearable latents with token-level EMG/IMU interaction.

This is an isolated successor to semantic residual distillation.  The
privileged VIVE teacher and shared decoder are retained, but the deployable
student no longer pools each modality before they can interact.  It uses:

* causal physical-sensor x lookback attention for EMG;
* bidirectional token-level cross-attention between EMG and IMU;
* an EMG-owned intent block, IMU-owned motion block, and fused residual block;
* a deterministic student latent (the teacher may remain variational).

``student_forward`` retains the established deployment boundary: causal EMG,
causal IMU, and a padding mask are its only inputs.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .channel_horizon_distillation import EMGChannelTimeGate
from .layers import PatchTransformerEncoder, masked_mean
from .semantic_residual_distillation import (
    IntentResidualHead,
    SemanticResidualDistillationModel,
    _apply_endpoint_correction,
)


class SensorLagAttention(nn.Module):
    """Summarise EMG using a learned distribution over sensor x causal lag."""

    def __init__(
        self,
        emg_channels: int,
        sensor_count: int,
        d_model: int,
        lag_edges_ms: list[float],
        effective_rate_hz: float,
        hidden: int,
    ) -> None:
        super().__init__()
        if sensor_count < 1 or emg_channels % sensor_count:
            raise ValueError("EMG features must divide evenly across sensors")
        if len(lag_edges_ms) < 2:
            raise ValueError("lag_edges_ms needs at least two edges")
        edges = torch.tensor(lag_edges_ms, dtype=torch.float32)
        if edges[0] != 0 or not bool(torch.all(edges[1:] > edges[:-1])):
            raise ValueError("lag_edges_ms must increase strictly from zero")
        self.emg_channels = int(emg_channels)
        self.sensor_count = int(sensor_count)
        self.features_per_sensor = emg_channels // sensor_count
        self.effective_rate_hz = float(effective_rate_hz)
        self.register_buffer("lag_edges_ms", edges)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.features_per_sensor),
            nn.Linear(self.features_per_sensor, d_model),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.sensor_bias = nn.Parameter(torch.zeros(sensor_count, 1))
        self.lag_bias = nn.Parameter(torch.zeros(1, len(lag_edges_ms) - 1))

    def forward(
        self, emg: torch.Tensor, time_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = emg.shape
        if channels != self.emg_channels:
            raise ValueError(f"expected {self.emg_channels} EMG features, got {channels}")
        grouped = emg.reshape(
            batch, steps, self.features_per_sensor, self.sensor_count
        ).permute(0, 1, 3, 2)
        embedded = self.feature_projection(grouped)

        positions = torch.arange(steps, device=emg.device).view(1, steps)
        masked_positions = positions.masked_fill(~time_mask, -1)
        last_valid = masked_positions.max(dim=1).values.clamp_min(0)
        age_ms = (
            (last_valid[:, None] - positions).clamp_min(0).to(emg.dtype)
            * (1000.0 / self.effective_rate_hz)
        )
        summaries: list[torch.Tensor] = []
        available: list[torch.Tensor] = []
        for low, high in zip(self.lag_edges_ms[:-1], self.lag_edges_ms[1:]):
            selected = time_mask & (age_ms >= low) & (age_ms < high)
            weights = selected[:, :, None, None].to(embedded.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            summaries.append((embedded * weights).sum(dim=1) / count)
            available.append(selected.any(dim=1))
        summary = torch.stack(summaries, dim=2)  # [B, sensor, lag, D]
        valid_bins = torch.stack(available, dim=1)[:, None, :].expand(
            -1, self.sensor_count, -1
        )
        logits = self.scorer(summary).squeeze(-1)
        logits = logits + self.sensor_bias + self.lag_bias
        logits = logits.masked_fill(~valid_bins, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
        context = (summary * attention.unsqueeze(-1)).sum(dim=(1, 2))
        return context, attention


class TemporalCrossAttentionStudentEncoder(nn.Module):
    """Token fusion with explicit EMG-intent and IMU-motion ownership."""

    def __init__(
        self,
        config: dict[str, Any],
        emg_channels: int,
        imu_channels: int,
        trajectory_steps: int,
    ) -> None:
        super().__init__()
        model = config["model"]
        settings = model["temporal_cross_attention"]
        width = int(model["d_model"])
        latent_dim = int(model["latent_dim"])
        factor = model["factor_latent"]
        self.intent_dim = int(factor["intent_dim"])
        self.motion_dim = int(factor["motion_dim"])
        self.residual_dim = latent_dim - self.intent_dim - self.motion_dim
        if min(self.intent_dim, self.motion_dim, self.residual_dim) <= 0:
            raise ValueError("factor dimensions must leave a residual latent block")

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
        self.emg_encoder = PatchTransformerEncoder(emg_channels, **common)
        self.imu_encoder = PatchTransformerEncoder(imu_channels, **common)
        channel = model["channel_time_attention"]
        sensor_count = len(config["data"].get("sensors", []))
        self.channel_gate = EMGChannelTimeGate(
            emg_channels,
            sensor_count,
            hidden=int(channel.get("hidden", 32)),
            temperature=float(channel.get("temperature", 1.0)),
            dropout_probability=float(channel.get("sensor_dropout_probability", 0.0)),
        )
        effective_rate = float(config["data"]["sample_rate_hz"]) / max(
            1, int(config["data"].get("decimation", 1))
        )
        self.lag_attention = SensorLagAttention(
            emg_channels,
            sensor_count,
            width,
            list(map(float, settings["lag_edges_ms"])),
            effective_rate,
            hidden=int(settings.get("lag_hidden", 64)),
        )
        cross_heads = int(settings.get("num_heads", model["num_heads"]))
        cross_dropout = float(settings.get("dropout", model["dropout"]))
        self.emg_from_imu = nn.MultiheadAttention(
            width, cross_heads, dropout=cross_dropout, batch_first=True
        )
        self.imu_from_emg = nn.MultiheadAttention(
            width, cross_heads, dropout=cross_dropout, batch_first=True
        )
        self.emg_cross_norm = nn.LayerNorm(width)
        self.imu_cross_norm = nn.LayerNorm(width)
        self.intent_context = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU()
        )
        self.motion_context = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.interaction_context = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(),
            nn.Dropout(float(model["dropout"])),
        )
        self.to_intent = nn.Linear(width, self.intent_dim)
        self.to_motion = nn.Linear(width, self.motion_dim)
        self.to_residual = nn.Linear(width, self.residual_dim)
        self.emg_to_full_latent = nn.Linear(width, latent_dim)
        self.imu_motion_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, trajectory_steps * 3),
        )
        self.trajectory_steps = int(trajectory_steps)
        self.imu_dropout_probability = float(model.get("imu_modality_dropout", 0.0))

    @staticmethod
    def _safe_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        safe = mask.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        return ~safe

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        apply_imu_dropout: bool = False,
        apply_channel_dropout: bool = False,
    ) -> dict[str, torch.Tensor]:
        gated_emg, channel_attention = self.channel_gate(
            emg, time_mask, apply_channel_dropout=apply_channel_dropout
        )
        emg_tokens, emg_pool, emg_mask = self.emg_encoder(gated_emg, time_mask)
        imu_tokens, imu_pool, imu_mask = self.imu_encoder(imu, time_mask)
        lag_context, lag_attention = self.lag_attention(gated_emg, time_mask)

        keep = torch.ones(
            imu_tokens.size(0), 1, 1, device=imu_tokens.device, dtype=imu_tokens.dtype
        )
        if apply_imu_dropout and self.training and self.imu_dropout_probability > 0.0:
            keep = (
                torch.rand_like(keep) >= self.imu_dropout_probability
            ).to(imu_tokens.dtype)
        fused_imu_tokens = imu_tokens * keep
        fused_imu_pool = imu_pool * keep.squeeze(1)

        emg_cross, emg_cross_weights = self.emg_from_imu(
            emg_tokens,
            fused_imu_tokens,
            fused_imu_tokens,
            key_padding_mask=self._safe_padding_mask(imu_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        imu_cross, imu_cross_weights = self.imu_from_emg(
            fused_imu_tokens,
            emg_tokens,
            emg_tokens,
            key_padding_mask=self._safe_padding_mask(emg_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        emg_cross_pool = masked_mean(
            self.emg_cross_norm(emg_tokens + emg_cross), emg_mask
        )
        imu_cross_pool = masked_mean(
            self.imu_cross_norm(fused_imu_tokens + imu_cross), imu_mask
        )
        intent_context = self.intent_context(
            torch.cat([emg_pool, lag_context], dim=-1)
        )
        motion_context = self.motion_context(fused_imu_pool)
        interaction = self.interaction_context(
            torch.cat([emg_cross_pool, imu_cross_pool], dim=-1)
        )
        mu = torch.cat(
            [
                self.to_intent(intent_context),
                self.to_motion(motion_context),
                self.to_residual(interaction),
            ],
            dim=-1,
        )
        emg_mu = self.emg_to_full_latent(intent_context)
        # Compatibility tensors only. With student_noise_scale=0 and both
        # Gaussian-distillation weights=0, the deployable latent is deterministic.
        log_variance = torch.zeros_like(mu)
        emg_log_variance = torch.zeros_like(emg_mu)
        return {
            "mu": mu,
            "log_variance": log_variance,
            "emg_mu": emg_mu,
            "emg_log_variance": emg_log_variance,
            "imu_trajectory": self.imu_motion_head(imu_pool).reshape(
                -1, self.trajectory_steps, 3
            ),
            # Direct deterministic task heads can consume these contexts
            # without passing through a Gaussian latent or teacher bridge.
            "intent_context": intent_context,
            "motion_context": motion_context,
            "interaction_context": interaction,
            "fused_imu_trajectory": self.imu_motion_head(
                fused_imu_pool
            ).reshape(-1, self.trajectory_steps, 3),
            "channel_attention": channel_attention,
            "lag_attention": lag_attention,
            "emg_from_imu_attention": emg_cross_weights,
            "imu_from_emg_attention": imu_cross_weights,
        }


class TemporalCrossAttentionDistillationModel(
    SemanticResidualDistillationModel
):
    """Semantic model with a deterministic, factor-owned token-fusion student."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        steps = int(config["model"]["teacher_trajectory_steps"])
        self.student = TemporalCrossAttentionStudentEncoder(
            config, emg_channels, imu_channels, trajectory_steps=steps
        )
        semantic = config["model"].get("semantic_residual", {})
        intent_dim = int(config["model"]["factor_latent"]["intent_dim"])
        hidden = int(semantic.get("head_width", 96))
        maximum = float(semantic.get("maximum_logit_delta", 1.5))
        self.student.fused_endpoint_residual = IntentResidualHead(
            2 * intent_dim, hidden, maximum
        )
        self.student.emg_endpoint_residual = IntentResidualHead(
            intent_dim, hidden, maximum
        )

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
    ) -> dict[str, Any]:
        """Run the deterministic wearable student; sample arguments are ignored."""
        del sample, noise_scale
        if apply_channel_dropout is None:
            apply_channel_dropout = apply_imu_dropout
        encoded = self.student(
            emg,
            imu,
            time_mask,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        latent = encoded["mu"]
        emg_latent = encoded["emg_mu"]
        outputs: dict[str, Any] = {
            **self.decoder(latent),
            "latent": latent,
            "mu": latent,
            "log_variance": encoded["log_variance"],
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": self.guidance(latent),
        }
        emg_outputs = {
            **self.decoder(emg_latent),
            "latent": emg_latent,
            "mu": emg_latent,
            "log_variance": encoded["emg_log_variance"],
            "guidance": self.guidance(emg_latent),
        }
        fused_intent = latent[:, : self.intent_dim]
        emg_intent = emg_latent[:, : self.intent_dim]
        _apply_endpoint_correction(
            outputs,
            self.student.fused_endpoint_residual(
                torch.cat([fused_intent, emg_intent], dim=-1)
            ),
        )
        _apply_endpoint_correction(
            emg_outputs, self.student.emg_endpoint_residual(emg_intent)
        )
        if include_emg_only:
            outputs["emg_only"] = emg_outputs
        return outputs
