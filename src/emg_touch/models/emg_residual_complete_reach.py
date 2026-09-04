"""Temporal EMG residual correction for the soft-routed complete reach model.

The successful soft-routed student is retained as the base.  A new per-path
query decoder attends to causal EMG tokens and predicts only the 3D correction
left unexplained by that base.  The residual head is zero-initialized, so
loading a soft-routed checkpoint reproduces its trajectory exactly before the
new branch is trained.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .soft_routed_complete_reach import SoftRoutedCompleteReachModel


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class TemporalEMGResidualHead(nn.Module):
    """Attend from 3D path queries to causal EMG tokens."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["emg_temporal_residual"]
        width = int(model["d_model"])
        self.steps = int(model["teacher_trajectory_steps"])
        self.residual_limit_m = float(settings.get("residual_limit_m", 0.15))
        if self.residual_limit_m <= 0.0:
            raise ValueError("residual_limit_m must be positive")
        heads = int(settings.get("num_heads", model["num_heads"]))
        dropout = float(settings.get("dropout", model["dropout"]))
        self.path_queries = nn.Parameter(torch.empty(self.steps, width))
        nn.init.normal_(self.path_queries, std=0.02)
        self.motion_query = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(width)
        self.context_mlp = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.path_residual_head = nn.Linear(width, 3)
        self.path_gate_head = nn.Linear(width, 1)
        self.endpoint_residual_head = nn.Linear(width, 3)
        self.endpoint_gate_head = nn.Linear(width, 1)
        self.axis_direction_head = nn.Linear(width, 3 * 3)

        # Loading the previous best checkpoint must initially give the same
        # path. Learning, rather than random initialization, earns a change.
        nn.init.zeros_(self.path_residual_head.weight)
        nn.init.zeros_(self.path_residual_head.bias)
        nn.init.zeros_(self.endpoint_residual_head.weight)
        nn.init.zeros_(self.endpoint_residual_head.bias)
        initial = _gate_logit(float(settings.get("correction_gate_initial", 0.25)))
        nn.init.zeros_(self.path_gate_head.weight)
        nn.init.constant_(self.path_gate_head.bias, initial)
        nn.init.zeros_(self.endpoint_gate_head.weight)
        nn.init.constant_(self.endpoint_gate_head.bias, initial)

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        return ~safe

    @staticmethod
    def _progress(reference: torch.Tensor) -> torch.Tensor:
        progress = torch.linspace(
            0.0,
            1.0,
            reference.size(1),
            device=reference.device,
            dtype=reference.dtype,
        )
        progress = progress.square() * (3.0 - 2.0 * progress)
        return progress.view(1, -1, 1)

    def forward(
        self,
        emg_tokens: torch.Tensor,
        emg_token_mask: torch.Tensor,
        motion_context: torch.Tensor,
        base_trajectory: torch.Tensor,
        base_endpoint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = emg_tokens.size(0)
        queries = self.path_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.motion_query(motion_context).unsqueeze(1)
        attended, attention = self.cross_attention(
            queries,
            emg_tokens,
            emg_tokens,
            key_padding_mask=self._safe_padding_mask(emg_token_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.context_norm(queries + attended)
        context = context + self.context_mlp(context)
        raw_path = self.residual_limit_m * torch.tanh(
            self.path_residual_head(context)
        )
        path_gate = torch.sigmoid(self.path_gate_head(context))
        progress = self._progress(base_trajectory)
        provisional = base_trajectory + progress * path_gate * raw_path

        endpoint_context = context.mean(dim=1)
        raw_endpoint = self.residual_limit_m * torch.tanh(
            self.endpoint_residual_head(endpoint_context)
        )
        endpoint_gate = torch.sigmoid(
            self.endpoint_gate_head(endpoint_context)
        )
        endpoint = base_endpoint + endpoint_gate * raw_endpoint
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )
        residual = trajectory - base_trajectory
        return {
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "pre_emg_residual_trajectory": base_trajectory,
            "pre_emg_residual_endpoint": base_endpoint,
            "emg_temporal_residual": residual,
            "emg_temporal_endpoint_residual": endpoint - base_endpoint,
            "emg_temporal_raw_path_residual": raw_path,
            "emg_temporal_raw_endpoint_residual": raw_endpoint,
            "emg_temporal_path_gate": path_gate.squeeze(-1),
            "emg_temporal_endpoint_gate": endpoint_gate.squeeze(-1),
            "emg_temporal_attention": attention,
            "axis_direction_logits": self.axis_direction_head(
                endpoint_context
            ).reshape(-1, 3, 3),
        }


class EMGResidualCompleteReachModel(SoftRoutedCompleteReachModel):
    """Soft-routed model plus a temporally resolved EMG-to-3D residual."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.emg_temporal_residual_head = TemporalEMGResidualHead(config)
        self.residual_warmup = False

    def train(self, mode: bool = True) -> "EMGResidualCompleteReachModel":
        super().train(mode)
        if mode and self.residual_warmup:
            # Keep dropout and normalization in the loaded base deterministic
            # while the zero-initialized residual learns its first correction.
            self.student.eval()
            self.student.emg_temporal_residual_head.train()
            self.teacher.eval()
            self.decoder.eval()
            if self.guidance is not None:
                self.guidance.eval()
        return self

    def _apply_temporal_residual(
        self,
        encoded: dict[str, torch.Tensor],
        base: dict[str, torch.Tensor],
        *,
        emg_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        motion_context = (
            torch.zeros_like(encoded["motion_context"])
            if emg_only
            else encoded["motion_context"]
        )
        return self.student.emg_temporal_residual_head(
            encoded["emg_tokens"],
            encoded["emg_token_mask"],
            motion_context,
            base["trajectory"],
            base["endpoint_3d"],
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
        factor = encoded["mu"]
        screen, decoder_latent, guidance = self._screen_decode(factor)
        base_motion = self.student.soft_routed_reach_heads(
            encoded["intent_context"],
            encoded["motion_context"],
            encoded["fused_imu_trajectory"],
        )
        residual_motion = self._apply_temporal_residual(encoded, base_motion)
        outputs: dict[str, Any] = {
            **screen,
            **base_motion,
            **residual_motion,
            "latent": decoder_latent,
            "mu": decoder_latent,
            "factor_latent": factor,
            "decoder_latent": decoder_latent,
            "log_variance": torch.zeros_like(decoder_latent),
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": guidance,
            # Private forward-state keys let isolated successor heads reuse
            # the causal encoding without running the full student twice.
            "_emg_tokens": encoded["emg_tokens"],
            "_emg_token_mask": encoded["emg_token_mask"],
            "_motion_context": encoded["motion_context"],
        }
        if include_emg_only:
            emg_factor = encoded["emg_mu"]
            emg_screen, emg_latent, emg_guidance = self._screen_decode(
                emg_factor, emg_only=True
            )
            zero_motion = torch.zeros_like(encoded["motion_context"])
            zero_base = torch.zeros_like(encoded["fused_imu_trajectory"])
            emg_base = self.student.soft_routed_reach_heads(
                encoded["intent_context"],
                zero_motion,
                zero_base,
                intent_gradient_scale=1.0,
            )
            emg_residual = self._apply_temporal_residual(
                encoded, emg_base, emg_only=True
            )
            outputs["emg_only"] = {
                **emg_screen,
                **emg_base,
                **emg_residual,
                "latent": emg_latent,
                "mu": emg_latent,
                "factor_latent": emg_factor,
                "decoder_latent": emg_latent,
                "log_variance": torch.zeros_like(emg_latent),
                "guidance": emg_guidance,
            }
        return outputs
