"""Deterministic wearable heads with no VAE sampling in deployment.

The proven deterministic bridge is retained for screen coordinates; it is a
fixed feed-forward adapter, not a stochastic VAE. The 3D path continues to use
the direct IMU base and bounded EMG correction that improved tracking.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .complete_reach_distillation import CompleteReachDistillationModel


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class DeterministicReachHeads(nn.Module):
    """Direct IMU-base 3D correction head conditioned on EMG intent."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["deterministic_complete_reach"]
        width = int(model["d_model"])
        hidden = int(settings.get("hidden", width))
        dropout = float(settings.get("dropout", model["dropout"]))
        self.steps = int(model["teacher_trajectory_steps"])
        self.correction_limit_m = float(settings.get("correction_limit_m", 0.20))
        self.correction_adapter = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.path_correction_head = nn.Linear(hidden, self.steps * 3)
        self.endpoint_correction_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.path_correction_head.weight)
        nn.init.zeros_(self.path_correction_head.bias)
        nn.init.zeros_(self.endpoint_correction_head.weight)
        nn.init.zeros_(self.endpoint_correction_head.bias)
        initial = float(settings.get("correction_gate_initial", 0.15))
        self.path_correction_logit = nn.Parameter(torch.tensor(_gate_logit(initial)))
        self.endpoint_correction_logit = nn.Parameter(
            torch.tensor(_gate_logit(initial))
        )

    @staticmethod
    def _progress(reference: torch.Tensor) -> torch.Tensor:
        value = torch.linspace(
            0.0, 1.0, reference.size(1),
            device=reference.device, dtype=reference.dtype,
        )
        value = value.square() * (3.0 - 2.0 * value)
        return value.view(1, -1, 1)

    def forward(
        self,
        intent_context: torch.Tensor,
        motion_context: torch.Tensor,
        imu_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        correction_context = self.correction_adapter(torch.cat([
            motion_context, intent_context.detach()
        ], dim=-1))
        path_correction = self.correction_limit_m * torch.tanh(
            self.path_correction_head(correction_context).reshape(-1, self.steps, 3)
        )
        endpoint_correction = self.correction_limit_m * torch.tanh(
            self.endpoint_correction_head(correction_context)
        )
        path_gate = torch.sigmoid(self.path_correction_logit)
        endpoint_gate = torch.sigmoid(self.endpoint_correction_logit)
        base = imu_base - imu_base[:, :1]
        progress = self._progress(base)
        provisional = base + progress * path_gate * path_correction
        endpoint = base[:, -1] + endpoint_gate * endpoint_correction
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )
        return {
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "imu_base_trajectory": base,
            "path_correction": path_correction,
            "endpoint_correction": endpoint_correction,
            "path_correction_gate": path_gate.expand(intent_context.size(0)),
            "endpoint_correction_gate": endpoint_gate.expand(intent_context.size(0)),
            "deterministic_intent_context": intent_context,
            "deterministic_motion_context": motion_context,
        }


class DeterministicCompleteReachModel(CompleteReachDistillationModel):
    """Teacher-assisted training, deterministic wearable-only deployment."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.deterministic_heads = DeterministicReachHeads(config)

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
            emg, imu, time_mask,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        decoded = self.student.deterministic_heads(
            encoded["intent_context"], encoded["motion_context"],
            encoded["fused_imu_trajectory"],
        )
        factor_latent = encoded["mu"]
        decoder_latent = self.student.teacher_latent_bridge(factor_latent)
        screen_decoded, guidance = self._student_decode(
            factor_latent, decoder_latent
        )
        outputs: dict[str, Any] = {
            **screen_decoded,
            **decoded,
            # These deterministic factors are never sampled. The bridge only
            # adapts coordinates for the established screen decoder.
            "latent": factor_latent,
            "mu": factor_latent,
            "factor_latent": factor_latent,
            "decoder_latent": decoder_latent,
            "log_variance": torch.zeros_like(factor_latent),
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": guidance,
        }
        if include_emg_only:
            zero_motion = torch.zeros_like(encoded["motion_context"])
            zero_base = torch.zeros_like(encoded["fused_imu_trajectory"])
            emg_decoded = self.student.deterministic_heads(
                encoded["intent_context"], zero_motion, zero_base
            )
            emg_latent = encoded["emg_mu"]
            emg_decoder_latent = self.student.emg_teacher_latent_bridge(
                emg_latent
            )
            emg_screen, emg_guidance = self._student_decode(
                emg_latent, emg_decoder_latent
            )
            outputs["emg_only"] = {
                **emg_screen,
                **emg_decoded,
                "latent": emg_latent,
                "mu": emg_latent,
                "factor_latent": emg_latent,
                "decoder_latent": emg_decoder_latent,
                "log_variance": torch.zeros_like(emg_latent),
                "guidance": emg_guidance,
            }
        return outputs
