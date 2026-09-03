"""Deterministic wearable heads with no VAE in the prediction pathway."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .complete_reach_distillation import CompleteReachDistillationModel
from .grid_point import (
    SpatialPointHead,
    decode_grid_outputs,
    finalize_point_prediction,
)


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class DeterministicReachHeads(nn.Module):
    """Direct EMG screen head and IMU-base 3D correction head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["deterministic_complete_reach"]
        width = int(model["d_model"])
        hidden = int(settings.get("hidden", width))
        dropout = float(settings.get("dropout", model["dropout"]))
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.steps = int(model["teacher_trajectory_steps"])
        self.correction_limit_m = float(settings.get("correction_limit_m", 0.20))

        self.screen_adapter = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.point_head = SpatialPointHead(
            hidden, grid_width, grid_height, dropout,
            direct_prediction=True, zero_initialize=False,
        )
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
        screen = self.point_head(self.screen_adapter(intent_context))
        screen.update(decode_grid_outputs(
            screen["heatmap_logits"], screen["offset_logits"],
            self.grid_width, self.grid_height,
        ))
        finalize_point_prediction(screen)

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
            **screen,
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
            # Compatibility/reporting fields from the predecessor. Zero is
            # literal here: neither direct head uses teacher-decoder context.
            "screen_shared_gate": torch.zeros(
                intent_context.size(0), device=intent_context.device,
                dtype=intent_context.dtype,
            ),
            "motion_shared_gate": torch.zeros(
                intent_context.size(0), device=intent_context.device,
                dtype=intent_context.dtype,
            ),
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
        outputs: dict[str, Any] = {
            **decoded,
            # Compatibility fields for auxiliary factor probes only. They do
            # not drive either prediction head and are never sampled.
            "latent": factor_latent,
            "mu": factor_latent,
            "factor_latent": factor_latent,
            "decoder_latent": factor_latent,
            "log_variance": torch.zeros_like(factor_latent),
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": self.guidance(factor_latent),
        }
        if include_emg_only:
            zero_motion = torch.zeros_like(encoded["motion_context"])
            zero_base = torch.zeros_like(encoded["fused_imu_trajectory"])
            emg_decoded = self.student.deterministic_heads(
                encoded["intent_context"], zero_motion, zero_base
            )
            emg_latent = encoded["emg_mu"]
            outputs["emg_only"] = {
                **emg_decoded,
                "latent": emg_latent,
                "mu": emg_latent,
                "factor_latent": emg_latent,
                "decoder_latent": emg_latent,
                "log_variance": torch.zeros_like(emg_latent),
                "guidance": self.guidance(emg_latent),
            }
        return outputs
