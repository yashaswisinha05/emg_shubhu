"""Task-separated screen intent and 3D motion prediction.

The screen branch remains driven by the EMG-owned intent latent.  The 3D
branch begins with the directly supervised IMU trajectory and permits EMG
intent to make only a bounded correction.  The screen context is detached
before the 3D correction adapter so trajectory gradients cannot damage the
screen representation.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .complete_reach_distillation import (
    CompleteReachDecoder,
    CompleteReachDistillationModel,
)


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class TaskSeparatedCompleteReachDecoder(CompleteReachDecoder):
    """Screen head plus bounded intent-conditioned 3D correction heads."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model = config["model"]
        settings = model["task_separated_complete_reach"]
        hidden = int(
            model.get("rolling_dual_head", {}).get(
                "hidden", model.get("decoder_width", 128)
            )
        )
        dropout = float(settings.get("dropout", model["dropout"]))
        self.correction_limit_m = float(settings.get("correction_limit_m", 0.20))
        if self.correction_limit_m <= 0.0:
            raise ValueError("correction_limit_m must be positive")
        self.correction_adapter = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.path_correction_head = nn.Linear(hidden, self.trajectory_steps * 3)
        self.endpoint_correction_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.path_correction_head.weight)
        nn.init.zeros_(self.path_correction_head.bias)
        nn.init.zeros_(self.endpoint_correction_head.weight)
        nn.init.zeros_(self.endpoint_correction_head.bias)
        initial = float(settings.get("correction_gate_initial", 0.10))
        self.path_correction_logit = nn.Parameter(torch.tensor(_gate_logit(initial)))
        self.endpoint_correction_logit = nn.Parameter(
            torch.tensor(_gate_logit(initial))
        )

    def forward(
        self,
        factor_latent: torch.Tensor,
        decoder_latent: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(factor_latent, decoder_latent, horizon_ms)
        # The 3D head may read screen intent, but its loss cannot rewrite the
        # screen representation. This is the hard task-separation boundary.
        correction_context = self.correction_adapter(torch.cat([
            outputs["motion_context"], outputs["screen_context"].detach()
        ], dim=-1))
        outputs["path_correction"] = self.correction_limit_m * torch.tanh(
            self.path_correction_head(correction_context).reshape(
                -1, self.trajectory_steps, 3
            )
        )
        outputs["endpoint_correction"] = self.correction_limit_m * torch.tanh(
            self.endpoint_correction_head(correction_context)
        )
        outputs["path_correction_gate"] = torch.sigmoid(
            self.path_correction_logit
        ).expand(factor_latent.size(0))
        outputs["endpoint_correction_gate"] = torch.sigmoid(
            self.endpoint_correction_logit
        ).expand(factor_latent.size(0))
        return outputs


class TaskSeparatedCompleteReachModel(CompleteReachDistillationModel):
    """One wearable model with independently owned 2D and 3D predictions."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.endpoint_decoder = TaskSeparatedCompleteReachDecoder(config)

    @staticmethod
    def _smooth_progress(reference: torch.Tensor) -> torch.Tensor:
        steps = reference.size(1)
        progress = torch.linspace(
            0.0, 1.0, steps, device=reference.device, dtype=reference.dtype
        )
        progress = progress.square() * (3.0 - 2.0 * progress)
        return progress.view(1, steps, 1)

    def _compose_3d(
        self, outputs: dict[str, Any], imu_base: torch.Tensor
    ) -> None:
        base = imu_base - imu_base[:, :1]
        progress = self._smooth_progress(base)
        path_gate = outputs["path_correction_gate"].view(-1, 1, 1)
        endpoint_gate = outputs["endpoint_correction_gate"].view(-1, 1)
        provisional = base + progress * path_gate * outputs["path_correction"]
        endpoint = base[:, -1] + endpoint_gate * outputs["endpoint_correction"]
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )
        outputs["imu_base_trajectory"] = base
        outputs["trajectory"] = trajectory
        outputs["complete_trajectory"] = trajectory
        outputs["endpoint_3d"] = endpoint

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
        outputs = super().student_forward(
            emg,
            imu,
            time_mask,
            sample=sample,
            noise_scale=noise_scale,
            include_emg_only=include_emg_only,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        self._compose_3d(outputs, outputs["imu_trajectory"])
        if include_emg_only:
            emg_outputs = outputs["emg_only"]
            zero_base = torch.zeros_like(outputs["imu_trajectory"])
            self._compose_3d(emg_outputs, zero_base)
        return outputs
