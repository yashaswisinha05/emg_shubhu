"""Parameter-efficient per-user adaptation of the acceleration reach model."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .emg_acceleration_complete_reach import EMGAccelerationCompleteReachModel


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class CandidateResidualAdapter(nn.Module):
    """Low-rank screen/path correction learned from one candidate's trials."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model.get("candidate_personalization", {})
        latent_dim = int(model["latent_dim"])
        rank = int(settings.get("rank", 12))
        steps = int(model["teacher_trajectory_steps"])
        self.screen_limit = float(settings.get("screen_residual_limit", 0.12))
        self.path_limit_m = float(settings.get("path_residual_limit_m", 0.10))
        self.steps = steps
        self.adapter = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank),
            nn.GELU(),
        )
        self.screen_head = nn.Linear(rank, 2)
        self.path_head = nn.Linear(rank, steps * 3)
        self.endpoint_head = nn.Linear(rank, 3)
        initial = _gate_logit(float(settings.get("gate_initial", 0.25)))
        self.screen_gate_logit = nn.Parameter(torch.tensor(initial))
        self.path_gate_logit = nn.Parameter(torch.tensor(initial))
        self.endpoint_gate_logit = nn.Parameter(torch.tensor(initial))

        # Exact preservation of the supplied population checkpoint.
        for head in (self.screen_head, self.path_head, self.endpoint_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _progress(reference: torch.Tensor) -> torch.Tensor:
        progress = torch.linspace(
            0.0, 1.0, reference.size(1),
            device=reference.device, dtype=reference.dtype,
        )
        progress = progress.square() * (3.0 - 2.0 * progress)
        return progress.view(1, -1, 1)

    def forward(
        self,
        factor_latent: torch.Tensor,
        base_prediction: torch.Tensor,
        base_trajectory: torch.Tensor,
        base_endpoint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.adapter(factor_latent)
        screen_raw = self.screen_limit * torch.tanh(self.screen_head(context))
        screen_gate = torch.sigmoid(self.screen_gate_logit)
        prediction = (base_prediction + screen_gate * screen_raw).clamp(0.0, 1.0)

        path_raw = self.path_limit_m * torch.tanh(
            self.path_head(context).reshape(-1, self.steps, 3)
        )
        endpoint_raw = self.path_limit_m * torch.tanh(self.endpoint_head(context))
        path_gate = torch.sigmoid(self.path_gate_logit)
        endpoint_gate = torch.sigmoid(self.endpoint_gate_logit)
        progress = self._progress(base_trajectory)
        provisional = base_trajectory + progress * path_gate * path_raw
        endpoint = base_endpoint + endpoint_gate * endpoint_raw
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )
        return {
            "prediction": prediction,
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "pre_personalization_prediction": base_prediction,
            "pre_personalization_trajectory": base_trajectory,
            "pre_personalization_endpoint": base_endpoint,
            "personalization_screen_residual": prediction - base_prediction,
            "personalization_path_residual": trajectory - base_trajectory,
            "personalization_endpoint_residual": endpoint - base_endpoint,
            "personalization_screen_raw": screen_raw,
            "personalization_path_raw": path_raw,
            "personalization_endpoint_raw": endpoint_raw,
            "personalization_screen_gate": screen_gate.expand(factor_latent.size(0)),
            "personalization_path_gate": path_gate.expand(factor_latent.size(0)),
            "personalization_endpoint_gate": endpoint_gate.expand(
                factor_latent.size(0)
            ),
        }


class PersonalizedCompleteReachModel(EMGAccelerationCompleteReachModel):
    """Frozen population representation plus one low-rank candidate adapter."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.candidate_personalization = CandidateResidualAdapter(config)
        self.personalization_warmup = False

    def train(self, mode: bool = True) -> "PersonalizedCompleteReachModel":
        super().train(mode)
        if mode and self.personalization_warmup:
            self.student.eval()
            self.student.candidate_personalization.train()
            self.teacher.eval()
            self.decoder.eval()
            if self.guidance is not None:
                self.guidance.eval()
        return self

    def _personalize(self, outputs: dict[str, Any]) -> None:
        correction = self.student.candidate_personalization(
            outputs["factor_latent"],
            outputs["prediction"],
            outputs["trajectory"],
            outputs["endpoint_3d"],
        )
        outputs.update(correction)

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
            emg, imu, time_mask, sample=sample, noise_scale=noise_scale,
            include_emg_only=include_emg_only,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        self._personalize(outputs)
        if include_emg_only:
            self._personalize(outputs["emg_only"])
        return outputs
