"""Semantic teacher alignment and wearable-only endpoint correction.

This experiment extends the channel+horizon student without changing its
deployment boundary.  The teacher is used only while computing training
losses; :meth:`student_forward` still receives EMG, IMU, and a causal mask.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .channel_horizon_distillation import (
    ChannelHorizonLatentDistillationModel,
)


class IntentResidualHead(nn.Module):
    """Predict a bounded correction to frozen-decoder endpoint logits."""

    def __init__(
        self, input_dim: int, hidden: int, maximum_logit_delta: float
    ) -> None:
        super().__init__()
        self.maximum_logit_delta = float(maximum_logit_delta)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # The new experiment begins as the already-tested parent model.  A
        # correction must be learned rather than perturbing its first batch.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, intent: torch.Tensor) -> torch.Tensor:
        return self.maximum_logit_delta * torch.tanh(self.network(intent))


def _apply_endpoint_correction(
    outputs: dict[str, torch.Tensor], correction: torch.Tensor
) -> None:
    """Retain the parent's endpoint and expose a corrected direct prediction."""
    outputs["base_direct_logits"] = outputs["direct_logits"]
    outputs["base_prediction"] = outputs["prediction"]
    outputs["residual_logit_delta"] = correction
    outputs["direct_logits"] = outputs["direct_logits"] + correction
    outputs["direct_prediction"] = torch.sigmoid(outputs["direct_logits"])
    outputs["prediction"] = outputs["direct_prediction"]


class SemanticResidualDistillationModel(
    ChannelHorizonLatentDistillationModel
):
    """Channel+horizon model with intent-only residual endpoint heads.

    The fused correction sees the fused and EMG-only intent blocks.  This
    makes EMG an explicit input to the last-mile correction even when IMU is
    strong.  A second head gives the EMG-only auxiliary the same opportunity.
    Both heads live under ``student`` so the existing optimizer and checkpoint
    code include them automatically.
    """

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        settings = config["model"].get("semantic_residual", {})
        intent_dim = int(config["model"]["factor_latent"]["intent_dim"])
        hidden = int(settings.get("head_width", 96))
        maximum = float(settings.get("maximum_logit_delta", 1.5))
        self.student.fused_endpoint_residual = IntentResidualHead(
            2 * intent_dim, hidden, maximum
        )
        self.student.emg_endpoint_residual = IntentResidualHead(
            intent_dim, hidden, maximum
        )
        self.intent_dim = intent_dim

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
        # The fused residual always needs the EMG intent representation, but
        # callers that did not request the decoded EMG-only branch do not pay
        # for or receive it after the correction has been formed.
        outputs = super().student_forward(
            emg,
            imu,
            time_mask,
            sample=sample,
            noise_scale=noise_scale,
            include_emg_only=True,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        emg_outputs = outputs["emg_only"]
        fused_intent = outputs["mu"][:, : self.intent_dim]
        emg_intent = emg_outputs["mu"][:, : self.intent_dim]
        fused_delta = self.student.fused_endpoint_residual(
            torch.cat([fused_intent, emg_intent], dim=-1)
        )
        emg_delta = self.student.emg_endpoint_residual(emg_intent)
        _apply_endpoint_correction(outputs, fused_delta)
        _apply_endpoint_correction(emg_outputs, emg_delta)
        if not include_emg_only:
            del outputs["emg_only"]
        return outputs
