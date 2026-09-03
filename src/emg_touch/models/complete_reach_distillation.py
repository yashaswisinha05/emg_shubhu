"""Wearable prediction of one stable, complete reach at every observation.

Unlike the future-only rolling decoder, this model predicts the same
onset-relative path and endpoint whether it is queried 400 ms before touch or
at touch.  The deployable boundary remains EMG, IMU, and a causal padding
mask; VIVE is used only to construct training and evaluation labels.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .latent_distillation import SharedIntentDecoder
from .rolling_dual_head_distillation import (
    RollingDualHeadDecoder,
    RollingDualHeadDistillationModel,
)


class CompleteReachDecoder(RollingDualHeadDecoder):
    """Screen endpoint, full 3D reach, and explicit 3D endpoint heads."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        hidden = int(
            config["model"].get("rolling_dual_head", {}).get(
                "hidden", config["model"].get("decoder_width", 128)
            )
        )
        self.endpoint_3d_head = nn.Linear(hidden, 3)

    def initialise_output_heads(self, teacher: SharedIntentDecoder) -> None:
        super().initialise_output_heads(teacher)
        # The last teacher path point is already an endpoint predictor after
        # the privileged teacher has learned the complete-reach target.
        with torch.no_grad():
            self.endpoint_3d_head.weight.copy_(
                teacher.trajectory_head.weight[-3:]
            )
            self.endpoint_3d_head.bias.copy_(
                teacher.trajectory_head.bias[-3:]
            )

    def forward(
        self,
        factor_latent: torch.Tensor,
        decoder_latent: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(factor_latent, decoder_latent, horizon_ms)
        outputs["endpoint_3d"] = self.trajectory_limit_m * torch.tanh(
            self.endpoint_3d_head(outputs["motion_context"])
        )
        # Explicit names make the invariant clear to live consumers.  The
        # inherited ``trajectory`` key is retained for the established losses.
        outputs["complete_trajectory"] = outputs["trajectory"]
        return outputs


class CompleteReachDistillationModel(RollingDualHeadDistillationModel):
    """Causal EMG+IMU student predicting a cutoff-invariant complete reach."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.endpoint_decoder = CompleteReachDecoder(config)
