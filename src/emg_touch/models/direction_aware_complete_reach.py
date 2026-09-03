"""Direction-aware complete-reach decoding from causal EMG+IMU.

The complete-reach baseline keeps screen intent and 3D motion deliberately
separate.  This successor adds a small gated intent-to-motion residual so the
target-specific EMG context can disambiguate up/down and left/right motion,
plus an explicit three-class direction head for every VIVE axis.
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


class DirectionAwareCompleteReachDecoder(CompleteReachDecoder):
    """Fuse intent into motion and classify signed displacement per axis."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model = config["model"]
        settings = model["direction_aware_complete_reach"]
        hidden = int(
            model.get("rolling_dual_head", {}).get(
                "hidden", model.get("decoder_width", 128)
            )
        )
        dropout = float(settings.get("dropout", model["dropout"]))
        self.intent_to_motion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.intent_to_motion_logit = nn.Parameter(torch.tensor(
            _gate_logit(float(settings.get("intent_to_motion_initial", 0.15)))
        ))
        self.direction_motion_norm = nn.LayerNorm(hidden)
        # [batch, axis=(x,y,z), class=(negative, stationary, positive)]
        self.axis_direction_head = nn.Linear(hidden, 3 * 3)

    def forward(
        self,
        factor_latent: torch.Tensor,
        decoder_latent: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(factor_latent, decoder_latent, horizon_ms)
        gate = torch.sigmoid(self.intent_to_motion_logit)
        context = self.direction_motion_norm(
            outputs["motion_context"]
            + gate * self.intent_to_motion(outputs["screen_context"])
        )
        # Recompute all 3D outputs from the direction-aware context.  The
        # inherited path and endpoint heads retain their teacher warm start.
        outputs["trajectory"] = self.trajectory_limit_m * torch.tanh(
            self.trajectory_head(context).reshape(
                -1, self.trajectory_steps, 3
            )
        )
        outputs["complete_trajectory"] = outputs["trajectory"]
        outputs["endpoint_3d"] = self.trajectory_limit_m * torch.tanh(
            self.endpoint_3d_head(context)
        )
        outputs["axis_direction_logits"] = self.axis_direction_head(
            context
        ).reshape(-1, 3, 3)
        outputs["motion_context"] = context
        outputs["intent_to_motion_gate"] = gate.expand(factor_latent.size(0))
        return outputs


class DirectionAwareCompleteReachModel(CompleteReachDistillationModel):
    """Complete-reach student with explicit signed-direction supervision."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.endpoint_decoder = DirectionAwareCompleteReachDecoder(config)
