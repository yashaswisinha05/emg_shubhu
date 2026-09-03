"""Hard direction-constrained complete-reach decoding from EMG+IMU.

The direction-aware predecessor only *penalised* mirrored motion, so its
continuous trajectory head could still emit a locally reversed segment.  This
isolated successor changes the trajectory parameterisation itself:

* a straight-through categorical decision selects -1, 0, or +1 per VIVE axis;
* an explicit screen-intent residual helps choose that direction;
* the endpoint magnitude is non-negative and receives that selected sign;
* positive increments are accumulated and normalised from zero to one;
* the complete path is progress multiplied by the signed endpoint.

Consequently every axis begins at zero, ends exactly at the explicit endpoint,
and can never step away from that endpoint.  VIVE remains supervision only.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .direction_aware_complete_reach import (
    DirectionAwareCompleteReachDecoder,
    DirectionAwareCompleteReachModel,
)


class MonotonicCompleteReachDecoder(DirectionAwareCompleteReachDecoder):
    """Construct a hard-sign, axis-monotonic onset-to-touch trajectory."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model = config["model"]
        settings = model.get("monotonic_complete_reach", {})
        hidden = int(
            model.get("rolling_dual_head", {}).get(
                "hidden", model.get("decoder_width", 128)
            )
        )
        self.direction_temperature = float(
            settings.get("direction_temperature", 1.0)
        )
        if self.direction_temperature <= 0.0:
            raise ValueError("direction_temperature must be positive")
        self.minimum_increment = float(settings.get("minimum_increment", 1e-4))
        if self.minimum_increment < 0.0:
            raise ValueError("minimum_increment cannot be negative")
        self.progress_increment_head = nn.Linear(
            hidden, (self.trajectory_steps - 1) * 3
        )
        # This bypasses the predecessor's small scalar intent-to-motion gate
        # for the discrete decision only. It gives EMG-owned pointing intent a
        # direct opportunity to disambiguate up/down and left/right signs.
        self.axis_direction_screen_head = nn.Linear(hidden, 3 * 3)
        nn.init.zeros_(self.axis_direction_screen_head.weight)
        nn.init.zeros_(self.axis_direction_screen_head.bias)
        # Equal initial logits produce a smooth constant-speed path. Training
        # then only has to learn deviations from that stable starting shape.
        nn.init.zeros_(self.progress_increment_head.weight)
        nn.init.zeros_(self.progress_increment_head.bias)
        self.register_buffer(
            "axis_direction_values",
            torch.tensor([-1.0, 0.0, 1.0]),
            persistent=True,
        )

    def _hard_axis_signs(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard forward signs with softmax gradients for training."""
        probabilities = torch.softmax(
            logits / self.direction_temperature, dim=-1
        )
        hard = F.one_hot(
            probabilities.argmax(dim=-1), num_classes=3
        ).to(probabilities.dtype)
        if self.training:
            selection = hard + probabilities - probabilities.detach()
        else:
            selection = hard
        signs = (selection * self.axis_direction_values).sum(dim=-1)
        return signs, probabilities

    def forward(
        self,
        factor_latent: torch.Tensor,
        decoder_latent: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(factor_latent, decoder_latent, horizon_ms)
        context = outputs["motion_context"]
        raw_endpoint = outputs["endpoint_3d"]
        outputs["axis_direction_logits"] = (
            outputs["axis_direction_logits"]
            + self.axis_direction_screen_head(outputs["screen_context"]).reshape(
                -1, 3, 3
            )
        )
        signs, probabilities = self._hard_axis_signs(
            outputs["axis_direction_logits"]
        )

        # Reuse the teacher-warm-started endpoint head for distance while the
        # explicit categorical head owns direction.  abs(tanh(.)) is bounded
        # by trajectory_limit_m and keeps the prior endpoint scale useful.
        endpoint = signs * raw_endpoint.abs()
        raw_increments = self.progress_increment_head(context).reshape(
            -1, self.trajectory_steps - 1, 3
        )
        increments = F.softplus(raw_increments) + self.minimum_increment
        cumulative = torch.cumsum(increments, dim=1)
        progress = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        progress = torch.cat(
            [torch.zeros_like(progress[:, :1]), progress], dim=1
        )
        trajectory = progress * endpoint[:, None, :]

        outputs["unconstrained_endpoint_3d"] = raw_endpoint
        outputs["endpoint_3d"] = endpoint
        outputs["trajectory"] = trajectory
        outputs["complete_trajectory"] = trajectory
        outputs["axis_direction_probabilities"] = probabilities
        outputs["axis_direction_signs"] = signs
        outputs["trajectory_progress"] = progress
        return outputs


class MonotonicCompleteReachModel(DirectionAwareCompleteReachModel):
    """Wearable student whose decoded 3D path cannot reverse per axis."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.endpoint_decoder = MonotonicCompleteReachDecoder(config)
