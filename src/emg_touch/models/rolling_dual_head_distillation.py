"""Rolling wearable inference with explicitly separated screen and 3D heads.

The causal temporal encoder remains the proven teacher-bridge backbone.  Its
factor latent already assigns intent to an EMG-owned block and motion to an
IMU-owned block.  This module stops decoding both tasks from one shared hidden
vector:

* the screen branch is driven primarily by the intent block;
* the trajectory branch is driven primarily by the motion block and the
  wearables-predicted time-to-touch;
* each branch receives a small learnable residual from teacher decoder space.

``student_forward`` accepts only EMG, IMU, and a causal mask.  VIVE trajectory,
screen target, and true time-to-touch remain training/evaluation labels.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .grid_point import (
    SpatialPointHead,
    decode_grid_outputs,
    finalize_point_prediction,
)
from .latent_distillation import SharedIntentDecoder
from .teacher_bridge_distillation import TeacherBridgeDistillationModel


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class RollingDualHeadDecoder(nn.Module):
    """Specialised screen and motion decoders with gated shared residuals."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        factors = model["factor_latent"]
        settings = model.get("rolling_dual_head", {})
        latent_dim = int(model["latent_dim"])
        self.intent_dim = int(factors["intent_dim"])
        self.motion_dim = int(factors["motion_dim"])
        self.motion_start = self.intent_dim
        self.motion_end = self.motion_start + self.motion_dim
        if self.motion_end > latent_dim:
            raise ValueError("intent and motion dimensions exceed latent_dim")

        hidden = int(settings.get("hidden", model.get("decoder_width", 128)))
        dropout = float(settings.get("dropout", model["dropout"]))
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.trajectory_steps = int(model["teacher_trajectory_steps"])
        self.trajectory_limit_m = float(model.get("trajectory_limit_m", 0.8))
        centers = model.get("horizon_latent", {}).get(
            "bins_ms", [50.0, 100.0, 200.0, 300.0, 400.0]
        )
        self.horizon_scale_ms = max(map(float, centers))

        def branch(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.screen_semantic = branch(self.intent_dim)
        self.screen_shared = branch(latent_dim)
        self.screen_norm = nn.LayerNorm(hidden)
        self.screen_shared_logit = nn.Parameter(torch.tensor(
            _gate_logit(float(settings.get("screen_shared_initial", 0.20)))
        ))
        self.point_head = SpatialPointHead(
            hidden,
            grid_width,
            grid_height,
            dropout,
            direct_prediction=True,
            zero_initialize=False,
        )

        self.motion_semantic = branch(self.motion_dim)
        self.motion_shared = branch(latent_dim)
        self.horizon_embedding = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.motion_norm = nn.LayerNorm(hidden)
        self.motion_shared_logit = nn.Parameter(torch.tensor(
            _gate_logit(float(settings.get("motion_shared_initial", 0.25)))
        ))
        self.trajectory_head = nn.Linear(hidden, self.trajectory_steps * 3)

    def initialise_output_heads(self, teacher: SharedIntentDecoder) -> None:
        """Warm-start shared projections and outputs from the trained teacher."""
        # These branches intentionally have the same layout as the teacher
        # trunk.  The semantic intent/motion paths then learn task-specific
        # residual information instead of both heads starting from scratch.
        self.screen_shared.load_state_dict(teacher.trunk.state_dict())
        self.motion_shared.load_state_dict(teacher.trunk.state_dict())
        self.point_head.load_state_dict(teacher.point_head.state_dict())
        self.trajectory_head.load_state_dict(teacher.trajectory_head.state_dict())

    def _horizon_features(self, horizon_ms: torch.Tensor) -> torch.Tensor:
        normalized = (horizon_ms / self.horizon_scale_ms).clamp(0.0, 4.0)
        return torch.stack(
            [normalized, torch.log1p(normalized), torch.sqrt(normalized + 1e-6)],
            dim=-1,
        )

    def forward(
        self,
        factor_latent: torch.Tensor,
        decoder_latent: torch.Tensor,
        horizon_ms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intent = factor_latent[:, : self.intent_dim]
        motion = factor_latent[:, self.motion_start : self.motion_end]
        screen_gate = torch.sigmoid(self.screen_shared_logit)
        motion_gate = torch.sigmoid(self.motion_shared_logit)
        screen_context = self.screen_norm(
            self.screen_semantic(intent)
            + screen_gate * self.screen_shared(decoder_latent)
        )
        motion_context = self.motion_norm(
            self.motion_semantic(motion)
            + motion_gate * self.motion_shared(decoder_latent)
            + self.horizon_embedding(self._horizon_features(horizon_ms))
        )

        outputs = self.point_head(screen_context)
        outputs.update(decode_grid_outputs(
            outputs["heatmap_logits"],
            outputs["offset_logits"],
            self.grid_width,
            self.grid_height,
        ))
        finalize_point_prediction(outputs)
        outputs["trajectory"] = self.trajectory_limit_m * torch.tanh(
            self.trajectory_head(motion_context).reshape(
                -1, self.trajectory_steps, 3
            )
        )
        outputs["screen_context"] = screen_context
        outputs["motion_context"] = motion_context
        outputs["screen_shared_gate"] = screen_gate.expand(factor_latent.size(0))
        outputs["motion_shared_gate"] = motion_gate.expand(factor_latent.size(0))
        return outputs


class RollingDualHeadDistillationModel(TeacherBridgeDistillationModel):
    """Teacher-bridge student with task-owned rolling prediction heads."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.endpoint_decoder = RollingDualHeadDecoder(config)

    def initialise_student_decoder_from_teacher(self) -> None:
        self.student.endpoint_decoder.initialise_output_heads(self.decoder)

    def _student_decode(
        self, factor_latent: torch.Tensor, decoder_latent: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        guidance = self.guidance(factor_latent)
        outputs = self.student.endpoint_decoder(
            factor_latent,
            decoder_latent,
            guidance["horizon_expected_ms"],
        )
        return outputs, guidance

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
        """Predict screen XY and relative 3D motion from rolling wearables."""
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
        factor_latent = encoded["mu"]
        emg_factor_latent = encoded["emg_mu"]
        decoder_latent = self.student.teacher_latent_bridge(factor_latent)
        emg_decoder_latent = self.student.emg_teacher_latent_bridge(
            emg_factor_latent
        )
        decoded, guidance = self._student_decode(
            factor_latent, decoder_latent
        )
        outputs: dict[str, Any] = {
            **decoded,
            "latent": decoder_latent,
            "mu": decoder_latent,
            "factor_latent": factor_latent,
            "decoder_latent": decoder_latent,
            "log_variance": encoded["log_variance"],
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": guidance,
        }
        if include_emg_only:
            emg_decoded, emg_guidance = self._student_decode(
                emg_factor_latent, emg_decoder_latent
            )
            outputs["emg_only"] = {
                **emg_decoded,
                "latent": emg_decoder_latent,
                "mu": emg_decoder_latent,
                "factor_latent": emg_factor_latent,
                "decoder_latent": emg_decoder_latent,
                "log_variance": encoded["emg_log_variance"],
                "guidance": emg_guidance,
            }
        return outputs
