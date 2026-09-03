"""Teacher-space bridge and student-owned decoder for wearable inference.

The temporal student deliberately factorises EMG intent, IMU motion, and
cross-modal residual information.  Those coordinates need not match the
privileged teacher VAE coordinates, so feeding them into the frozen teacher
decoder creates an avoidable domain mismatch.  This model retains the
factorised latent for guidance and adds:

* a residual adapter into teacher decoder space;
* a student-owned copy of the teacher decoder that can adapt safely;
* no semantic endpoint residual (it hurt held-out error in the parent run).

The trained student still accepts causal EMG, IMU, and a time mask only.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .channel_horizon_distillation import (
    ChannelHorizonLatentDistillationModel,
)
from .temporal_cross_attention_distillation import (
    TemporalCrossAttentionStudentEncoder,
)


class ResidualTeacherLatentBridge(nn.Module):
    """Learn a stable correction from factor latents to teacher coordinates."""

    def __init__(self, latent_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, factor_latent: torch.Tensor) -> torch.Tensor:
        return factor_latent + self.network(factor_latent)


class TeacherBridgeDistillationModel(ChannelHorizonLatentDistillationModel):
    """Temporal factor student with a separately adaptable endpoint decoder."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        model = config["model"]
        steps = int(model["teacher_trajectory_steps"])
        self.student = TemporalCrossAttentionStudentEncoder(
            config, emg_channels, imu_channels, trajectory_steps=steps
        )
        latent_dim = int(model["latent_dim"])
        bridge = model.get("teacher_bridge", {})
        hidden = int(bridge.get("hidden", 128))
        dropout = float(bridge.get("dropout", model["dropout"]))
        self.student.teacher_latent_bridge = ResidualTeacherLatentBridge(
            latent_dim, hidden, dropout
        )
        self.student.emg_teacher_latent_bridge = ResidualTeacherLatentBridge(
            latent_dim, hidden, dropout
        )
        # This copy is refreshed from the best trained teacher immediately
        # before student training. It then adapts without changing the oracle.
        self.student.endpoint_decoder = copy.deepcopy(self.decoder)

    def initialise_student_decoder_from_teacher(self) -> None:
        self.student.endpoint_decoder.load_state_dict(self.decoder.state_dict())

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
        """Decode deterministic wearable factors through student-owned heads."""
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
        outputs: dict[str, Any] = {
            **self.student.endpoint_decoder(decoder_latent),
            "latent": decoder_latent,
            # ``mu`` remains the trainer's teacher-comparison field. The
            # explicitly named factor latent continues to drive guidance.
            "mu": decoder_latent,
            "factor_latent": factor_latent,
            "decoder_latent": decoder_latent,
            "log_variance": encoded["log_variance"],
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": self.guidance(factor_latent),
        }
        if include_emg_only:
            outputs["emg_only"] = {
                **self.student.endpoint_decoder(emg_decoder_latent),
                "latent": emg_decoder_latent,
                "mu": emg_decoder_latent,
                "factor_latent": emg_factor_latent,
                "decoder_latent": emg_decoder_latent,
                "log_variance": encoded["emg_log_variance"],
                "guidance": self.guidance(emg_factor_latent),
            }
        return outputs
