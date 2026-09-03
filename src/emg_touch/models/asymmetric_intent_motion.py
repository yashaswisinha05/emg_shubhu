"""One causal model with asymmetric intent-motion information routing.

Screen inference sees EMG intent plus detached IMU/cross-modal factors. 3D
inference sees IMU motion plus detached EMG intent. Thus both predictions use
both modalities in the forward pass while task gradients remain physiologically
owned: screen -> EMG intent, trajectory -> IMU motion.
"""
from __future__ import annotations

from typing import Any

import torch

from .complete_reach_distillation import CompleteReachDistillationModel
from .deterministic_complete_reach import DeterministicReachHeads


class AsymmetricIntentMotionModel(CompleteReachDistillationModel):
    """Unified screen+3D model with explicit cross-task gradient boundaries."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.asymmetric_motion_heads = DeterministicReachHeads(config)
        factors = config["model"]["factor_latent"]
        self.intent_dim = int(factors["intent_dim"])
        self.motion_dim = int(factors["motion_dim"])
        self.motion_end = self.intent_dim + self.motion_dim

    def _screen_factor(self, factor: torch.Tensor) -> torch.Tensor:
        """Keep forward information but route screen gradients to intent only."""
        return torch.cat([
            factor[:, : self.intent_dim],
            factor[:, self.intent_dim : self.motion_end].detach(),
            factor[:, self.motion_end :].detach(),
        ], dim=-1)

    def _screen_decode(
        self, factor: torch.Tensor, emg_only: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        routed = factor if emg_only else self._screen_factor(factor)
        bridge = (
            self.student.emg_teacher_latent_bridge
            if emg_only else self.student.teacher_latent_bridge
        )
        decoder_latent = bridge(routed)
        routed_guidance = self.guidance(routed)
        decoded = self.student.endpoint_decoder(
            routed,
            decoder_latent,
            routed_guidance["horizon_expected_ms"],
        )
        return decoded, decoder_latent

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
            emg,
            imu,
            time_mask,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        factor = encoded["mu"]
        screen, screen_latent = self._screen_decode(factor)
        motion = self.student.asymmetric_motion_heads(
            # DeterministicReachHeads detaches intent internally before 3D
            # correction, so trajectory gradients remain in the motion route.
            encoded["intent_context"],
            encoded["motion_context"],
            encoded["fused_imu_trajectory"],
        )
        outputs: dict[str, Any] = {
            **screen,
            **motion,
            "latent": screen_latent,
            "mu": screen_latent,
            "factor_latent": factor,
            "decoder_latent": screen_latent,
            "log_variance": torch.zeros_like(screen_latent),
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            # Auxiliary probes use the complete factors and do not define the
            # task-routing boundary.
            "guidance": self.guidance(factor),
        }
        if include_emg_only:
            emg_factor = encoded["emg_mu"]
            emg_screen, emg_latent = self._screen_decode(
                emg_factor, emg_only=True
            )
            zero_motion = torch.zeros_like(encoded["motion_context"])
            zero_base = torch.zeros_like(encoded["fused_imu_trajectory"])
            emg_motion = self.student.asymmetric_motion_heads(
                encoded["intent_context"], zero_motion, zero_base
            )
            outputs["emg_only"] = {
                **emg_screen,
                **emg_motion,
                "latent": emg_latent,
                "mu": emg_latent,
                "factor_latent": emg_factor,
                "decoder_latent": emg_latent,
                "log_variance": torch.zeros_like(emg_latent),
                "guidance": self.guidance(emg_factor),
            }
        return outputs
