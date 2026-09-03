"""Teacher-aligned deterministic reach model with soft task routing.

The deployed student consumes only causal EMG, IMU, and a padding mask.  VIVE
is confined to the privileged teacher and supervised labels.  Both task heads
see both wearable representations in the forward pass; cross-task gradients
are attenuated rather than removed, preserving useful co-adaptation without
letting either task dominate the other representation.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .complete_reach_distillation import CompleteReachDistillationModel


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``value`` unchanged while multiplying its backward gradient.

    ``scale=0`` is a stop-gradient boundary and ``scale=1`` is ordinary joint
    training.  Keeping the forward value identical makes routing ablations
    directly comparable at inference time.
    """
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"gradient scale must be in [0, 1], got {scale}")
    return value.detach() + scale * (value - value.detach())


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class SoftRoutedReachHeads(nn.Module):
    """Direct IMU-base path plus a softly routed EMG-intent correction."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model["soft_routed_complete_reach"]
        width = int(model["d_model"])
        hidden = int(settings.get("hidden", width))
        dropout = float(settings.get("dropout", model["dropout"]))
        self.steps = int(model["teacher_trajectory_steps"])
        self.correction_limit_m = float(settings.get("correction_limit_m", 0.20))
        if self.correction_limit_m <= 0.0:
            raise ValueError("correction_limit_m must be positive")
        self.intent_gradient_scale = float(
            settings.get("trajectory_intent_gradient_scale", 0.10)
        )
        self.correction_adapter = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.path_correction_head = nn.Linear(hidden, self.steps * 3)
        self.endpoint_correction_head = nn.Linear(hidden, 3)
        # Auxiliary signed-axis prediction makes mirrored up/down and
        # left/right reaches explicitly expensive during training.
        self.axis_direction_head = nn.Linear(hidden, 3 * 3)
        nn.init.zeros_(self.path_correction_head.weight)
        nn.init.zeros_(self.path_correction_head.bias)
        nn.init.zeros_(self.endpoint_correction_head.weight)
        nn.init.zeros_(self.endpoint_correction_head.bias)
        initial = float(settings.get("correction_gate_initial", 0.10))
        self.path_correction_logit = nn.Parameter(torch.tensor(_gate_logit(initial)))
        self.endpoint_correction_logit = nn.Parameter(
            torch.tensor(_gate_logit(initial))
        )

    @staticmethod
    def _progress(reference: torch.Tensor) -> torch.Tensor:
        progress = torch.linspace(
            0.0,
            1.0,
            reference.size(1),
            device=reference.device,
            dtype=reference.dtype,
        )
        progress = progress.square() * (3.0 - 2.0 * progress)
        return progress.view(1, -1, 1)

    def forward(
        self,
        intent_context: torch.Tensor,
        motion_context: torch.Tensor,
        imu_base: torch.Tensor,
        *,
        intent_gradient_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        gradient_scale = (
            self.intent_gradient_scale
            if intent_gradient_scale is None
            else float(intent_gradient_scale)
        )
        routed_intent = scale_gradient(intent_context, gradient_scale)
        correction_context = self.correction_adapter(
            torch.cat([motion_context, routed_intent], dim=-1)
        )
        path_correction = self.correction_limit_m * torch.tanh(
            self.path_correction_head(correction_context).reshape(
                -1, self.steps, 3
            )
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
        # The explicit endpoint and decoded path always agree exactly.
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )
        return {
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "imu_base_trajectory": base,
            "path_correction": path_correction,
            "endpoint_correction": endpoint_correction,
            "path_correction_gate": path_gate.expand(intent_context.size(0)),
            "endpoint_correction_gate": endpoint_gate.expand(
                intent_context.size(0)
            ),
            "axis_direction_logits": self.axis_direction_head(
                correction_context
            ).reshape(-1, 3, 3),
            "trajectory_intent_gradient_scale": torch.full(
                (intent_context.size(0),),
                gradient_scale,
                device=intent_context.device,
                dtype=intent_context.dtype,
            ),
            # Compatibility with the established direction diagnostics.
            "intent_to_motion_gate": torch.full(
                (intent_context.size(0),),
                gradient_scale,
                device=intent_context.device,
                dtype=intent_context.dtype,
            ),
        }


class SoftRoutedCompleteReachModel(CompleteReachDistillationModel):
    """One wearable student with teacher bridge and soft gradient ownership."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        settings = config["model"]["soft_routed_complete_reach"]
        self.screen_motion_gradient_scale = float(
            settings.get("screen_motion_gradient_scale", 0.10)
        )
        self.screen_residual_gradient_scale = float(
            settings.get("screen_residual_gradient_scale", 0.10)
        )
        # Validate eagerly so configuration errors fail before training.
        for scale in (
            self.screen_motion_gradient_scale,
            self.screen_residual_gradient_scale,
        ):
            if not 0.0 <= scale <= 1.0:
                raise ValueError(f"gradient scale must be in [0, 1], got {scale}")
        factors = config["model"]["factor_latent"]
        self.intent_dim = int(factors["intent_dim"])
        self.motion_end = self.intent_dim + int(factors["motion_dim"])
        self.student.soft_routed_reach_heads = SoftRoutedReachHeads(config)

    def _screen_factor(self, factor: torch.Tensor) -> torch.Tensor:
        """Preserve all factors while attenuating non-intent gradients."""
        return torch.cat(
            [
                factor[:, : self.intent_dim],
                scale_gradient(
                    factor[:, self.intent_dim : self.motion_end],
                    self.screen_motion_gradient_scale,
                ),
                scale_gradient(
                    factor[:, self.motion_end :],
                    self.screen_residual_gradient_scale,
                ),
            ],
            dim=-1,
        )

    def _screen_decode(
        self, factor: torch.Tensor, *, emg_only: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        # The EMG-only auxiliary must train the complete EMG route, whereas the
        # fused screen route softly owns intent and only weakly updates motion.
        routed = factor if emg_only else self._screen_factor(factor)
        bridge = (
            self.student.emg_teacher_latent_bridge
            if emg_only
            else self.student.teacher_latent_bridge
        )
        decoder_latent = bridge(routed)
        guidance = self.guidance(routed)
        decoded = self.student.endpoint_decoder(
            routed,
            decoder_latent,
            guidance["horizon_expected_ms"],
        )
        return decoded, decoder_latent, guidance

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
        screen, decoder_latent, guidance = self._screen_decode(factor)
        motion = self.student.soft_routed_reach_heads(
            encoded["intent_context"],
            encoded["motion_context"],
            encoded["fused_imu_trajectory"],
        )
        outputs: dict[str, Any] = {
            **screen,
            **motion,
            "latent": decoder_latent,
            "mu": decoder_latent,
            "factor_latent": factor,
            "decoder_latent": decoder_latent,
            "log_variance": torch.zeros_like(decoder_latent),
            "imu_trajectory": encoded["imu_trajectory"],
            "channel_attention": encoded["channel_attention"],
            "lag_attention": encoded["lag_attention"],
            "emg_from_imu_attention": encoded["emg_from_imu_attention"],
            "imu_from_emg_attention": encoded["imu_from_emg_attention"],
            "guidance": guidance,
        }
        if include_emg_only:
            emg_factor = encoded["emg_mu"]
            emg_screen, emg_latent, emg_guidance = self._screen_decode(
                emg_factor, emg_only=True
            )
            zero_motion = torch.zeros_like(encoded["motion_context"])
            zero_base = torch.zeros_like(encoded["fused_imu_trajectory"])
            # The EMG-only auxiliary uses a full-strength gradient so it
            # maintains a genuinely useful intent pathway.
            emg_motion = self.student.soft_routed_reach_heads(
                encoded["intent_context"],
                zero_motion,
                zero_base,
                intent_gradient_scale=1.0,
            )
            outputs["emg_only"] = {
                **emg_screen,
                **emg_motion,
                "latent": emg_latent,
                "mu": emg_latent,
                "factor_latent": emg_factor,
                "decoder_latent": emg_latent,
                "log_variance": torch.zeros_like(emg_latent),
                "guidance": emg_guidance,
            }
        return outputs
