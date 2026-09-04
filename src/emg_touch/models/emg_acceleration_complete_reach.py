"""EMG-driven acceleration dynamics on top of the best complete-reach model.

The temporal EMG residual model remains the base.  This successor predicts a
bounded *acceleration* residual from causal EMG tokens and integrates it twice
to obtain a trajectory correction.  Its final projection is zero-initialized,
so loading an EMG-residual checkpoint exactly preserves the old prediction
until the dynamics branch earns a correction during training.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .emg_residual_complete_reach import EMGResidualCompleteReachModel


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


def finite_difference(values: torch.Tensor, duration_s: torch.Tensor) -> torch.Tensor:
    """Differentiate a BxTxD sequence using one duration per sequence."""
    if values.size(1) < 2:
        return torch.zeros_like(values)
    dt = duration_s.clamp_min(1e-3) / float(values.size(1) - 1)
    dt = dt[:, None, None]
    result = torch.empty_like(values)
    result[:, 0] = (values[:, 1] - values[:, 0]) / dt[:, 0]
    result[:, -1] = (values[:, -1] - values[:, -2]) / dt[:, 0]
    if values.size(1) > 2:
        result[:, 1:-1] = (
            values[:, 2:] - values[:, :-2]
        ) / (2.0 * dt)
    return result


def trajectory_kinematics(
    trajectory: torch.Tensor, duration_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = finite_difference(trajectory, duration_s)
    acceleration = finite_difference(velocity, duration_s)
    return velocity, acceleration


class EMGAccelerationDynamicsHead(nn.Module):
    """Cross-attend to EMG and integrate its acceleration contribution."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model.get("emg_acceleration_dynamics", {})
        width = int(model["d_model"])
        self.steps = int(model["teacher_trajectory_steps"])
        self.acceleration_limit_mps2 = float(
            settings.get("acceleration_limit_mps2", 4.0)
        )
        self.correction_limit_m = float(settings.get("correction_limit_m", 0.12))
        self.minimum_duration_s = float(settings.get("minimum_duration_s", 0.45))
        self.maximum_duration_s = float(settings.get("maximum_duration_s", 1.60))
        if self.acceleration_limit_mps2 <= 0.0 or self.correction_limit_m <= 0.0:
            raise ValueError("acceleration and correction limits must be positive")
        if self.maximum_duration_s <= self.minimum_duration_s:
            raise ValueError("maximum_duration_s must exceed minimum_duration_s")

        heads = int(settings.get("num_heads", model["num_heads"]))
        dropout = float(settings.get("dropout", model["dropout"]))
        self.phase_queries = nn.Parameter(torch.empty(self.steps, width))
        nn.init.normal_(self.phase_queries, std=0.02)
        self.motion_query = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(width)
        self.context_mlp = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.acceleration_head = nn.Linear(width, 3)
        self.acceleration_gate_head = nn.Linear(width, 1)
        self.duration_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1),
        )

        # Exact checkpoint preservation at initialization.
        nn.init.zeros_(self.acceleration_head.weight)
        nn.init.zeros_(self.acceleration_head.bias)
        initial = _gate_logit(float(settings.get("correction_gate_initial", 0.20)))
        nn.init.zeros_(self.acceleration_gate_head.weight)
        nn.init.constant_(self.acceleration_gate_head.bias, initial)
        nn.init.zeros_(self.duration_head[-1].weight)
        nn.init.zeros_(self.duration_head[-1].bias)

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        return ~safe

    @staticmethod
    def _integrate(
        acceleration: torch.Tensor, duration_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trapezoidally integrate with zero initial residual state."""
        dt = duration_s.clamp_min(1e-3) / float(acceleration.size(1) - 1)
        dt = dt[:, None, None]
        acceleration_average = 0.5 * (
            acceleration[:, 1:] + acceleration[:, :-1]
        )
        velocity_steps = acceleration_average * dt
        velocity = torch.cat([
            torch.zeros_like(acceleration[:, :1]),
            torch.cumsum(velocity_steps, dim=1),
        ], dim=1)
        velocity_average = 0.5 * (velocity[:, 1:] + velocity[:, :-1])
        position_steps = velocity_average * dt
        position = torch.cat([
            torch.zeros_like(acceleration[:, :1]),
            torch.cumsum(position_steps, dim=1),
        ], dim=1)
        return velocity, position

    def forward(
        self,
        emg_tokens: torch.Tensor,
        emg_token_mask: torch.Tensor,
        motion_context: torch.Tensor,
        base_trajectory: torch.Tensor,
        base_endpoint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = emg_tokens.size(0)
        queries = self.phase_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.motion_query(motion_context).unsqueeze(1)
        attended, attention = self.cross_attention(
            queries,
            emg_tokens,
            emg_tokens,
            key_padding_mask=self._safe_padding_mask(emg_token_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.context_norm(queries + attended)
        context = context + self.context_mlp(context)

        raw_acceleration = self.acceleration_limit_mps2 * torch.tanh(
            self.acceleration_head(context)
        )
        gate = torch.sigmoid(self.acceleration_gate_head(context))
        acceleration = gate * raw_acceleration
        pooled = context.mean(dim=1)
        duration_unit = torch.sigmoid(self.duration_head(pooled)).squeeze(-1)
        duration_s = self.minimum_duration_s + duration_unit * (
            self.maximum_duration_s - self.minimum_duration_s
        )
        residual_velocity, integrated_position = self._integrate(
            acceleration, duration_s
        )
        bounded_position = self.correction_limit_m * torch.tanh(
            integrated_position / self.correction_limit_m
        )
        trajectory = base_trajectory + bounded_position
        endpoint = trajectory[:, -1]
        base_velocity, base_acceleration = trajectory_kinematics(
            base_trajectory, duration_s
        )
        velocity, final_acceleration = trajectory_kinematics(
            trajectory, duration_s
        )
        return {
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "pre_acceleration_trajectory": base_trajectory,
            "pre_acceleration_endpoint": base_endpoint,
            "emg_acceleration_raw": raw_acceleration,
            "emg_acceleration_residual": acceleration,
            "emg_acceleration_gate": gate.squeeze(-1),
            "emg_acceleration_attention": attention,
            "emg_integrated_velocity_residual": residual_velocity,
            "emg_integrated_position_residual": bounded_position,
            "predicted_movement_duration_s": duration_s,
            "pre_acceleration_velocity": base_velocity,
            "pre_acceleration_dynamics": base_acceleration,
            "dynamics_velocity": velocity,
            "dynamics_acceleration": final_acceleration,
        }


class EMGAccelerationCompleteReachModel(EMGResidualCompleteReachModel):
    """Best EMG-residual student plus an integrated acceleration correction."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.emg_acceleration_dynamics_head = EMGAccelerationDynamicsHead(
            config
        )
        self.acceleration_warmup = False

    def train(self, mode: bool = True) -> "EMGAccelerationCompleteReachModel":
        super().train(mode)
        if mode and self.acceleration_warmup:
            self.student.eval()
            self.student.emg_acceleration_dynamics_head.train()
            self.teacher.eval()
            self.decoder.eval()
            if self.guidance is not None:
                self.guidance.eval()
        return self

    def _apply_acceleration_dynamics(
        self,
        encoded: dict[str, torch.Tensor],
        base: dict[str, torch.Tensor],
        *,
        emg_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        motion_context = (
            torch.zeros_like(encoded["motion_context"])
            if emg_only
            else encoded["motion_context"]
        )
        return self.student.emg_acceleration_dynamics_head(
            encoded["emg_tokens"],
            encoded["emg_token_mask"],
            motion_context,
            base["trajectory"],
            base["endpoint_3d"],
        )

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
        # Ask the parent for the established screen and EMG-residual 3D path.
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
        encoded = {
            "emg_tokens": outputs.pop("_emg_tokens"),
            "emg_token_mask": outputs.pop("_emg_token_mask"),
            "motion_context": outputs.pop("_motion_context"),
        }
        dynamics = self._apply_acceleration_dynamics(encoded, outputs)
        outputs.update(dynamics)
        if include_emg_only:
            emg_dynamics = self._apply_acceleration_dynamics(
                encoded, outputs["emg_only"], emg_only=True
            )
            outputs["emg_only"].update(emg_dynamics)
        return outputs
