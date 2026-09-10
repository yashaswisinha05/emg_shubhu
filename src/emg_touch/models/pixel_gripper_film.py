"""Small participant adapter restricted to pixel and gripper predictions."""
from __future__ import annotations

import torch
from torch import nn


class GroupedFiLM(nn.Module):
    """Identity-initialized feature modulation with few participant parameters."""

    def __init__(self, dimension, groups=16):
        super().__init__()
        if dimension % groups:
            raise ValueError("feature dimension must be divisible by FiLM groups")
        self.dimension, self.groups = dimension, groups
        self.gamma_delta = nn.Parameter(torch.zeros(groups))
        self.beta = nn.Parameter(torch.zeros(groups))

    def forward(self, values):
        repeats = self.dimension // self.groups
        gamma = (1 + self.gamma_delta).repeat_interleave(repeats)
        beta = self.beta.repeat_interleave(repeats)
        return values * gamma + beta


class PixelGripperFiLM(nn.Module):
    """Freeze a base model and adapt only its pixel and gripper branches."""

    def __init__(self, base_model, groups=16):
        super().__init__()
        self.base = base_model.eval().requires_grad_(False)
        dimension = base_model.context_fusion[0].out_features
        self.pixel_film = GroupedFiLM(dimension, groups)
        self.gripper_film = GroupedFiLM(dimension, groups)
        self.pixel_matrix_delta = nn.Parameter(torch.zeros(2, 2))
        self.pixel_bias = nn.Parameter(torch.zeros(2))
        self.gripper_log_temperature = nn.Parameter(torch.zeros(()))
        self.gripper_bias = nn.Parameter(torch.zeros(2))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def calibration_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("base.")]

    def calibration_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if not name.startswith("base.")}

    def load_calibration_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if not name.startswith("base.")]
        if missing or unexpected:
            raise ValueError(f"invalid calibration state: missing={missing}, "
                             f"unexpected={unexpected}")

    def regularization(self):
        terms = [self.pixel_film.gamma_delta, self.pixel_film.beta,
                 self.gripper_film.gamma_delta, self.gripper_film.beta,
                 self.pixel_matrix_delta, self.pixel_bias,
                 self.gripper_log_temperature, self.gripper_bias]
        return sum(value.square().mean() for value in terms)

    def forward(self, emg, imu):
        with torch.no_grad():
            result = self.base(emg, imu)
        context = result["context_features"].detach()

        adapted_pixel = self.pixel_film(context)
        direct = self.base.click_direct(adapted_pixel).sigmoid()
        endpoint = self.base.endpoint_to_click(
            result["final_position"].detach()).sigmoid()
        blend = self.base.click_blend_logit.sigmoid()
        raw_click = torch.lerp(direct, endpoint, blend)
        matrix = torch.eye(2, device=raw_click.device) + self.pixel_matrix_delta
        result["click_uncalibrated"] = result["click"]
        result["click"] = raw_click @ matrix.transpose(0, 1) + self.pixel_bias

        adapted_gripper = self.gripper_film(context)
        delta_logits = (self.base.gripper_context(adapted_gripper)
                        - self.base.gripper_context(context))
        temperature = self.gripper_log_temperature.exp().clamp(.1, 10.)
        result["gripper_state_logits_uncalibrated"] = result["gripper_state_logits"]
        result["gripper_state_logits"] = (
            result["gripper_state_logits"] + delta_logits
        ) / temperature + self.gripper_bias
        result["gripper_calibration_features"] = adapted_gripper
        result["gripper_temperature"] = temperature
        return result


class GripperHardContrastiveFiLM(PixelGripperFiLM):
    """Gripper-only FiLM plus a zero-initialized low-rank residual adapter."""

    def __init__(self, base_model, groups=16, rank=8):
        super().__init__(base_model, groups)
        dimension = base_model.context_fusion[0].out_features
        if rank <= 0 or rank > dimension:
            raise ValueError("adapter rank must be in [1, feature dimension]")
        self.gripper_residual = nn.Sequential(
            nn.Linear(dimension, rank, bias=False), nn.GELU(),
            nn.Linear(rank, dimension, bias=False))
        nn.init.zeros_(self.gripper_residual[-1].weight)
        # This variant must leave pixels exactly unchanged.
        for parameter in (self.pixel_film.gamma_delta, self.pixel_film.beta,
                          self.pixel_matrix_delta, self.pixel_bias):
            parameter.requires_grad_(False)

    def gripper_calibration_parameters(self):
        names = ("gripper_film.", "gripper_residual.",
                 "gripper_log_temperature", "gripper_bias")
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("base.") and name.startswith(names)
                and parameter.requires_grad]

    def regularization(self):
        values = [self.gripper_film.gamma_delta, self.gripper_film.beta,
                  self.gripper_log_temperature, self.gripper_bias]
        residual = self.gripper_residual[-1].weight
        return sum(value.square().mean() for value in values) + residual.square().mean()

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        context = result["context_features"].detach()
        features = self.gripper_film(context)
        features = features + self.gripper_residual(features)
        delta_logits = (self.base.gripper_context(features)
                        - self.base.gripper_context(context))
        temperature = self.gripper_log_temperature.exp().clamp(.1, 10.)
        result["gripper_state_logits"] = (
            result["gripper_state_logits_uncalibrated"] + delta_logits
        ) / temperature + self.gripper_bias
        result["gripper_calibration_features"] = features
        return result


class GripperLogitCalibration(nn.Module):
    """Two-parameter post-hoc calibration; the base classifier stays frozen."""

    def __init__(self, base_model, log_temperature=0., close_bias=0.):
        super().__init__()
        self.base = base_model.eval().requires_grad_(False)
        self.log_temperature = nn.Parameter(torch.tensor(float(log_temperature)))
        self.close_bias = nn.Parameter(torch.tensor(float(close_bias)))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, emg, imu):
        with torch.no_grad():
            result = self.base(emg, imu)
        temperature = self.log_temperature.exp().clamp(.05, 20.)
        bias = torch.stack((-self.close_bias / 2, self.close_bias / 2))
        result["gripper_state_logits_uncalibrated"] = result["gripper_state_logits"]
        result["gripper_state_logits"] = result["gripper_state_logits"] / temperature + bias
        result["gripper_temperature"] = temperature
        result["gripper_close_bias"] = self.close_bias
        return result
