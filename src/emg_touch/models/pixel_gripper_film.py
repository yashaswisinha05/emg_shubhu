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
        result["gripper_temperature"] = temperature
        return result
