"""State-attention V2: stronger EMG state and late geometric pixel decoding."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .reach_grasp_patch_transformer import CausalPatchBranch, causal_mask
from .reach_grasp_react_intent import CausalEMGIntent
from .reach_grasp_state_attention import EMGChannelEncoder


class StateConditionedAttentionV2(nn.Module):
    """Position/future backbone with an EMG-only state-conditioned correction."""

    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, future_steps=20,
                 predict_click=True, predict_final_pose=False,
                 react_context=100):
        super().__init__()
        if modality not in {"emg", "imu", "emg+imu"}:
            raise ValueError(f"unknown modality: {modality}")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        if predict_final_pose:
            raise ValueError("V2 intentionally has no endpoint head")
        self.modality, self.future_steps = modality, future_steps
        self.emg = EMGChannelEncoder(width, patch, stride, layers, heads, dropout)
        self.imu = CausalPatchBranch(48, width, patch, stride, layers, heads, dropout)

        # Restore the high-resolution causal decoder used by the strong
        # classifier: local convolution + sustained context + masked EMG branch.
        self.state_local = nn.Sequential(
            nn.Conv1d(width, width, 5, groups=width), nn.GELU(),
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 2, 1))
        self.state_context = nn.Linear(width, 2)
        self.state_blend_logit = nn.Parameter(torch.tensor(-2.))
        self.react = CausalEMGIntent(width=width // 2, heads=heads, layers=layers,
                                     dropout=dropout, context=react_context)
        self.react_current = nn.Linear(width // 2, 2)
        nn.init.zeros_(self.react_current.weight)
        nn.init.zeros_(self.react_current.bias)
        self.imu_state_prior = nn.Parameter(torch.zeros(2))
        self.state_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU())

        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        self.correction = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.correction_gate_logit = nn.Parameter(torch.tensor(-2.))
        self.state_film = nn.Linear(width, width * 2)
        nn.init.zeros_(self.state_film.weight)
        nn.init.zeros_(self.state_film.bias)
        self.output_norm = nn.LayerNorm(width)

        self.position = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.click_direct = self.position_to_click = self.pixel_blend = None
        if predict_click:
            self.click_direct = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
            self.position_to_click = nn.Sequential(
                nn.Linear(3, width), nn.GELU(), nn.Linear(width, 2))
            self.pixel_blend = nn.Linear(width, 1)
            nn.init.zeros_(self.pixel_blend.weight)
            nn.init.constant_(self.pixel_blend.bias, -1.)

        self.future_position = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 3))
        horizon = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        self.register_buffer("horizon_basis", torch.stack((
            horizon, horizon.square(), torch.sin(math.pi * horizon),
            torch.cos(math.pi * horizon)), -1))

    def forward(self, emg, imu, conditioning_probability=None):
        ef, imf = self.emg(emg), self.imu.forward_features(imu)
        branch = self.react(torch.zeros_like(emg) if self.modality == "imu" else emg)
        if self.modality == "imu":
            state_logits = self.imu_state_prior.view(1, 1, 2).expand(
                emg.shape[0], emg.shape[1], -1)
        else:
            local = self.state_local(F.pad(
                ef["local"].transpose(1, 2), (4, 0))).transpose(1, 2)
            state_logits = (self.state_context(ef["context"])
                            + self.state_blend_logit.sigmoid() * local
                            + self.react_current(branch["features"]))
        state_probability = state_logits.softmax(-1)
        if conditioning_probability is None:
            conditioning_probability = state_probability
        elif conditioning_probability.shape != state_probability.shape:
            raise ValueError("conditioning_probability must match state logits")
        # An external frozen classifier is a teacher/condition, never a route
        # for pose gradients back into that classifier.
        conditioning_probability = conditioning_probability.detach()
        state = self.state_embedding(conditioning_probability)

        if self.modality == "emg":
            base, attended = ef["context"], ef["context"]
            cross_weights = base.new_zeros(*base.shape[:2], base.shape[1])
        elif self.modality == "imu":
            base, attended = imf["context"], torch.zeros_like(imf["context"])
            cross_weights = base.new_zeros(*base.shape[:2], base.shape[1])
        else:
            base = imf["context"]
            attended, cross_weights = self.cross_attention(
                base + state, ef["context"], ef["context"],
                attn_mask=causal_mask(base.shape[1], base.device),
                need_weights=True, average_attn_weights=True)
        correction_gate = self.correction_gate_logit.sigmoid()
        if self.modality == "imu":
            correction_gate = correction_gate * 0
        motion = base + correction_gate * self.correction(attended)
        gamma, beta = self.state_film(state).chunk(2, -1)
        if self.modality == "imu":
            gamma, beta = torch.zeros_like(gamma), torch.zeros_like(beta)
        fused = self.output_norm(motion * (1 + gamma) + beta)
        position = self.position(fused)

        direct = from_position = blend = click = None
        if self.click_direct is not None:
            direct = self.click_direct(fused).sigmoid()
            # Pixel supervision cannot bend the metric XYZ representation.
            from_position = self.position_to_click(position.detach()).sigmoid()
            blend = self.pixel_blend(fused).sigmoid()
            click = torch.lerp(direct, from_position, blend)

        basis = self.horizon_basis.view(1, 1, self.future_steps, 4)
        future_input = torch.cat((
            fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1),
            basis.expand(*fused.shape[:2], -1, -1)), -1)
        identity = position.new_tensor([1., 0., 0., 0., 1., 0.])
        return {
            "gripper_state_logits": state_logits,
            "react_gripper_state_logits": branch["holding_logits"],
            "react_reconstruction": branch["reconstruction"],
            "state_probability": state_probability,
            "conditioning_state_probability": conditioning_probability,
            "position": position,
            "click": click,
            "click_direct": direct,
            "click_from_position": from_position,
            "click_position_blend": blend,
            "future_position": self.future_position(future_input),
            "orientation_6d": identity.view(1, 1, 6).expand(*position.shape[:2], 6),
            "future_orientation_6d": identity.view(1, 1, 1, 6).expand(
                *position.shape[:2], self.future_steps, 6),
            "final_position": None,
            "final_orientation_6d": None,
            "context_features": fused,
            "emg_channel_attention": ef["channel_attention"],
            "cross_attention_weights": cross_weights,
            "emg_correction_gate": correction_gate,
        }
