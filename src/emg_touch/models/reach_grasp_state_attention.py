"""Causal EMG-state-conditioned attention for position and screen intent."""
from __future__ import annotations

import math

import torch
from torch import nn

from .reach_grasp_patch_transformer import CausalPatchBranch, causal_mask


class EMGChannelEncoder(nn.Module):
    """Attend over the four physical sensors before causal temporal encoding."""

    def __init__(self, width, patch, stride, layers, heads, dropout):
        super().__init__()
        # Per sensor: short RMS, long RMS, and their two validity flags.
        self.sensor_projection = nn.Sequential(
            nn.Linear(4, width), nn.LayerNorm(width), nn.GELU())
        self.sensor_embedding = nn.Parameter(torch.zeros(4, width))
        self.score = nn.Sequential(
            nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))
        self.temporal = CausalPatchBranch(
            width, width, patch, stride, layers, heads, dropout)

    def forward(self, emg):
        sensors = torch.stack([
            torch.stack((emg[..., sensor], emg[..., 4 + sensor],
                         emg[..., 8 + sensor], emg[..., 12 + sensor]), -1)
            for sensor in range(4)], -2)
        valid = emg[..., 8:12].bool() & emg[..., 12:16].bool()
        tokens = self.sensor_projection(sensors) + self.sensor_embedding
        tokens = tokens * valid[..., None]
        scores = self.score(tokens).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(-1)
        # An all-invalid frame carries no learned sensor-identity shortcut.
        available = valid.any(-1, keepdim=True)
        pooled = (tokens * weights[..., None]).sum(-2) * available
        features = self.temporal.forward_features(pooled)
        features["channel_attention"] = weights * available
        return features


class StateConditionedAttentionModel(nn.Module):
    """EMG-only hand state conditions a conservative EMG correction to IMU.

    The IMU representation is the base in fused mode.  Both the FiLM state
    modulation and cross-modal correction start as exact no-ops, preventing
    random EMG features from damaging the strong inertial solution at epoch 0.
    """

    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, future_steps=20,
                 predict_click=True, predict_final_pose=False,
                 react_context=100):
        super().__init__()
        del react_context
        if modality not in {"emg", "imu", "emg+imu"}:
            raise ValueError(f"unknown modality: {modality}")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        if predict_final_pose:
            raise ValueError("this architecture intentionally has no endpoint head")
        self.modality, self.future_steps = modality, future_steps
        self.emg = EMGChannelEncoder(width, patch, stride, layers, heads, dropout)
        self.imu = CausalPatchBranch(48, width, patch, stride, layers, heads, dropout)

        self.state_local = nn.Linear(width, 2)
        self.state_context = nn.Linear(width, 2)
        self.state_blend_logit = nn.Parameter(torch.tensor(-1.))
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
        self.click = (nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, 2)) if predict_click else None)
        self.future_position = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 3))
        horizon = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        self.register_buffer("horizon_basis", torch.stack((
            horizon, horizon.square(), torch.sin(math.pi * horizon),
            torch.cos(math.pi * horizon)), -1))

    def forward(self, emg, imu):
        ef = self.emg(emg)
        imf = self.imu.forward_features(imu)
        if self.modality == "imu":
            state_logits = self.imu_state_prior.view(1, 1, 2).expand(
                emg.shape[0], emg.shape[1], -1)
        else:
            state_logits = (self.state_context(ef["context"])
                            + self.state_blend_logit.sigmoid()
                            * self.state_local(ef["local"]))
        state_probability = state_logits.softmax(-1)
        state = self.state_embedding(state_probability)

        if self.modality == "emg":
            base = ef["context"]
            attended = ef["context"]
            cross_weights = torch.zeros(
                (*base.shape[:2], base.shape[1]), device=base.device, dtype=base.dtype)
        elif self.modality == "imu":
            base = imf["context"]
            attended = torch.zeros_like(base)
            cross_weights = torch.zeros(
                (*base.shape[:2], base.shape[1]), device=base.device, dtype=base.dtype)
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

        basis = self.horizon_basis.view(1, 1, self.future_steps, 4)
        future_input = torch.cat((
            fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1),
            basis.expand(*fused.shape[:2], -1, -1)), -1)
        position = self.position(fused)
        # Identity placeholders preserve the common evaluation schema. They
        # are neither trained nor reported by this position-only experiment.
        identity = position.new_tensor([1., 0., 0., 0., 1., 0.])
        orientation = identity.view(1, 1, 6).expand(*position.shape[:2], 6)
        future_orientation = identity.view(1, 1, 1, 6).expand(
            *position.shape[:2], self.future_steps, 6)
        return {
            "gripper_state_logits": state_logits,
            "state_probability": state_probability,
            "position": position,
            "click": self.click(fused).sigmoid() if self.click is not None else None,
            "future_position": self.future_position(future_input),
            "orientation_6d": orientation,
            "future_orientation_6d": future_orientation,
            "final_position": None,
            "final_orientation_6d": None,
            "context_features": fused,
            "emg_channel_attention": ef["channel_attention"],
            "cross_attention_weights": cross_weights,
            "emg_correction_gate": correction_gate,
        }
