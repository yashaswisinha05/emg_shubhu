"""Minimal deployable model retained by the final component ablation."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from emg_touch.models.reach_grasp_patch_transformer import CausalPatchBranch


class CausalEMGStateContext(nn.Module):
    """Lean causal EMG state context without reconstruction/classification heads."""

    def __init__(self, inputs=16, width=64, heads=4, layers=4,
                 dropout=.1, context=100):
        super().__init__()
        self.context = context
        self.history = (context - 1) * layers
        self.signal = nn.Linear(inputs, width)
        self.state_mask = nn.Parameter(torch.zeros(width))
        self.modality = nn.Parameter(torch.randn(2, width) * .02)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False)

    def forward(self, emg):
        batch, frames, _ = emg.shape
        signal = self.signal(emg)
        time = torch.arange(frames, device=emg.device, dtype=signal.dtype)
        frequency = torch.exp(torch.arange(
            0, signal.shape[-1], 2, device=emg.device, dtype=signal.dtype)
            * (-math.log(10000.) / signal.shape[-1]))
        phase = time[:, None] * frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        state_token = self.state_mask.view(1, 1, -1).expand(batch, frames, -1)
        tokens = torch.stack((signal + self.modality[0],
                              state_token + self.modality[1]), 2)
        tokens = (tokens + position[None, :, None]).flatten(1, 2)
        chunks = []
        stamps_all = torch.arange(frames, device=emg.device)
        for start in range(0, frames, self.context):
            end = min(frames, start + self.context)
            left = max(0, start - self.history)
            stamps = stamps_all[left:end].repeat_interleave(2)
            lag = stamps[:, None] - stamps[None, :]
            blocked = (lag < 0) | (lag >= self.context)
            encoded = self.encoder(tokens[:, 2 * left:2 * end], mask=blocked)
            chunks.append(encoded[:, 2 * (start - left):])
        encoded = torch.cat(chunks, 1).reshape(batch, frames, 2, -1)
        return encoded[:, :, 1]


class MinimalEMGIMUModel(nn.Module):
    """Causal EMG+IMU model with no unused prediction heads.

    Inputs contain standardized values followed by per-channel validity flags:
    8+8 EMG features and 24+24 IMU features. Outputs are open/close logits,
    current XYZ, normalized pixel XY, and 10--200 ms future XYZ.
    """

    modality = "emg+imu"

    def __init__(self, width=128, patch=16, stride=4, layers=4, heads=4,
                 dropout=.1, react_context=100, adapter_layers=2,
                 future_steps=20):
        super().__init__()
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        self.model_args = dict(
            width=width, patch=patch, stride=stride, layers=layers, heads=heads,
            dropout=dropout, react_context=react_context,
            adapter_layers=adapter_layers, future_steps=future_steps)
        encoder = dict(width=width, patch=patch, stride=stride, layers=layers,
                       heads=heads, dropout=dropout)
        self.emg = CausalPatchBranch(16, **encoder)
        self.imu = CausalPatchBranch(48, **encoder)

        # State fusion preserves both sharp local evidence and longer context.
        self.context_gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 2))
        self.context_fusion = nn.Sequential(
            nn.Linear(width * 4, width * 2), nn.LayerNorm(width * 2),
            nn.GELU(), nn.Dropout(dropout))
        self.local_gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 2))
        self.local_fusion = nn.Sequential(
            nn.Linear(width * 4, width), nn.LayerNorm(width),
            nn.GELU(), nn.Dropout(dropout))
        self.gripper_context = nn.Linear(width * 2, 2)
        self.gripper_local = nn.Sequential(
            nn.Conv1d(width, width, 5, groups=width), nn.GELU(),
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 2, 1))
        self.gripper_blend_logit = nn.Parameter(torch.tensor(-2.))
        self.react = CausalEMGStateContext(
            inputs=16,
            width=width // 2, heads=heads, layers=layers,
            dropout=dropout, context=react_context)
        self.react_current = nn.Linear(width // 2, 2)
        nn.init.zeros_(self.react_current.weight)
        nn.init.zeros_(self.react_current.bias)

        # Motion fusion: IMU base plus a bounded, learned EMG correction.
        recurrent_dropout = dropout if adapter_layers > 1 else 0.
        self.emg_adapter = nn.GRU(
            width, width, adapter_layers, batch_first=True,
            dropout=recurrent_dropout)
        self.imu_adapter = nn.GRU(
            width, width, adapter_layers, batch_first=True,
            dropout=recurrent_dropout)
        self.emg_adapter_output = nn.Linear(width, width)
        self.imu_adapter_output = nn.Linear(width, width)
        for layer in (self.emg_adapter_output, self.imu_adapter_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.emg_correction = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.Tanh())
        self.correction_gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.emg_correction[-2].weight)
        nn.init.zeros_(self.emg_correction[-2].bias)
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, -2.)
        self.fused_norm = nn.LayerNorm(width)

        self.position_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.pixel_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.future_head = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 3))
        self.future_steps = future_steps
        tau = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        self.register_buffer("horizon_basis", torch.stack((
            tau, tau.square(), torch.sin(math.pi * tau),
            torch.cos(math.pi * tau)), -1))

    @staticmethod
    def _fuse(emg, imu, gate, layer):
        weights = gate(torch.cat((emg, imu), -1).detach()).softmax(-1)
        feature = layer(torch.cat((
            emg * weights[..., :1], imu * weights[..., 1:],
            emg - imu, emg * imu), -1))
        return feature, weights

    def forward(self, emg, imu):
        emg_feature = self.emg.forward_features(emg)
        imu_feature = self.imu.forward_features(imu)
        state_context, context_weights = self._fuse(
            emg_feature["context"], imu_feature["context"],
            self.context_gate, self.context_fusion)
        state_local, local_weights = self._fuse(
            emg_feature["local"], imu_feature["local"],
            self.local_gate, self.local_fusion)
        local_logits = self.gripper_local(F.pad(
            state_local.transpose(1, 2), (4, 0))).transpose(1, 2)
        state_logits = (self.gripper_context(state_context)
                        + self.gripper_blend_logit.sigmoid() * local_logits)
        state_logits = state_logits + self.react_current(self.react(emg))

        self.emg_adapter.flatten_parameters()
        self.imu_adapter.flatten_parameters()
        emg_motion = emg_feature["context"] + self.emg_adapter_output(
            self.emg_adapter(emg_feature["context"])[0])
        imu_motion = imu_feature["context"] + self.imu_adapter_output(
            self.imu_adapter(imu_feature["context"])[0])
        correction = self.emg_correction(emg_motion)
        correction_gate = self.correction_gate(
            torch.cat((imu_motion, emg_motion), -1)).sigmoid()
        fused = self.fused_norm(imu_motion + correction_gate * correction)

        basis = self.horizon_basis.view(1, 1, self.future_steps, 4)
        future_input = torch.cat((
            fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1),
            basis.expand(*fused.shape[:2], -1, -1)), -1)
        identity = fused.new_tensor([1., 0., 0., 0., 1., 0.])
        return {
            "gripper_state_logits": state_logits,
            "position": self.position_head(fused),
            "click": self.pixel_head(fused).sigmoid(),
            "future_position": self.future_head(future_input),
            "orientation_6d": None,
            # Compatibility placeholder for the shared evaluator. It is
            # never optimized or exposed by the minimal inference API.
            "future_orientation_6d": identity.view(1, 1, 1, 6).expand(
                *fused.shape[:2], self.future_steps, 6),
            "final_position": None,
            "final_orientation_6d": None,
            "emg_correction_gate": correction_gate,
            "fusion_weights": context_weights,
            "local_fusion_weights": local_weights,
        }
