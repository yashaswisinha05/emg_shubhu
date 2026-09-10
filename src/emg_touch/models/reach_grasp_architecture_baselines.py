"""Capacity-controlled causal baselines for the reach/gripper benchmark."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .reach_grasp_patch_transformer import CausalPatchBranch, causal_mask


def _left_conv(layer, value):
    return layer(F.pad(value, (layer.kernel_size[0] - 1, 0)))


class ResidualCausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.dilation = dilation
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        residual = value
        padding = 2 * self.dilation
        value = self.conv1(F.pad(value, (padding, 0))).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm1(value))).transpose(1, 2)
        value = self.conv2(F.pad(value, (padding, 0))).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm2(value))).transpose(1, 2)
        return residual + value


class CausalInceptionBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        branch = max(8, width // 4)
        self.convs = nn.ModuleList([nn.Conv1d(width, branch, k) for k in (3, 7, 15)])
        self.pool = nn.Conv1d(width, branch, 1)
        self.mix = nn.Conv1d(branch * 4, width, 1)
        self.norm, self.dropout = nn.LayerNorm(width), nn.Dropout(dropout)

    def forward(self, value):
        branches = [_left_conv(layer, value) for layer in self.convs]
        pooled = F.max_pool1d(F.pad(value, (2, 0)), 3, stride=1)
        branches.append(self.pool(pooled))
        mixed = self.mix(torch.cat(branches, 1)).transpose(1, 2)
        return value + self.dropout(F.gelu(self.norm(mixed))).transpose(1, 2)


class ArchitectureBaseline(nn.Module):
    """Common output heads over one of eight deliberately simple backbones."""

    NAMES = {"constant", "feature_mlp", "gru", "lstm", "tcn", "inceptiontime",
             "early_patch_transformer", "mult_cross_attention"}

    def __init__(self, architecture, modality="emg+imu", width=128, patch=16,
                 stride=4, layers=4, heads=4, dropout=.1, future_steps=20,
                 predict_click=True, predict_final_pose=False, react_context=100,
                 constant_initialization=None):
        super().__init__()
        if architecture not in self.NAMES:
            raise ValueError(f"unknown comparison architecture: {architecture}")
        if modality not in {"emg", "imu", "emg+imu"}:
            raise ValueError(f"unknown modality: {modality}")
        if predict_final_pose:
            raise ValueError("comparison models intentionally omit endpoint heads")
        self.architecture, self.modality = architecture, modality
        self.width, self.future_steps = width, future_steps
        self.predict_click = predict_click
        inputs = 64

        if architecture == "constant":
            initial = constant_initialization or {}
            self.dummy = nn.Parameter(torch.zeros(()))
            self.register_buffer("constant_state", torch.as_tensor(
                initial.get("state_logits", [0., 0.]), dtype=torch.float32))
            self.register_buffer("constant_position", torch.as_tensor(
                initial.get("position", [0., 0., 0.]), dtype=torch.float32))
            self.register_buffer("constant_click", torch.as_tensor(
                initial.get("click", [.5, .5]), dtype=torch.float32))
            future = initial.get("future_position", [[0., 0., 0.]] * future_steps)
            self.register_buffer("constant_future", torch.as_tensor(future, dtype=torch.float32))
            return

        self.input = nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU())
        if architecture == "feature_mlp":
            self.feature = nn.Sequential(
                nn.Linear(inputs * 3, width * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())
        elif architecture in {"gru", "lstm"}:
            recurrent = nn.GRU if architecture == "gru" else nn.LSTM
            self.recurrent = recurrent(width, width, num_layers=layers,
                                       dropout=dropout if layers > 1 else 0,
                                       batch_first=True, bidirectional=False)
        elif architecture == "tcn":
            self.temporal = nn.ModuleList([
                ResidualCausalBlock(width, 2 ** index, dropout)
                for index in range(layers)])
        elif architecture == "inceptiontime":
            self.temporal = nn.ModuleList([
                CausalInceptionBlock(width, dropout) for _ in range(layers)])
        elif architecture == "early_patch_transformer":
            self.patch_encoder = CausalPatchBranch(
                inputs, width, patch, stride, layers, heads, dropout)
        else:
            self.emg_encoder = CausalPatchBranch(
                16, width, patch, stride, layers, heads, dropout)
            self.imu_encoder = CausalPatchBranch(
                48, width, patch, stride, layers, heads, dropout)
            self.cross_attention = nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True)
            self.cross_output = nn.Sequential(
                nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

        self.state_head = nn.Linear(width, 2)
        self.position_head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                           nn.Linear(width, 3))
        self.click_head = (nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                         nn.Linear(width, 2)) if predict_click else None)
        self.future_head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                         nn.Linear(width, future_steps * 3))

    def _inputs(self, emg, imu):
        if self.modality == "emg":
            imu = torch.zeros_like(imu)
        elif self.modality == "imu":
            emg = torch.zeros_like(emg)
        return torch.cat((emg, imu), -1)

    def _features(self, emg, imu):
        values = self._inputs(emg, imu)
        if self.architecture == "feature_mlp":
            count = torch.arange(1, values.shape[1] + 1, device=values.device,
                                 dtype=values.dtype).view(1, -1, 1)
            mean = values.cumsum(1) / count
            delta = torch.cat((torch.zeros_like(values[:, :1]),
                               values[:, 1:] - values[:, :-1]), 1)
            return self.feature(torch.cat((values, mean, delta), -1))
        if self.architecture == "early_patch_transformer":
            return self.patch_encoder(values)
        if self.architecture == "mult_cross_attention":
            e, i = self.emg_encoder(emg), self.imu_encoder(imu)
            if self.modality == "emg":
                return e
            if self.modality == "imu":
                return i
            attended, _ = self.cross_attention(
                i, e, e, attn_mask=causal_mask(i.shape[1], i.device),
                need_weights=False)
            return self.cross_output(torch.cat((i, attended), -1))
        features = self.input(values)
        if self.architecture in {"gru", "lstm"}:
            self.recurrent.flatten_parameters()
            return self.recurrent(features)[0]
        channel = features.transpose(1, 2)
        for layer in self.temporal:
            channel = layer(channel)
        return channel.transpose(1, 2)

    def forward(self, emg, imu):
        batch, length = emg.shape[:2]
        identity = emg.new_tensor([1., 0., 0., 0., 1., 0.])
        if self.architecture == "constant":
            zero = self.dummy * 0
            state = self.constant_state.view(1, 1, 2).expand(batch, length, 2) + zero
            position = self.constant_position.view(1, 1, 3).expand(batch, length, 3) + zero
            click = (self.constant_click.view(1, 1, 2).expand(batch, length, 2) + zero
                     if self.predict_click else None)
            future = self.constant_future.view(1, 1, self.future_steps, 3).expand(
                batch, length, self.future_steps, 3) + zero
        else:
            features = self._features(emg, imu)
            state = self.state_head(features)
            position = self.position_head(features)
            click = self.click_head(features).sigmoid() if self.click_head is not None else None
            future = self.future_head(features).reshape(
                batch, length, self.future_steps, 3)
        return {
            "gripper_state_logits": state,
            "position": position,
            "click": click,
            "future_position": future,
            "orientation_6d": identity.view(1, 1, 6).expand(batch, length, 6),
            "future_orientation_6d": identity.view(1, 1, 1, 6).expand(
                batch, length, self.future_steps, 6),
            "final_position": None,
            "final_orientation_6d": None,
        }
