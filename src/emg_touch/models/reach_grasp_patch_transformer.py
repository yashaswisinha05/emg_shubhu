"""Chronos-style patch transformer adapted for causal frame-level interaction."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class CausalPatchBranch(nn.Module):
    """Overlapping patch transformer plus a local path for sharp transitions."""
    def __init__(self, inputs: int, width: int, patch: int = 16, stride: int = 4,
                 layers: int = 4, heads: int = 4, dropout: float = .1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.input = nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU())
        self.local = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=5, groups=width),
            nn.Conv1d(width, width, kernel_size=9, groups=width),
        ])
        self.patch_embed = nn.Conv1d(width, width, patch, stride=stride)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, nn.LayerNorm(width))
        self.output = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

    @staticmethod
    def positions(length, width, device, dtype):
        time = torch.arange(length, device=device, dtype=dtype)[:, None]
        scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                          * (-math.log(10000.) / width))
        result = torch.zeros(length, width, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(time * scale)
        result[:, 1::2] = torch.cos(time * scale[:result[:, 1::2].shape[1]])
        return result

    def forward_features(self, values):
        frames = self.input(values)
        channel = frames.transpose(1, 2)
        local = 0
        for layer in self.local:
            local = local + layer(F.pad(channel, (layer.kernel_size[0] - 1, 0)))
        local = F.gelu(local).transpose(1, 2)
        # Left padding means patch token k sees only samples ending at k*stride.
        tokens = self.patch_embed(F.pad(channel, (self.patch - 1, 0))).transpose(1, 2)
        tokens = tokens + self.positions(tokens.shape[1], tokens.shape[2],
                                         tokens.device, tokens.dtype)
        tokens = self.transformer(tokens, mask=causal_mask(tokens.shape[1], tokens.device))
        # Hold the latest completed causal patch representation between endpoints.
        expanded = tokens.repeat_interleave(self.stride, dim=1)[:, :values.shape[1]]
        if expanded.shape[1] < values.shape[1]:
            expanded = torch.cat([expanded, expanded[:, -1:].expand(
                -1, values.shape[1] - expanded.shape[1], -1)], 1)
        return {"local": local,
                "context": self.output(torch.cat([local, expanded], -1))}

    def forward(self, values):
        return self.forward_features(values)["context"]


class ReachGraspPatchTransformer(nn.Module):
    """Equal-width encoders; causal gate chooses complementary evidence per frame."""
    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1):
        super().__init__()
        self.modality = modality
        kwargs = dict(width=width, patch=patch, stride=stride, layers=layers,
                      heads=heads, dropout=dropout)
        self.emg = CausalPatchBranch(16, **kwargs)
        self.imu = CausalPatchBranch(48, **kwargs)
        self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                  nn.Linear(width, 2))
        self.fusion = nn.Sequential(nn.Linear(width * 4, width * 2), nn.LayerNorm(width * 2),
                                    nn.GELU(), nn.Dropout(dropout))
        self.interaction = nn.Linear(width * 2, 3)
        self.position = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                      nn.Linear(width, 3))

    def forward(self, emg, imu):
        e, i = self.emg(emg), self.imu(imu)
        if self.modality == "emg":
            i = torch.zeros_like(i)
        elif self.modality == "imu":
            e = torch.zeros_like(e)
        weights = self.gate(torch.cat([e, i], -1).detach()).softmax(-1)
        if self.modality == "emg":
            weights = torch.stack([torch.ones_like(weights[..., 0]),
                                   torch.zeros_like(weights[..., 1])], -1)
        elif self.modality == "imu":
            weights = torch.stack([torch.zeros_like(weights[..., 0]),
                                   torch.ones_like(weights[..., 1])], -1)
        fused = self.fusion(torch.cat([e * weights[..., 0:1], i * weights[..., 1:2],
                                      e - i, e * i], -1))
        return {"logits": self.interaction(fused), "position": self.position(fused),
                "fusion_weights": weights}
