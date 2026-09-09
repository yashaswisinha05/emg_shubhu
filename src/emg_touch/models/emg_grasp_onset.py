"""Causal, EMG-only grasp-onset detector."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrameLayerNorm(nn.Module):
    """Normalize channels independently at each time step (causal)."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalResidualBlock(nn.Module):
    """Dilated depthwise residual block with explicit left-only padding."""

    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=5, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width * 2, kernel_size=1)
        self.output = nn.Conv1d(width, width, kernel_size=1)
        self.norm = FrameLayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.pad(x, (4 * self.dilation, 0)))
        x = F.glu(self.mix(x), dim=1)
        x = self.dropout(self.output(x))
        return self.norm(residual + x)


class EMGGraspOnsetDetector(nn.Module):
    """Multi-scale temporal CNN dedicated to grasp onset.

    Input follows the repository convention: eight normalized EMG features
    (20/50 ms RMS for four sensors) followed by eight validity flags. First
    differences are formed internally and causally.
    """

    def __init__(self, width: int = 96, dropout: float = .15,
                 dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.width = width
        self.dilations = tuple(dilations)
        self.input = nn.Sequential(
            nn.Conv1d(24, width, kernel_size=1), nn.GELU(), FrameLayerNorm(width))
        self.blocks = nn.ModuleList(
            CausalResidualBlock(width, dilation, dropout)
            for dilation in self.dilations)
        self.head = nn.Sequential(
            nn.Conv1d(width * len(self.dilations), width, kernel_size=1),
            nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, 1, kernel_size=1))

    def forward(self, packed_emg: torch.Tensor) -> dict[str, torch.Tensor]:
        if packed_emg.shape[-1] != 16:
            raise ValueError("expected 8 EMG values followed by 8 validity flags")
        values, valid = packed_emg[..., :8], packed_emg[..., 8:]
        delta = torch.cat([torch.zeros_like(values[:, :1]),
                           values[:, 1:] - values[:, :-1]], dim=1)
        # A difference spanning missing data is not evidence of activation.
        pair_valid = valid * torch.cat([torch.zeros_like(valid[:, :1]),
                                        valid[:, :-1]], dim=1)
        features = torch.cat([values * valid, delta * pair_valid, valid], dim=-1)
        x = self.input(features.transpose(1, 2))
        scales = []
        for block in self.blocks:
            x = block(x)
            scales.append(x)
        logits = self.head(torch.cat(scales, dim=1)).transpose(1, 2)
        return {"grasp_logit": logits[..., 0]}
