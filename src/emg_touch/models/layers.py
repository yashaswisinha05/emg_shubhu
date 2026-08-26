from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm1d(nn.LayerNorm):
    """Layer normalization over channels independently at every time step."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(inputs, (left_padding, 0)))


class ResidualCausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, 1),
            # Per-time-step normalization is invariant to variable-length batch padding.
            ChannelLayerNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class MultiScaleCausalStem(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, d_model, 1)
        self.branches = nn.ModuleList(
            [ResidualCausalBlock(d_model, kernel, dilation=1, dropout=dropout) for kernel in kernel_sizes]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(d_model * len(kernel_sizes), d_model, 1),
            ChannelLayerNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs.transpose(1, 2))
        return self.output_projection(torch.cat([branch(hidden) for branch in self.branches], dim=1))


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        patch_length: int,
        patch_stride: int,
        kernel_sizes: list[int],
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.stem = MultiScaleCausalStem(input_dim, d_model, kernel_sizes, dropout)
        self.patch_projection = nn.Conv1d(
            d_model, d_model, kernel_size=patch_length, stride=patch_stride
        )
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            # norm_first=True cannot use PyTorch's nested-tensor fast path.
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, inputs: torch.Tensor, time_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.stem(inputs)
        tokens = self.patch_projection(hidden).transpose(1, 2)
        if tokens.size(1) <= self.position.size(1):
            position = self.position[:, : tokens.size(1)]
        else:
            # Interpolate learned positions for unusually long, valid trajectories.
            position = F.interpolate(
                self.position.transpose(1, 2),
                size=tokens.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        tokens = tokens + position
        patch_mask = time_mask.unfold(1, self.patch_length, self.patch_stride).any(dim=-1)
        # Transformer attention cannot accept a row where every key is masked.
        safe_mask = patch_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        tokens = self.transformer(tokens, src_key_padding_mask=~safe_mask)
        tokens = self.norm(tokens)
        weights = patch_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return tokens, pooled, patch_mask


class MultiScaleLookback(nn.Module):
    def __init__(
        self,
        d_model: int,
        lookbacks_s: list[float],
        sample_rate_hz: float,
        patch_stride: int,
    ) -> None:
        super().__init__()
        self.patch_counts = [
            max(1, math.ceil(seconds * sample_rate_hz / patch_stride)) for seconds in lookbacks_s
        ]
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(lookbacks_s)),
        )

    def forward(
        self,
        emg_context: torch.Tensor,
        imu_tokens: torch.Tensor,
        imu_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summaries = []
        for count in self.patch_counts:
            tokens = imu_tokens[:, -count:]
            mask = imu_patch_mask[:, -count:].unsqueeze(-1).to(tokens.dtype)
            summary = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            summaries.append(summary)
        summaries_tensor = torch.stack(summaries, dim=1)
        weights = torch.softmax(self.gate(emg_context), dim=-1)
        context = torch.sum(summaries_tensor * weights.unsqueeze(-1), dim=1)
        return context, weights


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
