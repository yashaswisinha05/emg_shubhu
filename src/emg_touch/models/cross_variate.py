"""Cross-variate multi-scale patch model (MCV-PatchTST adapted to EMG/IMU).

Three ideas from Qarni et al., MCV-PatchTST, adapted rather than copied:

1. Multi-scale patch decomposition. Applied *coarse* rather than fine: the EMG
   RMS envelope on this dataset has a ~540 ms autocorrelation timescale, five
   times slower than the existing 108 ms patch, so shorter patches would only
   resolve noise. Default scales are {16, 32, 64} samples = {108, 216, 432} ms.

2. Cross-variate patch attention over *physical* variate tokens - each EMG
   channel and each of the four IMU sensors - instead of over all 92 raw
   channels. The paper's O(N C^2 d) term is 8464 at C=92 but 64 at C=8, matching
   the paper's own cost, and the resulting 8x8 attention map is interpretable.
   Applied once, before temporal encoding, as the paper prescribes.

3. Positional encoding keyed to physical seconds-to-cutoff. The paper indexes a
   fixed-length lookback axis; these trajectories are variable-length causal
   prefixes, so a normalized index would make the same token mean different
   physical times across trials and destroy the fixed electromechanical delay.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .layers import MultiScaleCausalStem

IMU_SENSORS = 4


_BIN_CACHE: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _bin_boundaries(
    source_length: int, length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous, order-preserving partition of source_length into `length`
    bins of near-equal width (paper Eq. 7), cached per shape and device.
    """
    key = (source_length, length, device)
    cached = _BIN_CACHE.get(key)
    if cached is not None:
        return cached
    # Integer bin edges via Python ints (exact, no float64 needed - MPS has no
    # float64 support), then moved to the target device as plain index tensors.
    edges = [(index * source_length) // length for index in range(length + 1)]
    starts_list = edges[:-1]
    ends_list = [max(edges[index + 1], edges[index] + 1) for index in range(length)]
    ends_list[-1] = source_length
    starts = torch.tensor(starts_list, device=device, dtype=torch.long)
    ends = torch.tensor(ends_list, device=device, dtype=torch.long)
    _BIN_CACHE[key] = (starts, ends)
    return starts, ends


def _align(tokens: torch.Tensor, length: int) -> torch.Tensor:
    """Bin-averaged alignment along the patch-token dimension (paper Eq. 6-8).

    Equivalent to F.adaptive_avg_pool1d, implemented as an explicit partition
    and matmul instead: the MPS backend rejects adaptive pooling whenever the
    input size does not evenly divide the output size, which stride-based
    multi-scale patch counts rarely satisfy. This is backend-portable and
    exactly reproduces the same contiguous, non-overlapping bin averaging.
    """
    source_length = tokens.size(1)
    if source_length == length:
        return tokens
    starts, ends = _bin_boundaries(source_length, length, tokens.device)
    weights = torch.zeros(length, source_length, device=tokens.device, dtype=tokens.dtype)
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        weights[index, start:end] = 1.0 / (end - start)
    return torch.einsum("qs,bsd->bqd", weights, tokens)


def _align_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    """A bin is valid iff any source position it covers is valid."""
    source_length = mask.size(1)
    if source_length == length:
        return mask
    starts, ends = _bin_boundaries(source_length, length, mask.device)
    out = torch.zeros(mask.size(0), length, dtype=torch.bool, device=mask.device)
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        out[:, index] = mask[:, start:end].any(dim=1)
    return out


class MultiScalePatchEmbedder(nn.Module):
    """Stem, K parallel patch projections, scale alignment, gated fusion."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_lengths: tuple[int, ...],
        patch_stride: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_lengths = tuple(int(p) for p in patch_lengths)
        self.patch_stride = int(patch_stride)
        self.minimum_samples = max(self.patch_lengths)
        self.stem = MultiScaleCausalStem(input_dim, d_model, kernel_sizes, dropout)
        # Each scale gets its own projection, as in the paper: W_e^(k).
        self.projections = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=length, stride=self.patch_stride)
                for length in self.patch_lengths
            ]
        )
        scales = len(self.patch_lengths)
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(d_model * scales),
            nn.Linear(d_model * scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, scales),
        )
        # The stem ends in LayerNorm, but the per-scale Conv1d projections
        # that follow it do not, so the gated-fusion output has no normalized
        # scale. Two independently trained embedders (this one, and a second
        # instance for the other modality) can then drift to very different
        # output norms - measured at 2.2x on a real checkpoint (IMU tokens
        # vs EMG tokens) - which biases the unnormalized dot-product
        # cross-variate attention toward whichever modality's norm is larger,
        # independent of content. Matches PatchTransformerEncoder's own
        # final-norm convention.
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """values/mask: (B, T, C). Returns tokens (B, N, d), mask (B, N), gate (B, K)."""
        # The longest patch needs at least that many samples; pad the past so
        # short trajectories stay valid and the padding stays causally masked.
        if values.size(1) < self.minimum_samples:
            pad = self.minimum_samples - values.size(1)
            values = F.pad(values, (0, 0, pad, 0))
            mask = F.pad(mask, (0, 0, pad, 0), value=False)
        time_mask = mask.any(dim=-1)
        hidden = self.stem(torch.cat([values, mask.to(values.dtype)], dim=-1))

        scale_tokens: list[torch.Tensor] = []
        scale_masks: list[torch.Tensor] = []
        for length, projection in zip(self.patch_lengths, self.projections):
            tokens = projection(hidden).transpose(1, 2)
            patch_mask = time_mask.unfold(1, length, self.patch_stride).any(dim=-1)
            scale_tokens.append(tokens)
            scale_masks.append(patch_mask)

        aligned_length = min(tokens.size(1) for tokens in scale_tokens)
        aligned = [_align(tokens, aligned_length) for tokens in scale_tokens]
        aligned_masks = [_align_mask(m, aligned_length) for m in scale_masks]
        patch_mask = torch.stack(aligned_masks, dim=0).any(dim=0)

        # Gate conditioned on per-scale channel statistics (paper Eq. 9).
        weights = torch.stack(aligned_masks, dim=0).to(values.dtype)
        summaries = [
            (tokens * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
            for tokens, w in zip(aligned, weights)
        ]
        gate = torch.softmax(self.scale_gate(torch.cat(summaries, dim=-1)), dim=-1)
        fused = sum(
            gate[:, index : index + 1].unsqueeze(-1) * tokens
            for index, tokens in enumerate(aligned)
        )
        fused = self.output_norm(fused)
        return fused, patch_mask, gate


class TimeToCutoffEncoding(nn.Module):
    """Positional embedding indexed by physical seconds remaining to the cutoff."""

    def __init__(self, d_model: int, num_frequencies: int = 6) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.tensor([2.0**index * math.pi for index in range(num_frequencies)]),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * num_frequencies + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, patches: int, lengths: torch.Tensor, sample_rate: float
    ) -> torch.Tensor:
        """Returns (B, N, d). Token q is centred at (q + 0.5)/N of the valid prefix."""
        device = lengths.device
        centres = (torch.arange(patches, device=device, dtype=torch.float32) + 0.5) / patches
        valid = lengths.to(torch.float32).clamp_min(1.0).unsqueeze(-1)
        seconds_to_cutoff = (valid - centres.unsqueeze(0) * valid) / sample_rate
        scaled = seconds_to_cutoff.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [seconds_to_cutoff.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1
        )
        return self.projection(features)


class CrossVariateBackbone(nn.Module):
    """Eight physical variate tokens, mixed once, then encoded channel-independently."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        d_model = int(model["d_model"])
        dropout = float(model["dropout"])
        self.d_model = d_model
        self.sample_rate = float(data["sample_rate_hz"])
        from ..data.grid_trajectory import (
            emg_channel_count,
            grid_imu_sensor_indices,
        )

        # Each EMG channel is its own variate token, so derived antagonist
        # channels increase the variate count alongside the four electrodes.
        self.emg_channels = emg_channel_count(data)
        self.variates = self.emg_channels + IMU_SENSORS

        sensor_indices = torch.tensor(grid_imu_sensor_indices(data), dtype=torch.long)
        self.register_buffer("sensor_indices", sensor_indices, persistent=False)
        channels_per_sensor = int((sensor_indices == 0).sum())
        self.channels_per_sensor = channels_per_sensor

        patch_lengths = tuple(model.get("patch_lengths", (16, 32, 64)))
        stride = int(model["patch_stride"])
        kernels = list(model["tcn_kernel_sizes"])
        # One embedder per modality, shared across that modality's variates: the
        # four electrodes are homogeneous, as are the four IMU sensors.
        self.emg_embedder = MultiScalePatchEmbedder(
            2, d_model, patch_lengths, stride, kernels, dropout
        )
        self.imu_embedder = MultiScalePatchEmbedder(
            2 * channels_per_sensor, d_model, patch_lengths, stride, kernels, dropout
        )
        self.modality_embedding = nn.Parameter(torch.zeros(self.variates, d_model))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

        self.position = TimeToCutoffEncoding(d_model)
        self.cross_variate = nn.MultiheadAttention(
            d_model,
            num_heads=int(model.get("cross_variate_heads", 2)),
            dropout=dropout,
            batch_first=True,
        )
        self.cross_variate_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(model["num_heads"]),
            dim_feedforward=int(model["ffn_dim"]),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=int(model["num_layers"]), enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.variate_score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

    def _imu_by_sensor(
        self, imu: torch.Tensor, imu_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, 88) -> (B, 4, T, 22), grouping channels by physical sensor."""
        order = torch.argsort(self.sensor_indices, stable=True)
        batch, time = imu.size(0), imu.size(1)
        grouped = imu.index_select(-1, order).reshape(
            batch, time, IMU_SENSORS, self.channels_per_sensor
        )
        grouped_mask = imu_mask.index_select(-1, order).reshape(
            batch, time, IMU_SENSORS, self.channels_per_sensor
        )
        return grouped.permute(0, 2, 1, 3), grouped_mask.permute(0, 2, 1, 3)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = emg.size(0)

        emg_values = emg.permute(0, 2, 1).unsqueeze(-1).reshape(batch * self.emg_channels, -1, 1)
        emg_masks = emg_mask.permute(0, 2, 1).unsqueeze(-1).reshape(batch * self.emg_channels, -1, 1)
        emg_tokens, emg_patch_mask, emg_gate = self.emg_embedder(emg_values, emg_masks)

        imu_grouped, imu_grouped_mask = self._imu_by_sensor(imu, imu_mask)
        imu_values = imu_grouped.reshape(batch * IMU_SENSORS, imu.size(1), -1)
        imu_masks = imu_grouped_mask.reshape(batch * IMU_SENSORS, imu.size(1), -1)
        imu_tokens, imu_patch_mask, imu_gate = self.imu_embedder(imu_values, imu_masks)

        # Both modalities must share one patch axis before channel mixing.
        patches = min(emg_tokens.size(1), imu_tokens.size(1))
        emg_tokens = _align(emg_tokens, patches).reshape(batch, self.emg_channels, patches, -1)
        imu_tokens = _align(imu_tokens, patches).reshape(batch, IMU_SENSORS, patches, -1)
        emg_patch_mask = _align_mask(emg_patch_mask, patches).reshape(batch, self.emg_channels, patches)
        imu_patch_mask = _align_mask(imu_patch_mask, patches).reshape(batch, IMU_SENSORS, patches)

        tokens = torch.cat([emg_tokens, imu_tokens], dim=1)
        token_mask = torch.cat([emg_patch_mask, imu_patch_mask], dim=1)
        tokens = tokens + self.modality_embedding[None, :, None, :]
        tokens = tokens + self.position(patches, lengths, self.sample_rate).unsqueeze(1)

        # Cross-variate attention: one controlled mix across the eight physical
        # variates at each aligned patch position (paper Eq. 10).
        flat = tokens.permute(0, 2, 1, 3).reshape(batch * patches, self.variates, -1)
        flat_mask = token_mask.permute(0, 2, 1).reshape(batch * patches, self.variates)
        safe_mask = flat_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        mixed, attention = self.cross_variate(
            flat, flat, flat, key_padding_mask=~safe_mask, need_weights=True,
            average_attn_weights=True,
        )
        flat = self.cross_variate_norm(flat + mixed)
        tokens = flat.reshape(batch, patches, self.variates, -1).permute(0, 2, 1, 3)

        # Channel-independent temporal encoding, shared weights across variates.
        sequence = tokens.reshape(batch * self.variates, patches, -1)
        sequence_mask = token_mask.reshape(batch * self.variates, patches).clone()
        sequence_mask[~sequence_mask.any(dim=1), 0] = True
        encoded = self.temporal_norm(
            self.temporal(sequence, src_key_padding_mask=~sequence_mask)
        )

        weights = sequence_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled.reshape(batch, self.variates, -1)

        available = token_mask.any(dim=-1)
        scores = self.variate_score(pooled).squeeze(-1).masked_fill(~available, -1e4)
        variate_attention = torch.softmax(scores, dim=-1)
        context = self.output((variate_attention.unsqueeze(-1) * pooled).sum(dim=1))

        channel_attention = attention.reshape(
            batch, patches, self.variates, self.variates
        ).mean(dim=1)
        scale_gate = torch.cat(
            [
                emg_gate.reshape(batch, self.emg_channels, -1).mean(dim=1),
                imu_gate.reshape(batch, IMU_SENSORS, -1).mean(dim=1),
            ],
            dim=-1,
        )
        return context, variate_attention, channel_attention, scale_gate
