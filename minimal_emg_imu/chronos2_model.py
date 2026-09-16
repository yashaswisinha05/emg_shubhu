"""Frozen pretrained Chronos-2 encoder ablation for the minimal model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from minimal_emg_imu.model import MinimalEMGIMUModel


def install_causal_time_attention(chronos_model) -> list:
    """Prevent Chronos-2 context tokens from attending to later patches."""
    if getattr(chronos_model, "_emg_touch_causal", False):
        return []

    def add_causal_mask(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        if hidden is None or mask is None:
            raise RuntimeError("unexpected Chronos-2 time-attention call")
        length = hidden.shape[-2]
        blocked = torch.zeros(
            (1, 1, length, length), dtype=hidden.dtype,
            device=hidden.device)
        blocked.masked_fill_(torch.ones(
            (length, length), dtype=torch.bool,
            device=hidden.device).triu(1), torch.finfo(hidden.dtype).min)
        kwargs["attention_mask"] = mask + blocked
        return args, kwargs

    handles = []
    for block in chronos_model.encoder.block:
        handles.append(block.layer[0].register_forward_pre_hook(
            add_causal_mask, with_kwargs=True))
    chronos_model._emg_touch_causal = True
    return handles


class FrozenChronos2Branch(nn.Module):
    """Project frozen, causally aligned Chronos-2 patches to frame features.

    The foundation model is deliberately not registered as a child module:
    its weights are frozen, shared by the two sensor branches, and omitted
    from task checkpoints. The official checkpoint is loaded separately.
    """

    def __init__(self, chronos_model, inputs: int, width: int = 128,
                 context_frames: int = 512, chunk_frames: int = 128):
        super().__init__()
        config = chronos_model.chronos_config
        self.inputs = inputs
        self.patch = int(config.input_patch_size)
        self.stride = int(config.input_patch_stride)
        self.context_frames = context_frames
        self.chunk_frames = chunk_frames
        if self.patch != self.stride:
            raise ValueError("this ablation expects non-overlapping Chronos-2 patches")
        if context_frames < self.patch or context_frames % self.patch:
            raise ValueError("context_frames must be a multiple of the Chronos-2 patch size")
        if chunk_frames < 1 or chunk_frames > context_frames:
            raise ValueError("chunk_frames must be in [1, context_frames]")
        object.__setattr__(self, "_foundation", chronos_model)
        self.projection = nn.Sequential(
            nn.Linear(int(chronos_model.config.d_model), width),
            nn.LayerNorm(width), nn.GELU())
        self.cold_start = nn.Parameter(torch.zeros(width))

    @property
    def foundation(self):
        return object.__getattribute__(self, "_foundation")

    def _encode_window(self, values: torch.Tensor) -> torch.Tensor:
        batch, frames, channels = values.shape
        if frames != self.context_frames:
            raise ValueError("internal Chronos-2 window has the wrong length")
        # Chronos-2 treats sensor channels as related variates. Group IDs
        # permit cross-channel attention only within the same trial.
        context = values.transpose(1, 2).reshape(batch * channels, frames)
        group_ids = torch.arange(batch, device=values.device).repeat_interleave(channels)
        foundation = self.foundation
        foundation.eval()
        model_dtype = next(foundation.parameters()).dtype
        with torch.no_grad():
            encoded, _, _, patch_count = foundation.encode(
                context=context.to(dtype=model_dtype),
                group_ids=group_ids, num_output_patches=1)
        patches = encoded.last_hidden_state[:, :patch_count]
        patches = patches.reshape(batch, channels, patch_count, -1).mean(1)
        patches = self.projection(patches.to(dtype=values.dtype))

        # A patch covering samples [a,b] becomes available only at b. Hold
        # that representation until the next complete patch arrives.
        cold = self.cold_start.view(1, 1, -1).expand(batch, self.patch - 1, -1)
        aligned = torch.cat((cold, patches.repeat_interleave(self.stride, 1)), 1)
        return aligned[:, :self.context_frames]

    def forward_features(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, channels = values.shape
        if channels != self.inputs:
            raise ValueError(f"expected {self.inputs} inputs, received {channels}")
        chunks = []
        for start in range(0, frames, self.chunk_frames):
            end = min(frames, start + self.chunk_frames)
            left = max(0, end - self.context_frames)
            window = values[:, left:end]
            pad = self.context_frames - window.shape[1]
            window = F.pad(window, (0, 0, pad, 0))
            encoded = self._encode_window(window)
            chunks.append(encoded[:, -(end - start):])
        context = torch.cat(chunks, 1)[:, :frames]
        return {"local": context, "context": context}

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(values)["context"]


class Chronos2EMGIMUModel(MinimalEMGIMUModel):
    """Minimal four-head model with a shared frozen Chronos-2 encoder."""

    def __init__(self, chronos_model, width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, react_context=100,
                 adapter_layers=2, future_steps=20, context_frames=512,
                 chunk_frames=128, foundation_id="amazon/chronos-2"):
        super().__init__(
            width=width, patch=patch, stride=stride, layers=layers,
            heads=heads, dropout=dropout, react_context=react_context,
            adapter_layers=adapter_layers, future_steps=future_steps)
        install_causal_time_attention(chronos_model)
        branch = dict(width=width, context_frames=context_frames,
                      chunk_frames=chunk_frames)
        self.emg = FrozenChronos2Branch(chronos_model, 16, **branch)
        self.imu = FrozenChronos2Branch(chronos_model, 48, **branch)
        self.model_args.update(
            context_frames=context_frames, chunk_frames=chunk_frames,
            foundation_id=foundation_id)
