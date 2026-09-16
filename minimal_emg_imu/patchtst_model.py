"""Causal Hugging Face PatchTST encoder ablation for the minimal model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PatchTSTConfig, PatchTSTModel

from minimal_emg_imu.model import MinimalEMGIMUModel


class CausalPatchTSTBranch(nn.Module):
    """PatchTST features aligned causally to the 100-Hz frame grid.

    Hugging Face PatchTST is bidirectional by default.  Forward pre-hooks add
    an additive upper-triangular mask to every temporal self-attention layer.
    Fixed past-only windows keep memory bounded for complete recorded trials.
    """

    def __init__(self, inputs: int, width: int = 128, patch: int = 16,
                 stride: int = 4, layers: int = 4, heads: int = 4,
                 dropout: float = .1, context_frames: int = 600,
                 chunk_frames: int = 128):
        super().__init__()
        if context_frames < patch:
            raise ValueError("context_frames must be at least patch")
        if chunk_frames < 1 or chunk_frames > context_frames:
            raise ValueError("chunk_frames must be in [1, context_frames]")
        self.inputs = inputs
        self.width = width
        self.patch = patch
        self.stride = stride
        self.context_frames = context_frames
        self.chunk_frames = chunk_frames
        # Prefixing patch-1 zeros makes patch token j end at frame j*stride.
        model_context = context_frames + patch - 1
        config = PatchTSTConfig(
            num_input_channels=inputs,
            context_length=model_context,
            patch_length=patch,
            patch_stride=stride,
            num_hidden_layers=layers,
            d_model=width,
            num_attention_heads=heads,
            ffn_dim=width * 4,
            norm_type="layernorm",
            attention_dropout=dropout,
            positional_dropout=dropout,
            path_dropout=dropout,
            ff_dropout=dropout,
            activation_function="gelu",
            pre_norm=True,
            positional_encoding_type="sincos",
            use_cls_token=False,
            scaling=False,  # inputs are already training-set standardized
            do_mask_input=False,
            channel_attention=False,
            prediction_length=20,
        )
        self.encoder = PatchTSTModel(config)
        self._causal_hook_handles = []
        for layer in self.encoder.encoder.layers:
            self._causal_hook_handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._add_causal_mask, with_kwargs=True))

    @staticmethod
    def _add_causal_mask(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise RuntimeError("PatchTST attention did not receive hidden states")
        length = hidden.shape[-2]
        mask = torch.zeros((1, 1, length, length),
                           dtype=hidden.dtype, device=hidden.device)
        mask = mask.masked_fill(
            torch.ones((length, length), dtype=torch.bool,
                       device=hidden.device).triu(1),
            torch.finfo(hidden.dtype).min)
        kwargs["attention_mask"] = mask
        return args, kwargs

    def _encode_window(self, values: torch.Tensor) -> torch.Tensor:
        """Encode exactly context_frames and return frame-aligned features."""
        if values.shape[1] != self.context_frames:
            raise ValueError("internal PatchTST window has the wrong length")
        prefix = values.new_zeros(values.shape[0], self.patch - 1, self.inputs)
        encoded = self.encoder(past_values=torch.cat((prefix, values), 1))
        # [B, channels, patches, width] -> one feature per patch.
        patches = encoded.last_hidden_state.mean(1)
        frames = patches.repeat_interleave(self.stride, dim=1)
        if frames.shape[1] < self.context_frames:
            frames = torch.cat((frames, frames[:, -1:].expand(
                -1, self.context_frames - frames.shape[1], -1)), 1)
        return frames[:, :self.context_frames]

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
            chunks.append(encoded[:, -min(end - start, encoded.shape[1]):])
        context = torch.cat(chunks, 1)[:, :frames]
        # PatchTST supplies the encoder feature for both downstream routes;
        # the existing state head still applies its own causal local conv.
        return {"local": context, "context": context}

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(values)["context"]


class PatchTSTEMGIMUModel(MinimalEMGIMUModel):
    """Minimal four-head model with only the primary encoders replaced."""

    def __init__(self, width=128, patch=16, stride=4, layers=4, heads=4,
                 dropout=.1, react_context=100, adapter_layers=2,
                 future_steps=20, context_frames=600, chunk_frames=128):
        super().__init__(
            width=width, patch=patch, stride=stride, layers=layers,
            heads=heads, dropout=dropout, react_context=react_context,
            adapter_layers=adapter_layers, future_steps=future_steps)
        branch = dict(
            width=width, patch=patch, stride=stride, layers=layers,
            heads=heads, dropout=dropout, context_frames=context_frames,
            chunk_frames=chunk_frames)
        self.emg = CausalPatchTSTBranch(16, **branch)
        self.imu = CausalPatchTSTBranch(48, **branch)
        self.model_args.update(
            context_frames=context_frames, chunk_frames=chunk_frames)

