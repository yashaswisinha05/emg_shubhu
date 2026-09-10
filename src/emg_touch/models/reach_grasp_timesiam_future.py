"""TimeSiam-inspired masked-future model for wearable reach prediction.

This is an independent adaptation, not a copy of the authors' implementation.
It keeps the selected gripper/pose model and replaces its flat multi-horizon
future projector with shared lineage encoding and past-to-future cross-attention.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .reach_grasp_gripper_pixel import GoalConsistentGripperPoseModel


class TimeSiamFutureGripperPoseModel(GoalConsistentGripperPoseModel):
    """Causal wearable model with lineage-conditioned masked-future queries."""

    def __init__(self, future_steps=20, lineage_context=25, **kwargs):
        if future_steps <= 0 or lineage_context <= 0:
            raise ValueError("future_steps and lineage_context must be positive")
        # The parent future projector is intentionally absent. This class owns
        # a separate decoder while retaining every current-task head.
        super().__init__(future_steps=0, **kwargs)
        width = kwargs.get("width", 128)
        heads = kwargs.get("heads", 4)
        dropout = kwargs.get("dropout", .1)
        self.future_steps = future_steps
        self.lineage_context = lineage_context
        self.future_pose = None

        self.lineage = nn.Embedding(future_steps + 1, width)
        self.context_token = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width))
        shared = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.siamese_encoder = nn.TransformerEncoder(shared, 1, nn.LayerNorm(width))
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True)
        refine = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.future_refine = nn.TransformerEncoder(refine, 1, nn.LayerNorm(width))
        self.future_pose_projector = nn.Linear(width, 9)
        # Pre-training target order: 8 EMG values followed by 24 IMU values.
        self.future_wearable_projector = nn.Linear(width, 32)

    def _lineage_decode(self, context):
        batch, length, _ = context.shape
        width = self.lineage.embedding_dim
        token = self.context_token(context)

        # For each frame t, expose only [t-context+1, ..., t]. Padding exists
        # before the trial starts and is explicitly hidden from attention.
        padded = F.pad(token, (0, 0, self.lineage_context - 1, 0))
        memory = padded.unfold(1, self.lineage_context, 1).permute(0, 1, 3, 2)
        memory = memory.reshape(batch * length, self.lineage_context, width)
        memory = memory + self.lineage.weight[0]
        memory = self.siamese_encoder(memory)

        horizons = self.lineage.weight[1:self.future_steps + 1]
        query = token.reshape(batch * length, 1, width) + horizons.unsqueeze(0)
        query = self.siamese_encoder(query)

        t = torch.arange(length, device=context.device)[:, None]
        k = torch.arange(self.lineage_context, device=context.device)[None, :]
        padding = k < (self.lineage_context - 1 - t).clamp_min(0)
        padding = padding.unsqueeze(0).expand(batch, -1, -1).reshape(
            batch * length, self.lineage_context)
        attended, _ = self.cross_attention(
            query, memory, memory, key_padding_mask=padding, need_weights=False)
        decoded = self.future_refine(query + attended)
        return decoded.reshape(batch, length, self.future_steps, width)

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        decoded = self._lineage_decode(result["context_features"])
        pose = self.future_pose_projector(decoded)
        result["future_position"] = pose[..., :3]
        result["future_orientation_6d"] = pose[..., 3:]
        result["future_wearable"] = self.future_wearable_projector(decoded)
        result["lineage_features"] = decoded
        return result
