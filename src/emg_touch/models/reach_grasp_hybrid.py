"""Patch-context pose/holding with frame-resolution causal event heads."""
from __future__ import annotations

import torch
from torch import nn

from .reach_grasp_patch_transformer import CausalPatchBranch


class ReachGraspHybrid(nn.Module):
    """Use global patch context for state/pose and local features for boundaries.

    The event head never receives a future patch token: ``local`` is produced by
    left-padded depthwise convolutions before patch aggregation.
    """
    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1):
        super().__init__()
        self.modality = modality
        kwargs = dict(width=width, patch=patch, stride=stride, layers=layers,
                      heads=heads, dropout=dropout)
        self.emg = CausalPatchBranch(16, **kwargs)
        self.imu = CausalPatchBranch(48, **kwargs)
        self.context_gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                          nn.Linear(width, 2))
        self.context_fusion = nn.Sequential(
            nn.Linear(width * 4, width * 2), nn.LayerNorm(width * 2),
            nn.GELU(), nn.Dropout(dropout))
        self.local_gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                        nn.Linear(width, 2))
        self.local_fusion = nn.Sequential(
            nn.Linear(width * 4, width), nn.LayerNorm(width), nn.GELU(),
            nn.Dropout(dropout))
        self.holding = nn.Linear(width * 2, 1)
        self.events = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, padding=0, groups=width),
            nn.GELU(), nn.Conv1d(width, width, kernel_size=1), nn.GELU(),
            nn.Conv1d(width, 2, kernel_size=1))
        self.position = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                      nn.Linear(width, 3))

    def _mask_and_weights(self, e, i, gate):
        if self.modality == "emg":
            i = torch.zeros_like(i)
        elif self.modality == "imu":
            e = torch.zeros_like(e)
        weights = gate(torch.cat([e, i], -1).detach()).softmax(-1)
        if self.modality == "emg":
            weights = torch.stack([torch.ones_like(weights[..., 0]),
                                   torch.zeros_like(weights[..., 1])], -1)
        elif self.modality == "imu":
            weights = torch.stack([torch.zeros_like(weights[..., 0]),
                                   torch.ones_like(weights[..., 1])], -1)
        return e, i, weights

    @staticmethod
    def _fuse(e, i, weights, layer):
        return layer(torch.cat([e * weights[..., 0:1], i * weights[..., 1:2],
                                e - i, e * i], -1))

    def forward(self, emg, imu):
        ef, inf = self.emg.forward_features(emg), self.imu.forward_features(imu)
        ec, ic, context_weights = self._mask_and_weights(
            ef["context"], inf["context"], self.context_gate)
        context = self._fuse(ec, ic, context_weights, self.context_fusion)
        el, il, local_weights = self._mask_and_weights(
            ef["local"], inf["local"], self.local_gate)
        local = self._fuse(el, il, local_weights, self.local_fusion)
        # Explicit left padding keeps the high-resolution event convolution causal.
        event_logits = self.events(torch.nn.functional.pad(
            local.transpose(1, 2), (4, 0))).transpose(1, 2)
        logits = torch.cat([self.holding(context), event_logits], -1)
        return {"logits": logits, "position": self.position(context),
                "fusion_weights": context_weights,
                "local_fusion_weights": local_weights,
                "context_features": context, "local_features": local,
                "emg_context_features": ef["context"]}
