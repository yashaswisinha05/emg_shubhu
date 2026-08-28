"""Learnable per-participant arm-length scale and endpoint-affine offset.

The physics branch's anthropometry (arm.LINK_LENGTH, arm3.LINK_LENGTH, and
the associated masses) is one fixed, literature-average set of constants
applied to every participant. A person with different arm proportions than
that average has a systematically wrong physics_prediction geometry for the
same joint angles - not something more training epochs fix, since the error
is in the anthropometry assumption itself, not the learned weights.

Diagnosed directly (grid_fusion_physics3, physics_weight both at 0.1 and
1.0x): physics_blend stayed frozen at its ~0.018 init in every run, and a
direct read of raw_blend.grad on real batches showed no consistent
incentive there - not because physics_prediction is close and just
underweighted, but because it visibly sits far from both the click target
and the fusion prediction in real trial playback. This is the one lever
that attacks that root cause rather than the blend weight around it.
"""
from __future__ import annotations

import json

import torch
from torch import nn


class ParticipantCalibration(nn.Module):
    """Per-participant scale (endpoint reach) and offset (post-affine)."""

    def __init__(self, split_file: str) -> None:
        super().__init__()
        with open(split_file, encoding="utf-8") as handle:
            split = json.load(handle)
        participants = sorted(split["subject_counts"].keys())
        self.participants = participants
        self.index = {name: position for position, name in enumerate(participants)}
        # One extra slot for a subject not seen at split-build time - falls
        # back to the population-average anthropometry (scale 1.0, offset 0)
        # since that slot only ever gets a gradient from batches it's never
        # queried on.
        self.unknown_index = len(participants)
        count = len(participants) + 1
        # raw=0 -> sigmoid(0)=0.5 -> scale=1.0 exactly, the midpoint of
        # [0.8, 1.2] - an unpersonalised participant reproduces the shared
        # anthropometry exactly, matching this project's convention of
        # zero-initialising new learnable knobs so they must earn any effect
        # (same rationale as the physics_blend and residual_torque inits).
        self.raw_scale = nn.Parameter(torch.zeros(count))
        self.offset = nn.Embedding(count, 2)
        nn.init.zeros_(self.offset.weight)

    def indices_for(self, subjects: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.index.get(name, self.unknown_index) for name in subjects],
            device=device,
            dtype=torch.long,
        )

    def scale(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-trial endpoint-reach scale, (batch,), bounded to [0.8, 1.2]."""
        return 0.8 + 0.4 * torch.sigmoid(self.raw_scale[indices])

    def offset_for(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-trial additive screen-space offset, (batch, 2)."""
        return self.offset(indices)
