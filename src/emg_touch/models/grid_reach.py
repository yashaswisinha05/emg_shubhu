"""Grid+offset screen-touch prediction from wearables.

The reach-target model (train_reach_target_model.py) reached the screen
target with a single causal GRU and three linear heads, direct-regressing
the coordinate: 329-460 px per condition. The original screen-touch
pipeline in this project, on a different dataset, reached 184 px with two
architectural choices this model borrows rather than reinvents:

  a patch transformer instead of a GRU (PatchTransformerEncoder, already
  built and tested in this repository for exactly this kind of causal
  EMG/IMU sequence)

  a CenterNet-style grid+offset head instead of direct regression
  (SpatialPointHead / decode_grid_outputs / grid_point_loss, likewise
  already built and tested). Direct coordinate regression has to get the
  right real number from an unconstrained continuous space; grid+offset
  turns most of the problem into a classification (which of grid_width x
  grid_height cells) with a small residual regression (where inside that
  cell), which is an easier optimisation landscape and is why the original
  pipeline used it as the default over plain regression.

grid_point_loss is imported and called unmodified. It only reads
outputs["heatmap_logits"/"offsets"/"candidates"/"soft_prediction"/
"direct_prediction"] and batch["target"/"canvas_size"/"loss_weight"] - none
of which are coupled to the old dataset's feature layout - so this reuses
the exact tested loss composition (heatmap cross-entropy + offset Huber +
pixel Charbonnier + a Wasserstein-like transport term) rather than a
reimplementation that could drift from it.

What is deliberately NOT carried over: the original's residual-EMG-onto-a
-frozen-pretrained-IMU-backbone architecture (GridFusionRegressor). That
scheme depends on a separate pretraining run (grid_imu trained alone, then
frozen) which this dataset and task have no equivalent of yet. Here EMG and
IMU are each given their own patch-transformer encoder of equal capacity and
fused by concatenation before a single head - simpler, trainable end to end,
and a fair starting point before any asymmetric pretraining scheme is worth
building. Prediction mode is direct_aux_grid regardless (matching the
original's best-performing setting): the grid loss is auxiliary supervision
that regularises the same trunk the direct head reads from, and the direct
head's regression is the reported prediction.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .grid_point import SpatialPointHead, decode_grid_outputs, finalize_point_prediction
from .layers import PatchTransformerEncoder


class ModalityEncoder(nn.Module):
    """One PatchTransformerEncoder, pooled to a single context vector."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        patch_length: int,
        patch_stride: int,
        kernel_sizes: list[int],
    ) -> None:
        super().__init__()
        self.encoder = PatchTransformerEncoder(
            input_dim=channels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            patch_length=patch_length,
            patch_stride=patch_stride,
            kernel_sizes=kernel_sizes,
        )

    def forward(self, values: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        _, pooled, _ = self.encoder(values, time_mask)
        return pooled


class GridReachModel(nn.Module):
    """EMG (+IMU) -> grid+offset screen touch prediction. Tracker never enters."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int,
        use_imu: bool = True,
    ) -> None:
        super().__init__()
        model_config = config["model"]
        width = int(model_config["d_model"])
        self.use_imu = use_imu
        common = dict(
            d_model=width,
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            dropout=float(model_config["dropout"]),
            patch_length=int(model_config["patch_length"]),
            patch_stride=int(model_config["patch_stride"]),
            kernel_sizes=list(model_config["tcn_kernel_sizes"]),
        )
        self.emg_encoder = ModalityEncoder(emg_channels, **common)
        if use_imu:
            self.imu_encoder = ModalityEncoder(imu_channels, **common)
        context_dim = width * 2 if use_imu else width
        grid_width, grid_height = map(int, model_config.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.head = SpatialPointHead(
            context_dim,
            grid_width,
            grid_height,
            float(model_config["dropout"]),
            direct_prediction=True,
            # Not a residual onto a pretrained base - this trunk is the
            # only source of both the grid and the direct prediction, so it
            # needs real gradient on every head from step one. Zero-init is
            # for a branch that must start as a no-op; this branch IS the
            # model.
            zero_initialize=False,
        )

    def forward(
        self, emg: torch.Tensor, imu: torch.Tensor, time_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        emg_context = self.emg_encoder(emg, time_mask)
        if self.use_imu:
            imu_context = self.imu_encoder(imu, time_mask)
            context = torch.cat([emg_context, imu_context], dim=-1)
        else:
            context = emg_context
        raw = self.head(context)
        decoded = decode_grid_outputs(
            raw["heatmap_logits"], raw["offset_logits"], self.grid_width, self.grid_height
        )
        outputs = {**raw, **decoded}
        finalize_point_prediction(outputs)
        return outputs
