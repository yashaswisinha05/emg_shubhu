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


class PointingBottleneckModel(nn.Module):
    """One objective, no sampling, no adversarial split - a deliberate retreat.

    Formerly PointingIntentVAE: VAE sampling (mu/log_variance -> reparameterised
    z) plus a kinematic/anticipatory/residual latent split trained by a GRL
    adversarial pair, on top of grid_point_loss. On the real dataset that
    model reached 426 px, worse than the plain deterministic GridReachModel's
    406 px, which was itself worse than a smaller GRU's 392.7 px - three
    architectures, ranked exactly inversely with how much machinery each one
    stacked onto the same encoder.

    Two mechanisms measurably explain the degradation, not speculation:

      sigma drifted toward the N(0,I) prior instead of staying informative
      (0.135 at init -> 0.5-0.9 by epoch 1, and it never came back down).
      For KL = 0.5(mu^2 + sigma^2 - 1 - log sigma^2), d(KL)/d(log sigma^2) =
      0.5(sigma^2 - 1), which is NEGATIVE whenever sigma < 1 - the gradient
      always pushes sigma back toward 1 unless the reconstruction loss wants
      a precise, informative z badly enough to overpower it. Here it did
      not, so every training step injected z = mu + sigma*eps with sigma
      approaching 1 into the point head - close to pure noise relative to
      mu's own spread (mu_std sat at 0.4-0.6 the whole run).

      the GRL adversarial pair and the 10%-of-steps anticipatory dropout
      are two more sources of training-time noise/competition for gradient,
      stacked on top of the sampling noise above.

    None of this was wrong to try - it is exactly the mechanism that helped
    on the displacement task, where the encoder demonstrably had structured
    signal to organise. On the pointing task, three independent
    measurements (the ablation sweep, the plain architecture swap, and the
    anticipatory-gain measurement itself reading -4.4 px) now agree that
    EMG's marginal information here is at or below noise. Stacking
    regularisation machinery onto a signal that thin does not reveal more
    of it; it adds noise the optimiser has to fight through instead.

    What remains: the encoder (unchanged - still the architecture that
    reached 406 px on its own) feeding the grid+offset head DIRECTLY,
    trained by grid_point_loss alone - functionally GridReachModel, plus
    the ReduceLROnPlateau schedule described in the training script.

    A compressed bottleneck (context_dim -> a narrower latent before the
    head) was tried first as a capacity-regularisation lever and measured,
    not assumed: an 8-example overfit test that GridReachModel's identical
    check collapses to 24 px in 300 steps plateaued around 250 px through a
    256->40 bottleneck and did not go lower. Compressing by more than 6x
    right before a head that has to emit 40 heatmap logits plus 80 offset
    values was too aggressive - it constrained even trivial memorisation,
    which is a bad sign for fitting real signal. Removed rather than kept
    on the strength of the idea alone once the numbers said otherwise.

    The EMG-unique-contribution measurement this model's predecessor
    produced (silencing z_anticipatory) is gone, honestly rather than kept
    as a decoration: nothing trains the bottleneck to place EMG's surplus
    in a nameable subspace anymore, so zeroing part of it would measure the
    cost of destroying an arbitrary compressed feature, not EMG's
    contribution to anything. Getting that measurement back requires the
    auxiliary losses this class deliberately removes - a real trade this
    project can revisit if EMG turns out to matter once IMU is disentangled
    at the mechanism level, but not before.
    """

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
        self.point_head = SpatialPointHead(
            context_dim, grid_width, grid_height, float(model_config["dropout"]),
            direct_prediction=True, zero_initialize=False,
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

        raw = self.point_head(context)
        decoded = decode_grid_outputs(
            raw["heatmap_logits"], raw["offset_logits"], self.grid_width, self.grid_height
        )
        outputs = {**raw, **decoded}
        finalize_point_prediction(outputs)
        return outputs
