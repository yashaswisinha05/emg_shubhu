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

from .disentangle import gradient_reversal
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


class PointingIntentVAE(nn.Module):
    """VAE + kinematic/anticipatory disentanglement, decoding to the pointing task.

    Combines two lines of work that had been developed separately in this
    project: the VAE + kinematic/anticipatory latent split (anticipatory_vae.py,
    built for the displacement-forecast task) and the transformer + grid+offset
    architecture (GridReachModel above, built for the screen-touch task). They
    had never been joined - the grid model has no latent, the VAE model has no
    grid head - so the anticipatory split's "EMG's unique contribution"
    measurement could only ever run on displacement, never on the actual
    target this project cares about.

    The join is mostly substitution, reusing what already exists rather than
    rederiving it:

      encoder       ModalityEncoder x{1,2} (from GridReachModel) - same
                    architecture that produced the 406 px result
      latent        to_mean/to_log_variance -> kinematic/anticipatory/residual
                    split (same shapes and same reasoning as
                    anticipatory_vae.py: the split targets the delay between
                    EMG and the motion it drives, not EMG vs acceleration as
                    two factors to separate)
      disentangle   anticipatory_losses(), kinematic_state() - imported
                    unmodified from anticipatory_vae.py; they only read
                    outputs["kinematic_prediction"/"kinematic_adversarial"/
                    "session_adversarial"], nothing rollout-specific
      decoder       SpatialPointHead + decode_grid_outputs (from
                    GridReachModel) reading the sampled latent, instead of
                    an attractor rollout - the pointing task has no physical
                    trajectory to integrate toward, only a point to localise
      reconstruction grid_point_loss(), unmodified, same as GridReachModel

    One deliberate departure from anticipatory_vae.py: the KL prior is
    N(0, I), not the virtual-leader attractor. That prior was built from
    measured tracker kinematics, and the tracker is excluded from every
    wearable-mode model in this project on principle - it is the label, not
    an input, and building a prior FROM it would be a narrow leak of the
    same information the model is supposed to predict. N(0, I) is also this
    project's own precedent: the original grid_fusion_vae, before the
    attractor prior existed, used exactly this.

    The measurement this buys, directly on the screen target for the first
    time: silence z_anticipatory (measure_anticipatory=True), re-decode
    through the same head, and compare pixel error. The gap is EMG's unique
    contribution to WHICH TARGET, not to how the hand is moving - the
    question this project has been trying to answer since being asked to
    run it on the pointing task rather than displacement.
    """

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int,
        use_imu: bool = True,
    ) -> None:
        super().__init__()
        model_config = config["model"]
        settings = config.get("virtual_leader", {})
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

        self.kinematic_dim = int(settings.get("kinematic_dim", 16))
        self.anticipatory_dim = int(settings.get("anticipatory_dim", 16))
        self.residual_dim = int(settings.get("residual_dim", 8))
        self.anticipatory_dropout = float(settings.get("anticipatory_dropout", 0.1))
        latent_dim = self.kinematic_dim + self.anticipatory_dim + self.residual_dim
        self.latent_dim = latent_dim

        def head(outputs: int, bias: float = 0.0) -> nn.Sequential:
            layers = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Linear(context_dim, 128),
                nn.GELU(), nn.Linear(128, outputs),
            )
            nn.init.normal_(layers[-1].weight, std=0.01)
            nn.init.constant_(layers[-1].bias, bias)
            return layers

        self.to_mean = head(latent_dim)
        # sigma ~ exp(0.5*-4) ~= 0.135 at init. Zero-bias (sigma ~ 1) was
        # measured to collapse the posterior in this project's first VAE
        # (latent SNR ~0.03, mu_std flat across training) - this bias is
        # that fix, applied here from the start rather than found again.
        self.to_log_variance = head(latent_dim, bias=-4.0)

        grid_width, grid_height = map(int, model_config.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.point_head = SpatialPointHead(
            latent_dim, grid_width, grid_height, float(model_config["dropout"]),
            direct_prediction=True, zero_initialize=False,
        )

        state_dim = 6  # velocity (3) + scaled acceleration (3)
        self.kinematic_predictor = nn.Sequential(
            nn.LayerNorm(self.kinematic_dim), nn.Linear(self.kinematic_dim, 64),
            nn.GELU(), nn.Linear(64, state_dim),
        )
        self.kinematic_discriminator = nn.Sequential(
            nn.LayerNorm(self.anticipatory_dim + self.residual_dim),
            nn.Linear(self.anticipatory_dim + self.residual_dim, 64),
            nn.GELU(), nn.Linear(64, state_dim),
        )
        sessions = int(settings.get("session_count", 0))
        self.session_discriminator = (
            nn.Sequential(
                nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 64), nn.GELU(),
                nn.Linear(64, sessions),
            )
            if sessions > 0 else None
        )

    def split(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.kinematic_dim
        b = a + self.anticipatory_dim
        return latent[..., :a], latent[..., a:b], latent[..., b:]

    def _decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.point_head(latent)
        decoded = decode_grid_outputs(
            raw["heatmap_logits"], raw["offset_logits"], self.grid_width, self.grid_height
        )
        outputs = {**raw, **decoded}
        finalize_point_prediction(outputs)
        return outputs

    def forward(
        self, emg: torch.Tensor, imu: torch.Tensor, time_mask: torch.Tensor,
        strength: float = 1.0, measure_anticipatory: bool = False,
    ) -> dict[str, torch.Tensor]:
        emg_context = self.emg_encoder(emg, time_mask)
        if self.use_imu:
            imu_context = self.imu_encoder(imu, time_mask)
            context = torch.cat([emg_context, imu_context], dim=-1)
        else:
            context = emg_context

        mean = self.to_mean(context)
        log_variance = self.to_log_variance(context).clamp(-10.0, 2.0)
        if self.training:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latent = mean

        kinematic, anticipatory, residual = self.split(latent)
        if self.training and self.anticipatory_dropout > 0.0:
            # Matches anticipatory_vae.py exactly: `anticipatory` is
            # reassigned to its dropped-out value here, so everything below
            # (the decoder AND the disentanglement losses) reads the
            # post-dropout split, not the original. Training with the
            # subspace occasionally zeroed is what makes silencing it at
            # evaluation in-distribution rather than an out-of-distribution
            # surprise - the fix already made once for the trajectory model,
            # carried over rather than re-discovered.
            keep = (
                torch.rand(anticipatory.shape[:-1] + (1,), device=latent.device)
                >= self.anticipatory_dropout
            ).to(latent.dtype)
            anticipatory = anticipatory * keep
            latent = torch.cat([kinematic, anticipatory, residual], dim=-1)

        outputs = self._decode(latent)
        outputs["latent_mu"] = mean
        outputs["latent_log_variance"] = log_variance
        outputs["z_kinematic"] = kinematic
        outputs["z_anticipatory"] = anticipatory
        outputs["z_residual"] = residual
        outputs["kinematic_prediction"] = self.kinematic_predictor(kinematic)
        outputs["kinematic_adversarial"] = self.kinematic_discriminator(
            gradient_reversal(torch.cat([anticipatory, residual], dim=-1), strength)
        )
        if self.session_discriminator is not None:
            outputs["session_adversarial"] = self.session_discriminator(
                gradient_reversal(latent, strength)
            )

        if measure_anticipatory:
            # Re-decode with the anticipatory subspace forced to zero. The
            # gap between this and the ordinary prediction is the part of
            # WHICH TARGET the model could not have named without EMG's
            # surplus over current motion - measured directly on the screen
            # coordinate rather than on displacement.
            muted = torch.cat([kinematic, torch.zeros_like(anticipatory), residual], dim=-1)
            muted_outputs = self._decode(muted)
            outputs["prediction_without_anticipatory"] = muted_outputs["prediction"]
        return outputs


def standard_kl_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), per sample, unreduced.

    Against a fixed isotropic prior rather than the virtual-leader attractor
    anticipatory_vae.py uses: that prior is built from measured tracker
    kinematics, and the tracker is label-only in every wearable-mode model
    in this project - deriving a prior from it would leak the target through
    the back door. N(0, I) is also this project's own earlier precedent
    (grid_fusion_vae, before the attractor prior was added).
    """
    mu = outputs["latent_mu"]
    log_variance = outputs["latent_log_variance"]
    return 0.5 * (
        mu.square() + log_variance.exp() - 1.0 - log_variance
    ).sum(dim=-1)
