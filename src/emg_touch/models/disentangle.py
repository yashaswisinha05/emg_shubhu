"""Guided disentanglement: factor-aligned latent subspaces with adversarial leakage suppression.

Follows the guided-disentanglement formulation for multi-factor EMG: split
the latent into one subspace per generative factor plus a residual, then for
each factor k

    predictor      F_k(z_k)          pulls factor k INTO its own subspace
    discriminator  G_k(GRL(z_\k))    pushes factor k OUT of every other one

with the gradient reversal layer flipping the sign on the way back, so the
discriminator gets better at reading the factor while the encoder is driven
to make that impossible. The objective adds

    + lambda_cls * L_cls  -  lambda_adv * L_adv

Why this rather than a beta-VAE. Raising beta buys statistical independence
between latent dimensions, but independence is not the property wanted here -
it says nothing about WHICH dimension carries what, and it pays for the
structure with reconstruction quality. Supervised alignment plus adversarial
suppression targets the actual requirement directly, and it is available here
because the factor labels already exist in the dataset.

The factor worth spending this on in this project is participant identity.
Every model here is evaluated on a session-level split - held-out
participants, unseen electrode application - and cross-participant transfer
has been the persistent weakness. A GRL discriminator that tries to read the
session identity out of the destination subspace, and fails, is precisely a
destination representation that does not depend on who was wearing the
sensors. That is a targeted attack on the measured failure mode rather than a
generic regulariser.

Configuration (a1..a4, b1..b3, mix1..mix7) is available as a second factor
and is a genuine nuisance variable in the same way.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        return -ctx.strength * gradient, None


def gradient_reversal(values: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Identity forward, negated gradient backward."""
    return _GradientReversal.apply(values, strength)


def reversal_strength(step: int, total_steps: int, maximum: float = 1.0) -> float:
    """Ramp the reversal strength from 0 to `maximum` over training.

    Standard for adversarial domain adaptation and not cosmetic: at full
    strength from step one the discriminator is still random, so the encoder
    is pushed away from whatever noise it happens to read, which
    destabilises the representation before it means anything. The ramp lets
    the subspaces form first and the suppression tighten afterwards.
    """
    if total_steps <= 0:
        return maximum
    progress = min(max(step / float(total_steps), 0.0), 1.0)
    return float(maximum * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0))


class FactorHead(nn.Module):
    """Predictor plus adversarial discriminator for one factor."""

    def __init__(
        self,
        own_dim: int,
        other_dim: int,
        classes: int,
        hidden: int = 64,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.regression = regression
        outputs = classes
        self.predictor = nn.Sequential(
            nn.LayerNorm(own_dim), nn.Linear(own_dim, hidden), nn.GELU(),
            nn.Linear(hidden, outputs),
        )
        self.discriminator = nn.Sequential(
            nn.LayerNorm(other_dim), nn.Linear(other_dim, hidden), nn.GELU(),
            nn.Linear(hidden, outputs),
        )

    def forward(
        self, own: torch.Tensor, other: torch.Tensor, strength: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor(own), self.discriminator(gradient_reversal(other, strength))


class GuidedDisentangler(nn.Module):
    """Partitions a latent vector and applies factor-guided losses.

    factors: name -> (width, class count, is_regression). A width of 0 means
    the factor is only suppressed, never predicted - useful for a nuisance
    variable with no subspace of its own.
    """

    def __init__(self, factors: dict[str, tuple[int, int, bool]], residual_dim: int = 0) -> None:
        super().__init__()
        self.factor_names = list(factors)
        self.widths = {name: factors[name][0] for name in self.factor_names}
        self.residual_dim = residual_dim
        total = sum(self.widths.values()) + residual_dim
        self.latent_dim = total
        self.heads = nn.ModuleDict()
        offset = 0
        self.slices: dict[str, tuple[int, int]] = {}
        for name in self.factor_names:
            width = self.widths[name]
            self.slices[name] = (offset, offset + width)
            offset += width
        for name in self.factor_names:
            width, classes, regression = factors[name]
            if width <= 0:
                continue
            self.heads[name] = FactorHead(
                own_dim=width,
                other_dim=total - width,
                classes=classes,
                regression=regression,
            )

    def subspace(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.slices[name]
        return latent[..., start:end]

    def complement(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.slices[name]
        # The complement always includes the residual subspace, so the
        # residual is pushed away from every factor too and is left carrying
        # only what no factor explains.
        return torch.cat([latent[..., :start], latent[..., end:]], dim=-1)

    def forward(
        self,
        latent: torch.Tensor,
        labels: dict[str, torch.Tensor],
        strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        classification = latent.new_zeros(())
        adversarial = latent.new_zeros(())
        details: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            if name not in labels:
                continue
            own = self.subspace(latent, name)
            other = self.complement(latent, name)
            predicted, adversarial_prediction = head(own, other, strength)
            target = labels[name]
            if head.regression:
                own_loss = F.smooth_l1_loss(predicted, target)
                adversarial_loss = F.smooth_l1_loss(adversarial_prediction, target)
            else:
                own_loss = F.cross_entropy(predicted, target)
                adversarial_loss = F.cross_entropy(adversarial_prediction, target)
            classification = classification + own_loss
            adversarial = adversarial + adversarial_loss
            details[f"cls_{name}"] = own_loss.detach()
            details[f"adv_{name}"] = adversarial_loss.detach()
        return {
            "classification": classification,
            "adversarial": adversarial,
            **details,
        }


def disentanglement_loss(
    outputs: dict[str, torch.Tensor], config: dict[str, Any]
) -> torch.Tensor:
    """lambda_cls * L_cls + lambda_adv * L_adv.

    Both terms are ADDED here, unlike the sign in the source formulation,
    because the gradient reversal layer already negates the adversarial
    gradient reaching the encoder. Subtracting as well would double-negate
    it and train the encoder to help the discriminator - the exact opposite
    of the intent. The discriminator's own parameters sit downstream of the
    reversal and so are still trained to maximise accuracy.
    """
    settings = config.get("loss", {})
    return (
        float(settings.get("disentangle_cls_weight", 1.0)) * outputs["classification"]
        + float(settings.get("disentangle_adv_weight", 1.0)) * outputs["adversarial"]
    )
