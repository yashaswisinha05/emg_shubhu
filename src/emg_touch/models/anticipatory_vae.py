"""Trajectory VAE with a kinematic / anticipatory latent split.

The point of this model is to answer a question the plain trajectory VAE
cannot: when the model beats constant-velocity extrapolation, is that the
muscle signal, or just a recurrent encoder watching the recent shape of the
trajectory? Ablating EMG answers it with one number from a separate run.
This answers it continuously, from the architecture, per trial.

Why EMG and acceleration are NOT two factors to separate. They sit on one
causal chain - neural drive drives muscle force drives joint torque drives
M(q)qddot = tau drives acceleration - so asking a discriminator to separate
them is asking it to separate a cause from its own effect, and it would be
fighting the physics rather than revealing structure.

The exploitable asymmetry is the delay along that chain. Electromechanical
delay is 40-80 ms, so at time t the measured acceleration reflects neural
drive from ~60 ms ago while EMG reflects drive now. EMG therefore carries
information the kinematics have not expressed yet, and that surplus - not
EMG itself - is the thing worth isolating. Hence:

    z_kin   supervised to reconstruct the CURRENT motion state. Whatever the
            kinematics already determine is pulled in here.
    z_ant   held behind a gradient reversal layer against that same
            prediction, so it is actively prevented from carrying anything
            the current motion explains. It still feeds the destination, so
            the only way it can earn its keep is by carrying drive that has
            not been executed yet.
    z_res   nuisance; suppressed alongside z_ant and never predicted from.

The measurement this buys: zero z_ant at evaluation and re-roll the
trajectory. The increase in error is EMG's unique contribution, in
centimetres, on every trial - reported as `anticipatory_gain_m`. If it is
zero, EMG adds nothing here regardless of how well the model scores, and
that is worth knowing directly rather than inferring from a separate
ablation run.

A session discriminator is available on the same machinery. Every model in
this project is scored on held-out sessions and cross-participant transfer
has been the persistent weakness, so a discriminator that cannot read
session identity out of the latent is exactly a participant-invariant
representation.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .disentangle import gradient_reversal
from .trajectory_intent_vae import VirtualLeaderTrajectoryVAE

# Accelerations run ~10x larger than velocities in these reaches, so the
# kinematic target is scaled to put both on a comparable footing. Without it
# the reconstruction term is effectively an acceleration-only loss and z_kin
# never learns to hold velocity.
ACCELERATION_SCALE = 10.0


def kinematic_state(velocity: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
    """The current motion state z_kin is asked to reproduce.

    Position is deliberately excluded. It is an absolute world coordinate, so
    predicting it would let z_kin encode where in the room the hand happens
    to be - a nuisance the destination genuinely depends on, which would make
    the adversarial term fight the task instead of the redundancy.
    """
    return torch.cat([velocity, acceleration / ACCELERATION_SCALE], dim=-1)


class AnticipatoryTrajectoryVAE(VirtualLeaderTrajectoryVAE):
    """Structured latent over the attractor decoder."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        settings = config.get("virtual_leader", {})
        context_dim = int(
            settings.get("context_dim", 2 * int(config["model"]["d_model"]))
        )
        self.kinematic_dim = int(settings.get("kinematic_dim", 16))
        self.anticipatory_dim = int(settings.get("anticipatory_dim", 16))
        self.residual_dim = int(settings.get("residual_dim", 8))
        latent_dim = self.kinematic_dim + self.anticipatory_dim + self.residual_dim
        self.latent_dim = latent_dim

        self.to_latent = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        # The destination heads now read the structured latent rather than
        # the raw context, so everything reaching the destination has passed
        # through the partition.
        self.posterior_mean = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, self.position_dim),
        )
        self.posterior_log_variance = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, self.position_dim),
        )
        self.dynamics = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 2),
        )
        # Small random, never zero: a zero final weight makes d(head)/d(latent)
        # exactly zero and silently severs the encoder. Measured that failure
        # once already in this project; the bias still carries the sane start.
        for head, bias in (
            (self.posterior_mean, 0.0),
            (self.posterior_log_variance, -6.0),
            (self.dynamics, 0.0),
        ):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.constant_(head[-1].bias, bias)

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
            if sessions > 0
            else None
        )

    def split(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.kinematic_dim
        b = a + self.anticipatory_dim
        return latent[..., :a], latent[..., a:b], latent[..., b:]

    def _destination(
        self, latent: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = position + self.posterior_mean(latent)
        log_variance = self.posterior_log_variance(latent).clamp(-10.0, 2.0)
        return mean, log_variance

    def forward(
        self,
        context: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        horizon: int | None = None,
        strength: float = 1.0,
        measure_anticipatory: bool = False,
    ) -> dict[str, torch.Tensor]:
        horizon = int(horizon or self.horizon)
        latent = self.to_latent(context)
        kinematic, anticipatory, residual = self.split(latent)

        eta, drag = self.attractor_parameters(latent)
        mean, log_variance = self._destination(latent, position)
        if self.training:
            sampled = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            sampled = mean
        predicted = self.rollout(sampled, position, velocity, eta, drag, horizon)

        displacement = (
            acceleration + drag.unsqueeze(-1) * velocity
        ) / eta.unsqueeze(-1)
        limit = float(self.prior_displacement_limit)
        magnitude = displacement.norm(dim=-1, keepdim=True)
        displacement = displacement * (limit / magnitude.clamp_min(limit))

        outputs: dict[str, torch.Tensor] = {
            "trajectory": predicted,
            "destination_mu": mean,
            "destination_log_variance": log_variance,
            "destination_sigma": torch.exp(0.5 * log_variance),
            "prior_mu": (position + displacement).detach(),
            "eta": eta,
            "drag": drag,
            "latent": latent,
            "z_kinematic": kinematic,
            "z_anticipatory": anticipatory,
            "z_residual": residual,
            # Predictor pulls the current motion state INTO z_kin; the
            # discriminator behind the reversal pushes it OUT of the rest.
            "kinematic_prediction": self.kinematic_predictor(kinematic),
            "kinematic_adversarial": self.kinematic_discriminator(
                gradient_reversal(torch.cat([anticipatory, residual], dim=-1), strength)
            ),
        }
        if self.session_discriminator is not None:
            outputs["session_adversarial"] = self.session_discriminator(
                gradient_reversal(latent, strength)
            )

        if measure_anticipatory:
            # Re-roll with the anticipatory subspace silenced. The gap is the
            # part of the prediction that current kinematics could not have
            # produced - EMG's unique contribution, in metres.
            muted = torch.cat(
                [kinematic, torch.zeros_like(anticipatory), residual], dim=-1
            )
            muted_eta, muted_drag = self.attractor_parameters(muted)
            muted_mean, _ = self._destination(muted, position)
            outputs["trajectory_without_anticipatory"] = self.rollout(
                muted_mean, position, velocity, muted_eta, muted_drag, horizon
            )
        return outputs


def anticipatory_losses(
    outputs: dict[str, torch.Tensor],
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    config: dict[str, Any],
    session: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Factor-guided terms, all ADDED.

    The adversarial terms are added rather than subtracted because the
    reversal layer already negates the gradient reaching the encoder.
    Subtracting as well would double-negate and train the encoder to help
    the discriminator - the opposite of the intent. The discriminators' own
    weights sit downstream of the reversal and are still trained to be as
    accurate as they can.
    """
    settings = config.get("loss", {})
    target = kinematic_state(velocity, acceleration)
    predicted = F.smooth_l1_loss(outputs["kinematic_prediction"], target)
    adversarial = F.smooth_l1_loss(outputs["kinematic_adversarial"], target)
    total = (
        float(settings.get("kinematic_predict_weight", 1.0)) * predicted
        + float(settings.get("kinematic_adversarial_weight", 0.5)) * adversarial
    )
    result = {
        "kinematic_predict": predicted,
        "kinematic_adversarial": adversarial,
        "disentangle": total,
    }
    if session is not None and "session_adversarial" in outputs:
        session_loss = F.cross_entropy(outputs["session_adversarial"], session)
        result["session_adversarial"] = session_loss
        result["disentangle"] = total + float(
            settings.get("session_adversarial_weight", 0.2)
        ) * session_loss
    return result
