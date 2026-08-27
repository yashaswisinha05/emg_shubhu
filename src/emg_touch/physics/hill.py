"""Differentiable Hill-type muscle model.

Follows Katibeh et al. (Neural Comput & Applic 2025, doi:10.1007/s00521-024-10813-y),
"Simultaneous and continuous estimation of upper limb kinematics of shoulder
press movements: state-space EMG model", with two deliberate departures:

- Activation dynamics are added. The paper feeds filtered EMG straight in as
  activation. Electromechanical delay is ~40-80 ms against reaches of ~1 s
  here, and that lead is precisely the quantity that could help at the early
  prediction cutoffs, so it is modelled explicitly rather than discarded.

- The muscle set differs. The paper uses anterior deltoid plus two triceps
  heads; this dataset records AD, LD, BB, TB - a complete agonist/antagonist
  pair at each joint. The affine length-angle relation of the paper's Eq. 5 is
  kept, but biceps and triceps are treated as biarticular.

Every physiological parameter is a learnable tensor initialised at a
literature value and constrained to a plausible range, so the model can be
fitted end to end while remaining interpretable.
"""
from __future__ import annotations

import torch
from torch import nn

# Paper Eq. 4: active force-length polynomial, valid on 0.5 <= l <= 1.5.
FORCE_LENGTH_COEFFICIENTS = (-2.06, 6.16, -3.13)

# Index order matches schema.SENSORS: S0=AD, S4=LD, S8=BB, S12=TB.
MUSCLE_NAMES = ("AD", "LD", "BB", "TB")
# Nominal peak isometric force (N), paper Table 2 where available; the biceps
# value is a standard literature figure since the paper has no biceps.
NOMINAL_FORCE = (120.0, 120.0, 100.0, 120.0)
# Sign of each muscle's moment arm at (shoulder, elbow). Zero means the muscle
# does not span that joint. AD/LD act at the shoulder only; BB flexes both,
# TB extends both.
MOMENT_ARM_SIGNS = (
    (1.0, 0.0),
    (1.0, 0.0),
    (0.5, 1.0),
    (-0.5, -1.0),
)


def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map an unconstrained parameter into (low, high)."""
    return low + (high - low) * torch.sigmoid(raw)


class ActivationDynamics(nn.Module):
    """EMG envelope -> muscle activation.

    Electromechanical delay, a first-order activation filter with distinct
    rise and fall time constants, and the standard exponential nonlinearity.
    All are causal, so activation at time t depends only on EMG up to t.
    """

    def __init__(self, muscles: int = 4, sample_rate_hz: float = 148.148) -> None:
        super().__init__()
        self.sample_rate_hz = float(sample_rate_hz)
        self.muscles = muscles
        # Raw parameters; physiological ranges are imposed in the properties.
        self.raw_delay = nn.Parameter(torch.zeros(muscles))
        self.raw_activation_tau = nn.Parameter(torch.zeros(muscles))
        self.raw_deactivation_tau = nn.Parameter(torch.zeros(muscles))
        self.raw_shape = nn.Parameter(torch.zeros(muscles))

    @property
    def delay_s(self) -> torch.Tensor:
        return _bounded(self.raw_delay, 0.02, 0.10)      # 20-100 ms

    @property
    def activation_tau_s(self) -> torch.Tensor:
        return _bounded(self.raw_activation_tau, 0.01, 0.05)

    @property
    def deactivation_tau_s(self) -> torch.Tensor:
        return _bounded(self.raw_deactivation_tau, 0.03, 0.12)

    @property
    def shape(self) -> torch.Tensor:
        return _bounded(self.raw_shape, -3.0, -0.05)     # A in (-3, 0)

    def forward(self, excitation: torch.Tensor) -> torch.Tensor:
        """excitation: (B, T, M) in roughly [0, 1]. Returns activation (B, T, M)."""
        batch, steps, muscles = excitation.shape
        dt = 1.0 / self.sample_rate_hz

        # Electromechanical delay by fractional-sample shift, kept
        # differentiable by linear interpolation between integer lags.
        lag = self.delay_s * self.sample_rate_hz
        low = torch.floor(lag).clamp(min=0)
        frac = (lag - low).view(1, 1, muscles)
        delayed = torch.zeros_like(excitation)
        for muscle in range(muscles):
            shift = int(low[muscle].item())
            a = torch.nn.functional.pad(
                excitation[:, :, muscle], (shift, 0)
            )[:, :steps]
            b = torch.nn.functional.pad(
                excitation[:, :, muscle], (shift + 1, 0)
            )[:, :steps]
            delayed[:, :, muscle] = (
                (1.0 - frac[0, 0, muscle]) * a + frac[0, 0, muscle] * b
            )

        # First-order activation filter with rise/fall asymmetry.
        activation = torch.zeros_like(delayed)
        state = torch.zeros(batch, muscles, device=excitation.device, dtype=excitation.dtype)
        rise, fall = self.activation_tau_s, self.deactivation_tau_s
        for step in range(steps):
            target = delayed[:, step]
            tau = torch.where(target > state, rise, fall).clamp_min(1e-4)
            state = state + dt * (target - state) / tau
            activation[:, step] = state

        # Exponential nonlinearity (Zajac). Shape A < 0 gives the usual
        # concave relation between neural drive and force-generating capacity.
        shape = self.shape.view(1, 1, muscles)
        return (torch.exp(shape * activation) - 1.0) / (torch.exp(shape) - 1.0)


class HillMuscle(nn.Module):
    """Muscle length from joint angles, then Hill force, then joint torque."""

    def __init__(self, muscles: int = 4) -> None:
        super().__init__()
        self.muscles = muscles
        signs = torch.tensor(MOMENT_ARM_SIGNS[:muscles], dtype=torch.float32)
        self.register_buffer("moment_arm_signs", signs, persistent=False)
        # Moment-arm magnitudes (m), paper Table 3 scale.
        self.raw_moment_arm = nn.Parameter(torch.zeros(muscles, 2))
        # Peak isometric force scale, multiplying the nominal values.
        self.raw_force_scale = nn.Parameter(torch.zeros(muscles))
        self.register_buffer(
            "nominal_force",
            torch.tensor(NOMINAL_FORCE[:muscles], dtype=torch.float32),
            persistent=False,
        )
        # Reference (optimal) muscle length offset, normalising l/l0 to ~1.
        self.raw_length_offset = nn.Parameter(torch.zeros(muscles))

    @property
    def moment_arm(self) -> torch.Tensor:
        return _bounded(self.raw_moment_arm, 0.01, 0.09) * self.moment_arm_signs

    @property
    def peak_force(self) -> torch.Tensor:
        return self.nominal_force * _bounded(self.raw_force_scale, 0.25, 4.0)

    def normalised_length(self, angles: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 5: length is affine in the joint angles it spans.

        angles: (..., 2) = (shoulder, elbow). Returns (..., M) normalised so
        that the neutral posture sits near the optimal length of 1.0.
        """
        # A muscle shortens as the joint it flexes rotates positively, hence
        # the negative sign on the moment-arm contribution.
        contribution = -(angles.unsqueeze(-2) * self.moment_arm).sum(dim=-1)
        offset = 1.0 + _bounded(self.raw_length_offset, -0.3, 0.3)
        return contribution + offset

    def force(
        self, activation: torch.Tensor, angles: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Hill force (paper Eqs. 3-4). Shapes (..., M)."""
        length = self.normalised_length(angles)
        q0, q1, q2 = FORCE_LENGTH_COEFFICIENTS
        active = q0 + q1 * length + q2 * length * length
        # The polynomial is only valid on [0.5, 1.5]; outside it the active
        # element produces no force.
        in_range = ((length >= 0.5) & (length <= 1.5)).to(length.dtype)
        active = torch.clamp(active, min=0.0) * in_range
        passive = torch.exp(torch.clamp(10.0 * length - 15.0, max=4.0))
        # Contraction velocity along each muscle, from joint angular velocity.
        contraction = -(velocity.unsqueeze(-2) * self.moment_arm).sum(dim=-1)
        # Hill force-velocity: shortening (negative) weakens the muscle.
        force_velocity = torch.clamp(1.0 - contraction, min=0.05, max=1.8)
        return (active * force_velocity * activation + passive) * self.peak_force

    def torque(
        self, activation: torch.Tensor, angles: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Joint torque, (..., 2). Moment arms are the length-angle Jacobian."""
        muscle_force = self.force(activation, angles, velocity)
        return torch.einsum("...m,mj->...j", muscle_force, self.moment_arm)
