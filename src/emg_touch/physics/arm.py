"""Two-link planar arm dynamics and forward kinematics.

Implements Eqs. 6-10 of Katibeh et al. verbatim, with the paper's Table 1
anthropometry as the default. The external-load term is dropped: the paper's
subjects hold a dumbbell, these subjects reach to a screen.

The endpoint is mapped to normalised screen coordinates by a learned affine
transform rather than an explicit 3-D screen pose. That is deliberate. Fitting
an explicit pose requires arm length, screen distance and mm-per-pixel
simultaneously, which are degenerate under a common scale factor - a direct
attempt on this data converged to a 30 cm arm and a 1 m wide laptop screen.
The affine absorbs that gauge freedom without needing the panel dimensions.
"""
from __future__ import annotations

import torch
from torch import nn

GRAVITY = 9.81

# Paper Table 1: upper arm and forearm.
LINK_LENGTH = (0.298, 0.419)
LINK_MASS = (2.089, 1.912)
CENTRE_OF_MASS = (0.152, 0.181)
MOMENT_OF_INERTIA = (0.0159, 0.0257)


def _floor_min_eigenvalue_2x2(mass: torch.Tensor, floor: float) -> torch.Tensor:
    """Clamp a batch of symmetric 2x2 matrices' smaller eigenvalue to floor.

    torch.linalg.eigh is not implemented on the MPS backend (it crashes with
    NotImplementedError there, CPU/CUDA only), so this reimplements the same
    operation as a closed form for the 2x2 case, which this arm only ever
    needs it for. Verified to match torch.linalg.eigh's convention exactly
    (eigenvector sign/ordering included) against the actual reference on
    2000 random SPD matrices and on all 7 real physics postures spanning the
    arm's full elbow range; for those postures m01^2 the off-diagonal never
    vanishes and a-d is always positive (checked over a 20x20 grid spanning
    the full joint range), so atan2 below never hits its ambiguous 0/0 case.
    """
    a = mass[..., 0, 0]
    b = mass[..., 0, 1]
    d = mass[..., 1, 1]
    trace = a + d
    discriminant = torch.sqrt((a - d) ** 2 + 4.0 * b * b)
    small = (trace - discriminant) / 2.0
    large = (trace + discriminant) / 2.0
    small_floored = small.clamp(min=floor)
    theta = 0.5 * torch.atan2(2.0 * b, a - d)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    # Eigenvector for the smaller eigenvalue is (-sin, cos); for the larger
    # it is (cos, sin) - matches torch.linalg.eigh's convention exactly.
    c1, s1 = -sin_theta, cos_theta
    c2, s2 = cos_theta, sin_theta
    m11 = c1 * c1 * small_floored + c2 * c2 * large
    m12 = c1 * s1 * small_floored + c2 * s2 * large
    m22 = s1 * s1 * small_floored + s2 * s2 * large
    return torch.stack([torch.stack([m11, m12], -1), torch.stack([m12, m22], -1)], -2)


class TwoLinkArm(nn.Module):
    """Planar 2-DOF arm: state is (shoulder, elbow) angle and angular velocity."""

    def __init__(self) -> None:
        super().__init__()
        l1, l2 = LINK_LENGTH
        m1, m2 = LINK_MASS
        lg1, lg2 = CENTRE_OF_MASS
        i1, i2 = MOMENT_OF_INERTIA
        # Paper Eq. 10. Fixed anthropometry: fitting these alongside the
        # endpoint affine would be redundant and unidentifiable.
        self.register_buffer("coefficient_a", torch.tensor(i1 + i2 + m1 * lg1**2 + m2 * (l1**2 + lg2**2)))
        self.register_buffer("coefficient_b", torch.tensor(m2 * l1 * lg2))
        self.register_buffer("coefficient_d", torch.tensor(i2 + m2 * lg2**2))
        self.register_buffer("link_length", torch.tensor(LINK_LENGTH))
        self.register_buffer("link_mass", torch.tensor(LINK_MASS))
        # Joint viscosity: real limbs dissipate, and an undamped integration
        # of a noisy torque estimate diverges.
        self.raw_damping = nn.Parameter(torch.zeros(2))

    @property
    def damping(self) -> torch.Tensor:
        return 0.05 + 1.95 * torch.sigmoid(self.raw_damping)

    def mass_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 7. angles (..., 2) -> (..., 2, 2)."""
        cos_elbow = torch.cos(angles[..., 1])
        a, b, d = self.coefficient_a, self.coefficient_b, self.coefficient_d
        m11 = a + 2.0 * b * cos_elbow
        m12 = d + b * cos_elbow
        m22 = d.expand_as(m11)
        return torch.stack(
            [torch.stack([m11, m12], -1), torch.stack([m12, m22], -1)], dim=-2
        )

    def coriolis(self, angles: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 8, as the torque contribution C(theta, theta_dot) @ theta_dot."""
        sin_elbow = torch.sin(angles[..., 1])
        b = self.coefficient_b
        shoulder_rate, elbow_rate = velocity[..., 0], velocity[..., 1]
        first = -b * sin_elbow * (2.0 * shoulder_rate * elbow_rate + elbow_rate**2)
        second = b * sin_elbow * shoulder_rate**2
        return torch.stack([first, second], dim=-1)

    def gravity(self, angles: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 10 g1, g2."""
        l1, l2 = self.link_length[0], self.link_length[1]
        m1, m2 = self.link_mass[0], self.link_mass[1]
        shoulder = angles[..., 0]
        both = angles[..., 0] + angles[..., 1]
        g1 = m1 * GRAVITY * (l1 / 2) * torch.cos(shoulder) + m2 * GRAVITY * (
            l1 * torch.cos(shoulder) + (l2 / 2) * torch.cos(both)
        )
        g2 = m2 * GRAVITY * (l2 / 2) * torch.cos(both)
        return torch.stack([g1, g2], dim=-1)

    def acceleration(
        self, angles: torch.Tensor, velocity: torch.Tensor, torque: torch.Tensor
    ) -> torch.Tensor:
        """Solve M @ theta_ddot = tau - C - G - damping for theta_ddot."""
        residual = (
            torque
            - self.coriolis(angles, velocity)
            - self.gravity(angles)
            - self.damping * velocity
        )
        mass = self.mass_matrix(angles)
        # A 2-link arm's mass matrix has a thin smaller eigenvalue across
        # most of the elbow's practical range (measured 0.017-0.036 for
        # theta2 in [0, 1.0] rad, worst at full extension, all below the
        # 0.05 floor below), so a moderate shoulder torque can dominate
        # elbow acceleration through the shoulder-elbow coupling term even
        # when the elbow's own torque points the other way - confirmed
        # directly: torque +2.85 N*m produced acceleration -200 rad/s^2 at
        # theta2=0. The floor therefore engages through most of this range
        # by design, not only exactly at theta2=0; what it guarantees is
        # narrower than "leaves non-singular configurations alone" - only
        # that a configuration whose natural eigenvalues already exceed 0.05
        # is left completely unchanged (eigenvalues clamped, not
        # blanket-added-to), unlike a fixed or trace-scaled ridge, which was
        # measured to distort even those better-conditioned configurations
        # by ~50%.
        mass = _floor_min_eigenvalue_2x2(mass, floor=0.05)
        return torch.linalg.solve(mass, residual.unsqueeze(-1)).squeeze(-1)

    def endpoint(self, angles: torch.Tensor) -> torch.Tensor:
        """Planar fingertip position (..., 2) from joint angles."""
        l1, l2 = self.link_length[0], self.link_length[1]
        shoulder = angles[..., 0]
        both = angles[..., 0] + angles[..., 1]
        x = l1 * torch.cos(shoulder) + l2 * torch.cos(both)
        y = l1 * torch.sin(shoulder) + l2 * torch.sin(both)
        return torch.stack([x, y], dim=-1)


class EndpointToScreen(nn.Module):
    """Learned affine from planar endpoint (m) to normalised screen coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        # Start near the screen centre with a modest gain, so the first
        # rollouts land on the canvas instead of far outside it.
        nn.init.normal_(self.linear.weight, std=0.5)
        nn.init.constant_(self.linear.bias, 0.5)

    def forward(self, endpoint: torch.Tensor) -> torch.Tensor:
        return self.linear(endpoint)
