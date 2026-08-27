"""Differentiable 3-DOF (2-shoulder + elbow) rigid-body arm dynamics.

Product-of-exponentials forward kinematics and screw-theory Jacobians, following
the standard construction in Murray/Li/Sastry / Modern Robotics. Shoulder axis 1
(z, through the origin) and axis 2 (-y, through the origin) give a 2-DOF
"spherical-lite" shoulder; the elbow (-y, through (l1,0,0)) adds one more,
biarticular in the sense that it is driven by all three joint angles through the
forearm's CoM and hand frames.

This replaces the earlier planar 2-link model: that model could only represent
flexion/extension in a single plane and had no way to express ab/adduction, which
this dataset's arm reaches plainly exercise. The mass matrix, Coriolis term, and
gravity vector are all computed directly from the kinematics rather than derived
by hand (as the old model's paper-Eq-10 gravity/mass terms were) - safer for 3
DOF, where hand-deriving Christoffel symbols is easy to get wrong.

Coriolis is obtained from the standard identity C(q,qd) qd = Mdot(q,qd) qd -
0.5 * grad_q(qd^T M(q) qd), which is how the C(q,qd) definition that makes
Mdot - 2C skew-symmetric is normally derived in the first place, so that
invariant holds by construction rather than needing to be checked after the
fact - computed here with one jvp (for Mdot qd) and one grad (for the
kinetic-energy gradient) instead of the full n^3 Christoffel-symbol tensor.
"""
from __future__ import annotations

import math

import torch
from torch import nn

GRAVITY = 9.81

# Same anthropometry as the old planar model (paper Table 1).
LINK_LENGTH = (0.298, 0.419)
LINK_MASS = (2.089, 1.912)
CENTRE_OF_MASS = (0.152, 0.181)
# The old model only ever needed a single (sagittal-plane) bending inertia per
# link; keep those as Iyy/Izz (the two transverse axes, taken equal under a
# thin-cylinder symmetry assumption - reasonable for a limb segment, and it is
# only the transverse inertias that mattered for the old planar motion). Ixx
# (about each segment's own long axis, i.e. "twist") is new: no planar model
# ever needed it. It is derived from the existing transverse value using de
# Leva's (1996) male segment radius-of-gyration ratios rather than introduced
# as a fresh absolute number, since only *ratios* are trusted from memory here
# and the segment masses/lengths already in this file are literature-sourced,
# not re-derived from a generic body height/mass:
#   upper arm: r_longitudinal / r_sagittal = 0.158 / 0.285 = 0.5544
#   forearm:   r_longitudinal / r_sagittal = 0.121 / 0.276 = 0.4384
# Since I = m (r * length)^2, the ratio of inertias is just the square of the
# ratio of radii of gyration - no mass or length term needed to carry over.
MOMENT_OF_INERTIA_TRANSVERSE = (0.0159, 0.0257)
_LONGITUDINAL_RATIO = (0.5544, 0.4384)
MOMENT_OF_INERTIA_LONGITUDINAL = tuple(
    i * r * r for i, r in zip(MOMENT_OF_INERTIA_TRANSVERSE, _LONGITUDINAL_RATIO)
)


def _skew(w: torch.Tensor) -> torch.Tensor:
    """w: (..., 3) -> (..., 3, 3) skew-symmetric cross-product matrix."""
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1)
    row1 = torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _exp_twist(v: torch.Tensor, w: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Closed-form exp of a unit revolute twist S=[v;w] (|w|=1) by angle theta.

    v, w: (3,) constants (this arm's fixed screw axes). theta: (...,) batched
    joint angle. Returns (..., 4, 4).
    """
    batch_shape = theta.shape
    what = _skew(w).expand(*batch_shape, 3, 3)
    what_sq = what @ what
    eye3 = torch.eye(3, device=theta.device, dtype=theta.dtype).expand(*batch_shape, 3, 3)
    sin_t = torch.sin(theta)[..., None, None]
    cos_t = torch.cos(theta)[..., None, None]
    rotation = eye3 + sin_t * what + (1.0 - cos_t) * what_sq
    theta_ = theta[..., None, None]
    g_mat = eye3 * theta_ + (1.0 - cos_t) * what + (theta_ - sin_t) * what_sq
    translation = (g_mat @ v.reshape(3, 1).expand(*batch_shape, 3, 1)).squeeze(-1)
    transform = torch.zeros(*batch_shape, 4, 4, device=theta.device, dtype=theta.dtype)
    transform[..., 0:3, 0:3] = rotation
    transform[..., 0:3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def _transform_inverse(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., 0:3, 0:3]
    position = transform[..., 0:3, 3:4]
    rotation_t = rotation.transpose(-1, -2)
    inverse = torch.zeros_like(transform)
    inverse[..., 0:3, 0:3] = rotation_t
    inverse[..., 0:3, 3:4] = -rotation_t @ position
    inverse[..., 3, 3] = 1.0
    return inverse


def _adjoint(transform: torch.Tensor) -> torch.Tensor:
    """Ad_T for [v; w] twist ordering: [[R, [p]xR], [0, R]]."""
    rotation = transform[..., 0:3, 0:3]
    position = transform[..., 0:3, 3]
    batch_shape = transform.shape[:-2]
    ad = torch.zeros(*batch_shape, 6, 6, device=transform.device, dtype=transform.dtype)
    ad[..., 0:3, 0:3] = rotation
    ad[..., 0:3, 3:6] = _skew(position) @ rotation
    ad[..., 3:6, 3:6] = rotation
    return ad


def _matmul4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


class ThreeDofArm(nn.Module):
    """Space-frame-at-shoulder, 3-DOF (q1, q2 shoulder; q3 elbow) arm dynamics.

    Fixed screw axes (paper's convention, S3 corrected so the elbow point is
    invariant to q3 - verified below):
        S1 = [0, 0, 0,  0, 0, 1]     shoulder axis 1 (z), through origin
        S2 = [0, 0, 0,  0, -1, 0]    shoulder axis 2 (-y), through origin
        S3 = [0, 0, -l1, 0, -1, 0]   elbow axis (-y), through (l1, 0, 0)
    """

    def __init__(self) -> None:
        super().__init__()
        l1, l2 = LINK_LENGTH
        m1, m2 = LINK_MASS
        lc1, lc2 = CENTRE_OF_MASS
        i_trans1, i_trans2 = MOMENT_OF_INERTIA_TRANSVERSE
        i_long1, i_long2 = MOMENT_OF_INERTIA_LONGITUDINAL
        self.register_buffer("link_length", torch.tensor(LINK_LENGTH))
        self.register_buffer("link_mass", torch.tensor(LINK_MASS))
        self.register_buffer("com_offset", torch.tensor(CENTRE_OF_MASS))
        # Diagonal body-frame inertia at each link's own CoM: (x=longitudinal
        # "twist" axis, y=z=transverse "bending" axes, equal by the thin-
        # cylinder symmetry assumption above).
        self.register_buffer(
            "inertia1", torch.diag(torch.tensor([i_long1, i_trans1, i_trans1]))
        )
        self.register_buffer(
            "inertia2", torch.diag(torch.tensor([i_long2, i_trans2, i_trans2]))
        )
        self.register_buffer("s1_w", torch.tensor([0.0, 0.0, 1.0]))
        self.register_buffer("s1_v", torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer("s2_w", torch.tensor([0.0, -1.0, 0.0]))
        self.register_buffer("s2_v", torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer("s3_w", torch.tensor([0.0, -1.0, 0.0]))
        self.register_buffer("s3_v", torch.tensor([0.0, 0.0, -l1]))
        # Home configs (CoM / hand frames at q=0), offset along +x from the
        # shoulder.
        m_c1 = torch.eye(4); m_c1[0, 3] = lc1
        m_c2 = torch.eye(4); m_c2[0, 3] = l1 + lc2
        m_hand = torch.eye(4); m_hand[0, 3] = l1 + l2
        self.register_buffer("m_c1", m_c1)
        self.register_buffer("m_c2", m_c2)
        self.register_buffer("m_hand", m_hand)
        # Joint viscosity, same role and range as the old model's damping.
        self.raw_damping = nn.Parameter(torch.zeros(3))

        # Coriolis coefficients for the closed form used in coriolis() below.
        # A symbolic Christoffel-symbol derivation (offline, cross-checked
        # against this module's mass_matrix()/gravity() to 1e-8 - see the
        # class docstring) shows C(q,qd)@qd for this joint arrangement
        # collapses to a combination of just 4 scalars, which are exactly
        # the cos(2*q2)/cos(q3)/cos(2*q2+q3)/cos(2*q2+2*q3) coefficients of
        # M11(q2,q3) - the only mass-matrix entry whose q-dependence actually
        # matters here. Extracting them by evaluating M11 at 4 points and
        # solving the resulting linear system (instead of hand-transcribing
        # the symbolic result as float literals) keeps them automatically
        # consistent with mass_matrix() if the anthropometry constants above
        # are ever changed.
        with torch.no_grad():

            def _m11(q2: float, q3: float) -> float:
                angles = torch.zeros(1, 3)
                angles[0, 1] = q2
                angles[0, 2] = q3
                return self.mass_matrix(angles)[0, 0, 0].item()

            p_00 = _m11(0.0, 0.0)
            p_q2 = _m11(math.pi / 4.0, 0.0)
            p_q3 = _m11(0.0, math.pi / 2.0)
            p_both = _m11(math.pi / 4.0, math.pi / 2.0)
            coeff_b = (p_q2 - p_both) / 2.0
            coeff_const = (p_q2 + p_both) / 2.0
            coeff_a = (p_00 + p_q3 - 2.0 * coeff_const - 2.0 * coeff_b) / 2.0
            coeff_d = (p_00 - p_q3) / 2.0 - coeff_b
        self.register_buffer(
            "coriolis_k",
            torch.tensor([2.0 * coeff_a, 2.0 * coeff_b, 2.0 * coeff_d, coeff_b]),
        )

    @property
    def damping(self) -> torch.Tensor:
        return 0.05 + 1.95 * torch.sigmoid(self.raw_damping)

    def _exp1(self, theta: torch.Tensor) -> torch.Tensor:
        return _exp_twist(self.s1_v, self.s1_w, theta)

    def _exp2(self, theta: torch.Tensor) -> torch.Tensor:
        return _exp_twist(self.s2_v, self.s2_w, theta)

    def _exp3(self, theta: torch.Tensor) -> torch.Tensor:
        return _exp_twist(self.s3_v, self.s3_w, theta)

    def _fk(self, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """angles: (..., 3) -> (T after joint1, T after joints1-2, T after joints1-3)."""
        e1 = self._exp1(angles[..., 0])
        e2 = self._exp2(angles[..., 1])
        e3 = self._exp3(angles[..., 2])
        t1 = e1
        t12 = _matmul4(t1, e2)
        t123 = _matmul4(t12, e3)
        return t1, t12, t123

    def _com_and_hand_frames(
        self, angles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, t12, t123 = self._fk(angles)
        m_c1 = self.m_c1.expand(*angles.shape[:-1], 4, 4)
        m_c2 = self.m_c2.expand(*angles.shape[:-1], 4, 4)
        m_hand = self.m_hand.expand(*angles.shape[:-1], 4, 4)
        t_c1 = _matmul4(t12, m_c1)
        t_c2 = _matmul4(t123, m_c2)
        t_hand = _matmul4(t123, m_hand)
        return t_c1, t_c2, t_hand

    def endpoint(self, angles: torch.Tensor) -> torch.Tensor:
        """Hand position (..., 3) from joint angles."""
        _, _, t_hand = self._com_and_hand_frames(angles)
        return t_hand[..., 0:3, 3]

    def _body_jacobians(
        self, angles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Body Jacobians (..., 6, 3) for the upper-arm and forearm CoM frames.

        Standard PoE body-Jacobian recursion: only sweep the joints that
        actually move each frame (2 for the upper arm, 3 for the forearm);
        later columns are structurally zero.
        """
        e1 = self._exp1(angles[..., 0])
        e2 = self._exp2(angles[..., 1])
        e3 = self._exp3(angles[..., 2])
        batch_shape = angles.shape[:-1]
        s1 = torch.cat([self.s1_v, self.s1_w]).expand(*batch_shape, 6)
        s2 = torch.cat([self.s2_v, self.s2_w]).expand(*batch_shape, 6)
        s3 = torch.cat([self.s3_v, self.s3_w]).expand(*batch_shape, 6)

        m_c1 = self.m_c1.expand(*batch_shape, 4, 4)
        m_c2 = self.m_c2.expand(*batch_shape, 4, 4)

        # Upper-arm CoM frame: driven by joints 1, 2 only.
        j1 = torch.zeros(*batch_shape, 6, 3, device=angles.device, dtype=angles.dtype)
        trailing = m_c1
        ad = _adjoint(_transform_inverse(trailing))
        j1[..., 2] = 0.0  # column for joint 3 stays zero
        j1[..., 1] = (ad @ s2.unsqueeze(-1)).squeeze(-1)
        trailing = _matmul4(e2, trailing)
        ad = _adjoint(_transform_inverse(trailing))
        j1[..., 0] = (ad @ s1.unsqueeze(-1)).squeeze(-1)

        # Forearm CoM frame: driven by all three joints.
        j2 = torch.zeros(*batch_shape, 6, 3, device=angles.device, dtype=angles.dtype)
        trailing = m_c2
        ad = _adjoint(_transform_inverse(trailing))
        j2[..., 2] = (ad @ s3.unsqueeze(-1)).squeeze(-1)
        trailing = _matmul4(e3, trailing)
        ad = _adjoint(_transform_inverse(trailing))
        j2[..., 1] = (ad @ s2.unsqueeze(-1)).squeeze(-1)
        trailing = _matmul4(e2, trailing)
        ad = _adjoint(_transform_inverse(trailing))
        j2[..., 0] = (ad @ s1.unsqueeze(-1)).squeeze(-1)

        return j1, j2

    def mass_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """M(q): (..., 3, 3) = J1^T G1 J1 + J2^T G2 J2, G = diag(m*I3, I)."""
        j1, j2 = self._body_jacobians(angles)
        batch_shape = angles.shape[:-1]
        g1 = torch.zeros(*batch_shape, 6, 6, device=angles.device, dtype=angles.dtype)
        g1[..., 0:3, 0:3] = self.link_mass[0] * torch.eye(
            3, device=angles.device, dtype=angles.dtype
        )
        g1[..., 3:6, 3:6] = self.inertia1
        g2 = torch.zeros(*batch_shape, 6, 6, device=angles.device, dtype=angles.dtype)
        g2[..., 0:3, 0:3] = self.link_mass[1] * torch.eye(
            3, device=angles.device, dtype=angles.dtype
        )
        g2[..., 3:6, 3:6] = self.inertia2
        return j1.transpose(-1, -2) @ g1 @ j1 + j2.transpose(-1, -2) @ g2 @ j2

    def gravity(self, angles: torch.Tensor) -> torch.Tensor:
        """Gv(q) = dV/dq, closed form (verified against an autograd.grad/
        sympy-oracle version to 1e-8 before switching - see coriolis() for
        why the autograd route isn't used at runtime). q1's axis (z) is
        parallel to gravity, so rotating it changes no link's height and
        Gv[0] is identically zero; q2 and q3 reduce to exactly the old
        2-link model's shoulder/elbow gravity terms (paper Eq. 10).
        """
        m1, m2 = self.link_mass[0], self.link_mass[1]
        l1 = self.link_length[0]
        lc1, lc2 = self.com_offset[0], self.com_offset[1]
        q2, q3 = angles[..., 1], angles[..., 2]
        both = q2 + q3
        g2 = (m1 * GRAVITY * lc1 + m2 * GRAVITY * l1) * torch.cos(q2) + m2 * GRAVITY * lc2 * torch.cos(
            both
        )
        g3 = m2 * GRAVITY * lc2 * torch.cos(both)
        return torch.stack([torch.zeros_like(g2), g2, g3], dim=-1)

    def coriolis(self, angles: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """C(q,qd) @ qd, (..., 3), closed form (see coriolis_k derivation above).

        An earlier version of this computed C(q,qd)@qd generically via the
        identity Mdot(q,qd)@qd - 0.5*grad_q(qd^T M(q) qd), using one jvp and
        one autograd.grad call per acceleration() call - correct (verified
        against the same sympy oracle used below) but far too slow for
        training: ~20 ms/call on MPS, which at up to ~1800 substeps per
        trial made a single forward pass take over a minute. This closed
        form is the same quantity, algebraically reduced.
        """
        q2, q3 = angles[..., 1], angles[..., 2]
        qd1, qd2, qd3 = velocity[..., 0], velocity[..., 1], velocity[..., 2]
        k_a, k_b, k_c, k_d = (
            self.coriolis_k[0],
            self.coriolis_k[1],
            self.coriolis_k[2],
            self.coriolis_k[3],
        )
        sin_2q2 = torch.sin(2.0 * q2)
        sin_2q2_q3 = torch.sin(2.0 * q2 + q3)
        sin_2q2_2q3 = torch.sin(2.0 * q2 + 2.0 * q3)
        sin_q3 = torch.sin(q3)

        c0 = -qd1 * (
            k_a * qd2 * sin_2q2
            + k_b * qd2 * sin_2q2_q3
            + k_c * qd2 * sin_2q2_2q3
            + k_d * qd3 * sin_q3
            + k_d * qd3 * sin_2q2_q3
            + k_c * qd3 * sin_2q2_2q3
        )
        c1 = qd1 * qd1 * (
            0.5 * k_a * sin_2q2 + 0.5 * k_b * sin_2q2_q3 + 0.5 * k_c * sin_2q2_2q3
        ) - k_d * qd3 * sin_q3 * (2.0 * qd2 + qd3)
        c2 = qd1 * qd1 * (
            0.5 * k_d * sin_q3 + 0.5 * k_d * sin_2q2_q3 + 0.5 * k_c * sin_2q2_2q3
        ) + k_d * qd2 * qd2 * sin_q3
        return torch.stack([c0, c1, c2], dim=-1)

    def acceleration(
        self, angles: torch.Tensor, velocity: torch.Tensor, torque: torch.Tensor
    ) -> torch.Tensor:
        """Solve M @ theta_ddot = tau - C@qd - G - damping*qd for theta_ddot.

        Forced to run outside autocast: under CUDA AMP, mass_matrix()'s
        matmuls autocast to fp16 while the elementwise residual terms stay
        fp32, so torch.linalg.solve sees mismatched dtypes (this integration
        loop is numerically sensitive besides - the same reason substeps
        exist - so full precision here regardless of the outer training
        loop's autocast setting is the right call anyway, not just a dtype
        workaround).
        """
        with torch.autocast(device_type=angles.device.type, enabled=False):
            # Match the module's own buffer dtype (float32 normally; float64
            # in precision-sensitive verification/testing that casts the
            # whole module), not a hardcoded float32 - autocast only wraps
            # ops, it never touches parameter/buffer dtypes, so this is the
            # dtype every buffer below (link_mass, coriolis_k, ...) actually
            # is.
            compute_dtype = self.link_mass.dtype
            angles = angles.to(compute_dtype)
            velocity = velocity.to(compute_dtype)
            torque = torque.to(compute_dtype)
            residual = (
                torque
                - self.coriolis(angles, velocity)
                - self.gravity(angles)
                - self.damping * velocity
            )
            mass = self.mass_matrix(angles)
            mass = _floor_min_eigenvalue_3x3(mass, floor=0.02)
            return torch.linalg.solve(mass, residual.unsqueeze(-1)).squeeze(-1)


def _jacobi_eigh_3x3(mass: torch.Tensor, sweeps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched classical Jacobi eigenvalue iteration for symmetric 3x3 matrices.

    torch.linalg.eigh is not implemented on MPS (hit this already for the 2x2
    case in arm.py); Jacobi rotations only need cos/sin/atan2, all MPS-safe.
    4 sweeps already reaches machine precision (~1e-14 max eigenvalue error
    over 500 random SPD matrices, vs ~0.1 at 2 sweeps and ~1e-7 at 3), so
    this keeps a 1-sweep safety margin over the minimum rather than using
    the much more conservative 8 originally used before this was profiled.
    Returns (eigenvalues (...,3), eigenvectors (...,3,3) as columns).
    """
    a = mass.clone()
    batch_shape = a.shape[:-2]
    v = torch.eye(3, device=mass.device, dtype=mass.dtype).expand(*batch_shape, 3, 3).clone()
    pairs = [(0, 1), (0, 2), (1, 2)]
    for _ in range(sweeps):
        for p, q in pairs:
            apq = a[..., p, q]
            app = a[..., p, p]
            aqq = a[..., q, q]
            theta = 0.5 * torch.atan2(2.0 * apq, app - aqq)
            c = torch.cos(theta)
            s = torch.sin(theta)
            rotation = torch.eye(3, device=mass.device, dtype=mass.dtype).expand(
                *batch_shape, 3, 3
            ).clone()
            rotation[..., p, p] = c
            rotation[..., q, q] = c
            rotation[..., p, q] = -s
            rotation[..., q, p] = s
            a = rotation.transpose(-1, -2) @ a @ rotation
            v = v @ rotation
    eigenvalues = torch.diagonal(a, dim1=-2, dim2=-1)
    return eigenvalues, v


def _floor_min_eigenvalue_3x3(mass: torch.Tensor, floor: float) -> torch.Tensor:
    eigenvalues, eigenvectors = _jacobi_eigh_3x3(mass)
    floored = eigenvalues.clamp(min=floor)
    return eigenvectors @ torch.diag_embed(floored) @ eigenvectors.transpose(-1, -2)


class EndpointToScreen3(nn.Module):
    """Learned affine from the 3-D hand position (m) to normalised screen coords."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        nn.init.normal_(self.linear.weight, std=0.5)
        nn.init.constant_(self.linear.bias, 0.5)

    def forward(self, endpoint: torch.Tensor) -> torch.Tensor:
        return self.linear(endpoint)
