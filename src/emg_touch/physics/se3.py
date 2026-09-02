"""Differentiable SO(3)/SE(3) operations for the tracked 6-DoF end effector.

The Vive export carries a full pose - position plus quat_w/x/y/z - and the
pipeline currently loads the quaternion and ignores it. Using it correctly
means not treating it as four Euclidean numbers, for two concrete reasons:

  Double cover. q and -q are the same rotation. An L2 loss between them
  reports a large error for two identical poses, and a mean over quaternions
  spanning both hemispheres can cancel to nearly zero. Every function here
  aligns to a hemisphere first.

  Curvature. The unit quaternions are a sphere, not a vector space. Adding
  two of them leaves the manifold, and the Euclidean distance between two
  rotations is not the angle between them - it understates large rotations
  and has the wrong gradient. Differences are taken as a relative rotation
  and mapped into the tangent space with the logarithm, where the vector
  operations the rest of the model uses are actually valid.

The practical payoff for this project is that a destination expressed in the
tangent space at the *current* pose is a relative pose. That makes the latent
translation- and orientation-invariant by construction - the same reach
toward the same target produces the same latent wherever in the workspace it
starts - rather than something the encoder has to learn to be invariant to.

The attractor generalises to the manifold without changing shape:

    xddot = eta * log(r . x^-1) - rho * xdot

with log the SE(3) logarithm. The Euclidean version already used here is
exactly this with log(r . x^-1) = r - x, which is what the translation part
reduces to.

Conventions: quaternions are (w, x, y, z), scalar first, matching the
export's column order. Rotation vectors are axis * angle in radians.
Everything is batched over leading dimensions and differentiable, with the
small-angle limits handled explicitly so gradients stay finite at identity -
where a reach that has not started yet actually sits.
"""
from __future__ import annotations

import torch

# Below this angle the series expansions are used instead of the closed
# forms, whose 0/0 would otherwise produce NaN gradients exactly at the
# identity rotation.
SMALL_ANGLE = 1e-6


def normalise_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def align_hemisphere(q: torch.Tensor) -> torch.Tensor:
    """Flip q to the w >= 0 hemisphere.

    q and -q name the same rotation, so this picks one representative and
    makes every downstream comparison take the short way round. Without it a
    trial whose tracker happened to report the far-side sign looks maximally
    different from an identical one.
    """
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quaternion_log(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion -> rotation vector (axis * angle), in R^3.

    This is the map into the tangent space at the identity. The result is a
    genuine vector: it can be scaled, averaged and regressed with ordinary
    Euclidean machinery, which is the whole reason for going through it.
    """
    q = align_hemisphere(normalise_quaternion(q))
    scalar = q[..., 0].clamp(-1.0, 1.0)
    vector = q[..., 1:]
    sine = vector.norm(dim=-1)
    angle = 2.0 * torch.atan2(sine, scalar)
    # scale = angle / sin(angle/2); as angle -> 0 this tends to 2, and the
    # closed form is 0/0 there. torch.where alone would still backpropagate a
    # NaN through the unused branch, so the denominator is made safe first.
    safe_sine = torch.where(sine > SMALL_ANGLE, sine, torch.ones_like(sine))
    scale = torch.where(
        sine > SMALL_ANGLE, angle / safe_sine, torch.full_like(sine, 2.0)
    )
    return vector * scale.unsqueeze(-1)


def quaternion_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Rotation vector -> unit quaternion. Inverse of quaternion_log."""
    angle = rotation_vector.norm(dim=-1)
    half = 0.5 * angle
    # sin(angle/2)/angle -> 1/2 as angle -> 0, same 0/0 care as above.
    safe_angle = torch.where(angle > SMALL_ANGLE, angle, torch.ones_like(angle))
    scale = torch.where(
        angle > SMALL_ANGLE,
        torch.sin(half) / safe_angle,
        torch.full_like(angle, 0.5),
    )
    return torch.cat(
        [torch.cos(half).unsqueeze(-1), rotation_vector * scale.unsqueeze(-1)],
        dim=-1,
    )


def relative_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rotation taking `source` to `target`, as target . source^-1."""
    return normalise_quaternion(
        quaternion_multiply(
            normalise_quaternion(target), quaternion_conjugate(normalise_quaternion(source))
        )
    )


def rotation_geodesic(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle in radians between two rotations - the true geodesic distance.

    This is what an orientation loss should minimise. The Euclidean distance
    between the raw quaternions is not proportional to it and is not even
    monotonic across the double cover.
    """
    return quaternion_log(relative_rotation(source, target)).norm(dim=-1)


def pose_log(
    source_position: torch.Tensor,
    source_rotation: torch.Tensor,
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
) -> torch.Tensor:
    """Relative 6-DoF pose as a tangent vector at the source pose.

    Returns (3 translation, 3 rotation) concatenated. Translation is left in
    the world frame rather than rotated into the source frame: the tracked
    hand moves toward a screen fixed in the world, so a world-frame offset is
    the quantity the attractor should pull along. Rotating it into the hand
    frame would make the destination depend on wrist angle, which is the
    opposite of the invariance wanted here.
    """
    translation = target_position - source_position
    rotation = quaternion_log(relative_rotation(source_rotation, target_rotation))
    return torch.cat([translation, rotation], dim=-1)


def pose_exp(
    source_position: torch.Tensor,
    source_rotation: torch.Tensor,
    tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a tangent vector at a pose. Inverse of pose_log."""
    position = source_position + tangent[..., :3]
    rotation = normalise_quaternion(
        quaternion_multiply(quaternion_exp(tangent[..., 3:]), normalise_quaternion(source_rotation))
    )
    return position, rotation


def geodesic_pose_error(
    predicted_position: torch.Tensor,
    predicted_rotation: torch.Tensor,
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    rotation_weight: float = 0.1,
) -> torch.Tensor:
    """Translation error in metres plus weighted rotation error in radians.

    The two have different units, so rotation_weight sets the exchange rate
    and is a real modelling choice rather than a detail: at 0.1, one radian
    of orientation error costs the same as 10 cm of position error. It should
    be set from how much wrist angle actually matters for the task, not left
    at a default.
    """
    translation = (predicted_position - target_position).norm(dim=-1)
    rotation = rotation_geodesic(predicted_rotation, target_rotation)
    return translation + rotation_weight * rotation
