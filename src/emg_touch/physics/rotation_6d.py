"""Continuous 6D rotation representation and orientation metrics."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def quaternion_to_matrix_numpy(quaternion):
    """Convert scalar-first (w, x, y, z) quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotation_6d_numpy(matrix):
    """Store the first two matrix columns without an Euler discontinuity."""
    return np.asarray(matrix)[..., :, :2].reshape(*np.asarray(matrix).shape[:-2], 6)


def rotation_6d_to_matrix(rotation):
    """Gram-Schmidt map from two predicted axes to an SO(3) matrix."""
    axes = rotation.reshape(*rotation.shape[:-1], 3, 2)
    first = F.normalize(axes[..., 0], dim=-1, eps=1e-6)
    second = axes[..., 1] - (first * axes[..., 1]).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def orientation_errors_numpy(predicted_6d, true_6d):
    predicted = rotation_6d_to_matrix(torch.as_tensor(predicted_6d)).numpy()
    true = rotation_6d_to_matrix(torch.as_tensor(true_6d)).numpy()
    relative = np.swapaxes(predicted, -1, -2) @ true
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    geodesic = np.degrees(np.arccos(cosine))
    # ZYX convention: yaw is rotation around the VIVE/world z axis.
    predicted_yaw = np.arctan2(predicted[..., 1, 0], predicted[..., 0, 0])
    true_yaw = np.arctan2(true[..., 1, 0], true[..., 0, 0])
    yaw = np.degrees(np.abs(np.arctan2(np.sin(predicted_yaw - true_yaw),
                                       np.cos(predicted_yaw - true_yaw))))
    return geodesic, yaw
