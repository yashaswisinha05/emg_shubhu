"""Auxiliary future-motion objective for the neuromuscular decoder."""
import torch
from torch.nn import functional as F


def future_imu_delta_loss(output, batch):
    """Predict future IMU change; future samples are labels, never inputs."""
    prediction = output["future_imu_delta"]
    imu = batch["imu"][..., :24]
    valid = batch["imu"][..., 24:].bool() & batch["imu_usable"][..., None]
    total = prediction.sum() * 0
    count = prediction.new_zeros(())
    for horizon in range(1, min(prediction.shape[2] + 1, imu.shape[1])):
        target = imu[:, horizon:] - imu[:, :-horizon]
        mask = valid[:, horizon:] & valid[:, :-horizon]
        error = F.smooth_l1_loss(
            prediction[:, :-horizon, horizon - 1], target, reduction="none")
        total = total + error[mask].sum()
        count = count + mask.sum()
    return total / count.clamp_min(1)
