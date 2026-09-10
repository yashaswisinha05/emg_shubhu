"""Losses for state-conditioned position attention."""
import torch
from torch.nn import functional as F


def masked_mean(value, mask):
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0


def future_position_losses(output, batch, usable=None):
    """Supervised future position and prediction-to-arrival consistency."""
    if usable is None:
        usable = batch["emg_usable"] & batch["imu_usable"]
    supervised = output["future_position"].sum() * 0
    consistent = output["future_position"].sum() * 0
    supervised_count = output["future_position"].new_zeros(())
    consistent_count = output["future_position"].new_zeros(())
    for horizon in range(1, output["future_position"].shape[2] + 1):
        if horizon >= usable.shape[1]:
            break
        source = usable[:, :-horizon]
        target_valid = batch["pose_mask"][:, horizon:, 0].bool()
        valid = source & target_valid
        prediction = output["future_position"][:, :-horizon, horizon - 1]
        truth = batch["pose"][:, horizon:]
        supervised = supervised + F.smooth_l1_loss(
            prediction[valid], truth[valid], reduction="sum")
        supervised_count = supervised_count + valid.sum() * 3

        arrival_valid = valid & usable[:, horizon:]
        arrival = output["position"][:, horizon:].detach()
        consistent = consistent + F.smooth_l1_loss(
            prediction[arrival_valid], arrival[arrival_valid], reduction="sum")
        consistent_count = consistent_count + arrival_valid.sum() * 3
    return (supervised / supervised_count.clamp_min(1),
            consistent / consistent_count.clamp_min(1))
