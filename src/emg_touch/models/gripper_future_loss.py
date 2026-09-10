"""Targets for causal reconstruction of a withheld future pose block."""
import torch
from torch.nn import functional as F


def future_pairs(output, batch):
    """100 Hz grid: output[t,h] predicts label[t+h+1], within each trial.

    Missing VIVE labels and padding never become targets. Use the same paired
    wearable-valid support as the existing trainer for modality comparisons.
    Future wearable availability is irrelevant: those inputs are withheld.
    """
    usable = batch["emg_usable"] & batch["imu_usable"]
    length = usable.shape[1]
    for horizon in range(1, output["future_position"].shape[2] + 1):
        if horizon >= length:
            break
        source = usable[:, :-horizon]
        pvalid = source & batch["pose_mask"][:, horizon:, 0].bool()
        rvalid = source & batch["orientation_mask"][:, horizon:].bool()
        yield (horizon, output["future_position"][:, :-horizon, horizon - 1],
               batch["pose"][:, horizon:], pvalid,
               output["future_orientation_6d"][:, :-horizon, horizon - 1],
               batch["orientation"][:, horizon:], rvalid)


def future_pose_loss(output, batch):
    position = output["future_position"].sum() * 0
    rotation = output["future_orientation_6d"].sum() * 0
    np, nr = 0, 0
    for _, pred, truth, pmask, rpred, rtruth, rmask in future_pairs(output, batch):
        position = position + F.smooth_l1_loss(pred[pmask], truth[pmask], reduction="sum")
        rotation = rotation + F.smooth_l1_loss(rpred[rmask], rtruth[rmask], reduction="sum")
        np = np + pmask.sum() * 3
        nr = nr + rmask.sum() * 6
    return position / torch.as_tensor(np).clamp_min(1) + rotation / torch.as_tensor(nr).clamp_min(1)
