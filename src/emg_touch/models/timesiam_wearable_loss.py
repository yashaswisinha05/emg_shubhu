"""Self-supervised future-wearable reconstruction for TimeSiam adaptation."""
import torch
from torch.nn import functional as F


def future_wearable_loss(output, batch, modality):
    prediction = output["future_wearable"]
    horizons = prediction.shape[2]
    total = prediction.sum() * 0
    count = prediction.new_zeros(())
    streams = []
    if modality in ("emg", "emg+imu"):
        streams.append((prediction[..., :8], batch["emg"][..., :8],
                        batch["emg"][..., 8:].bool(), batch["emg_usable"]))
    if modality in ("imu", "emg+imu"):
        streams.append((prediction[..., 8:], batch["imu"][..., :24],
                        batch["imu"][..., 24:].bool(), batch["imu_usable"]))
    for predicted, target, target_valid, source_valid in streams:
        length = target.shape[1]
        for horizon in range(1, min(horizons + 1, length)):
            mask = source_valid[:, :-horizon, None] & target_valid[:, horizon:]
            error = F.smooth_l1_loss(
                predicted[:, :-horizon, horizon - 1], target[:, horizon:], reduction="none")
            total = total + error[mask].sum()
            count = count + mask.sum()
    return total / count.clamp_min(1)
