"""Causal, physically meaningful training augmentation for EMG and IMUs."""
from __future__ import annotations

import math
import torch


class PhysiologicalWearableAugmenter:
    """Simulate placement, gain, noise, dropout, and short telemetry gaps.

    Values enter in standardized-value + validity-mask form.  Augmentation is
    applied only to the value half; masks are updated when evidence is removed.
    No sample is moved earlier in time, so the causal task cannot see future
    measurements.
    """
    def __init__(self, strength=1.):
        if strength < 0:
            raise ValueError("augmentation strength cannot be negative")
        self.strength = float(strength)

    @staticmethod
    def _stats(stats, key, value):
        mean = torch.as_tensor(stats[key]["mean"], device=value.device,
                               dtype=value.dtype)
        std = torch.as_tensor(stats[key]["std"], device=value.device,
                              dtype=value.dtype)
        return mean, std

    @staticmethod
    def _masked_standardize(raw, mean, std, valid):
        return torch.where(valid, (raw - mean) / std, torch.zeros_like(raw))

    def _missing_spans(self, value, valid, probability):
        batch, frames, _ = value.shape
        maximum = max(1, min(frames, round(10 * self.strength)))
        for index in range(batch):
            if torch.rand((), device=value.device) < probability:
                length = int(torch.randint(1, maximum + 1, (), device=value.device))
                start = int(torch.randint(0, max(1, frames - length + 1), (),
                            device=value.device))
                value[index, start:start + length] = 0
                valid[index, start:start + length] = False

    def _emg(self, packed, stats, allow_branch_dropout):
        channels = packed.shape[-1] // 2
        value, valid = packed[..., :channels].clone(), packed[..., channels:].bool().clone()
        mean, std = self._stats(stats, "emg", value)
        raw = value * std + mean
        batch = len(raw)
        # The two RMS scales belonging to one physical sensor share gain.
        sensor_gain = torch.exp(torch.randn(batch, 1, 4, device=raw.device,
                                             dtype=raw.dtype) * (.20 * self.strength))
        gain = torch.cat([sensor_gain, sensor_gain], -1)
        raw = raw * gain
        value = self._masked_standardize(raw, mean, std, valid)
        value = value + torch.randn_like(value) * (.035 * self.strength) * valid
        for item in range(batch):
            if torch.rand((), device=value.device) < .10 * self.strength:
                sensor = int(torch.randint(0, 4, (), device=value.device))
                for channel in (sensor, sensor + 4):
                    value[item, :, channel] = 0
                    valid[item, :, channel] = False
            if allow_branch_dropout and torch.rand((), device=value.device) < .03 * self.strength:
                value[item] = 0
                valid[item] = False
        self._missing_spans(value, valid, .20 * self.strength)
        return torch.cat([value, valid.to(value.dtype)], -1)

    def _imu(self, packed, stats, allow_branch_dropout):
        channels = packed.shape[-1] // 2
        value, valid = packed[..., :channels].clone(), packed[..., channels:].bool().clone()
        mean, std = self._stats(stats, "imu", value)
        raw = (value * std + mean).reshape(*value.shape[:2], 4, 2, 3)
        shaped_valid = valid.reshape(*valid.shape[:2], 4, 2, 3)
        batch = len(raw)
        # A small independent mounting rotation per physical sensor is shared
        # by its accelerometer and gyroscope triads.
        axis = torch.randn(batch, 4, 3, device=raw.device, dtype=raw.dtype)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        angle = torch.randn(batch, 4, device=raw.device, dtype=raw.dtype)
        angle = angle * math.radians(4.) * self.strength
        zero = torch.zeros_like(angle)
        x, y, z = axis.unbind(-1)
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(batch, 4, 3, 3)
        identity = torch.eye(3, device=raw.device, dtype=raw.dtype)[None, None]
        rotation = identity + angle.sin()[..., None, None] * skew
        rotation = rotation + (1 - angle.cos())[..., None, None] * (skew @ skew)
        raw = torch.einsum("btskj,bsij->btski", raw, rotation)
        raw = raw.reshape(*value.shape)
        valid = shaped_valid.reshape(*valid.shape)
        value = self._masked_standardize(raw, mean, std, valid)
        value = value + torch.randn_like(value) * (.025 * self.strength) * valid
        # Slowly varying sensor bias in standardized units.
        drift_end = torch.randn(batch, 1, channels, device=value.device,
                                dtype=value.dtype) * (.04 * self.strength)
        ramp = torch.linspace(0, 1, value.shape[1], device=value.device,
                              dtype=value.dtype)[None, :, None]
        value = value + drift_end * ramp * valid
        for item in range(batch):
            if torch.rand((), device=value.device) < .06 * self.strength:
                sensor = int(torch.randint(0, 4, (), device=value.device))
                sl = slice(sensor * 6, sensor * 6 + 6)
                value[item, :, sl] = 0
                valid[item, :, sl] = False
            if allow_branch_dropout and torch.rand((), device=value.device) < .03 * self.strength:
                value[item] = 0
                valid[item] = False
        self._missing_spans(value, valid, .12 * self.strength)
        return torch.cat([value, valid.to(value.dtype)], -1)

    def __call__(self, batch, stats, modality):
        if not self.strength:
            return batch
        allow_branch_dropout = modality == "emg+imu"
        batch["emg"] = self._emg(batch["emg"], stats, allow_branch_dropout)
        batch["imu"] = self._imu(batch["imu"], stats, allow_branch_dropout)
        for key in ("emg", "imu"):
            channels = batch[key].shape[-1] // 2
            batch[key + "_usable"] = batch[key][..., channels:].mean(-1) >= .75
        return batch
