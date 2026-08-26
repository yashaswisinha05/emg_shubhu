from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import EMGEncoder, IMUEncoder


class MultimodalMaskedPretrainer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        d_model = int(model["d_model"])
        patch_length = int(model["patch_length"])
        self.patch_length = patch_length
        self.patch_stride = int(model["patch_stride"])
        self.emg_encoder = EMGEncoder(model)
        self.imu_encoder = IMUEncoder(model)
        self.emg_decoder = nn.Linear(d_model, patch_length * 4)
        self.imu_decoder = nn.Linear(d_model, patch_length * 24)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    @staticmethod
    def random_time_mask(observed: torch.Tensor, ratio: float) -> torch.Tensor:
        random_values = torch.rand_like(observed, dtype=torch.float32)
        return (random_values < ratio) & observed

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
        mask_ratio: float = 0.4,
    ) -> dict[str, torch.Tensor]:
        emg_time_mask = self.random_time_mask(emg_mask.any(-1), mask_ratio)
        imu_time_mask = self.random_time_mask(imu_mask.any(-1), mask_ratio)
        masked_emg = emg.masked_fill(emg_time_mask.unsqueeze(-1), 0.0)
        masked_imu = imu.masked_fill(imu_time_mask.unsqueeze(-1), 0.0)
        effective_emg_mask = emg_mask & ~emg_time_mask.unsqueeze(-1)
        effective_imu_mask = imu_mask & ~imu_time_mask.unsqueeze(-1)
        emg_tokens, emg_context, _ = self.emg_encoder(masked_emg, effective_emg_mask)
        imu_tokens, imu_context, _ = self.imu_encoder(masked_imu, effective_imu_mask)
        emg_reconstruction = self.emg_decoder(emg_tokens).reshape(
            emg.size(0), -1, 4, self.patch_length
        )
        imu_reconstruction = self.imu_decoder(imu_tokens).reshape(
            imu.size(0), -1, 24, self.patch_length
        )
        emg_target = emg.unfold(1, self.patch_length, self.patch_stride)
        imu_target = imu.unfold(1, self.patch_length, self.patch_stride)
        emg_observed = emg_mask.unfold(1, self.patch_length, self.patch_stride)
        imu_observed = imu_mask.unfold(1, self.patch_length, self.patch_stride)
        emg_selected = emg_time_mask.unfold(1, self.patch_length, self.patch_stride).any(-1)
        imu_selected = imu_time_mask.unfold(1, self.patch_length, self.patch_stride).any(-1)
        emg_weights = emg_observed & emg_selected.unsqueeze(-1).unsqueeze(-1)
        imu_weights = imu_observed & imu_selected.unsqueeze(-1).unsqueeze(-1)
        emg_loss = ((emg_reconstruction - emg_target).square() * emg_weights).sum() / emg_weights.sum().clamp_min(1)
        imu_loss = ((imu_reconstruction - imu_target).square() * imu_weights).sum() / imu_weights.sum().clamp_min(1)
        emg_normalized = F.normalize(emg_context, dim=-1)
        imu_normalized = F.normalize(imu_context, dim=-1)
        logits = emg_normalized @ imu_normalized.transpose(0, 1) / self.temperature.clamp(0.02, 1.0)
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return {
            "loss": emg_loss + imu_loss + 0.1 * contrastive,
            "emg_reconstruction": emg_loss,
            "imu_reconstruction": imu_loss,
            "contrastive": contrastive,
        }

