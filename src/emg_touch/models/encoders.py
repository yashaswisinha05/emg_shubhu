from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import PatchTransformerEncoder


class EMGEncoder(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = PatchTransformerEncoder(
            input_dim=8,
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ffn_dim=model_config["ffn_dim"],
            dropout=model_config["dropout"],
            patch_length=model_config["patch_length"],
            patch_stride=model_config["patch_stride"],
            kernel_sizes=model_config["tcn_kernel_sizes"],
        )

    def forward(
        self, emg: torch.Tensor, emg_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([emg, emg_mask.to(emg.dtype)], dim=-1)
        return self.encoder(inputs, emg_mask.any(dim=-1))


class IMUEncoder(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(model_config["d_model"])
        sensor_dim = max(16, d_model // 4)
        self.sensor_projection = nn.Linear(12, sensor_dim)
        self.sensor_embedding = nn.Parameter(torch.zeros(1, 1, 4, sensor_dim))
        nn.init.trunc_normal_(self.sensor_embedding, std=0.02)
        self.encoder = PatchTransformerEncoder(
            input_dim=sensor_dim * 4,
            d_model=d_model,
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ffn_dim=model_config["ffn_dim"],
            dropout=model_config["dropout"],
            patch_length=model_config["patch_length"],
            patch_stride=model_config["patch_stride"],
            kernel_sizes=model_config["tcn_kernel_sizes"],
        )

    def forward(
        self, imu: torch.Tensor, imu_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = imu.shape
        values = imu.reshape(batch, time, 4, 6)
        masks = imu_mask.reshape(batch, time, 4, 6).to(imu.dtype)
        sensor_inputs = torch.cat([values, masks], dim=-1)
        sensor_tokens = self.sensor_projection(sensor_inputs) + self.sensor_embedding
        flattened = sensor_tokens.reshape(batch, time, -1)
        return self.encoder(flattened, imu_mask.any(dim=-1))

