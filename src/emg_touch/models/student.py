from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .encoders import EMGEncoder
from .heads import ConditionalCVAEHead, MDNHead


class EMGStudent(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        data_config = config["data"]
        d_model = int(model_config["d_model"])
        self.head_type = str(model_config["head"]).lower()
        self.emg_encoder = EMGEncoder(model_config)
        self.context_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        if self.head_type == "mdn":
            self.coordinate_head: nn.Module = MDNHead(d_model, model_config["mdn_components"])
        elif self.head_type == "cvae":
            self.coordinate_head = ConditionalCVAEHead(d_model, model_config["cvae_latent_dim"])
        else:
            raise ValueError(f"Unknown probabilistic head: {self.head_type}")
        grid_x, grid_y = model_config["screen_grid"]
        self.zone_head = nn.Linear(d_model, int(grid_x * grid_y))
        self.future_samples = int(data_config["future_imu_samples"])
        self.future_imu_head = nn.Linear(d_model, self.future_samples * 24)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        _, pooled, _ = self.emg_encoder(emg, emg_mask)
        context = self.context_projection(pooled)
        distribution = self.coordinate_head(context, target) if self.head_type == "cvae" else self.coordinate_head(context)
        prediction = (
            distribution["mean"]
            if self.head_type == "cvae"
            else MDNHead.expected_value(distribution)
        )
        return {
            "context": context,
            "distribution": distribution,
            "prediction": prediction,
            "zone_logits": self.zone_head(context),
            "future_imu": self.future_imu_head(context).reshape(-1, self.future_samples, 24),
        }

    def sample(self, outputs: dict[str, Any], count: int) -> torch.Tensor:
        if self.head_type == "mdn":
            return MDNHead.sample(outputs["distribution"], count)
        return self.coordinate_head.sample(outputs["context"], count)

