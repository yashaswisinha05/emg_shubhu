from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .encoders import EMGEncoder, IMUEncoder
from .heads import ConditionalCVAEHead, MDNHead
from .layers import MultiScaleLookback, masked_mean


class MultimodalTeacher(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        data_config = config["data"]
        d_model = int(model_config["d_model"])
        self.head_type = str(model_config["head"]).lower()
        self.emg_encoder = EMGEncoder(model_config)
        self.imu_encoder = IMUEncoder(model_config)
        self.cross_attention = nn.MultiheadAttention(
            d_model, model_config["num_heads"], dropout=model_config["dropout"], batch_first=True
        )
        self.lookback = MultiScaleLookback(
            d_model=d_model,
            lookbacks_s=model_config["imu_lookbacks_s"],
            sample_rate_hz=float(data_config["sample_rate_hz"]),
            patch_stride=int(model_config["patch_stride"]),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        if self.head_type == "mdn":
            self.coordinate_head: nn.Module = MDNHead(d_model, model_config["mdn_components"])
        elif self.head_type == "cvae":
            self.coordinate_head = ConditionalCVAEHead(d_model, model_config["cvae_latent_dim"])
        else:
            raise ValueError(f"Unknown probabilistic head: {self.head_type}")
        grid_x, grid_y = model_config["screen_grid"]
        self.zone_head = nn.Linear(d_model, int(grid_x * grid_y))
        self.future_imu_head = nn.Linear(
            d_model, int(data_config["future_imu_samples"]) * 24
        )
        self.future_samples = int(data_config["future_imu_samples"])

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
        target: torch.Tensor | None = None,
        drop_imu: bool = False,
    ) -> dict[str, Any]:
        emg_tokens, emg_context, emg_patch_mask = self.emg_encoder(emg, emg_mask)
        if drop_imu:
            imu = torch.zeros_like(imu)
            imu_mask = torch.zeros_like(imu_mask)
        imu_tokens, _, imu_patch_mask = self.imu_encoder(imu, imu_mask)
        safe_key_mask = imu_patch_mask.clone()
        empty = ~safe_key_mask.any(dim=1)
        if empty.any():
            safe_key_mask[empty, 0] = True
            imu_tokens = imu_tokens.clone()
            imu_tokens[empty, 0] = 0.0
        attended, _ = self.cross_attention(
            query=emg_tokens,
            key=imu_tokens,
            value=imu_tokens,
            key_padding_mask=~safe_key_mask,
            need_weights=False,
        )
        attended_context = masked_mean(attended, emg_patch_mask)
        imu_context, lookback_weights = self.lookback(emg_context, imu_tokens, imu_patch_mask)
        context = self.fusion(torch.cat([emg_context, attended_context, imu_context], dim=-1))
        distribution = self.coordinate_head(context, target) if self.head_type == "cvae" else self.coordinate_head(context)
        prediction = (
            distribution["mean"]
            if self.head_type == "cvae"
            else MDNHead.expected_value(distribution)
        )
        future_imu = self.future_imu_head(emg_context).reshape(-1, self.future_samples, 24)
        return {
            "context": context,
            "emg_context": emg_context,
            "distribution": distribution,
            "prediction": prediction,
            "zone_logits": self.zone_head(context),
            "future_imu": future_imu,
            "lookback_weights": lookback_weights,
        }

    def sample(self, outputs: dict[str, Any], count: int) -> torch.Tensor:
        if self.head_type == "mdn":
            return MDNHead.sample(outputs["distribution"], count)
        return self.coordinate_head.sample(outputs["context"], count)

