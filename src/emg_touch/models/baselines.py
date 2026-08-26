from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import MultiScaleCausalStem


class CausalTCNRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        d_model = int(model_config["d_model"])
        self.stem = MultiScaleCausalStem(
            input_dim=8,
            d_model=d_model,
            kernel_sizes=model_config["tcn_kernel_sizes"],
            dropout=model_config["dropout"],
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(model_config["dropout"]), nn.Linear(d_model, 2)
        )

    def forward(self, emg: torch.Tensor, emg_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(torch.cat([emg, emg_mask.to(emg.dtype)], dim=-1))
        time_mask = emg_mask.any(dim=-1).unsqueeze(1).to(hidden.dtype)
        pooled = (hidden * time_mask).sum(dim=-1) / time_mask.sum(dim=-1).clamp_min(1.0)
        prediction = torch.sigmoid(self.head(pooled))
        return {"prediction": prediction, "context": pooled}


class PatchTSTRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import PatchTSTConfig, PatchTSTForRegression
        except ImportError as error:
            raise ImportError("Install transformers to use the PatchTST baseline") from error
        model_config = config["model"]
        data_config = config["data"]
        patch_config = PatchTSTConfig(
            num_input_channels=4,
            context_length=data_config["context_length"],
            patch_length=model_config["patch_length"],
            patch_stride=model_config["patch_stride"],
            num_targets=2,
            d_model=model_config["d_model"],
            num_hidden_layers=model_config["num_layers"],
            num_attention_heads=model_config["num_heads"],
            ffn_dim=model_config["ffn_dim"],
            attention_dropout=model_config["dropout"],
            ff_dropout=model_config["dropout"],
            head_dropout=model_config["dropout"],
            use_cls_token=True,
            channel_attention=True,
            loss="mse",
        )
        self.model = PatchTSTForRegression(patch_config)

    def forward(self, emg: torch.Tensor, emg_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(past_values=emg, past_observed_mask=emg_mask)
        return {"prediction": torch.sigmoid(outputs.regression_outputs)}

