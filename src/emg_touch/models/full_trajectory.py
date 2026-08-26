from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .baselines import CausalTCNRegressor
from .encoders import EMGEncoder, IMUEncoder
from .layers import masked_mean


def coordinate_head(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
    )


class FullEMGTCNRegressor(CausalTCNRegressor):
    """Deterministic complete-trajectory EMG baseline."""


class FullEMGPatchRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["model"]["d_model"])
        self.encoder = EMGEncoder(config["model"])
        self.head = coordinate_head(d_model, float(config["model"]["dropout"]))

    def forward(self, emg: torch.Tensor, emg_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, context, _ = self.encoder(emg, emg_mask)
        return {"prediction": torch.sigmoid(self.head(context)), "context": context}


class FullIMUPatchRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["model"]["d_model"])
        self.encoder = IMUEncoder(config["model"])
        self.head = coordinate_head(d_model, float(config["model"]["dropout"]))

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, context, _ = self.encoder(imu, imu_mask)
        return {"prediction": torch.sigmoid(self.head(context)), "context": context}


class FullEMGResidualRegressor(nn.Module):
    """Frozen IMU predictor plus a correction computed only from EMG.

    This controlled model is intentionally less expressive than cross-modal fusion:
    the residual head never receives IMU features.  Consequently, differences
    between touch-relative window conditions cannot come from a second trainable
    IMU pathway.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        d_model = int(model["d_model"])
        self.emg_encoder = EMGEncoder(model)
        self.imu_encoder = IMUEncoder(model)
        self.imu_head = coordinate_head(d_model, float(model["dropout"]))
        self.emg_delta = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(d_model, 2, bias=False),
        )
        nn.init.zeros_(self.emg_delta[-1].weight)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, emg_context, _ = self.emg_encoder(emg, emg_mask)
        _, imu_context, _ = self.imu_encoder(imu, imu_mask)
        imu_logits = self.imu_head(imu_context)
        emg_delta = self.emg_delta(emg_context)
        return {
            "prediction": torch.sigmoid(imu_logits + emg_delta),
            "context": emg_context,
            "imu_logits": imu_logits,
            "emg_delta": emg_delta,
        }


class FullMultimodalRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        d_model = int(model["d_model"])
        self.emg_encoder = EMGEncoder(model)
        self.imu_encoder = IMUEncoder(model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            int(model["num_heads"]),
            dropout=float(model["dropout"]),
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(model["dropout"]),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Preserve a strong IMU-only solution and let EMG learn a residual correction.
        self.imu_head = coordinate_head(d_model, float(model["dropout"]))
        self.fusion_delta = nn.Linear(d_model, 2)
        nn.init.zeros_(self.fusion_delta.weight)
        nn.init.zeros_(self.fusion_delta.bias)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        emg_tokens, emg_context, emg_patch_mask = self.emg_encoder(emg, emg_mask)
        imu_tokens, imu_context, imu_patch_mask = self.imu_encoder(imu, imu_mask)
        safe_imu_mask = imu_patch_mask.clone()
        empty = ~safe_imu_mask.any(dim=1)
        if empty.any():
            safe_imu_mask[empty, 0] = True
            imu_tokens = imu_tokens.clone()
            imu_tokens[empty, 0] = 0.0
        attended, _ = self.cross_attention(
            query=emg_tokens,
            key=imu_tokens,
            value=imu_tokens,
            key_padding_mask=~safe_imu_mask,
            need_weights=False,
        )
        attended_context = masked_mean(attended, emg_patch_mask)
        context = self.fusion(
            torch.cat([emg_context, imu_context, attended_context], dim=-1)
        )
        imu_logits = self.imu_head(imu_context)
        fusion_delta = self.fusion_delta(context)
        return {
            "prediction": torch.sigmoid(imu_logits + fusion_delta),
            "context": context,
            "imu_logits": imu_logits,
            "fusion_delta": fusion_delta,
        }


def build_full_trajectory_model(kind: str, config: dict[str, Any]) -> nn.Module:
    kind = kind.lower()
    if kind == "emg_tcn":
        return FullEMGTCNRegressor(config)
    if kind == "emg_patch":
        return FullEMGPatchRegressor(config)
    if kind == "imu_patch":
        return FullIMUPatchRegressor(config)
    if kind == "emg_residual":
        return FullEMGResidualRegressor(config)
    if kind == "multimodal":
        return FullMultimodalRegressor(config)
    raise ValueError(f"Unknown full-trajectory model kind: {kind}")


def forward_full_trajectory_model(
    model: nn.Module, batch: dict[str, Any], kind: str
) -> dict[str, torch.Tensor]:
    if kind in {"emg_tcn", "emg_patch"}:
        return model(batch["emg"], batch["emg_mask"])
    if kind == "imu_patch":
        return model(batch["imu"], batch["imu_mask"])
    if kind in {"emg_residual", "multimodal"}:
        return model(batch["emg"], batch["emg_mask"], batch["imu"], batch["imu_mask"])
    raise ValueError(f"Unknown full-trajectory model kind: {kind}")
