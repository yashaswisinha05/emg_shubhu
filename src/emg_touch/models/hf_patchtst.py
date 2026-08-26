from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTModel

from ..data.grid_trajectory import grid_imu_feature_dim
from .grid_point import (
    ElapsedTimeConditioner,
    SpatialPointHead,
    decode_grid_outputs,
    elapsed_from_batch,
    finalize_point_prediction,
)


HF_PATCHTST_MODEL_KINDS = (
    "hf_patchtst_imu",
    "hf_patchtst_emg",
    "hf_patchtst_fusion",
)


def right_align_fixed_context(
    values: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align each causal prefix in the fixed context required by PatchTST.

    If a recording is longer than the configured context, only its oldest samples
    are discarded. Therefore the complete movement endpoint is always retained.
    """

    batch, _, channels = values.shape
    fixed_values = values.new_zeros((batch, context_length, channels))
    fixed_mask = mask.new_zeros((batch, context_length, channels))
    for index in range(batch):
        source_length = int(lengths[index])
        retained = min(source_length, context_length)
        source_start = source_length - retained
        target_start = context_length - retained
        fixed_values[index, target_start:] = values[
            index, source_start:source_length
        ]
        fixed_mask[index, target_start:] = mask[
            index, source_start:source_length
        ]
    fixed_values = fixed_values.masked_fill(~fixed_mask, 0.0)
    return fixed_values, fixed_mask


class HFPatchTSTPointRegressor(nn.Module):
    """Touch regressor backed by the exact Hugging Face PatchTSTModel class."""

    def __init__(self, kind: str, config: dict[str, Any]) -> None:
        super().__init__()
        if kind not in HF_PATCHTST_MODEL_KINDS:
            raise ValueError(f"Unknown Hugging Face PatchTST kind: {kind}")
        self.kind = kind
        data = config["data"]
        model = config["model"]
        settings = config["hf_patchtst"]
        imu_channels = grid_imu_feature_dim(data)
        if kind == "hf_patchtst_imu":
            input_channels = imu_channels
        elif kind == "hf_patchtst_emg":
            input_channels = 4
        else:
            input_channels = 4 + imu_channels

        self.context_length = int(settings["context_length"])
        self.input_channels = input_channels
        self.use_cls_token = bool(settings.get("use_cls_token", True))
        self.hf_config = PatchTSTConfig(
            num_input_channels=input_channels,
            context_length=self.context_length,
            patch_length=int(settings["patch_length"]),
            patch_stride=int(settings["patch_stride"]),
            num_hidden_layers=int(settings["num_hidden_layers"]),
            d_model=int(settings["d_model"]),
            num_attention_heads=int(settings["num_attention_heads"]),
            channel_attention=bool(settings.get("channel_attention", True)),
            share_embedding=bool(settings.get("share_embedding", True)),
            ffn_dim=int(settings["ffn_dim"]),
            norm_type=str(settings.get("norm_type", "layernorm")),
            norm_eps=float(settings.get("norm_eps", 1e-5)),
            attention_dropout=float(settings.get("attention_dropout", 0.0)),
            positional_dropout=float(settings.get("positional_dropout", 0.0)),
            path_dropout=float(settings.get("path_dropout", 0.0)),
            ff_dropout=float(settings.get("ff_dropout", 0.0)),
            bias=bool(settings.get("bias", True)),
            activation_function=str(settings.get("activation_function", "gelu")),
            pre_norm=bool(settings.get("pre_norm", True)),
            positional_encoding_type=str(
                settings.get("positional_encoding_type", "sincos")
            ),
            use_cls_token=self.use_cls_token,
            scaling=settings.get("scaling", None),
            do_mask_input=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # This is deliberately the library model, not the project's custom
        # MaskAwarePatchEncoder.
        self.patchtst = PatchTSTModel(self.hf_config)

        hidden_dim = int(settings["d_model"])
        task_dim = int(model["d_model"])
        self.context_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(input_channels * hidden_dim),
            nn.Linear(input_channels * hidden_dim, task_dim),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
        )
        self.time_conditioner = (
            ElapsedTimeConditioner(task_dim)
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        self.head = SpatialPointHead(
            task_dim,
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
        )

    def _signals(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "hf_patchtst_imu":
            return batch["imu"], batch["imu_mask"]
        if self.kind == "hf_patchtst_emg":
            return batch["emg"], batch["emg_mask"]
        return (
            torch.cat([batch["emg"], batch["imu"]], dim=-1),
            torch.cat([batch["emg_mask"], batch["imu_mask"]], dim=-1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values, mask = self._signals(batch)
        fixed_values, fixed_mask = right_align_fixed_context(
            values, mask, batch["lengths"], self.context_length
        )
        model_output = self.patchtst(
            past_values=fixed_values,
            past_observed_mask=fixed_mask,
            return_dict=True,
        )
        hidden = model_output.last_hidden_state
        if self.use_cls_token:
            channel_context = hidden[:, :, 0, :]
        else:
            channel_context = hidden.mean(dim=2)
        context = self.context_projection(channel_context)
        if self.time_conditioner is not None:
            context = context + self.time_conditioner(elapsed_from_batch(batch))
        outputs = self.head(context)
        outputs.update(
            decode_grid_outputs(
                outputs["heatmap_logits"],
                outputs["offset_logits"],
                self.grid_width,
                self.grid_height,
            )
        )
        finalize_point_prediction(outputs)
        outputs["context"] = context
        return outputs


def build_hf_patchtst_model(
    kind: str, config: dict[str, Any]
) -> HFPatchTSTPointRegressor:
    return HFPatchTSTPointRegressor(kind, config)
