from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..data.grid_trajectory import (
    emg_channel_count,
    grid_imu_feature_dim,
    grid_imu_sensor_indices,
)
from ..physics.rollout import PhysicsBranch
from ..physics.rollout3 import PhysicsBranch3
from .cross_variate import CrossVariateBackbone
from .layers import PatchTransformerEncoder


GRID_MODEL_KINDS = (
    "grid_imu",
    "grid_emg",
    "grid_fusion",
    "grid_emg_first",
    "grid_crossvar",
    "grid_fusion_physics",
    "grid_fusion_physics3",
    "grid_fusion_vae",
)


def masked_channel_statistics(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(values.dtype)
    count = valid.sum(dim=1).clamp_min(1.0)
    coverage = valid.mean(dim=1)
    mean = (values * valid).sum(dim=1) / count
    mean_abs = (values.abs() * valid).sum(dim=1) / count
    variance = ((values - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count
    split = max(1, values.size(1) // 2)

    def part_abs(part_values: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        weights = part_mask.to(part_values.dtype)
        return (part_values.abs() * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    early = part_abs(values[:, :split], mask[:, :split])
    late = part_abs(values[:, split:], mask[:, split:])
    statistics = torch.stack(
        [coverage, mean_abs, variance.sqrt(), late - early], dim=-1
    )
    return statistics, mask.any(dim=1)


def masked_softmax(
    logits: torch.Tensor, available: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    safe_logits = logits.masked_fill(~available, -1e4)
    probabilities = torch.softmax(safe_logits, dim=dim) * available.to(logits.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class HierarchicalChannelAttention(nn.Module):
    """Dynamic sensor-then-feature attention with global importance priors."""

    def __init__(
        self,
        channels: int,
        sensor_indices: tuple[int, ...],
        hidden_dim: int,
        residual_strength: float,
    ) -> None:
        super().__init__()
        if len(sensor_indices) != channels:
            raise ValueError("Every channel must have a sensor index")
        self.channels = channels
        self.sensor_count = max(sensor_indices) + 1
        self.residual_strength = float(residual_strength)
        self.register_buffer(
            "sensor_indices", torch.tensor(sensor_indices, dtype=torch.long)
        )
        self.channel_score = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sensor_score = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_channel_logits = nn.Parameter(torch.zeros(channels))
        self.global_sensor_logits = nn.Parameter(torch.zeros(self.sensor_count))

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        statistics, channel_available = masked_channel_statistics(values, mask)
        channel_logits = self.channel_score(statistics).squeeze(-1)
        channel_logits = channel_logits + self.global_channel_logits.unsqueeze(0)
        sensor_statistics = []
        sensor_available = []
        feature_probabilities = torch.zeros_like(channel_logits)

        for sensor in range(self.sensor_count):
            selected = self.sensor_indices == sensor
            selected_available = channel_available[:, selected]
            weights = selected_available.to(values.dtype)
            sensor_statistics.append(
                (statistics[:, selected] * weights.unsqueeze(-1)).sum(dim=1)
                / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            sensor_available.append(selected_available.any(dim=1))
            feature_probabilities[:, selected] = masked_softmax(
                channel_logits[:, selected], selected_available
            )

        sensor_statistics_tensor = torch.stack(sensor_statistics, dim=1)
        sensor_available_tensor = torch.stack(sensor_available, dim=1)
        sensor_logits = self.sensor_score(sensor_statistics_tensor).squeeze(-1)
        sensor_logits = sensor_logits + self.global_sensor_logits.unsqueeze(0)
        sensor_probabilities = masked_softmax(
            sensor_logits, sensor_available_tensor
        )

        channel_probabilities = torch.zeros_like(channel_logits)
        for sensor in range(self.sensor_count):
            selected = self.sensor_indices == sensor
            channel_probabilities[:, selected] = (
                sensor_probabilities[:, sensor].unsqueeze(1)
                * feature_probabilities[:, selected]
            )
        channel_probabilities = (
            channel_probabilities * channel_available.to(values.dtype)
        )
        # Attention probabilities sum to one and are easy to interpret. Scale
        # them only for the residual input gate so uniform attention is the
        # identity transform instead of shrinking every signal channel.
        available_count = channel_available.sum(dim=1).clamp_min(1)
        importance = channel_probabilities * available_count.unsqueeze(1)
        factor = (
            1.0 - self.residual_strength
            + self.residual_strength * importance
        )
        weighted = values * factor.unsqueeze(1)
        weighted = weighted.masked_fill(~mask, 0.0)
        return weighted, sensor_probabilities, channel_probabilities


class ElapsedTimeConditioner(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, elapsed_s: torch.Tensor) -> torch.Tensor:
        elapsed = elapsed_s.clamp(0.0, 10.0)
        features = torch.stack(
            [elapsed / 2.0, torch.log1p(elapsed), torch.sqrt(elapsed + 1e-6)],
            dim=-1,
        )
        return self.network(features)


def elapsed_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "prefix_elapsed_s" in batch:
        return batch["prefix_elapsed_s"]
    return batch["movement_duration_s"]


def gather_tail(
    values: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a right-aligned causal endpoint window from left-padded batches."""

    batch, _, channels = values.shape
    relative = torch.arange(sample_count, device=values.device).unsqueeze(0)
    indices = lengths.unsqueeze(1) - sample_count + relative
    valid_time = (indices >= 0) & (indices < lengths.unsqueeze(1))
    safe_indices = indices.clamp(0, values.size(1) - 1)
    gather_indices = safe_indices.unsqueeze(-1).expand(batch, sample_count, channels)
    selected_values = values.gather(1, gather_indices)
    selected_mask = mask.gather(1, gather_indices) & valid_time.unsqueeze(-1)
    selected_values = selected_values.masked_fill(~selected_mask, 0.0)
    return selected_values, selected_mask


class MaskAwarePatchEncoder(nn.Module):
    def __init__(self, input_dim: int, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = PatchTransformerEncoder(
            input_dim=input_dim * 2,
            d_model=int(model_config["d_model"]),
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            dropout=float(model_config["dropout"]),
            patch_length=int(model_config["patch_length"]),
            patch_stride=int(model_config["patch_stride"]),
            kernel_sizes=list(model_config["tcn_kernel_sizes"]),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([values, mask.to(values.dtype)], dim=-1)
        return self.encoder(inputs, mask.any(dim=-1))


def endpoint_quality(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(values.dtype)
    count = valid.sum(dim=(1, 2)).clamp_min(1.0)
    coverage = valid.mean(dim=(1, 2))
    mean_abs = (values.abs() * valid).sum(dim=(1, 2)) / count
    mean = (values * valid).sum(dim=(1, 2)) / count
    variance = ((values - mean[:, None, None]).square() * valid).sum(
        dim=(1, 2)
    ) / count
    split = max(1, values.size(1) // 2)

    def masked_abs_mean(part_values: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        weights = part_mask.to(part_values.dtype)
        return (part_values.abs() * weights).sum(dim=(1, 2)) / weights.sum(
            dim=(1, 2)
        ).clamp_min(1.0)

    early = masked_abs_mean(values[:, :split], mask[:, :split])
    late = masked_abs_mean(values[:, split:], mask[:, split:])
    endpoint_change = late - early
    return torch.stack([coverage, mean_abs, variance.sqrt(), endpoint_change], dim=-1)


class SpatialPointHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        grid_width: int,
        grid_height: int,
        dropout: float,
        direct_prediction: bool = False,
        zero_initialize: bool = False,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        cells = grid_width * grid_height
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heatmap = nn.Linear(d_model, cells)
        self.offsets = nn.Linear(d_model, cells * 2)
        self.direct = nn.Linear(d_model, 2) if direct_prediction else None
        # VAE-style: the same trunk that already encodes context for the mean
        # (mu = direct) also encodes a spread (sigma) - the "keep encoding"
        # extra head the touch-location distillation idea needs, reusing the
        # trunk rather than adding a second one. Deliberately not wired into
        # mu at all yet (mu stays exactly the existing deterministic point,
        # unaffected): the first thing worth validating is whether sigma
        # calibrates sensibly at all (does it shrink as more causal EMG/IMU
        # history arrives, does ~68% of targets land within mu+-sigma) before
        # letting it influence training itself.
        self.log_sigma = (
            nn.Linear(d_model, 2) if (direct_prediction and predict_uncertainty) else None
        )
        if zero_initialize:
            nn.init.zeros_(self.heatmap.weight)
            nn.init.zeros_(self.heatmap.bias)
            nn.init.zeros_(self.offsets.weight)
            nn.init.zeros_(self.offsets.bias)
            if self.direct is not None:
                nn.init.zeros_(self.direct.weight)
                nn.init.zeros_(self.direct.bias)
        if self.log_sigma is not None:
            # Zero weight regardless of zero_initialize - a freshly added
            # capability should start inert, same convention as every other
            # new head this project has added. Bias set so sigma starts at
            # ~0.12 normalised units (roughly this model's typical pixel
            # error scale), not an arbitrary default - the model still has
            # to earn any deviation from that through the NLL loss.
            nn.init.zeros_(self.log_sigma.weight)
            nn.init.constant_(self.log_sigma.bias, -2.12)  # log(0.12)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(context)
        batch = context.size(0)
        outputs = {
            "heatmap_logits": self.heatmap(hidden),
            "offset_logits": self.offsets(hidden).reshape(batch, -1, 2),
        }
        if self.direct is not None:
            outputs["direct_logits"] = self.direct(hidden)
        if self.log_sigma is not None:
            # hidden.detach(): reads off the same encoding mu reads off
            # (the "keep encoding" reuse), but must not let sigma's
            # gradient back into `shared` itself - shared's weights feed
            # mu too, so gradient reaching them from the sigma branch
            # changes what mu computes on the next step even though
            # nothing ever touches `direct`'s own weights directly. Found
            # this the hard way: detaching only the downstream mu *value*
            # inside the NLL loss (grid_training.py) verified clean on
            # `direct.weight.grad` in isolation, but a real 35-epoch run
            # still measurably hurt accuracy (184 -> 204px test) - the
            # leak was one level further upstream than that check looked.
            outputs["direct_log_sigma"] = self.log_sigma(hidden.detach())
        return outputs


def decode_grid_outputs(
    heatmap_logits: torch.Tensor,
    offset_logits: torch.Tensor,
    grid_width: int,
    grid_height: int,
) -> dict[str, torch.Tensor]:
    probabilities = torch.softmax(heatmap_logits, dim=-1)
    offsets = 0.5 * torch.tanh(offset_logits)
    cell_indices = torch.arange(
        grid_width * grid_height, device=heatmap_logits.device
    )
    cell_x = (cell_indices % grid_width).to(heatmap_logits.dtype)
    cell_y = torch.div(cell_indices, grid_width, rounding_mode="floor").to(
        heatmap_logits.dtype
    )
    candidate_x = (cell_x[None, :] + 0.5 + offsets[:, :, 0]) / grid_width
    candidate_y = (cell_y[None, :] + 0.5 + offsets[:, :, 1]) / grid_height
    candidates = torch.stack([candidate_x, candidate_y], dim=-1).clamp(0.0, 1.0)
    soft_prediction = (probabilities.unsqueeze(-1) * candidates).sum(dim=1)
    predicted_cell = heatmap_logits.argmax(dim=-1)
    hard_prediction = candidates.gather(
        1, predicted_cell[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return {
        "prediction": hard_prediction,
        "soft_prediction": soft_prediction,
        "probabilities": probabilities,
        "offsets": offsets,
        "candidates": candidates,
        "predicted_cell": predicted_cell,
        "heatmap_confidence": probabilities.max(dim=-1).values,
        "heatmap_entropy": entropy,
    }


def finalize_point_prediction(outputs: dict[str, torch.Tensor]) -> None:
    """Use direct continuous XY when configured, retaining the grid prediction."""

    if "direct_logits" not in outputs:
        return
    outputs["grid_prediction"] = outputs["prediction"]
    outputs["direct_prediction"] = torch.sigmoid(outputs["direct_logits"])
    outputs["prediction"] = outputs["direct_prediction"]
    if "direct_log_sigma" in outputs:
        # Clamped in log-space before exp, not after: bounds sigma to
        # roughly [0.0025, 7.4] normalised units - wide enough to never bind
        # during normal training, narrow enough that a bad batch can't send
        # sigma to 0 or inf and break the NLL loss that trains it.
        outputs["direct_sigma"] = torch.exp(outputs["direct_log_sigma"].clamp(-6.0, 2.0))


class IMUGridBackbone(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        d_model = int(model["d_model"])
        input_dim = grid_imu_feature_dim(data)
        attention = config.get("channel_attention", {})
        self.input_attention = (
            HierarchicalChannelAttention(
                input_dim,
                grid_imu_sensor_indices(data),
                int(attention.get("hidden_dim", 32)),
                float(attention.get("residual_strength", 0.5)),
            )
            if bool(attention.get("enabled", False))
            else None
        )
        self.sample_rate = float(data["sample_rate_hz"])
        self.tail_samples = max(
            int(model["patch_length"]),
            int(math.ceil(float(model.get("imu_endpoint_s", 0.5)) * self.sample_rate)),
        )
        self.full_encoder = MaskAwarePatchEncoder(input_dim, model)
        self.tail_encoder = MaskAwarePatchEncoder(input_dim, model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        sensor_attention = None
        channel_attention = None
        if self.input_attention is not None:
            imu, sensor_attention, channel_attention = self.input_attention(
                imu, imu_mask
            )
        _, full_context, _ = self.full_encoder(imu, imu_mask)
        tail, tail_mask = gather_tail(
            imu, imu_mask, lengths, self.tail_samples
        )
        _, tail_context, _ = self.tail_encoder(tail, tail_mask)
        context = self.fusion(torch.cat([full_context, tail_context], dim=-1))
        return context, sensor_attention, channel_attention


class EMGEndpointBackbone(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        d_model = int(model["d_model"])
        emg_channels = emg_channel_count(data)
        attention = config.get("channel_attention", {})
        self.input_attention = (
            HierarchicalChannelAttention(
                emg_channels,
                tuple(range(emg_channels)),
                int(attention.get("hidden_dim", 32)),
                float(attention.get("residual_strength", 0.5)),
            )
            if bool(attention.get("enabled", False))
            else None
        )
        sample_rate = float(data["sample_rate_hz"])
        minimum = int(model["patch_length"])
        self.samples_500 = max(
            minimum, int(math.ceil(float(model.get("emg_long_s", 0.5)) * sample_rate))
        )
        self.samples_300 = max(
            minimum, int(math.ceil(float(model.get("emg_short_s", 0.3)) * sample_rate))
        )
        self.encoder_500 = MaskAwarePatchEncoder(emg_channels, model)
        self.encoder_300 = MaskAwarePatchEncoder(emg_channels, model)
        # Both endpoint windows discard most of a typical reach, so the
        # anticipatory pre-movement burst never reaches the head. When enabled,
        # a third encoder reads the entire causal prefix alongside them.
        self.full_context = bool(model.get("emg_full_context", False))
        self.full_encoder = (
            MaskAwarePatchEncoder(emg_channels, model) if self.full_context else None
        )
        windows = 3 if self.full_context else 2
        quality_dim = 4 * windows
        joint_dim = d_model * windows + quality_dim
        self.lookback_gate = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, windows),
        )
        self.fusion = nn.Sequential(
            nn.Linear(joint_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.reliability = nn.Sequential(
            nn.Linear(d_model + quality_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        channel_attention = None
        if self.input_attention is not None:
            emg, _, channel_attention = self.input_attention(emg, emg_mask)
        emg_500, mask_500 = gather_tail(
            emg, emg_mask, lengths, self.samples_500
        )
        emg_300, mask_300 = gather_tail(
            emg, emg_mask, lengths, self.samples_300
        )
        _, context_500, _ = self.encoder_500(emg_500, mask_500)
        _, context_300, _ = self.encoder_300(emg_300, mask_300)
        contexts = [context_500, context_300]
        qualities = [
            endpoint_quality(emg_500, mask_500),
            endpoint_quality(emg_300, mask_300),
        ]
        if self.full_encoder is not None:
            _, context_full, _ = self.full_encoder(emg, emg_mask)
            contexts.insert(0, context_full)
            qualities.insert(0, endpoint_quality(emg, emg_mask))
        quality = torch.cat(qualities, dim=-1)
        joint = torch.cat([*contexts, quality], dim=-1)
        lookback_weights = torch.softmax(self.lookback_gate(joint), dim=-1)
        selected = sum(
            lookback_weights[:, index : index + 1] * value
            for index, value in enumerate(contexts)
        )
        differences = [
            contexts[index] - contexts[index + 1]
            for index in range(len(contexts) - 1)
        ]
        context = self.fusion(
            torch.cat([selected, *differences, quality], dim=-1)
        )
        reliability = self.reliability(torch.cat([context, quality], dim=-1))
        return context, reliability, lookback_weights, quality, channel_attention


class GridIMURegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        self.backbone = IMUGridBackbone(config)
        self.time_conditioner = (
            ElapsedTimeConditioner(int(model["d_model"]))
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        self.head = SpatialPointHead(
            int(model["d_model"]),
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, sensor_attention, channel_attention = self.backbone(
            batch["imu"], batch["imu_mask"], batch["lengths"]
        )
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
        if sensor_attention is not None:
            outputs["imu_sensor_attention"] = sensor_attention
        if channel_attention is not None:
            outputs["imu_channel_attention"] = channel_attention
        return outputs


class GridEMGRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        self.backbone = EMGEndpointBackbone(config)
        self.time_conditioner = (
            ElapsedTimeConditioner(int(model["d_model"]))
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        self.head = SpatialPointHead(
            int(model["d_model"]),
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, reliability, lookback_weights, quality, channel_attention = self.backbone(
            batch["emg"], batch["emg_mask"], batch["lengths"]
        )
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
        outputs.update(
            {
                "context": context,
                "emg_reliability": reliability,
                "emg_lookback_weights": lookback_weights,
                "emg_quality": quality,
            }
        )
        if channel_attention is not None:
            outputs["emg_channel_attention"] = channel_attention
        return outputs


class GridFusionRegressor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        self.imu_model = GridIMURegressor(config)
        self.emg_backbone = EMGEndpointBackbone(config)
        self.emg_time_conditioner = (
            ElapsedTimeConditioner(int(model["d_model"]))
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        self.predict_uncertainty = bool(model.get("predict_uncertainty", False))
        self.emg_residual_head = SpatialPointHead(
            int(model["d_model"]),
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
            zero_initialize=True,
            predict_uncertainty=self.predict_uncertainty,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.imu_model(batch)
        (
            emg_context,
            reliability,
            lookback_weights,
            quality,
            emg_channel_attention,
        ) = self.emg_backbone(
            batch["emg"], batch["emg_mask"], batch["lengths"]
        )
        if self.emg_time_conditioner is not None:
            emg_context = emg_context + self.emg_time_conditioner(
                elapsed_from_batch(batch)
            )
        residual = self.emg_residual_head(emg_context)
        heatmap_logits = base["heatmap_logits"] + reliability * residual["heatmap_logits"]
        offset_logits = base["offset_logits"] + reliability.unsqueeze(-1) * residual[
            "offset_logits"
        ]
        outputs = {
            "heatmap_logits": heatmap_logits,
            "offset_logits": offset_logits,
            "base_prediction": base["prediction"],
            "emg_reliability": reliability,
            "emg_lookback_weights": lookback_weights,
            "emg_quality": quality,
            "emg_context": emg_context,
            # Exposed so a wrapper can encode from both modalities rather
            # than re-running the IMU backbone: the VAE variant needs the
            # joint representation, not just the EMG half.
            "imu_context": base["context"],
        }
        for key in ("imu_sensor_attention", "imu_channel_attention"):
            if key in base:
                outputs[key] = base[key]
        if emg_channel_attention is not None:
            outputs["emg_channel_attention"] = emg_channel_attention
        if self.direct_prediction:
            outputs["direct_logits"] = base["direct_logits"] + reliability * residual[
                "direct_logits"
            ]
        if "direct_log_sigma" in residual:
            # Uncertainty is scoped to the EMG residual head only, not
            # combined with a (nonexistent, since the IMU base head isn't
            # opted into this) base-model sigma - this is deliberately the
            # smallest testable version: does sigma calibrate sensibly at
            # all, before deciding how it should combine with anything else.
            outputs["direct_log_sigma"] = residual["direct_log_sigma"]
        outputs.update(
            decode_grid_outputs(
                heatmap_logits,
                offset_logits,
                self.grid_width,
                self.grid_height,
            )
        )
        finalize_point_prediction(outputs)
        return outputs


class GridEMGFirstRegressor(nn.Module):
    """EMG is the base predictor; IMU enters only as a gated residual.

    This mirrors GridFusionRegressor with the modalities exchanged. The IMU
    head is zero-initialized, so the model starts as grid_emg exactly and the
    learned gate measures how much IMU the task actually requires.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        d_model = int(model["d_model"])
        self.emg_model = GridEMGRegressor(config)
        self.imu_backbone = IMUGridBackbone(config)
        self.imu_time_conditioner = (
            ElapsedTimeConditioner(d_model)
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        self.imu_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.imu_residual_head = SpatialPointHead(
            d_model,
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
            zero_initialize=True,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.emg_model(batch)
        imu_context, sensor_attention, channel_attention = self.imu_backbone(
            batch["imu"], batch["imu_mask"], batch["lengths"]
        )
        if self.imu_time_conditioner is not None:
            imu_context = imu_context + self.imu_time_conditioner(
                elapsed_from_batch(batch)
            )
        gate = self.imu_gate(torch.cat([imu_context, base["context"]], dim=-1))
        residual = self.imu_residual_head(imu_context)
        heatmap_logits = base["heatmap_logits"] + gate * residual["heatmap_logits"]
        offset_logits = base["offset_logits"] + gate.unsqueeze(-1) * residual[
            "offset_logits"
        ]
        outputs = {
            "heatmap_logits": heatmap_logits,
            "offset_logits": offset_logits,
            "base_prediction": base["prediction"],
            "imu_gate": gate,
        }
        for key in (
            "emg_reliability",
            "emg_lookback_weights",
            "emg_quality",
            "emg_channel_attention",
        ):
            if key in base:
                outputs[key] = base[key]
        if sensor_attention is not None:
            outputs["imu_sensor_attention"] = sensor_attention
        if channel_attention is not None:
            outputs["imu_channel_attention"] = channel_attention
        if self.direct_prediction:
            outputs["direct_logits"] = base["direct_logits"] + gate * residual[
                "direct_logits"
            ]
        outputs.update(
            decode_grid_outputs(
                heatmap_logits,
                offset_logits,
                self.grid_width,
                self.grid_height,
            )
        )
        finalize_point_prediction(outputs)
        return outputs


class GridCrossVariateRegressor(nn.Module):
    """Multi-scale patching plus one cross-variate mix over eight sensor tokens.

    EMG and IMU are mixed at the patch level rather than combined as late,
    scalar-gated logits. Their per-trial errors are nearly uncorrelated
    (r = 0.09-0.21 on a1), so the complementary information is per trial and per
    time, which a single scalar reliability cannot route.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.direct_prediction = str(
            model.get("prediction_mode", "grid")
        ).lower() == "direct_aux_grid"
        self.backbone = CrossVariateBackbone(config)
        self.time_conditioner = (
            ElapsedTimeConditioner(int(model["d_model"]))
            if bool(config.get("continual", {}).get("enabled", False))
            else None
        )
        self.head = SpatialPointHead(
            int(model["d_model"]),
            grid_width,
            grid_height,
            float(model["dropout"]),
            direct_prediction=self.direct_prediction,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, variate_attention, cross_attention, scale_gate = self.backbone(
            batch["emg"],
            batch["emg_mask"],
            batch["imu"],
            batch["imu_mask"],
            batch["lengths"],
        )
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
        outputs["variate_attention"] = variate_attention
        outputs["cross_variate_attention"] = cross_attention
        outputs["scale_gate"] = scale_gate
        return outputs


class GridFusionPhysicsRegressor(nn.Module):
    """grid_fusion with a Hill-model physics branch sharing its encoder.

    The fusion coordinate head is unchanged and still carries the prediction.
    The physics branch integrates the Hill-driven two-link arm from EMG and
    emits its own endpoint estimate, supervised against the same click. It is
    auxiliary by construction: forward kinematics needs roughly 3 degrees of
    arm-orientation accuracy to match what the learned head already achieves,
    and IMU orientation here is 12-19 degrees, so physics cannot be trusted to
    carry the coordinate. What it can do is give EMG a structured pathway -
    activation, force, torque, joint angle - and shape the shared encoder
    through the auxiliary loss.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.fusion = GridFusionRegressor(config)
        self.physics = PhysicsBranch(config)
        physics = config.get("physics", {})
        # Blend weight for folding the physics endpoint into the reported
        # coordinate. Zero-initialised, so the model starts as exactly
        # grid_fusion and physics has to earn any influence.
        self.raw_blend = nn.Parameter(torch.tensor(float(physics.get("blend_init", -4.0))))

    @property
    def blend(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_blend)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.fusion(batch)
        physics = self.physics(
            batch["emg"],
            batch["emg_mask"],
            batch["lengths"],
            outputs["emg_context"],
            batch["subject"],
        )
        outputs.update(physics)
        outputs["fusion_prediction"] = outputs["prediction"]
        blend = self.blend
        outputs["physics_blend"] = blend.expand(outputs["prediction"].size(0))
        blended = (
            (1.0 - blend) * outputs["prediction"]
            + blend * physics["physics_prediction"].clamp(0.0, 1.0)
        )
        outputs["prediction"] = blended
        # grid_point_loss optimises direct_prediction while evaluation reads
        # prediction. Both must be the blended coordinate, otherwise the blend
        # weight receives no gradient and the model is scored on a quantity it
        # never trained.
        if "direct_prediction" in outputs:
            outputs["direct_prediction"] = blended
        return outputs


class GridFusionPhysics3Regressor(nn.Module):
    """grid_fusion with a 3-DOF rigid-body physics branch sharing its encoder.

    Same auxiliary-blend structure as GridFusionPhysicsRegressor, but the
    physics branch is arm3.ThreeDofArm (2-shoulder + elbow, screw-theory
    dynamics) driven by a plain torque MLP instead of the Hill muscle chain -
    see physics/rollout3.py for why.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.fusion = GridFusionRegressor(config)
        self.physics = PhysicsBranch3(config)
        physics = config.get("physics", {})
        d_model = int(config["model"]["d_model"])
        self.raw_blend = nn.Parameter(torch.tensor(float(physics.get("blend_init", -4.0))))
        # A single global blend scalar cannot express that physics may be
        # worth trusting on some trials and not others: measured across two
        # loss weights it simply froze at its initialisation, and a direct
        # gradient read found the per-batch incentive averaging to noise. This
        # gate lets the weight vary per trial on the same context the rest of
        # the branch sees. Zero-initialised, so the model starts at exactly
        # the previous global-scalar behaviour and any per-trial structure has
        # to be earned.
        self.blend_gate = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.blend_gate[-1].weight)
        nn.init.zeros_(self.blend_gate[-1].bias)

    @property
    def blend(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_blend)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.fusion(batch)
        physics = self.physics(
            batch["emg"],
            batch["emg_mask"],
            batch["lengths"],
            outputs["emg_context"],
            batch["subject"],
            batch["imu"],
            batch["imu_mask"],
        )
        outputs.update(physics)
        outputs["fusion_prediction"] = outputs["prediction"]
        blend = torch.sigmoid(
            self.raw_blend + self.blend_gate(outputs["emg_context"]).squeeze(-1)
        )
        outputs["physics_blend"] = blend
        blended = (
            (1.0 - blend).unsqueeze(-1) * outputs["prediction"]
            + blend.unsqueeze(-1) * physics["physics_prediction"].clamp(0.0, 1.0)
        )
        outputs["prediction"] = blended
        if "direct_prediction" in outputs:
            outputs["direct_prediction"] = blended
        return outputs


class LatentPointVAE(nn.Module):
    """Encoder -> q(z|x) = N(mu, sigma^2) -> sampled z -> screen point.

    A real variational bottleneck, not a second output head: z is sampled
    through the reparameterisation trick during training (mu alone at
    evaluation, the usual deterministic readout), and the KL term in
    grid_point_loss regularises q(z|x) toward N(0, I). Because the decoder
    has no path to the target except through z, every bit the prediction
    needs has to be carried by the latent and paid for in KL - that is the
    information bottleneck this is here to impose.

    latent_dim defaults to 3 deliberately. In the planned second stage this
    decoder is replaced by the 3R arm's forward kinematics and z becomes the
    three joint angles, so testing the bottleneck at its eventual width now
    means the accuracy cost of that width is already known before the
    kinematics are swapped in.
    """

    def __init__(
        self, context_dim: int, latent_dim: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_log_variance = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.to_mu.weight, std=0.01)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_log_variance.weight)
        # Bias -4 => sigma ~0.135 at initialisation, NOT sigma ~1 (bias 0).
        # Measured why: with sigma ~1 the encoder's mu spread across a batch
        # was ~0.034 after real steps, so the sampled z was ~30x more
        # injected noise than signal and the decoder had nothing usable to
        # read - the standard road into posterior collapse. Starting the
        # posterior tighter than the prior lets mu's signal dominate from the
        # first step; the KL term (which pays -log sigma^2 for being tight)
        # still pushes back toward the prior, so this sets where the search
        # starts, not where it is allowed to end up.
        nn.init.constant_(self.to_log_variance.bias, -4.0)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(context)
        mu = self.to_mu(hidden)
        # Clamped before exp: bounds sigma to roughly [0.018, 7.4], wide
        # enough never to bind in normal training but enough to stop a bad
        # batch sending the KL or the sample to inf.
        log_variance = self.to_log_variance(hidden).clamp(-8.0, 4.0)
        if self.training:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mu + standard_deviation * torch.randn_like(standard_deviation)
        else:
            latent = mu
        return self.decoder(latent), mu, log_variance


class GridFusionVAERegressor(nn.Module):
    """grid_fusion with its coordinate prediction routed through a VAE latent.

    The fusion model is kept intact and still drives the grid path
    (heatmap/offset losses), so the shared encoder keeps receiving the same
    supervision it always did. What changes is where the reported coordinate
    comes from: instead of fusion's own direct head, it is decoded from a
    sampled latent encoded from both modalities' contexts. fusion's direct
    head therefore stops contributing to the prediction and goes untrained -
    harmless, and left in place so this wrapper stays a pure addition rather
    than a modification of a model every other experiment depends on.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        self.fusion = GridFusionRegressor(config)
        d_model = int(model["d_model"])
        self.vae = LatentPointVAE(
            context_dim=2 * d_model,
            latent_dim=int(model.get("latent_dim", 3)),
            hidden=int(model.get("latent_hidden", 64)),
            dropout=float(model["dropout"]),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.fusion(batch)
        context = torch.cat(
            [outputs["imu_context"], outputs["emg_context"]], dim=-1
        )
        logits, mu, log_variance = self.vae(context)
        outputs["fusion_prediction"] = outputs["prediction"]
        outputs["latent_mu"] = mu
        outputs["latent_log_variance"] = log_variance
        outputs["latent_sigma"] = torch.exp(0.5 * log_variance)
        outputs["direct_logits"] = logits
        # grid_point_loss optimises direct_prediction and evaluation reads
        # prediction; both must be the VAE's output or the model would be
        # scored on a coordinate it never trained (the same wiring mistake
        # the physics branch hit earlier).
        outputs["direct_prediction"] = torch.sigmoid(logits)
        outputs["prediction"] = outputs["direct_prediction"]
        return outputs


def build_grid_model(kind: str, config: dict[str, Any]) -> nn.Module:
    if kind == "grid_imu":
        return GridIMURegressor(config)
    if kind == "grid_emg":
        return GridEMGRegressor(config)
    if kind == "grid_fusion":
        return GridFusionRegressor(config)
    if kind == "grid_emg_first":
        return GridEMGFirstRegressor(config)
    if kind == "grid_crossvar":
        return GridCrossVariateRegressor(config)
    if kind == "grid_fusion_physics":
        return GridFusionPhysicsRegressor(config)
    if kind == "grid_fusion_physics3":
        return GridFusionPhysics3Regressor(config)
    if kind == "grid_fusion_vae":
        return GridFusionVAERegressor(config)
    raise ValueError(f"Unknown grid model kind: {kind}")
