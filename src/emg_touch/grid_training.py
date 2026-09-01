from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data.grid_trajectory import emg_channel_names, grid_imu_feature_names
from .data.schema import SENSORS
from .metrics import merge_metric_batches
from .utils import move_batch_to_device


VARIATE_NAMES = ("AD", "LD", "BB", "TB", "S0", "S4", "S8", "S12")


def grid_targets(
    target: torch.Tensor,
    grid_width: int,
    grid_height: int,
    gaussian_sigma_cells: float,
) -> dict[str, torch.Tensor]:
    safe_target = target.clamp(0.0, 1.0 - 1e-7)
    scaled_x = safe_target[:, 0] * grid_width
    scaled_y = safe_target[:, 1] * grid_height
    cell_x = scaled_x.floor().long().clamp(0, grid_width - 1)
    cell_y = scaled_y.floor().long().clamp(0, grid_height - 1)
    cell_index = cell_y * grid_width + cell_x
    offset = torch.stack(
        [scaled_x - (cell_x.to(target.dtype) + 0.5), scaled_y - (cell_y.to(target.dtype) + 0.5)],
        dim=-1,
    )

    indices = torch.arange(grid_width * grid_height, device=target.device)
    anchor_x = (indices % grid_width).to(target.dtype) + 0.5
    anchor_y = torch.div(indices, grid_width, rounding_mode="floor").to(target.dtype) + 0.5
    distance_squared = (
        (anchor_x[None, :] - scaled_x[:, None]).square()
        + (anchor_y[None, :] - scaled_y[:, None]).square()
    )
    sigma = max(float(gaussian_sigma_cells), 1e-3)
    soft_heatmap = torch.exp(-0.5 * distance_squared / (sigma * sigma))
    soft_heatmap = soft_heatmap / soft_heatmap.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return {
        "cell_x": cell_x,
        "cell_y": cell_y,
        "cell_index": cell_index,
        "offset": offset,
        "soft_heatmap": soft_heatmap,
    }


def continual_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("continual", {}).get("enabled", False))


def _prefix_length(
    cue_offset_s: float,
    elapsed_s: float,
    sample_rate_hz: float,
    full_length: int,
) -> int:
    return max(
        2,
        min(
            full_length,
            int(np.floor((cue_offset_s + elapsed_s) * sample_rate_hz)) + 1,
        ),
    )


def _build_prefix_batch(
    batch: dict[str, Any],
    selections: list[tuple[int, int, float, float, bool, float]],
    sample_rate_hz: float,
) -> dict[str, Any] | None:
    """Create a padded batch of causal views.

    A selection contains source index, prefix length, elapsed movement time,
    progress (used only for weighting/reporting), endpoint flag and loss weight.
    """

    if not selections:
        return None
    device = batch["lengths"].device
    indices = torch.tensor([item[0] for item in selections], device=device)
    prefix_lengths = torch.tensor(
        [item[1] for item in selections], dtype=torch.long, device=device
    )
    maximum = int(prefix_lengths.max())
    result: dict[str, Any] = {}
    for key in ("emg", "emg_mask", "imu", "imu_mask"):
        source = batch[key]
        shape = (len(selections), maximum, source.size(2))
        target = source.new_zeros(shape)
        for output_index, (source_index, length, *_rest) in enumerate(selections):
            target[output_index, :length] = source[source_index, :length]
        result[key] = target
    result["lengths"] = prefix_lengths

    sequence_keys = {"emg", "emg_mask", "imu", "imu_mask", "lengths"}
    for key, value in batch.items():
        if key in sequence_keys:
            continue
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == len(batch["lengths"]):
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == len(batch["lengths"]):
            result[key] = [value[int(index)] for index in indices.cpu().tolist()]
        else:
            result[key] = value

    result["full_duration_s"] = result["duration_s"].clone()
    result["duration_s"] = (prefix_lengths.to(torch.float32) - 1.0) / float(
        sample_rate_hz
    )
    result["prefix_elapsed_s"] = torch.tensor(
        [item[2] for item in selections], dtype=torch.float32, device=device
    )
    result["prefix_progress"] = torch.tensor(
        [item[3] for item in selections], dtype=torch.float32, device=device
    )
    result["is_endpoint"] = torch.tensor(
        [item[4] for item in selections], dtype=torch.bool, device=device
    )
    result["loss_weight"] = torch.tensor(
        [item[5] for item in selections], dtype=torch.float32, device=device
    )
    result["requested_cutoff"] = [
        "touch" if item[4] else f"{item[2]:.1f}s" for item in selections
    ]
    return result


def make_continual_training_batch(
    batch: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    settings = config.get("continual", {})
    if not bool(settings.get("enabled", False)):
        return batch
    sample_rate = float(config["data"]["sample_rate_hz"])
    interval = float(settings.get("interval_s", 0.2))
    random_prefixes = int(settings.get("random_prefixes_per_trajectory", 2))
    include_start = bool(settings.get("include_start_prediction", True))
    include_endpoint = bool(settings.get("include_endpoint", True))
    minimum_weight = float(settings.get("minimum_prefix_weight", 0.5))
    start_weight = float(settings.get("start_prefix_weight", minimum_weight))
    weight_power = float(settings.get("prefix_weight_power", 1.0))
    selections: list[tuple[int, int, float, float, bool, float]] = []

    for index in range(len(batch["lengths"])):
        full_length = int(batch["lengths"][index])
        cue_offset = float(batch["cue_offset_s"][index])
        movement_duration = float(batch["movement_duration_s"][index])
        if include_start:
            selections.append(
                (
                    index,
                    _prefix_length(cue_offset, 0.0, sample_rate, full_length),
                    0.0,
                    0.0,
                    False,
                    start_weight,
                )
            )
        cutoff_count = max(0, int(np.floor((movement_duration - 1e-6) / interval)))
        candidates = interval * torch.arange(1, cutoff_count + 1)
        if len(candidates) and random_prefixes > 0:
            selected = candidates[
                torch.randperm(len(candidates))[: min(random_prefixes, len(candidates))]
            ]
            for cutoff_tensor in selected:
                elapsed = float(cutoff_tensor)
                progress = min(1.0, elapsed / max(movement_duration, 1e-6))
                weight = minimum_weight + (1.0 - minimum_weight) * (
                    progress**weight_power
                )
                selections.append(
                    (
                        index,
                        _prefix_length(
                            cue_offset, elapsed, sample_rate, full_length
                        ),
                        elapsed,
                        progress,
                        False,
                        weight,
                    )
                )
        if include_endpoint:
            selections.append(
                (index, full_length, movement_duration, 1.0, True, 1.0)
            )
    result = _build_prefix_batch(batch, selections, sample_rate)
    if result is None:
        raise ValueError("Continual training produced an empty prefix batch")
    return result


def make_fixed_continual_prefix_batch(
    batch: dict[str, Any], cutoff_s: float, config: dict[str, Any]
) -> dict[str, Any] | None:
    sample_rate = float(config["data"]["sample_rate_hz"])
    settings = config.get("continual", {})
    minimum_weight = float(settings.get("minimum_prefix_weight", 0.5))
    weight_power = float(settings.get("prefix_weight_power", 1.0))
    selections: list[tuple[int, int, float, float, bool, float]] = []
    for index in range(len(batch["lengths"])):
        movement_duration = float(batch["movement_duration_s"][index])
        if movement_duration + 1e-6 < cutoff_s:
            continue
        full_length = int(batch["lengths"][index])
        cue_offset = float(batch["cue_offset_s"][index])
        progress = min(1.0, cutoff_s / max(movement_duration, 1e-6))
        weight = minimum_weight + (1.0 - minimum_weight) * (
            progress**weight_power
        )
        selections.append(
            (
                index,
                _prefix_length(cue_offset, cutoff_s, sample_rate, full_length),
                cutoff_s,
                progress,
                False,
                weight,
            )
        )
    return _build_prefix_batch(batch, selections, sample_rate)


def continual_prefix_batches(
    loader: DataLoader, cutoff_s: float, config: dict[str, Any]
):
    for batch in loader:
        prefix = make_fixed_continual_prefix_batch(batch, cutoff_s, config)
        if prefix is not None:
            yield prefix


def grid_point_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    model_config = config["model"]
    loss_config = config.get("loss", {})
    grid_width, grid_height = map(int, model_config.get("grid_size", [8, 5]))
    targets = grid_targets(
        batch["target"],
        grid_width,
        grid_height,
        float(loss_config.get("gaussian_sigma_cells", 0.5)),
    )
    soft_fraction = float(loss_config.get("gaussian_soft_fraction", 1.0))
    if not 0.0 <= soft_fraction <= 1.0:
        raise ValueError("loss.gaussian_soft_fraction must lie in [0, 1]")
    one_hot = F.one_hot(
        targets["cell_index"], num_classes=grid_width * grid_height
    ).to(batch["target"].dtype)
    target_heatmap = (
        (1.0 - soft_fraction) * one_hot
        + soft_fraction * targets["soft_heatmap"]
    )

    # Give edge/corner examples slightly more influence. Without this, an
    # uncertain regressor can reduce average loss by shrinking predictions to
    # the screen centre. Normalize weights so the overall learning-rate scale
    # remains stable.
    radius = torch.linalg.vector_norm(
        (batch["target"] - 0.5) * 2.0, dim=-1
    ) / (2.0**0.5)
    sample_weights = 1.0 + float(loss_config.get("edge_weight", 0.0)) * radius
    if "loss_weight" in batch:
        sample_weights = sample_weights * batch["loss_weight"].to(
            sample_weights.dtype
        )
    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)

    def weighted_mean(per_sample: torch.Tensor) -> torch.Tensor:
        return (per_sample * sample_weights).mean()

    log_probabilities = F.log_softmax(outputs["heatmap_logits"], dim=-1)
    heatmap_loss = weighted_mean(
        -(target_heatmap * log_probabilities).sum(dim=-1)
    )

    predicted_target_offset = outputs["offsets"].gather(
        1, targets["cell_index"][:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    offset_per_sample = F.smooth_l1_loss(
        predicted_target_offset,
        targets["offset"],
        beta=float(loss_config.get("offset_huber_beta", 0.1)),
        reduction="none",
    ).mean(dim=-1)
    offset_loss = weighted_mean(offset_per_sample)

    coordinate_prediction = outputs.get(
        "direct_prediction", outputs["soft_prediction"]
    )
    pixel_delta = (
        coordinate_prediction - batch["target"]
    ) * batch["canvas_size"]
    pixel_unit = float(loss_config.get("pixel_normalizer_px", 80.0))
    scaled_delta = pixel_delta / pixel_unit
    pixel_per_sample = F.smooth_l1_loss(
        scaled_delta,
        torch.zeros_like(scaled_delta),
        beta=float(loss_config.get("pixel_huber_beta", 0.25)),
        reduction="none",
    ).mean(dim=-1)
    pixel_loss = weighted_mean(pixel_per_sample)
    epsilon = float(loss_config.get("charbonnier_epsilon_px", 1.0)) / pixel_unit
    radial_per_sample = torch.sqrt(
        scaled_delta.square().sum(dim=-1) + epsilon * epsilon
    )
    radial_loss = weighted_mean(radial_per_sample)

    # Wasserstein-like expected distance: probability placed on opposite sides
    # of the screen remains expensive instead of cancelling into a central
    # probability-weighted coordinate.
    candidate_delta = (
        outputs["candidates"] - batch["target"].unsqueeze(1)
    ) * batch["canvas_size"].unsqueeze(1)
    candidate_distance = torch.sqrt(
        (candidate_delta / pixel_unit).square().sum(dim=-1)
        + epsilon * epsilon
    )
    transport_per_sample = (
        outputs["probabilities"] * candidate_distance
    ).sum(dim=-1)
    transport_loss = weighted_mean(transport_per_sample)

    total = (
        float(loss_config.get("heatmap_weight", 1.0)) * heatmap_loss
        + float(loss_config.get("offset_weight", 1.0)) * offset_loss
        + float(loss_config.get("pixel_weight", 0.5)) * pixel_loss
        + float(loss_config.get("radial_weight", 0.25)) * radial_loss
        + float(loss_config.get("transport_weight", 0.0)) * transport_loss
    )

    # VAE-style uncertainty: mu (direct_prediction) is untouched by this -
    # it is still trained exactly as before, purely by the losses above.
    # sigma is trained only by this Gaussian NLL against the same target,
    # so it has to earn a calibrated spread rather than being handed one -
    # too small and a real miss costs the (target-mu)^2/sigma^2 term
    # dearly, too large and the log(sigma) term costs it for needless
    # doubt. Deliberately not wired into mu's own training at all yet -
    # first validate sigma calibrates sensibly on its own (roughly 68% of
    # targets within mu+-sigma, shrinking as more causal EMG/IMU history
    # arrives) before letting it influence anything else.
    # Virtual leader: supervise *every* decimated step's destination estimate
    # against the same click, not just the pooled one. This is the reason the
    # branch exists - it converts one supervision signal per trial into one
    # per timestep on exactly the predicted quantity, which is the thinness
    # every earlier temporal branch in this project ran into.
    #
    # Weighted by the branch's own per-step confidence so it is not forced to
    # claim the destination is already obvious in the first few samples of a
    # reach, when it genuinely is not - but the weights are normalised inside
    # the branch, so this cannot be minimised by simply declaring everything
    # unreliable.
    vl_loss = outputs["heatmap_logits"].new_zeros(())
    if "vl_destinations" in outputs:
        destinations = outputs["vl_destinations"]
        step_delta = (
            destinations - batch["target"].unsqueeze(1)
        ) * batch["canvas_size"].unsqueeze(1) / pixel_unit
        step_distance = torch.sqrt(
            step_delta.square().sum(dim=-1) + epsilon * epsilon
        )
        vl_per_sample = (outputs["vl_weights"] * step_distance).sum(dim=1)
        vl_loss = weighted_mean(vl_per_sample)
        total = total + float(loss_config.get("vl_weight", 0.0)) * vl_loss

    # VAE: KL(q(z|x) || N(0, I)), the price the model pays for every bit it
    # routes through the latent. Reported separately because its magnitude is
    # the diagnostic that matters - driven to ~0 the latent has collapsed to
    # the prior and carries nothing (the decoder would then be predicting a
    # constant), while a large value against poor accuracy means the
    # bottleneck is passing information it cannot use.
    kl_loss = outputs["heatmap_logits"].new_zeros(())
    if "latent_mu" in outputs:
        latent_mu = outputs["latent_mu"]
        latent_log_variance = outputs["latent_log_variance"]
        if "prior_mu" in outputs:
            # KL between two diagonal Gaussians: the posterior against a
            # data-derived prior (the virtual-leader attractor readout)
            # rather than N(0, I). This is where the attractor model actually
            # enters the objective - the belief about the destination is
            # pulled toward what the observed acceleration implies, weighted
            # automatically by how much the per-timestep estimates agree.
            # Floor on the prior's width, not just a numerical guard. The
            # attractor's per-step estimates grow more consistent as
            # loss.vl_weight trains them (measured: prior sigma 0.075 ->
            # 0.031 within 12 steps), and KL scales as 1/sigma_prior^2, so an
            # increasingly confident prior pulls the posterior arbitrarily
            # hard toward a branch that is standalone *worse* than fusion
            # (~250 px against 184 px). The floor caps how much precision the
            # prior may claim.
            prior_floor = float(loss_config.get("vl_prior_sigma_floor", 0.05))
            prior_log_variance = 2.0 * torch.log(
                outputs["prior_sigma"].clamp_min(prior_floor)
            )
            kl_per_sample = 0.5 * (
                prior_log_variance
                - latent_log_variance
                + (
                    latent_log_variance.exp()
                    + (latent_mu - outputs["prior_mu"]).square()
                )
                / prior_log_variance.exp()
                - 1.0
            ).sum(dim=-1)
        else:
            kl_per_sample = 0.5 * (
                latent_mu.square()
                + latent_log_variance.exp()
                - 1.0
                - latent_log_variance
            ).sum(dim=-1)
        kl_loss = weighted_mean(kl_per_sample)
        total = total + float(loss_config.get("kl_weight", 0.0)) * kl_loss

    nll_loss = outputs["heatmap_logits"].new_zeros(())
    if "direct_sigma" in outputs:
        sigma = outputs["direct_sigma"]
        # detach(): without it this term's (target-mu)^2 half would also
        # backprop into mu, contradicting "mu is trained exactly as before" -
        # sigma has to explain mu's existing errors, not reshape mu to suit
        # itself.
        mu = coordinate_prediction.detach()
        nll_per_sample = (
            torch.log(sigma) + 0.5 * ((batch["target"] - mu) / sigma).square()
        ).sum(dim=-1)
        nll_loss = weighted_mean(nll_per_sample)
        total = total + float(loss_config.get("nll_weight", 0.0)) * nll_loss

    # An EMG-first model may lean on the IMU residual instead of learning from
    # EMG. Charging for the gate keeps IMU a last resort and makes the mean
    # gate a reportable measure of how much IMU the task requires.
    # Physics branch: supervise the rollout's own endpoint against the same
    # click. The torque-magnitude charge keeps the branch physically honest -
    # for the Hill model it is the residual torque (the part the Hill terms
    # can't explain), for the plain-torque branch (rollout3) it is the whole
    # commanded torque, since there is no separate physiological term to defer
    # to; either way it discourages torque that only cancels out through
    # clamping/eigenvalue-flooring rather than actually driving sensible
    # motion.
    physics_loss = outputs["heatmap_logits"].new_zeros(())
    residual_loss = outputs["heatmap_logits"].new_zeros(())
    if "physics_prediction" in outputs:
        physics_delta = (
            outputs["physics_prediction"] - batch["target"]
        ) * batch["canvas_size"] / pixel_unit
        physics_loss = weighted_mean(
            torch.sqrt(physics_delta.square().sum(dim=-1) + epsilon * epsilon)
        )
        torque_term = outputs.get("physics_residual_torque", outputs.get("physics_torque"))
        # physics_residual_torque is (batch, joints); physics_torque is
        # (batch, decimated_steps, joints) - flatten every trailing dim so
        # both reduce to one scalar per trial before weighted_mean.
        residual_loss = weighted_mean(torque_term.square().flatten(1).mean(dim=-1))
        total = (
            total
            + float(loss_config.get("physics_weight", 0.1)) * physics_loss
            + float(loss_config.get("physics_residual_weight", 0.01)) * residual_loss
        )

    # A free endpoint-to-screen affine has enough degrees of freedom to fit
    # the target using only whichever input dimensions already vary a lot
    # (the well-measured shoulder axes), leaving no pressure on a poorly
    # varying one (the elbow) even when its trajectory is wrong - confirmed
    # directly on a trained checkpoint (biasing the elbow toward extension
    # made physics_prediction worse only because the affine was co-calibrated
    # for the folded endpoint distribution, not because folding is actually
    # correct).
    #
    # First version of this penalty charged ||W||^2 (overall magnitude) and
    # measurably did nothing at weight 0.05: elbow travel barely moved (0.033
    # -> 0.040 rad) and the bias-sweep verdict didn't change. On reflection
    # that penalises the wrong quantity - a uniform shrink doesn't create any
    # pressure to use the elbow's input column *more* than the shoulder's
    # columns, it just scales all three down together (or does nothing if the
    # elbow's column is already near zero, since there's little there to
    # shrink). What actually needs charging is the *imbalance* between
    # columns, not the overall scale, which the affine legitimately needs to
    # map metre-scale positions into normalised screen coordinates.
    #
    # Charges the coefficient of variation across the three input columns'
    # norms (std/mean of ||W[:,0]||, ||W[:,1]||, ||W[:,2]||) - scale
    # invariant, so it doesn't fight the affine's necessary overall gain,
    # only the gap between its most- and least-used input dimensions. Still
    # not a structural/identity target - it never says which physical axis
    # should carry more weight, only that none of the three should be
    # structurally ignored (an explicit physical target for this exact
    # 3D->2D map was tried once already and converged to a 30cm arm and a 1m
    # screen - arm.py).
    affine_penalty = outputs["heatmap_logits"].new_zeros(())
    if "physics_affine_weight" in outputs:
        column_norm = outputs["physics_affine_weight"].norm(dim=0)
        # 1e-6, not the pixel-space `epsilon` above - this ratio is
        # dimensionless (a coefficient of variation), unrelated units.
        affine_penalty = column_norm.std() / (column_norm.mean() + 1e-6)
        total = total + float(loss_config.get("affine_weight_penalty", 0.0)) * affine_penalty

    gate_penalty = outputs["heatmap_logits"].new_zeros(())
    gate_weight = float(loss_config.get("imu_gate_weight", 0.0))
    if "imu_gate" in outputs:
        gate_penalty = weighted_mean(outputs["imu_gate"].reshape(-1))
        if gate_weight > 0.0:
            total = total + gate_weight * gate_penalty

    return {
        "loss": total,
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "pixel_loss": pixel_loss,
        "radial_loss": radial_loss,
        "transport_loss": transport_loss,
        "imu_gate_penalty": gate_penalty,
        "affine_penalty": affine_penalty,
        "nll_loss": nll_loss,
        "kl_loss": kl_loss,
        "vl_loss": vl_loss,
        "physics_loss": physics_loss,
        "physics_residual_loss": residual_loss,
    }


@torch.no_grad()
def grid_validation_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pixel_error = 0.0
    weighted_pixel_error = 0.0
    endpoint_pixel_error = 0.0
    count = 0
    weight_total = 0.0
    endpoint_count = 0
    for batch in loader:
        views = [batch]
        if continual_enabled(config):
            for cutoff in config.get("continual", {}).get(
                "validation_cutoffs_s", [0.2, 0.6, 1.0]
            ):
                prefix = make_fixed_continual_prefix_batch(
                    batch, float(cutoff), config
                )
                if prefix is not None:
                    views.append(prefix)
        for view in views:
            device_batch = move_batch_to_device(view, device)
            outputs = model(device_batch)
            losses = grid_point_loss(outputs, device_batch, config)
            batch_size = int(device_batch["target"].size(0))
            pixel_error = torch.linalg.vector_norm(
                (outputs["prediction"] - device_batch["target"])
                * device_batch["canvas_size"],
                dim=-1,
            )
            view_weights = device_batch.get(
                "loss_weight", torch.ones_like(pixel_error)
            ).to(pixel_error.dtype)
            view_weight = float(view_weights.sum())
            total_loss += float(losses["loss"]) * view_weight
            total_pixel_error += float(pixel_error.sum())
            weighted_pixel_error += float((pixel_error * view_weights).sum())
            count += batch_size
            weight_total += view_weight
            if "is_endpoint" not in device_batch:
                endpoint_pixel_error += float(pixel_error.sum())
                endpoint_count += batch_size
    if count == 0:
        raise ValueError("Validation loader is empty")
    return {
        "total_loss": total_loss / max(weight_total, 1e-8),
        "mean_pixel_error": total_pixel_error / count,
        "weighted_mean_pixel_error": weighted_pixel_error
        / max(weight_total, 1e-8),
        "endpoint_mean_pixel_error": endpoint_pixel_error
        / max(endpoint_count, 1),
    }


def grid_validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    """Backward-compatible combined validation loss."""

    return grid_validation_scores(model, loader, device, config)["total_loss"]


@torch.no_grad()
def evaluate_grid_model(
    model: torch.nn.Module,
    loader: DataLoader,
    kind: str,
    device: torch.device,
    config: dict[str, Any],
    fold: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    grid_width, grid_height = map(int, config["model"].get("grid_size", [8, 5]))
    metric_batches: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    cell_correct: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        outputs = model(device_batch)
        prediction = outputs["prediction"].detach().cpu()
        target = batch["target"].cpu()
        canvas = batch["canvas_size"].cpu()
        button = batch["button_size"].cpu()
        targets = grid_targets(target, grid_width, grid_height, 0.5)
        predicted_cell = outputs["predicted_cell"].detach().cpu()
        correct = predicted_cell == targets["cell_index"]
        cell_correct.append(correct)
        confidence = outputs["heatmap_confidence"].detach().cpu()
        entropy = outputs["heatmap_entropy"].detach().cpu()
        confidences.append(confidence)
        entropies.append(entropy)
        metric_batches.append(
            {
                "prediction": prediction,
                "target": target,
                "canvas_size": canvas,
                "button_size": button,
            }
        )
        pixel_delta = (prediction - target) * canvas
        pixel_error = torch.linalg.vector_norm(pixel_delta, dim=-1)
        inside = (pixel_delta.abs() <= button / 2.0).all(dim=-1)
        reliability = outputs.get("emg_reliability")
        lookback = outputs.get("emg_lookback_weights")
        grid_prediction = outputs.get("grid_prediction")
        variate_attention = outputs.get("variate_attention")
        cross_variate = outputs.get("cross_variate_attention")
        scale_gate = outputs.get("scale_gate")
        if variate_attention is not None:
            variate_attention = variate_attention.detach().cpu()
        if cross_variate is not None:
            cross_variate = cross_variate.detach().cpu()
        if scale_gate is not None:
            scale_gate = scale_gate.detach().cpu()
        physics_prediction = outputs.get("physics_prediction")
        physics_angles = outputs.get("physics_angles")
        physics_blend = outputs.get("physics_blend")
        fusion_prediction = outputs.get("fusion_prediction")
        participant_scale = outputs.get("physics_participant_scale")
        participant_offset = outputs.get("physics_participant_offset")
        if physics_prediction is not None:
            physics_prediction = physics_prediction.detach().cpu()
            physics_angles = physics_angles.detach().cpu()
            physics_blend = physics_blend.detach().cpu()
            fusion_prediction = fusion_prediction.detach().cpu()
        if participant_scale is not None:
            participant_scale = participant_scale.detach().cpu()
            participant_offset = participant_offset.detach().cpu()
        direct_sigma = outputs.get("direct_sigma")
        if direct_sigma is not None:
            direct_sigma = direct_sigma.detach().cpu()
        vl_sigma = outputs.get("vl_sigma")
        vl_gate = outputs.get("vl_gate")
        vl_prediction = outputs.get("vl_prediction")
        # Guarded independently: the intent-VAE has a virtual-leader prior
        # (so vl_sigma/vl_prediction) but no gate, since the branch enters
        # through the KL rather than a blend.
        if vl_sigma is not None:
            vl_sigma = vl_sigma.detach().cpu()
        if vl_gate is not None:
            vl_gate = vl_gate.detach().cpu()
        if vl_prediction is not None:
            vl_prediction = vl_prediction.detach().cpu()
        latent_mu = outputs.get("latent_mu")
        latent_sigma = outputs.get("latent_sigma")
        if latent_mu is not None:
            latent_mu = latent_mu.detach().cpu()
            latent_sigma = latent_sigma.detach().cpu()
        imu_gate = outputs.get("imu_gate")
        if imu_gate is not None:
            imu_gate = imu_gate.detach().cpu().reshape(-1)
        imu_sensor_attention = outputs.get("imu_sensor_attention")
        imu_channel_attention = outputs.get("imu_channel_attention")
        emg_channel_attention = outputs.get("emg_channel_attention")
        if reliability is not None:
            reliability = reliability.detach().cpu().squeeze(-1)
        if lookback is not None:
            lookback = lookback.detach().cpu()
        if grid_prediction is not None:
            grid_prediction = grid_prediction.detach().cpu()
        if imu_sensor_attention is not None:
            imu_sensor_attention = imu_sensor_attention.detach().cpu()
        if imu_channel_attention is not None:
            imu_channel_attention = imu_channel_attention.detach().cpu()
        if emg_channel_attention is not None:
            emg_channel_attention = emg_channel_attention.detach().cpu()
        requested_cutoffs = batch.get("requested_cutoff")
        prefix_elapsed = batch.get("prefix_elapsed_s")
        prefix_progress = batch.get("prefix_progress")
        for index, trial_id in enumerate(batch["trial_id"]):
            if isinstance(requested_cutoffs, list):
                requested_cutoff = str(requested_cutoffs[index])
            else:
                requested_cutoff = "touch"
            record: dict[str, Any] = {
                "trial_id": trial_id,
                "subject": batch["subject"][index],
                "configuration": batch["configuration"][index],
                "requested_cutoff": requested_cutoff,
                "model_kind": kind,
                "duration_s": float(batch["duration_s"][index]),
                "target_x": float(target[index, 0]),
                "target_y": float(target[index, 1]),
                "prediction_x": float(prediction[index, 0]),
                "prediction_y": float(prediction[index, 1]),
                "target_cell": int(targets["cell_index"][index]),
                "predicted_cell": int(predicted_cell[index]),
                "cell_correct": bool(correct[index]),
                "heatmap_confidence": float(confidence[index]),
                "heatmap_entropy": float(entropy[index]),
                "pixel_error": float(pixel_error[index]),
                "inside_target_box": bool(inside[index]),
            }
            if fold is not None:
                record["fold"] = int(fold)
            if reliability is not None:
                record["emg_reliability"] = float(reliability[index])
            if imu_gate is not None:
                record["imu_gate"] = float(imu_gate[index])
            if direct_sigma is not None:
                record["direct_sigma_x"] = float(direct_sigma[index, 0])
                record["direct_sigma_y"] = float(direct_sigma[index, 1])
            if vl_sigma is not None:
                record["vl_sigma_x"] = float(vl_sigma[index, 0])
                record["vl_sigma_y"] = float(vl_sigma[index, 1])
            if vl_gate is not None:
                record["vl_gate"] = float(vl_gate[index])
            if vl_prediction is not None:
                record["vl_prediction_x"] = float(vl_prediction[index, 0])
                record["vl_prediction_y"] = float(vl_prediction[index, 1])
            if latent_mu is not None:
                # Per-dimension so phase 2 can check whether these track
                # anything pose-like before the kinematic decoder is wired in.
                for dim in range(latent_mu.size(-1)):
                    record[f"latent_mu_{dim}"] = float(latent_mu[index, dim])
                    record[f"latent_sigma_{dim}"] = float(latent_sigma[index, dim])
            if physics_prediction is not None:
                record["physics_prediction_x"] = float(physics_prediction[index, 0])
                record["physics_prediction_y"] = float(physics_prediction[index, 1])
                if physics_angles.size(-1) == 2:
                    record["physics_shoulder_rad"] = float(physics_angles[index, 0])
                    record["physics_elbow_rad"] = float(physics_angles[index, 1])
                else:
                    record["physics_shoulder1_rad"] = float(physics_angles[index, 0])
                    record["physics_shoulder2_rad"] = float(physics_angles[index, 1])
                    record["physics_elbow_rad"] = float(physics_angles[index, 2])
                record["physics_blend"] = float(physics_blend[index])
                record["fusion_prediction_x"] = float(fusion_prediction[index, 0])
                record["fusion_prediction_y"] = float(fusion_prediction[index, 1])
                if participant_scale is not None:
                    record["physics_participant_scale"] = float(participant_scale[index])
                    record["physics_participant_offset_x"] = float(participant_offset[index, 0])
                    record["physics_participant_offset_y"] = float(participant_offset[index, 1])
            if variate_attention is not None:
                for position, name in enumerate(VARIATE_NAMES):
                    record[f"variate_attention_{name}"] = float(
                        variate_attention[index, position]
                    )
            if cross_variate is not None:
                # Mass each variate sends to the other modality: the quantity
                # a scalar reliability gate cannot represent.
                emg_block = cross_variate[index, :4, 4:].sum(dim=-1)
                imu_block = cross_variate[index, 4:, :4].sum(dim=-1)
                record["cross_emg_to_imu"] = float(emg_block.mean())
                record["cross_imu_to_emg"] = float(imu_block.mean())
                for position, name in enumerate(VARIATE_NAMES):
                    for other, other_name in enumerate(VARIATE_NAMES):
                        record[f"cv_{name}_to_{other_name}"] = float(
                            cross_variate[index, position, other]
                        )
            if scale_gate is not None:
                labels = ("emg_p16", "emg_p32", "emg_p64", "imu_p16", "imu_p32", "imu_p64")
                for position in range(min(scale_gate.size(-1), len(labels))):
                    record[f"scale_{labels[position]}"] = float(
                        scale_gate[index, position]
                    )
            if lookback is not None:
                names = (
                    ("full", "500ms", "300ms")
                    if lookback.size(-1) == 3
                    else ("500ms", "300ms")
                )
                for window in range(lookback.size(-1)):
                    label = (
                        names[window] if window < len(names) else str(window)
                    )
                    record[f"emg_weight_{label}"] = float(
                        lookback[index, window]
                    )
            if grid_prediction is not None:
                record["grid_prediction_x"] = float(grid_prediction[index, 0])
                record["grid_prediction_y"] = float(grid_prediction[index, 1])
            if imu_sensor_attention is not None:
                for sensor_index, sensor in enumerate(SENSORS):
                    record[f"attention_imu_sensor_{sensor}"] = float(
                        imu_sensor_attention[index, sensor_index]
                    )
            if imu_channel_attention is not None:
                names = grid_imu_feature_names(config["data"])
                for channel_index, name in enumerate(names):
                    safe_name = (
                        name.lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                    record[f"attention_imu_channel_{channel_index:02d}_{safe_name}"] = float(
                        imu_channel_attention[index, channel_index]
                    )
            if emg_channel_attention is not None:
                # Not SENSORS: derived antagonist channels widen this stack
                # beyond the four electrodes, and naming from a fixed 4-tuple
                # silently drops them from every attention summary.
                names = emg_channel_names(config["data"])
                for channel_index in range(emg_channel_attention.size(-1)):
                    label = (
                        names[channel_index]
                        if channel_index < len(names)
                        else str(channel_index)
                    )
                    record[f"attention_emg_channel_{label}"] = float(
                        emg_channel_attention[index, channel_index]
                    )
            if prefix_elapsed is not None:
                record["prefix_elapsed_s"] = float(prefix_elapsed[index])
            if prefix_progress is not None:
                record["prefix_progress"] = float(prefix_progress[index])
            records.append(record)

    if not metric_batches:
        raise ValueError("Evaluation loader is empty")
    metrics = merge_metric_batches(metric_batches)
    errors = np.asarray([record["pixel_error"] for record in records])
    metrics.update(
        {
            "grid_cell_accuracy": float(torch.cat(cell_correct).float().mean()),
            "accuracy_within_50px": float(np.mean(errors <= 50.0)),
            "accuracy_within_100px": float(np.mean(errors <= 100.0)),
            "mean_heatmap_confidence": float(torch.cat(confidences).mean()),
            "mean_heatmap_entropy": float(torch.cat(entropies).mean()),
        }
    )
    return metrics, records


def grid_data_report(
    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        dataset = loader.dataset
        movement_durations = np.asarray(dataset.movement_durations)
        report[name] = {
            "accepted_trajectories": len(dataset),
            "excluded_trajectories": len(dataset.excluded),
            "minimum_duration_s": min(dataset.durations),
            "maximum_duration_s": max(dataset.durations),
            "minimum_movement_duration_s": float(movement_durations.min()),
            "median_movement_duration_s": float(np.median(movement_durations)),
            "maximum_movement_duration_s": float(movement_durations.max()),
            "eligible_at_200ms": int((movement_durations >= 0.2).sum()),
            "eligible_at_400ms": int((movement_durations >= 0.4).sum()),
            "excluded": dataset.excluded,
        }
    return report
