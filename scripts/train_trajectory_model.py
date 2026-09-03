#!/usr/bin/env python3
"""Train the goal-conditioned trajectory VAE on the tracked dataset.

    python scripts/train_trajectory_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_trajectory.yaml \\
        --device cuda --epochs 20

Reports every prediction against two baselines, because a trajectory error
in metres means nothing on its own:

  hold      the hand stays where it is at the cutoff. Any model that cannot
            beat this has learned nothing at all - and on a task where much
            of a trial is stationary, holding still is a genuinely strong
            baseline that a weak model can lose to.
  linear    the hand continues at its current measured velocity. This is the
            one that matters: it already uses position and velocity, so
            beating it is exactly the claim that EMG plus the attractor adds
            information beyond simple extrapolation. A model that ties with
            linear has learned to extrapolate and nothing more.

Both are computed on identical cutoffs and horizons, so the comparison is
like for like.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import build_tracked_loaders  # noqa: E402
from emg_touch.models.anticipatory_vae import (  # noqa: E402
    AnticipatoryTrajectoryVAE,
    anticipatory_losses,
)
from emg_touch.models.disentangle import reversal_strength  # noqa: E402
from emg_touch.models.trajectory_intent_vae import (  # noqa: E402
    VirtualLeaderTrajectoryVAE,
    trajectory_loss,
)
from emg_touch.utils import (  # noqa: E402
    AverageMeter,
    choose_device,
    save_json,
    seed_everything,
)


class TrajectoryEncoder(torch.nn.Module):
    """Causal encoder with an optional balanced, modality-separated pathway."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        data = config["data"]
        model = config["model"]
        from emg_touch.data.tracked_dataset import (
            emg_feature_count,
            imu_feature_count,
        )

        width = int(model["d_model"])
        emg_channels = emg_feature_count(data)
        # Derived EMG features increase EMG width but do not create more IMU
        # channels; IMU remains six axes for each physical sensor.
        imu_channels = imu_feature_count(data)
        self.separate_modalities = bool(model.get("separate_modality_encoders", False))
        self.imu_dropout = float(model.get("imu_modality_dropout", 0.0))
        if self.separate_modalities:
            self.emg_project = torch.nn.Linear(emg_channels, width)
            self.emg_encoder = torch.nn.GRU(width, width, num_layers=1, batch_first=True)
            self.emg_normalise = torch.nn.LayerNorm(width)
            self.imu_project = torch.nn.Linear(imu_channels, width)
            self.imu_encoder = torch.nn.GRU(width, width, num_layers=1, batch_first=True)
            self.imu_normalise = torch.nn.LayerNorm(width)
            # Each branch gets the same hidden width irrespective of its raw
            # channel count.  The fusion returns the legacy 2*width contract.
            self.fusion = torch.nn.Sequential(
                torch.nn.LayerNorm(4 * width),
                torch.nn.Linear(4 * width, 2 * width),
                torch.nn.GELU(),
            )
            self.emg_only_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(2 * width),
                torch.nn.Linear(2 * width, 2 * width),
                torch.nn.GELU(),
            )
        else:
            # 3 position + 3 velocity: retained for checkpoint-compatible
            # forecast-mode and legacy wearable baselines.
            self.project = torch.nn.Linear(emg_channels + imu_channels + 6, width)
            self.encoder = torch.nn.GRU(width, width, num_layers=1, batch_first=True)
            self.normalise = torch.nn.LayerNorm(width)
        self.context_dim = 2 * width

    @staticmethod
    def _pool(hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden[:, -1], hidden.mean(dim=1)], dim=-1)

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        return_modalities: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.separate_modalities:
            emg_hidden, _ = self.emg_encoder(self.emg_project(emg))
            imu_hidden, _ = self.imu_encoder(self.imu_project(imu))
            emg_context = self._pool(self.emg_normalise(emg_hidden))
            imu_context = self._pool(self.imu_normalise(imu_hidden))
            if self.training and self.imu_dropout > 0.0:
                keep = (
                    torch.rand(imu_context.size(0), 1, device=imu_context.device)
                    >= self.imu_dropout
                ).to(imu_context.dtype)
                imu_context = imu_context * keep
            context = self.fusion(torch.cat([emg_context, imu_context], dim=-1))
            emg_only = self.emg_only_projection(emg_context)
            return (context, emg_only) if return_modalities else context
        features = torch.cat([emg, imu, position, velocity], dim=-1)
        hidden, _ = self.encoder(self.project(features))
        hidden = self.normalise(hidden)
        # Last state plus a mean over the prefix: the last state carries the
        # instant the prediction is made from, the mean carries the whole
        # approach.
        context = self._pool(hidden)
        return (context, context) if return_modalities else context


class TrajectoryModel(torch.nn.Module):
    def __init__(self, config: dict, kind: str = "trajectory",
                 task: str = "forecast") -> None:
        super().__init__()
        self.kind = kind
        # forecast: the tracker's own history is an input, and the model
        #   extrapolates it. Legitimate for latency compensation in a system
        #   that already has a tracker, but it presupposes the very sensor an
        #   EMG interface exists to remove.
        # wearable: EMG and IMU only. The tracker is the label and never
        #   reaches the encoder. This is the setting the project is actually
        #   about, and it is strictly harder - with no measured velocity the
        #   initial motion state has to be inferred from the muscle signal
        #   rather than read off.
        self.task = task
        self.encoder = TrajectoryEncoder(config)
        settings = dict(config.get("virtual_leader", {}))
        settings["context_dim"] = self.encoder.context_dim
        merged = {**config, "virtual_leader": settings}
        self.vae = (
            AnticipatoryTrajectoryVAE(merged)
            if kind == "anticipatory"
            else VirtualLeaderTrajectoryVAE(merged)
        )
        # Without a tracker there is no measured velocity to start the
        # attractor rollout from, so it is predicted from the wearables. Small
        # random init rather than zero: a zero final weight would make the
        # head's gradient with respect to the context exactly zero and sever
        # it silently, which has already happened twice in this project.
        self.initial_velocity = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder.context_dim),
            torch.nn.Linear(self.encoder.context_dim, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 3),
        )
        torch.nn.init.normal_(self.initial_velocity[-1].weight, std=0.01)
        torch.nn.init.zeros_(self.initial_velocity[-1].bias)

    def forward(
        self,
        window: dict[str, torch.Tensor],
        horizon: int,
        strength: float = 1.0,
        measure_anticipatory: bool = False,
        include_emg_only: bool = False,
    ) -> dict:
        context, emg_context = self.encoder(
            window["emg"], window["imu"], window["position"], window["velocity"],
            return_modalities=True,
        )
        outputs = self._decode(
            context, window, horizon, strength, measure_anticipatory
        )
        if include_emg_only and self.encoder.separate_modalities:
            outputs["emg_only_outputs"] = self._decode(
                emg_context, window, horizon, strength, False
            )
        return outputs

    def _decode(
        self,
        context: torch.Tensor,
        window: dict[str, torch.Tensor],
        horizon: int,
        strength: float,
        measure_anticipatory: bool,
    ) -> dict:
        if self.task == "wearable":
            # Origin, not the tracked position: the model predicts
            # displacement, so its own frame starts at zero. Velocity and
            # acceleration come from the encoder rather than the tracker.
            start = torch.zeros_like(window["position"][:, -1])
            velocity = self.initial_velocity(context)
            acceleration = torch.zeros_like(velocity)
            arguments = (context, start, velocity, acceleration)
        else:
            arguments = (
                context,
                window["position"][:, -1],
                window["velocity"][:, -1],
                window["acceleration"],
            )
        if self.kind == "anticipatory":
            return self.vae(
                *arguments,
                horizon=horizon,
                strength=strength,
                measure_anticipatory=measure_anticipatory,
            )
        return self.vae(*arguments, horizon=horizon)


def make_window(
    batch: dict[str, torch.Tensor], horizon: int, minimum_prefix: int, generator,
    dt: float = 1.0, ablate: tuple[str, ...] = (), relative: bool = False,
    cutoff_offsets: tuple[int, ...] = (),
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Cut each trial randomly or at configured offsets from movement onset.

    Cutoffs are drawn past the onset because a stationary hand has no
    destination-directed acceleration for the attractor to read - predicting
    from the pre-buffer would be asking the model a question the physics
    cannot answer.
    """
    lengths = batch["lengths"]
    onsets = batch["onset"]
    prefixes, futures, offsets = [], [], []
    for row in range(len(lengths)):
        length = int(lengths[row])
        onset = int(onsets[row])
        if cutoff_offsets:
            # History length and time relative to movement onset are separate
            # concepts.  A negative offset is legal when the pre-buffer is
            # long enough, which finally tests the anticipatory EMG interval.
            offset = int(generator.choice(cutoff_offsets))
            cut = onset + offset
            if cut < minimum_prefix or cut + horizon > length:
                continue
        else:
            start = max(onset + minimum_prefix, minimum_prefix)
            latest = length - horizon
            if latest <= start:
                continue
            cut = int(generator.integers(start, latest))
            offset = cut - onset
        prefixes.append((row, cut))
        futures.append(row)
        offsets.append(offset)
    if not prefixes:
        return None

    # A fixed window, not the batch's smallest cut. Using the minimum threw
    # away every other trial's history down to whatever the unluckiest cut in
    # the batch happened to be (measured: 24 samples, ~0.19 s), which starves
    # the encoder of exactly the approach it is supposed to read intent from.
    prefix_length = min(minimum_prefix * 4, min(cut for _, cut in prefixes))
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu", "position", "velocity"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut in prefixes]
        )
    future = torch.stack(
        [batch["position"][row, cut : cut + horizon] for row, cut in prefixes]
    )
    future_mask = torch.stack(
        [batch["position_mask"][row, cut : cut + horizon] for row, cut in prefixes]
    )
    # Backward difference of the measured velocity: velocity is an
    # independent tracker estimate (verified, ~3e-01 relative to the
    # derivative of position), so only one differencing pass is needed.
    # Divided by dt: a bare velocity difference is delta-v per SAMPLE, not
    # per second, which at ~126 Hz is 126x too small. The attractor prior is
    # r = x + (xddot + rho xdot)/eta, so an acceleration that small makes the
    # acceleration term vanish next to the drag term - the prior silently
    # stops being an attractor readout at all and becomes a velocity
    # extrapolation, which is exactly the baseline it is supposed to beat.
    # Ablation: zero an input group everywhere it enters the encoder. The
    # model beating the linear baseline is only evidence about EMG if EMG is
    # what carries it - the encoder also sees the whole kinematic prefix
    # while the baseline sees only the last sample, so trajectory shape alone
    # could produce the same margin. Zeroing rather than removing keeps the
    # architecture and parameter count identical, so the comparison isolates
    # the information and not the model size.
    # The tracker is the LABEL. In wearable mode it must not reach the
    # encoder at all, but the baselines still need it to be meaningful, so
    # the unablated copy is kept separately and never shown to the model.
    window["_true_position"] = window["position"].clone()
    window["_true_velocity"] = window["velocity"].clone()
    if relative:
        # Predict displacement from the cutoff, not absolute world position.
        # Without a tracker the hand's absolute location is simply unknown,
        # so asking for it is ill-posed; where it is going NEXT is not.
        origin = window["_true_position"][:, -1:].clone()
        future = future - origin

    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])

    # How far into the movement this cutoff sits, in samples past onset.
    # Electromechanical delay means EMG leads motion by 40-80 ms, so whatever
    # EMG contributes is concentrated in the first samples after onset - and a
    # mean over uniformly sampled cutoffs, most of which land mid-reach where
    # kinematics dominate, would average that away to nothing.
    window["samples_past_onset"] = torch.tensor(
        offsets, dtype=torch.float32, device=batch["position"].device
    )
    # Original batch-row index for each surviving cutoff, in the same order
    # as every other window[...] tensor. Needed to re-index any per-trial
    # field read from the UNFILTERED batch (e.g. batch["session"]) after
    # make_window has dropped rows without a valid cutoff - passing such a
    # field through unindexed silently assumes nothing was dropped, which a
    # long horizon makes false on almost every batch (see the crash this
    # fixes: 16-row batch, one trial too short for a 1000 ms cutoff, 15-row
    # window, cross_entropy given a 16-row target against a 15-row input).
    window["_rows"] = torch.tensor(
        [row for row, _ in prefixes], dtype=torch.long, device=batch["position"].device
    )

    velocity = window["_true_velocity"] if relative else window["velocity"]
    window["acceleration"] = (
        (velocity[:, -1] - velocity[:, -2]) / dt
        if velocity.size(1) > 1
        else torch.zeros_like(velocity[:, -1])
    )
    return window, future, future_mask


def baselines(
    window: dict, future: torch.Tensor, dt: float, relative: bool = False,
    mean_profile: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Reference predictions.

    In wearable mode the baselines are computed from the tracker copy the
    model never sees, and they mean something different. `linear` needs a
    measured velocity, which a wearable-only system does not have - it is
    reported as an upper reference showing what a tracker would buy, NOT as
    a baseline the model is expected to beat.

    `mean_profile` is the baseline that matters there instead: the average
    displacement over the training set. Reaching is stereotyped, so a model
    can score well by reproducing the population's average reach with no
    per-trial inference at all. Beating the mean profile is what separates
    "inferred where THIS hand is going" from "reaches look alike".
    """
    horizon = future.size(1)
    position = window["_true_position"][:, -1]
    velocity = window["_true_velocity"][:, -1]
    steps = torch.arange(1, horizon + 1, device=future.device, dtype=future.dtype)
    origin = torch.zeros_like(position) if relative else position
    hold = origin.unsqueeze(1).expand(-1, horizon, -1)
    linear = origin.unsqueeze(1) + velocity.unsqueeze(1) * steps.view(1, -1, 1) * dt
    result = {"hold": hold, "linear": linear}
    if mean_profile is not None:
        result["mean_reach"] = mean_profile.unsqueeze(0).expand(
            future.size(0), -1, -1
        ).to(future.device)
    return result


def training_mean_profile(
    loader, horizon: int, minimum_prefix: int, dt: float, batches: int | None = None,
    cutoff_offsets: tuple[int, ...] = (),
) -> torch.Tensor:
    """Average displacement trajectory over the training set."""
    generator = np.random.default_rng(0)
    collected = []
    for index, batch in enumerate(loader):
        if batches is not None and index >= batches:
            break
        made = make_window(
            batch, horizon, minimum_prefix, generator, dt, relative=True,
            cutoff_offsets=cutoff_offsets,
        )
        if made is None:
            continue
        _, future, mask = made
        weights = mask.to(future.dtype).unsqueeze(-1)
        collected.append((future * weights).sum(0) / weights.sum(0).clamp_min(1.0))
    if not collected:
        return torch.zeros(horizon, 3)
    return torch.stack(collected).mean(0)


def displacement_error(predicted: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> tuple[float, float]:
    """Mean and final displacement error in metres, over valid steps.

    mask is (batch, horizon) - the collate emits one validity flag per
    timestep, not per coordinate axis.
    """
    weights = mask.to(predicted.dtype)
    distance = (predicted - target).norm(dim=-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    mean = ((distance * weights).sum(dim=1) / denominator).mean()
    final = distance[:, -1].mean()
    return float(mean), float(final)


def evaluate(model, loader, config, device, horizon, minimum_prefix, dt,
             ablate: tuple[str, ...] = (), relative: bool = False,
             mean_profile: torch.Tensor | dict[int, torch.Tensor] | None = None,
             cutoff_offsets: tuple[int, ...] = ()) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)  # fixed cutoffs, so runs compare
    paired_interventions = bool(
        config.get("evaluation", {}).get("paired_modality_interventions", False)
    )
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            # Evaluation visits every requested offset for every eligible
            # trial. Training randomly chooses among them, but evaluating one
            # random offset per trial would confound time with trial identity.
            evaluation_offsets = tuple((offset,) for offset in cutoff_offsets) or ((),)
            for selected_offsets in evaluation_offsets:
                made = make_window(
                    batch, horizon, minimum_prefix, generator, dt, ablate, relative,
                    selected_offsets,
                )
                if made is None:
                    continue
                window, future, future_mask = made
                anticipatory = getattr(model, "kind", "") == "anticipatory"
                outputs = model(
                    window, horizon, measure_anticipatory=anticipatory,
                    include_emg_only=True,
                )
                selected_profile = mean_profile
                if isinstance(mean_profile, dict):
                    selected_profile = mean_profile[selected_offsets[0]]
                predictions = {
                    "model": outputs["trajectory"],
                    **baselines(window, future, dt, relative, selected_profile),
                }
                if "trajectory_without_anticipatory" in outputs:
                    predictions["kinematic_only_latent"] = outputs[
                        "trajectory_without_anticipatory"
                    ]
                if "emg_only_outputs" in outputs:
                    predictions["emg_only"] = outputs["emg_only_outputs"]["trajectory"]

                if paired_interventions:
                    interventions = {
                        "without_emg": ("emg", torch.zeros_like(window["emg"])),
                        "without_imu": ("imu", torch.zeros_like(window["imu"])),
                    }
                    if window["emg"].size(0) > 1:
                        interventions["shuffled_emg"] = (
                            "emg", torch.roll(window["emg"], shifts=1, dims=0)
                        )
                    for label, (key, replacement) in interventions.items():
                        changed = {**window, key: replacement}
                        predictions[label] = model(changed, horizon)["trajectory"]

                past_onset = window.get("samples_past_onset")
                for name, prediction in predictions.items():
                    mean, final = displacement_error(prediction, future, future_mask)
                    totals.setdefault(f"{name}_mean_m", []).append(mean)
                    totals.setdefault(f"{name}_final_m", []).append(final)
                    if past_onset is not None:
                        per_sample = (
                            (prediction[:, : future.size(1)] - future).norm(dim=-1)
                            * future_mask
                        ).sum(dim=1) / future_mask.sum(dim=1).clamp_min(1.0)
                        for label, low, high in (
                            ("early", -1e9, 30.0),
                            ("mid", 30.0, 90.0),
                            ("late", 90.0, 1e9),
                        ):
                            chosen = (past_onset >= low) & (past_onset < high)
                            if chosen.any():
                                totals.setdefault(f"{name}_{label}_m", []).append(
                                    float(per_sample[chosen].mean())
                                )
                        for offset in past_onset.unique():
                            chosen = past_onset == offset
                            totals.setdefault(
                                f"{name}_offset_{int(offset.item()):+d}_m", []
                            ).append(float(per_sample[chosen].mean()))
    return {k: float(np.mean(v)) for k, v in totals.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/tracked_trajectory")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--horizon-ms", type=float,
        help="Override virtual_leader.horizon (default 254 ms / 32 samples). "
        "The tracked dataset has no post-touch buffer - touch is defined as "
        "the trial's last recorded sample - so a horizon this long needs "
        "cutoffs sitting that far before the RECORDING ends, not just "
        "before the hand stops. Run diagnose_horizon_feasibility.py first "
        "for anything beyond ~500 ms to see what fraction of trials can "
        "even produce a valid cutoff at the requested horizon.",
    )
    parser.add_argument(
        "--task",
        choices=("forecast", "wearable"),
        default="forecast",
        help="wearable: EMG+IMU only, tracker is the label and never an "
        "input, and the target is displacement from the cutoff. This is the "
        "setting an EMG interface actually operates in. forecast: the "
        "tracker's own history is an input and the model extrapolates it.",
    )
    parser.add_argument(
        "--cutoff-offset-ms", type=float, nargs="+",
        help="Prediction cutoff(s) relative to tracker-defined movement onset. "
        "Example: --cutoff-offset-ms -100 -50 0 50 100. History length is "
        "controlled independently by virtual_leader.minimum_prefix.",
    )
    parser.add_argument(
        "--separate-modalities", action="store_true",
        help="Use equal-width EMG and IMU encoders before fusion.",
    )
    parser.add_argument("--emg-only-weight", type=float)
    parser.add_argument("--imu-dropout", type=float)
    parser.add_argument("--emg-feature-windows-ms", type=float, nargs="+")
    parser.add_argument(
        "--emg-feature-kinds", nargs="+",
        choices=("rms", "waveform_length", "log_energy", "derivative"),
    )
    parser.add_argument(
        "--paired-modality-interventions", action="store_true",
        help="At validation/test time, score the same checkpoint with EMG "
        "zeroed, EMG shuffled between trials, and IMU zeroed.",
    )
    parser.add_argument(
        "--model",
        choices=("trajectory", "anticipatory"),
        default="trajectory",
        help="anticipatory adds the kinematic/anticipatory latent split and "
        "reports how much of the prediction survives with the anticipatory "
        "subspace silenced - EMG's unique contribution, in metres.",
    )
    parser.add_argument(
        "--ablate",
        default="",
        help="Comma-separated input groups to zero out: emg, imu, position, "
        "velocity. Use --ablate emg,imu to leave only the tracked kinematics, "
        "which tests whether the margin over the linear baseline comes from "
        "the muscle signal or merely from seeing the trajectory's history.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.separate_modalities:
        config["model"]["separate_modality_encoders"] = True
    if args.emg_only_weight is not None:
        config["loss"]["emg_only_weight"] = args.emg_only_weight
    if args.imu_dropout is not None:
        config["model"]["imu_modality_dropout"] = args.imu_dropout
    if args.emg_feature_windows_ms:
        config["data"]["emg_feature_windows_ms"] = args.emg_feature_windows_ms
    if args.emg_feature_kinds:
        config["data"]["emg_feature_kinds"] = args.emg_feature_kinds
    if args.paired_modality_interventions:
        config.setdefault("evaluation", {})["paired_modality_interventions"] = True
    if not 0.0 <= float(config["model"].get("imu_modality_dropout", 0.0)) < 1.0:
        parser.error("IMU modality dropout must be in [0, 1)")
    if float(config.get("loss", {}).get("emg_only_weight", 0.0)) < 0.0:
        parser.error("EMG-only loss weight must be non-negative")
    if bool(config.get("loss", {}).get("emg_only_weight", 0.0)) and not bool(
        config["model"].get("separate_modality_encoders", False)
    ):
        parser.error("EMG-only loss requires --separate-modalities")
    if args.task != "wearable" and bool(
        config["model"].get("separate_modality_encoders", False)
    ):
        parser.error("separate modality encoders currently require --task wearable")
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    seed_everything(int(config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, args.cache_dir
    )
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials  (split_by="
        f"{config['data'].get('split_by', 'session')})"
    )

    model = TrajectoryModel(config, args.model, args.task).to(device)
    ablate = tuple(x.strip() for x in args.ablate.split(",") if x.strip())
    relative = args.task == "wearable"
    global_step = 0
    ramp_steps = max(1, len(train_loader) * int(config["training"]["epochs"]) // 3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if relative:
        # Enforced rather than trusted to the caller: the whole point of this
        # mode is that the tracker cannot reach the encoder.
        ablate = tuple(sorted(set(ablate) | {"position", "velocity"}))
        print("TASK: wearable - EMG+IMU only, tracker is label-only, "
              "target is displacement from the cutoff")
    if ablate:
        print(f"ABLATION: zeroing {list(ablate)}")
    minimum_prefix = int(config["virtual_leader"].get("minimum_prefix", 16))
    rate = float(config["data"]["sample_rate_hz"]) / int(config["data"]["decimation"])
    dt = 1.0 / rate
    if args.horizon_ms:
        horizon = max(1, int(round(args.horizon_ms * rate / 1000.0)))
        print(f"horizon overridden to {args.horizon_ms:.0f} ms ({horizon} samples "
              f"at {rate:.1f} Hz) - was {int(config['virtual_leader']['horizon'])} "
              f"samples ({int(config['virtual_leader']['horizon']) / rate * 1000:.0f} ms)")
    else:
        horizon = int(config["virtual_leader"]["horizon"])
    cutoff_offsets = tuple(dict.fromkeys(
        int(round(value / 1000.0 / dt)) for value in (args.cutoff_offset_ms or [])
    ))
    if cutoff_offsets:
        print("CUTOFF OFFSETS: " + ", ".join(
            f"{offset * dt * 1000:+.1f} ms" for offset in cutoff_offsets
        ))
    # Needs horizon and dt, so computed once both exist. Only meaningful in
    # wearable mode, where beating the average reach is the real test.
    mean_profile = None
    if relative:
        mean_profile = (
            {
                offset: training_mean_profile(
                    train_loader, horizon, minimum_prefix, dt,
                    cutoff_offsets=(offset,),
                )
                for offset in cutoff_offsets
            }
            if cutoff_offsets
            else training_mean_profile(train_loader, horizon, minimum_prefix, dt)
        )
    clip = float(config["training"].get("gradient_clip_norm", 1.0))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    generator = np.random.default_rng(int(config.get("seed", 42)))

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        names = ["loss", "reconstruction", "kl"]
        emg_only_weight = float(config.get("loss", {}).get("emg_only_weight", 0.0))
        if emg_only_weight > 0.0:
            names += ["emg_only_reconstruction", "emg_only_kl"]
        if args.model == "anticipatory":
            names += ["kinematic_predict", "kinematic_adversarial"]
        meters = {k: AverageMeter() for k in names}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            made = make_window(
                batch, horizon, minimum_prefix, generator, dt, ablate, relative,
                cutoff_offsets,
            )
            if made is None:
                continue
            window, future, future_mask = made
            strength = reversal_strength(global_step, ramp_steps)
            outputs = model(
                window, horizon, strength=strength,
                include_emg_only=emg_only_weight > 0.0,
            )
            # future_mask is already (batch, horizon); trajectory_loss wants
            # exactly that, so no reduction over a coordinate axis here.
            losses = trajectory_loss(outputs, future, future_mask, config)
            if emg_only_weight > 0.0:
                if "emg_only_outputs" not in outputs:
                    raise ValueError(
                        "loss.emg_only_weight requires model.separate_modality_encoders"
                    )
                emg_losses = trajectory_loss(
                    outputs["emg_only_outputs"], future, future_mask, config
                )
                losses["loss"] = losses["loss"] + emg_only_weight * emg_losses["loss"]
                losses["emg_only_reconstruction"] = emg_losses["reconstruction"]
                losses["emg_only_kl"] = emg_losses["kl"]
            if args.model == "anticipatory":
                # The TRUE velocity, not the encoder's copy. In wearable
                # mode the encoder's velocity is zeroed, so passing it here
                # trained z_kin to reconstruct zero - the split collapsed and
                # the anticipatory gain read 0.02 cm because the partition was
                # measuring nothing, not because EMG carried nothing. Using
                # the tracker value as a supervision TARGET is legitimate: it
                # is ground truth, exactly like the trajectory, and never
                # reaches the model as an input.
                session = batch.get("session")
                if session is not None:
                    # Re-index to the rows make_window actually kept - see
                    # window["_rows"]'s own comment for why this is required
                    # rather than optional whenever a horizon is long enough
                    # to drop any trial from the batch.
                    session = session.index_select(0, window["_rows"])
                extra = anticipatory_losses(
                    outputs,
                    window["_true_velocity"][:, -1],
                    window["acceleration"],
                    config,
                    session=session,
                )
                losses = {**losses, **extra}
                losses["loss"] = losses["loss"] + extra["disentangle"]
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            for name, meter in meters.items():
                if name in losses:
                    meter.update(float(losses[name].detach()), int(future.size(0)))

        scores = evaluate(
            model, validation_loader, config, device, horizon, minimum_prefix, dt,
            ablate, relative, mean_profile, cutoff_offsets,
        )
        record = {"epoch": epoch, **{k: m.average for k, m in meters.items()}, **scores}
        history.append(record)
        print(
            f"epoch={epoch} loss={meters['loss'].average:.4f} "
            f"recon={meters['reconstruction'].average:.4f} "
            f"kl={meters['kl'].average:.3f} | val model={scores.get('model_mean_m', float('nan')):.4f} m "
            f"hold={scores.get('hold_mean_m', float('nan')):.4f} "
            f"linear={scores.get('linear_mean_m', float('nan')):.4f}"
        )
        if scores.get("model_mean_m", float("inf")) < best:
            best = scores["model_mean_m"]
            torch.save({"model_state": model.state_dict(), "config": config},
                       output / "best.pt")

    # Evaluate the BEST checkpoint, not whatever the last epoch left behind.
    # On the first real run the best validation score was epoch 1 and the
    # model degraded afterwards, so testing the final weights reported a
    # model that had already been damaged by the KL runaway.
    best_path = output / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        print(f"loaded best checkpoint (val {best:.4f} m) for test evaluation")
    test_scores = evaluate(
        model, test_loader, config, device, horizon, minimum_prefix, dt, ablate,
        relative, mean_profile, cutoff_offsets,
    )
    print()
    print("=== test ===")
    print("\n  by how early the cutoff was (samples past movement onset):")
    print(f"    {'':22}{'early <30':>12}{'mid 30-90':>12}{'late >90':>12}")
    for name in (
        "model", "emg_only", "kinematic_only_latent", "hold", "linear", "mean_reach"
    ):
        cells = []
        for label in ("early", "mid", "late"):
            value = test_scores.get(f"{name}_{label}_m")
            cells.append(f"{value * 100:9.2f} cm" if value else "         -")
        if any(c.strip() != "-" for c in cells):
            print(f"    {name:22}" + "".join(f"{c:>12}" for c in cells))
    print()
    if cutoff_offsets:
        print("\n  by exact cutoff relative to movement onset:")
        for name in (
            "model", "emg_only", "without_emg", "shuffled_emg", "without_imu",
            "hold", "mean_reach",
        ):
            cells = []
            for offset in cutoff_offsets:
                value = test_scores.get(f"{name}_offset_{offset:+d}_m")
                cells.append(
                    f"{offset * dt * 1000:+.0f} ms={value * 100:.2f} cm"
                    if value is not None else f"{offset * dt * 1000:+.0f} ms=-"
                )
            if any(test_scores.get(f"{name}_offset_{o:+d}_m") is not None
                   for o in cutoff_offsets):
                print(f"    {name:12} " + " | ".join(cells))
    print()
    for name in (
        "model", "emg_only", "without_emg", "shuffled_emg", "without_imu",
        "kinematic_only_latent", "hold", "linear", "mean_reach"
    ):
        mean = test_scores.get(f"{name}_mean_m")
        final = test_scores.get(f"{name}_final_m")
        if mean is not None:
            print(f"  {name:7} mean {mean * 100:6.2f} cm | final {final * 100:6.2f} cm")
    if test_scores.get("without_emg_mean_m") is not None:
        full = test_scores["model_mean_m"]
        print("\n  paired same-checkpoint modality effects (positive = modality helps):")
        print(f"    EMG removal:  "
              f"{(test_scores['without_emg_mean_m'] - full) * 100:+.2f} cm")
        if test_scores.get("shuffled_emg_mean_m") is not None:
            print(f"    EMG shuffle:  "
                  f"{(test_scores['shuffled_emg_mean_m'] - full) * 100:+.2f} cm")
        print(f"    IMU removal:  "
              f"{(test_scores['without_imu_mean_m'] - full) * 100:+.2f} cm")
    muted = test_scores.get("kinematic_only_latent_mean_m")
    if muted is not None and test_scores.get("model_mean_m"):
        gain = muted - test_scores["model_mean_m"]
        print(f"\n  anticipatory subspace is worth {gain * 100:.2f} cm "
              f"({gain / muted * 100:.1f}% of the muted error)")
        print("  that is the part of the prediction the current kinematics could")
        print("  not have produced - EMG's unique contribution, measured directly")
        print("  rather than inferred from a separate ablation run")
    if relative:
        # In wearable mode `linear` needs a measured velocity the system does
        # not have, so it is a reference for what a tracker would buy, not a
        # bar to clear. The honest bar is the average reach: beating it is
        # what distinguishes inferring where THIS hand is going from
        # reproducing the fact that reaches look alike.
        model_mean = test_scores.get("model_mean_m")
        reference = test_scores.get("mean_reach_mean_m")
        print("\n  wearable mode: the tracker is the label, so `linear` is not a")
        print("  baseline the model can be asked to beat - it uses the very sensor")
        print("  being replaced. The comparison that matters is the average reach.")
        if model_mean and reference:
            gain = (reference - model_mean) / reference * 100
            if gain > 0:
                print(f"\n  model is {gain:.1f}% BETTER than the average reach profile")
                print("  -> EMG+IMU carry per-trial information about where this")
                print("     particular reach is going")
            else:
                print(f"\n  model is {-gain:.1f}% WORSE than the average reach profile")
                print("  -> no per-trial information recovered from EMG+IMU; the")
                print("     model has learned the population's average reach")
        save_json({"history": history, "test": test_scores}, output / "results.json")
        return
    model_mean = test_scores.get("model_mean_m")
    linear_mean = test_scores.get("linear_mean_m")
    if model_mean and linear_mean:
        change = (linear_mean - model_mean) / linear_mean * 100.0
        verdict = "BETTER than" if change > 0 else "WORSE than"
        print(f"\n  model is {abs(change):.1f}% {verdict} constant-velocity extrapolation")
        print("  (that is the comparison that matters: linear already uses position")
        print("   and velocity, so beating it is the claim that EMG plus the")
        print("   attractor adds something extrapolation does not)")
    save_json({"history": history, "test": test_scores}, output / "results.json")


if __name__ == "__main__":
    main()
