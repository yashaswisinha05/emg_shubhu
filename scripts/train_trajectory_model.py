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
    """Causal encoder over EMG + IMU + tracked kinematics -> pooled context."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        data = config["data"]
        model = config["model"]
        from emg_touch.data.grid_trajectory import (
            emg_channel_count,
            grid_imu_feature_dim,
        )

        width = int(model["d_model"])
        emg_channels = len(data.get("sensors", ["S0", "S4", "S8", "S12"]))
        imu_channels = 6 * emg_channels  # raw ACC/GYRO per sensor
        # 3 position + 3 velocity: the kinematics the attractor is written in.
        self.project = torch.nn.Linear(emg_channels + imu_channels + 6, width)
        self.encoder = torch.nn.GRU(width, width, num_layers=1, batch_first=True)
        self.normalise = torch.nn.LayerNorm(width)
        self.context_dim = 2 * width

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([emg, imu, position, velocity], dim=-1)
        hidden, _ = self.encoder(self.project(features))
        hidden = self.normalise(hidden)
        # Last state plus a mean over the prefix: the last state carries the
        # instant the prediction is made from, the mean carries the whole
        # approach.
        return torch.cat([hidden[:, -1], hidden.mean(dim=1)], dim=-1)


class TrajectoryModel(torch.nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(config)
        settings = dict(config.get("virtual_leader", {}))
        settings["context_dim"] = self.encoder.context_dim
        self.vae = VirtualLeaderTrajectoryVAE({**config, "virtual_leader": settings})

    def forward(self, window: dict[str, torch.Tensor], horizon: int) -> dict:
        context = self.encoder(
            window["emg"], window["imu"], window["position"], window["velocity"]
        )
        return self.vae(
            context,
            window["position"][:, -1],
            window["velocity"][:, -1],
            window["acceleration"],
            horizon=horizon,
        )


def make_window(
    batch: dict[str, torch.Tensor], horizon: int, minimum_prefix: int, generator,
    dt: float = 1.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Cut each trial at a random point after movement onset.

    Cutoffs are drawn past the onset because a stationary hand has no
    destination-directed acceleration for the attractor to read - predicting
    from the pre-buffer would be asking the model a question the physics
    cannot answer.
    """
    lengths = batch["lengths"]
    onsets = batch["onset"]
    prefixes, futures = [], []
    for row in range(len(lengths)):
        length = int(lengths[row])
        start = max(int(onsets[row]) + minimum_prefix, minimum_prefix)
        latest = length - horizon
        if latest <= start:
            continue
        cut = int(generator.integers(start, latest))
        prefixes.append((row, cut))
        futures.append(row)
    if not prefixes:
        return None

    # A fixed window, not the batch's smallest cut. Using the minimum threw
    # away every other trial's history down to whatever the unluckiest cut in
    # the batch happened to be (measured: 24 samples, ~0.19 s), which starves
    # the encoder of exactly the approach it is supposed to read intent from.
    prefix_length = min(minimum_prefix * 4, min(cut for _, cut in prefixes))
    rows = torch.tensor([row for row, _ in prefixes], dtype=torch.long)
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
    velocity = window["velocity"]
    window["acceleration"] = (
        (velocity[:, -1] - velocity[:, -2]) / dt
        if velocity.size(1) > 1
        else torch.zeros_like(velocity[:, -1])
    )
    return window, future, future_mask


def baselines(window: dict, future: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
    horizon = future.size(1)
    position = window["position"][:, -1]
    velocity = window["velocity"][:, -1]
    steps = torch.arange(1, horizon + 1, device=future.device, dtype=future.dtype)
    hold = position.unsqueeze(1).expand(-1, horizon, -1)
    linear = position.unsqueeze(1) + velocity.unsqueeze(1) * steps.view(1, -1, 1) * dt
    return {"hold": hold, "linear": linear}


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


def evaluate(model, loader, config, device, horizon, minimum_prefix, dt) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)  # fixed cutoffs, so runs compare
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            made = make_window(batch, horizon, minimum_prefix, generator, dt)
            if made is None:
                continue
            window, future, future_mask = made
            outputs = model(window, horizon)
            for name, prediction in {
                "model": outputs["trajectory"],
                **baselines(window, future, dt),
            }.items():
                mean, final = displacement_error(prediction, future, future_mask)
                totals.setdefault(f"{name}_mean_m", []).append(mean)
                totals.setdefault(f"{name}_final_m", []).append(final)
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
    args = parser.parse_args()

    config = load_config(args.config)
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

    model = TrajectoryModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    horizon = int(config["virtual_leader"]["horizon"])
    minimum_prefix = int(config["virtual_leader"].get("minimum_prefix", 16))
    rate = float(config["data"]["sample_rate_hz"]) / int(config["data"]["decimation"])
    dt = 1.0 / rate
    clip = float(config["training"].get("gradient_clip_norm", 1.0))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    generator = np.random.default_rng(int(config.get("seed", 42)))

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        meters = {k: AverageMeter() for k in ("loss", "reconstruction", "kl")}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            made = make_window(batch, horizon, minimum_prefix, generator, dt)
            if made is None:
                continue
            window, future, future_mask = made
            outputs = model(window, horizon)
            # future_mask is already (batch, horizon); trajectory_loss wants
            # exactly that, so no reduction over a coordinate axis here.
            losses = trajectory_loss(outputs, future, future_mask, config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            for name, meter in meters.items():
                meter.update(float(losses[name].detach()), int(future.size(0)))

        scores = evaluate(
            model, validation_loader, config, device, horizon, minimum_prefix, dt
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
        model, test_loader, config, device, horizon, minimum_prefix, dt
    )
    print()
    print("=== test ===")
    for name in ("model", "hold", "linear"):
        mean = test_scores.get(f"{name}_mean_m")
        final = test_scores.get(f"{name}_final_m")
        if mean is not None:
            print(f"  {name:7} mean {mean * 100:6.2f} cm | final {final * 100:6.2f} cm")
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
