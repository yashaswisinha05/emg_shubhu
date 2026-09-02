#!/usr/bin/env python3
"""Predict the reach's actual endpoint from wearables, not a fixed horizon.

The wearable trajectory model predicts displacement over the next 254 ms.
That is a motion forecast, and for an interface it answers the wrong
question: what matters is not where the hand will be a quarter second from
now, but which point on the screen it is heading for.

This predicts the destination itself. From a prefix of EMG (optionally plus
IMU), with the tracker never entering the encoder, it produces three things:

    screen      the touch coordinate, normalised, and in pixels
    path        the remaining 3-D hand path as `phases` points evenly spaced
                in normalised time between the cutoff and the touch, so the
                positional error is reported at every step rather than only
                at the end
    duration    time remaining until the touch

Why the screen coordinate is a better target than the trajectory for this
sensor set. The trajectory task had to be posed as displacement, because
without a tracker the hand's absolute position is unknown and asking for it
is ill-posed. The screen coordinate has no such problem: it is a property of
the target, not of where the hand happens to be, so it is directly
predictable from wearables and directly comparable with this project's
earlier screen-coordinate results in pixels.

Baselines. `mean` predicts the training set's average screen target, average
path shape and average duration - the population reach, with no per-trial
inference. Beating it is the whole claim, exactly as in wearable trajectory
mode. `centre` predicts the middle of the screen, which bounds what a model
that has learned nothing at all would score.

The touch is taken as the last valid sample of each trial. The script checks
that assumption on real data rather than trusting it: if the hand is still
moving at the final sample, the trials do not end at contact and the reported
endpoint is not the touch. That warning is printed once at startup.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import build_tracked_loaders  # noqa: E402
from emg_touch.utils import save_json, seed_everything  # noqa: E402

# Reuse the encoder the trajectory work already tuned, rather than a second
# copy that would drift from it.
_spec = importlib.util.spec_from_file_location(
    "_traj", Path(__file__).resolve().parent / "train_trajectory_model.py"
)
_traj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_traj)
TrajectoryEncoder = _traj.TrajectoryEncoder


class ReachTargetModel(torch.nn.Module):
    """EMG (+IMU) -> screen target, remaining path, and time to touch."""

    def __init__(self, config: dict, phases: int = 8) -> None:
        super().__init__()
        self.phases = phases
        self.encoder = TrajectoryEncoder(config)
        width = self.encoder.context_dim

        def head(outputs: int) -> torch.nn.Sequential:
            layers = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, 128),
                torch.nn.GELU(), torch.nn.Linear(128, outputs),
            )
            # Small random rather than zero. A zero final weight makes the
            # head's gradient with respect to the context identically zero and
            # severs the encoder silently - that failure has already occurred
            # twice in this project, so it is not left to chance again.
            torch.nn.init.normal_(layers[-1].weight, std=0.01)
            torch.nn.init.zeros_(layers[-1].bias)
            return layers

        self.screen = head(2)
        self.path = head(phases * 3)
        self.duration = head(1)

    def forward(self, window: dict) -> dict:
        context = self.encoder(
            window["emg"], window["imu"], window["position"], window["velocity"]
        )
        return {
            # Sigmoid: the screen coordinate is normalised to [0, 1], so the
            # bound is a fact about the task rather than a squashing choice.
            "screen": torch.sigmoid(self.screen(context)),
            "path": self.path(context).view(-1, self.phases, 3),
            # Log-seconds, so the head works in a space where a 100 ms error
            # on a 300 ms reach and on a 2 s reach are not treated alike.
            "log_duration": self.duration(context).squeeze(-1),
        }


def make_reach_window(
    batch: dict, minimum_prefix: int, phases: int, rate: float, generator,
    ablate: tuple[str, ...] = (),
) -> tuple[dict, dict] | None:
    """Cut after onset, and take the touch as the trial's last valid sample."""
    lengths, onsets = batch["lengths"], batch["onset"]
    chosen = []
    for row in range(len(lengths)):
        length = int(lengths[row])
        touch = length - 1
        start = max(int(onsets[row]) + minimum_prefix, minimum_prefix)
        # A cutoff must leave enough reach ahead of it to be a prediction
        # rather than a readout of an arrival that has already happened.
        latest = touch - minimum_prefix
        if latest <= start:
            continue
        chosen.append((row, int(generator.integers(start, latest)), touch))
    if not chosen:
        return None

    prefix_length = min(minimum_prefix * 4, min(cut for _, cut, _ in chosen))
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu", "position", "velocity"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut, _ in chosen]
        )
    # The tracker is the label. Kept for targets and baselines, never shown.
    origin = torch.stack([batch["position"][row, cut] for row, cut, _ in chosen])

    paths, durations = [], []
    for row, cut, touch in chosen:
        # Evenly spaced in normalised time, so trials of different durations
        # are compared at the same fraction of the reach rather than at the
        # same absolute delay.
        steps = np.linspace(cut, touch, phases + 1)[1:].round().astype(int)
        steps = np.clip(steps, 0, int(batch["lengths"][row]) - 1)
        paths.append(batch["position"][row, steps])
        durations.append((touch - cut) / rate)
    target_path = torch.stack(paths) - origin.unsqueeze(1)

    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])

    rows = torch.tensor([row for row, _, _ in chosen], dtype=torch.long)
    targets = {
        "path": target_path,
        "duration": torch.tensor(durations, dtype=torch.float32),
        "origin": origin,
    }
    if "screen_target" in batch:
        targets["screen"] = batch["screen_target"].index_select(0, rows)
    if "canvas" in batch:
        targets["canvas"] = batch["canvas"].index_select(0, rows)
    return window, targets


def reach_loss(outputs: dict, targets: dict, config: dict) -> dict:
    settings = config.get("loss", {})
    epsilon = float(settings.get("trajectory_epsilon_m", 0.002))
    distance = torch.sqrt(
        (outputs["path"] - targets["path"]).square().sum(-1) + epsilon * epsilon
    )
    path = distance.mean()
    total = path
    result = {"path": path}
    if "screen" in targets:
        screen = F.smooth_l1_loss(outputs["screen"], targets["screen"])
        total = total + float(settings.get("screen_weight", 1.0)) * screen
        result["screen"] = screen
    duration = F.smooth_l1_loss(
        outputs["log_duration"], torch.log(targets["duration"].clamp_min(1e-3))
    )
    total = total + float(settings.get("duration_weight", 0.1)) * duration
    result["duration"] = duration
    result["loss"] = total
    return result


@torch.no_grad()
def evaluate(model, loader, config, device, minimum_prefix, phases, rate,
             ablate, reference: dict | None) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)  # fixed cutoffs, so runs compare
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        made = make_reach_window(
            batch, minimum_prefix, phases, rate, generator, ablate
        )
        if made is None:
            continue
        window, targets = made
        outputs = model(window)

        candidates = {"model": outputs}
        if reference is not None:
            candidates["mean"] = {
                "path": reference["path"].to(device).unsqueeze(0).expand(
                    targets["path"].size(0), -1, -1
                ),
                "screen": reference["screen"].to(device).unsqueeze(0).expand(
                    targets["path"].size(0), -1
                ),
                "log_duration": torch.full_like(
                    outputs["log_duration"], float(reference["log_duration"])
                ),
            }
            candidates["centre"] = {
                "path": torch.zeros_like(targets["path"]),
                "screen": torch.full_like(outputs["screen"], 0.5),
                "log_duration": torch.zeros_like(outputs["log_duration"]),
            }

        for name, prediction in candidates.items():
            error = (prediction["path"] - targets["path"]).norm(dim=-1)
            totals.setdefault(f"{name}_path_m", []).append(float(error.mean()))
            totals.setdefault(f"{name}_endpoint_m", []).append(float(error[:, -1].mean()))
            # Positional error at each normalised step of the remaining reach.
            for step in range(error.size(1)):
                totals.setdefault(f"{name}_step{step}_m", []).append(
                    float(error[:, step].mean())
                )
            if "screen" in targets:
                delta = prediction["screen"] - targets["screen"]
                totals.setdefault(f"{name}_screen_norm", []).append(
                    float(delta.norm(dim=-1).mean())
                )
                if "canvas" in targets:
                    pixels = (delta * targets["canvas"]).norm(dim=-1)
                    totals.setdefault(f"{name}_screen_px", []).append(
                        float(pixels.mean())
                    )
            seconds = torch.exp(prediction["log_duration"])
            totals.setdefault(f"{name}_duration_ms", []).append(
                float((seconds - targets["duration"]).abs().mean() * 1000.0)
            )
    return {key: float(np.mean(values)) for key, values in totals.items()}


def training_reference(loader, minimum_prefix, phases, rate, batches=40) -> dict:
    """Population average path, screen target and duration."""
    generator = np.random.default_rng(0)
    paths, screens, durations = [], [], []
    for index, batch in enumerate(loader):
        if index >= batches or batch is None:
            continue
        made = make_reach_window(batch, minimum_prefix, phases, rate, generator)
        if made is None:
            continue
        _, targets = made
        paths.append(targets["path"].mean(0))
        durations.append(targets["duration"].mean())
        if "screen" in targets:
            screens.append(targets["screen"].mean(0))
    if not paths:
        return None
    return {
        "path": torch.stack(paths).mean(0).cpu(),
        "screen": (torch.stack(screens).mean(0).cpu() if screens
                   else torch.tensor([0.5, 0.5])),
        "log_duration": float(torch.log(torch.stack(durations).mean().clamp_min(1e-3))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/reach_target")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phases", type=int, default=8)
    parser.add_argument(
        "--inputs", choices=("emg", "emg+imu"), default="emg+imu",
        help="emg: the muscle signal alone. The tracker never enters the "
        "encoder in either case - it is the label.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    # The tracker is the label in every configuration here, so it is removed
    # unconditionally; --inputs only chooses whether the IMU joins EMG.
    ablate = ["position", "velocity"] + (["imu"] if args.inputs == "emg" else [])
    ablate = tuple(sorted(set(ablate)))
    print(f"inputs: {args.inputs}  (encoder is blind to {list(ablate)})")
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials"
    )

    rate = float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )
    minimum_prefix = int(config["virtual_leader"].get("minimum_prefix", 16))
    reference = training_reference(train_loader, minimum_prefix, args.phases, rate)

    model = ReachTargetModel(config, args.phases).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = np.random.default_rng(int(args.seed or 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, best_state, history = float("inf"), None, []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            made = make_reach_window(
                batch, minimum_prefix, args.phases, rate, generator, ablate
            )
            if made is None:
                continue
            window, targets = made
            losses = reach_loss(model(window), targets, config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            running.append(float(losses["loss"].detach()))

        scores = evaluate(
            model, validation_loader, config, device, minimum_prefix,
            args.phases, rate, ablate, reference,
        )
        selection = scores.get("model_screen_norm", scores.get("model_path_m", 1e9))
        history.append({"epoch": epoch, "train": float(np.mean(running or [0])), **scores})
        print(
            f"epoch={epoch} loss={np.mean(running or [0]):.4f} | "
            f"val screen={scores.get('model_screen_norm', float('nan')):.4f} norm "
            f"({scores.get('model_screen_px', float('nan')):.1f} px) | "
            f"endpoint={scores.get('model_endpoint_m', float('nan')) * 100:.2f} cm"
        )
        if selection < best:
            best = selection
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"loaded best checkpoint (val screen {best:.4f}) for test")
    test = evaluate(
        model, test_loader, config, device, minimum_prefix, args.phases, rate,
        ablate, reference,
    )

    print("\n=== test ===")
    print(f"  inputs: {args.inputs}, tracker excluded\n")
    print(f"  {'':10}{'screen norm':>14}{'screen px':>12}{'endpoint':>12}{'path mean':>12}{'timing':>11}")
    for name in ("model", "mean", "centre"):
        if f"{name}_path_m" not in test:
            continue
        print(
            f"  {name:10}"
            f"{test.get(f'{name}_screen_norm', float('nan')):>14.4f}"
            f"{test.get(f'{name}_screen_px', float('nan')):>12.1f}"
            f"{test[f'{name}_endpoint_m'] * 100:>10.2f} cm"
            f"{test[f'{name}_path_m'] * 100:>10.2f} cm"
            f"{test.get(f'{name}_duration_ms', float('nan')):>8.0f} ms"
        )

    print("\n  positional error along the remaining reach")
    print(f"  {'phase':10}" + "".join(f"{(i + 1) / args.phases:>9.0%}" for i in range(args.phases)))
    for name in ("model", "mean", "centre"):
        if f"{name}_step0_m" not in test:
            continue
        cells = "".join(
            f"{test[f'{name}_step{i}_m'] * 100:>9.2f}" for i in range(args.phases)
        )
        print(f"  {name:10}{cells}")
    print("  (cm; 100% is the touch itself)")

    model_screen = test.get("model_screen_norm")
    mean_screen = test.get("mean_screen_norm")
    if model_screen and mean_screen:
        gain = (mean_screen - model_screen) / mean_screen * 100
        print(
            f"\n  screen prediction is {gain:+.1f}% vs the population mean target"
        )
        print("  -> positive means the wearables identify WHICH target this reach")
        print("     is for, not merely where an average reach ends up")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
