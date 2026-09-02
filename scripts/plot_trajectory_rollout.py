#!/usr/bin/env python3
"""Reconstruct the whole-trial global path from a trained wearable model.

Every trajectory number reported so far (4.40 cm mean / 8.44 cm at 254 ms)
is a SHORT-HORIZON forecast: one cutoff, one 254 ms window. This walks a
full trial and answers two related but different questions.

    python scripts/plot_trajectory_rollout.py \\
        --root "/media/.../emg_imu_vive" \\
        --checkpoint runs/.../best.pt \\
        --cache-dir artifacts/tracked_cache \\
        --model anticipatory --task wearable \\
        --device cuda --num-trials 6 --output-dir runs/rollout_check

anchored   Stitch consecutive 254 ms windows together, RE-GROUNDING to the
           true measured position at the start of every window - exactly
           what the existing short-horizon number already measures, just
           visualised across a whole trial instead of one cutoff. This is
           a consistency check, not a new claim: its mean/final error
           should land close to the numbers already reported.

blind      Anchor to the true position ONCE, at the first cutoff after
           onset, then chain the model's own predictions: each window's
           last predicted point becomes the next window's origin, with the
           tracker never consulted again. This is the honest answer to
           "how well can EMG+IMU alone reconstruct where the hand actually
           is in the room over a full reach" - dead reckoning, and it will
           drift, because every window's error compounds into the next
           one's starting point. How FAST it drifts (reported as drift at
           25/50/75/100% of the walked path) is the number that matters,
           not whether it eventually drifts at all.

The tracker is used in exactly two ways here, both outside the model: as
the one-time anchor for blind rollout, and as ground truth for plotting
and error - never as a model input, at any step. emg/imu at every window
are the real, causal recorded signal - nothing here is fabricated from a
previous prediction.

Prints a numeric summary always. Saves a PNG per trial (true path vs both
reconstructions, plus drift-vs-time for the blind rollout) if matplotlib is
importable; the numeric summary is unaffected if it is not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.data.tracked_dataset import build_tracked_loaders  # noqa: E402
from emg_touch.utils import choose_device  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_traj", Path(__file__).resolve().parent / "train_trajectory_model.py"
)
_traj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_traj)
TrajectoryModel = _traj.TrajectoryModel


def window_at_cutoff(
    batch: dict, row: int, cut: int, minimum_prefix: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    """One-row window ending at `cut`, position/velocity zeroed (wearable-only).

    Mirrors make_window's prefix-length rule (train_trajectory_model.py) so a
    checkpoint sees the same shape of input at inference it saw in training.
    """
    prefix_length = min(minimum_prefix * 4, cut)
    window = {}
    for key in ("emg", "imu"):
        window[key] = batch[key][row : row + 1, cut - prefix_length : cut].to(device)
    zero_shape = window["emg"].shape[:-1] + (3,)
    window["position"] = torch.zeros(zero_shape, device=device)
    window["velocity"] = torch.zeros(zero_shape, device=device)
    return window


@torch.no_grad()
def anchored_rollout(
    model, batch: dict, row: int, onset: int, length: int, minimum_prefix: int,
    horizon: int, device: torch.device,
) -> dict:
    """Stitch consecutive windows, re-grounding to the true position each time."""
    true_position = batch["position"][row].to(device)
    points, times, errors = [], [], []
    cut = max(onset + minimum_prefix, minimum_prefix)
    while cut + horizon <= length:
        window = window_at_cutoff(batch, row, cut, minimum_prefix, device)
        outputs = model(window, horizon)
        predicted = outputs["trajectory"][0]  # (horizon, 3), displacement from 0
        origin = true_position[cut]
        segment = origin.unsqueeze(0) + predicted
        truth = true_position[cut : cut + horizon]
        points.append(segment.cpu().numpy())
        times.append(np.arange(cut, cut + horizon))
        errors.append((segment - truth).norm(dim=-1).cpu().numpy())
        cut += horizon
    if not points:
        return {}
    return {
        "points": np.concatenate(points, axis=0),
        "times": np.concatenate(times, axis=0),
        "errors": np.concatenate(errors, axis=0),
    }


@torch.no_grad()
def blind_rollout(
    model, batch: dict, row: int, onset: int, length: int, minimum_prefix: int,
    horizon: int, device: torch.device,
) -> dict:
    """Anchor once; chain the model's own predictions with no further tracker use."""
    true_position = batch["position"][row].to(device)
    cut = max(onset + minimum_prefix, minimum_prefix)
    believed = true_position[cut].clone()  # the ONE-TIME anchor
    points, times, drift = [], [], []
    while cut + horizon <= length:
        window = window_at_cutoff(batch, row, cut, minimum_prefix, device)
        outputs = model(window, horizon)
        predicted = outputs["trajectory"][0]
        segment = believed.unsqueeze(0) + predicted
        points.append(segment.cpu().numpy())
        times.append(np.arange(cut, cut + horizon))
        truth = true_position[cut : cut + horizon]
        drift.append((segment - truth).norm(dim=-1).cpu().numpy())
        believed = segment[-1]  # the model's OWN point, never re-grounded
        cut += horizon
    if not points:
        return {}
    return {
        "points": np.concatenate(points, axis=0),
        "times": np.concatenate(times, axis=0),
        "drift": np.concatenate(drift, axis=0),
    }


def summarise_drift(drift: np.ndarray) -> dict:
    if len(drift) == 0:
        return {}
    fractions = (0.25, 0.5, 0.75, 1.0)
    return {
        f"drift_at_{int(f * 100)}pct_cm": float(drift[min(int(f * len(drift)) , len(drift) - 1)] * 100)
        for f in fractions
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Override the config stored in the checkpoint.")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/rollout_check")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model", choices=("trajectory", "anticipatory"), required=True,
        help="Must match how the checkpoint was trained.",
    )
    parser.add_argument(
        "--task", choices=("forecast", "wearable"), default="wearable",
        help="Must match how the checkpoint was trained. wearable is the "
        "setting this script exists to check.",
    )
    parser.add_argument("--num-trials", type=int, default=6)
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.config:
        from emg_touch.config import load_config

        config = load_config(args.config)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError(
            "checkpoint has no stored config and --config was not given"
        )

    model = TrajectoryModel(config, args.model, args.task).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader = build_tracked_loaders(config, args.root, Path(args.cache_dir))
    batch = next(iter(test_loader))
    if batch is None:
        print("test loader produced no usable batch", file=sys.stderr)
        sys.exit(2)

    horizon = int(config["virtual_leader"]["horizon"])
    minimum_prefix = int(config["virtual_leader"].get("minimum_prefix", 16))
    rate = float(config["data"]["sample_rate_hz"]) / max(1, int(config["data"].get("decimation", 10)))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        can_plot = True
    except ImportError:
        can_plot = False
        print("matplotlib not importable - numeric summary only, no PNGs\n")

    print(f"model={args.model} task={args.task} horizon={horizon} steps "
          f"({horizon / rate * 1000:.0f} ms per window)\n")
    print(f"{'trial':>6}{'len (s)':>9}{'windows':>9}{'anchored':>12}{'anchored':>10}"
          f"{'blind@25%':>11}{'blind@50%':>11}{'blind@75%':>11}{'blind@100%':>12}")
    print(f"{'':6}{'':9}{'':9}{'mean cm':>12}{'final cm':>10}{'cm':>11}{'cm':>11}{'cm':>11}{'cm':>12}")
    print("-" * 91)

    shown = 0
    for row in range(batch["position"].size(0)):
        if shown >= args.num_trials:
            break
        length = int(batch["lengths"][row])
        onset = int(batch["onset"][row])
        start = max(onset + minimum_prefix, minimum_prefix)
        if length - start < 2 * horizon:
            continue  # too short for a meaningful multi-window walk

        anchored = anchored_rollout(model, batch, row, onset, length, minimum_prefix, horizon, device)
        blind = blind_rollout(model, batch, row, onset, length, minimum_prefix, horizon, device)
        if not anchored or not blind:
            continue

        drift_summary = summarise_drift(blind["drift"])
        duration_s = (length - start) / rate
        n_windows = len(anchored["errors"]) // horizon
        print(
            f"{shown:>6}{duration_s:>9.2f}{n_windows:>9}"
            f"{anchored['errors'].mean() * 100:>12.2f}"
            f"{anchored['errors'][-1] * 100:>10.2f}"
            f"{drift_summary.get('drift_at_25pct_cm', float('nan')):>11.2f}"
            f"{drift_summary.get('drift_at_50pct_cm', float('nan')):>11.2f}"
            f"{drift_summary.get('drift_at_75pct_cm', float('nan')):>11.2f}"
            f"{drift_summary.get('drift_at_100pct_cm', float('nan')):>12.2f}"
        )

        if can_plot:
            true_position = batch["position"][row, start:length].cpu().numpy()
            # Two axes with the most spread - usually the screen-facing plane.
            spread = true_position.max(axis=0) - true_position.min(axis=0)
            axes_order = np.argsort(spread)[::-1][:2]
            labels = ["x", "y", "z"]

            figure, (plane, drift_axis) = plt.subplots(1, 2, figsize=(11, 5))
            plane.plot(
                true_position[:, axes_order[0]], true_position[:, axes_order[1]],
                "k-", linewidth=2, label="true path",
            )
            plane.plot(
                anchored["points"][:, axes_order[0]], anchored["points"][:, axes_order[1]],
                "b--", linewidth=1.2, alpha=0.8, label="anchored (re-grounded)",
            )
            plane.plot(
                blind["points"][:, axes_order[0]], blind["points"][:, axes_order[1]],
                "r:", linewidth=1.5, label="blind (dead reckoning)",
            )
            plane.scatter(*true_position[0, axes_order], c="green", marker="o", s=60, zorder=5, label="onset")
            plane.scatter(*true_position[-1, axes_order], c="black", marker="x", s=60, zorder=5, label="touch")
            plane.set_xlabel(labels[axes_order[0]] + " (m)")
            plane.set_ylabel(labels[axes_order[1]] + " (m)")
            plane.set_title(f"trial {shown}: global path")
            plane.legend(fontsize=8)
            plane.set_aspect("equal", adjustable="datalim")

            drift_axis.plot(blind["times"] / rate, blind["drift"] * 100, "r-")
            drift_axis.set_xlabel("time (s)")
            drift_axis.set_ylabel("blind drift from truth (cm)")
            drift_axis.set_title("dead-reckoning drift over the trial")

            figure.tight_layout()
            path = output / f"trial_{shown:02d}_rollout.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)

        shown += 1

    if shown == 0:
        print("\nno trial in the first batch was long enough for a multi-window "
              "walk - increase batch_size or loosen the length check")
    else:
        print(f"\n{shown} trial(s) shown. anchored mean/final should be close to "
              "this checkpoint's already-reported short-horizon numbers - if it "
              "is not, something about this script's windowing has drifted from "
              "training's own, and that mismatch is worth chasing before "
              "trusting the blind-rollout numbers.")
        if can_plot:
            print(f"PNGs written to {output}/")


if __name__ == "__main__":
    main()
