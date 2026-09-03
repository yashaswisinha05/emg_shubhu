#!/usr/bin/env python3
"""Replay a trial causally, emitting (mu, sigma) at every instant - not one cutoff.

Every number this project has reported comes from ONE cutoff per trial,
picked in advance. This is different: given a checkpoint (176 px point model
+ trained uncertainty head), it walks one test trial forward sample by
sample with a rolling causal buffer - never touching a sample after the
current instant - and re-predicts (mu_x, mu_y, sigma_x, sigma_y) at every
step. This is the actual shape a deployed interface would run in.

    python scripts/live_inference.py \\
        --root "/media/.../emg_imu_vive" \\
        --base-checkpoint runs/grid_leadwindow_emg_imu/best.pt \\
        --uncertainty-checkpoint runs/uncertainty_head/best.pt \\
        --config configs/tracked_grid_within.yaml \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --num-trials 3 --output-dir runs/live_check

--uncertainty-checkpoint is optional: without it, this reports mu only
(sigma columns come back empty), so it also works as a live check on any
plain GridReachModel checkpoint before an uncertainty head exists for it.

Prints, per trial, a table of (time-to-touch, mu in px, sigma in px, actual
error) at a chosen stride, plus a summary: does sigma actually SHRINK as
touch approaches, which is the property a live confidence estimate needs to
be useful for anything (a UI that shows a growing-then-shrinking circle) -
NOT just whether it is calibrated at any one instant, which train_uncertainty
_head.py's coverage numbers already checked.

Every prediction here only ever reads the causal buffer's own recorded
EMG/IMU. No prediction reads a sample after itself, and no prediction reads
the tracker - verified structurally below, not just asserted, the same
convention this project's other wearable-mode scripts follow.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    CANVAS_COLUMNS,
    build_tracked_loaders,
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.grid_reach import GridReachModel, UncertaintyHead  # noqa: E402
from emg_touch.utils import choose_device  # noqa: E402


def canvas_from_disk(root: str) -> tuple[float, float] | None:
    for path in sorted(Path(root).rglob("trial_*.csv"))[:1]:
        try:
            frame = pd.read_csv(path, nrows=64)
        except Exception:  # noqa: BLE001
            return None
        if not all(name in frame.columns for name in CANVAS_COLUMNS):
            return None
        values = (
            frame[list(CANVAS_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        )
        if len(values):
            return float(values[-1][0]), float(values[-1][1])
    return None


@torch.no_grad()
def replay_trial(
    base, head, emg: torch.Tensor, imu: torch.Tensor, onset: int, touch: int,
    minimum_prefix: int, patch_length: int, stride: int, canvas: torch.Tensor,
    target_px: torch.Tensor,
) -> list[dict]:
    """One causal forward pass per step. Buffer is emg[:cut]/imu[:cut] only."""
    records = []
    start = max(onset + minimum_prefix, patch_length)
    for cut in range(start, touch + 1, stride):
        # The causal invariant this whole function exists to hold: only
        # samples strictly before `cut` are visible at step `cut`.
        prefix_length = max(patch_length, min(minimum_prefix * 4, cut))
        window_emg = emg[cut - prefix_length : cut].unsqueeze(0)
        window_imu = imu[cut - prefix_length : cut].unsqueeze(0)
        mask = torch.ones(1, prefix_length, dtype=torch.bool, device=emg.device)

        outputs = base(window_emg, window_imu, mask)
        mu = outputs["prediction"][0]
        mu_px = mu * canvas
        record = {
            "sample": cut,
            "time_to_touch_ms": None,  # filled by the caller, which knows the rate
            "mu_x_px": float(mu_px[0]), "mu_y_px": float(mu_px[1]),
            "error_px": float((mu_px - target_px).norm()),
        }
        if head is not None:
            sigma = head(outputs["context"])[0]
            sigma_px = sigma * canvas
            record["sigma_x_px"] = float(sigma_px[0])
            record["sigma_y_px"] = float(sigma_px[1])
            record["sigma_radius_px"] = float(sigma_px.norm())
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--uncertainty-checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/live_check")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", choices=("emg", "emg+imu"), default="emg+imu")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--stride-ms", type=float, default=40.0,
                        help="How often to emit a prediction, in ms of wall time.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    _, _, test_loader = build_tracked_loaders(config, args.root, Path(args.cache_dir))
    batch = next(iter(test_loader))
    if batch is None:
        print("no usable batch", file=sys.stderr)
        sys.exit(2)

    use_imu = args.inputs == "emg+imu"
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    base = GridReachModel(config, emg_channels, imu_channels, use_imu).to(device)
    state = torch.load(args.base_checkpoint, map_location=device)
    base.load_state_dict(state["model_state"] if "model_state" in state else state)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    head = None
    if args.uncertainty_checkpoint:
        context_dim = int(config["model"]["d_model"]) * (2 if use_imu else 1)
        head = UncertaintyHead(context_dim).to(device)
        head_state = torch.load(args.uncertainty_checkpoint, map_location=device)
        head.load_state_dict(head_state["head_state"])
        head.eval()

    rate = float(config["data"]["sample_rate_hz"]) / max(1, int(config["data"].get("decimation", 10)))
    stride = max(1, int(round(args.stride_ms * rate / 1000.0)))
    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    fallback = canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback
        else torch.tensor([1.0, 1.0], device=device)
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shown = 0
    all_shrink_checks = []

    for row in range(batch["position"].size(0)):
        if shown >= args.num_trials:
            break
        length = int(batch["lengths"][row])
        onset = int(batch["onset"][row])
        touch = length - 1
        if touch - max(onset + minimum_prefix, patch_length) < 3 * stride:
            continue

        emg = batch["emg"][row, :length].to(device)
        imu = batch["imu"][row, :length].to(device)
        target = batch["screen_target"][row].to(device)
        canvas = batch["canvas"][row].to(device) if "canvas" in batch else canvas_tensor
        target_px = target * canvas

        records = replay_trial(
            base, head, emg, imu, onset, touch, minimum_prefix, patch_length,
            stride, canvas, target_px,
        )
        for record in records:
            record["time_to_touch_ms"] = (touch - record["sample"]) / rate * 1000.0

        print(f"\n=== trial {shown} ({len(records)} live predictions, "
              f"stride {args.stride_ms:.0f} ms) ===")
        header = f"{'t-to-touch (ms)':>16}{'mu_x':>8}{'mu_y':>8}{'error px':>10}"
        if head is not None:
            header += f"{'sigma_x':>9}{'sigma_y':>9}"
        print(header)
        for record in records[::max(1, len(records) // 12)]:
            line = (f"{record['time_to_touch_ms']:>16.0f}{record['mu_x_px']:>8.0f}"
                    f"{record['mu_y_px']:>8.0f}{record['error_px']:>10.1f}")
            if head is not None:
                line += f"{record['sigma_x_px']:>9.1f}{record['sigma_y_px']:>9.1f}"
            print(line)

        if head is not None and len(records) >= 4:
            radii = [r["sigma_radius_px"] for r in records]
            first_half = float(np.mean(radii[: len(radii) // 2]))
            second_half = float(np.mean(radii[len(radii) // 2 :]))
            shrinks = second_half < first_half
            all_shrink_checks.append(shrinks)
            print(f"  sigma radius: {first_half:.1f} px (far) -> {second_half:.1f} px "
                  f"(near touch)  {'shrinks (good)' if shrinks else 'does NOT shrink'}")

        pd.DataFrame(records).to_csv(output / f"trial_{shown:02d}_live.csv", index=False)
        shown += 1

    if shown == 0:
        print("no trial in the first batch was long enough to replay", file=sys.stderr)
        sys.exit(2)

    if all_shrink_checks:
        fraction = sum(all_shrink_checks) / len(all_shrink_checks)
        print(f"\n{shown} trial(s) replayed. sigma shrinks toward touch in "
              f"{sum(all_shrink_checks)}/{len(all_shrink_checks)} trials.")
        if fraction < 0.6:
            print("  Fewer than 60% shrink - sigma may not be tracking how much of the")
            print("  reach remains. Coverage from train_uncertainty_head.py can still be")
            print("  well-calibrated on average while this property fails; check both.")
    print(f"wrote per-trial CSVs to {output}/")


if __name__ == "__main__":
    main()
