#!/usr/bin/env python3
"""Let the network choose its own forecast horizon from EMG+IMU, per trial.

Every trajectory number so far used a horizon WE picked (254 ms, then
1000 ms, both hand-set via --horizon-ms). This trains a small head
(AdaptiveHorizonHead, src/emg_touch/models/adaptive_horizon.py) that reads
the same context the trajectory model already computes and outputs tau: how
far ahead, in this specific trial, the network estimates it can forecast
confidently.

    python scripts/train_adaptive_horizon_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_trajectory_emg_enhanced.yaml \\
        --cache-dir artifacts/tracked_cache \\
        --device cuda --epochs 20 --model anticipatory \\
        --tau-min-ms 100 --tau-max-ms 1000 \\
        --output-dir runs/adaptive_horizon

tau_max_ms=1000 is not arbitrary - scripts/diagnose_horizon_feasibility.py
measured this dataset directly and found >=95% of trials support a horizon
up to ~1000 ms; picking a tau_max the data cannot supply would only teach
the network to want something structurally unavailable.

NOT a discriminator. A gradient-reversal discriminator needs an adversarial
target; "predict further ahead" is a direct trade-off within one network,
not a two-player game, so this is one differentiable loss: reward larger
tau, penalised by whatever forecast error it costs at that specific horizon
(interpolate_at bridges continuous tau to the decoder's fixed integer
timesteps - see adaptive_horizon.py for why this needs to be differentiable
at all).

reach_weight is RAMPED, not constant, because it has to be - measured, not
assumed. An un-ramped run collapses to tau pinned at the SAME value for
every input regardless of context, gradient vanishing in the saturated
sigmoid before the network had any chance to learn that easy and hard
trials should differ (verified: two synthetic groups with genuinely
different achievable accuracy at long range, tau ended up 100.0 vs 100.0,
zero separation, un-ramped). Ramping reach_weight from 0 over the first
third of training - the same schedule this script already uses for the
anticipatory model's GRL reversal_strength - lets the accuracy-vs-context
relationship shape tau BEFORE the reach incentive is strong enough to
saturate it (verified: with the ramp, the same two synthetic groups
separated to 99.7 vs 10.4 - a real, usable signal, not a coincidence of
one weight value).

Reports the comparison that actually decides whether this is worth its
complexity: mean forecast error AT the network's own chosen tau, against
the SAME model's error at a FIXED tau=tau_min for every trial and a FIXED
tau=tau_max for every trial. If the adaptive number does not beat both
fixed extremes, letting the network choose bought nothing over picking one
sensible constant by hand.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import build_tracked_loaders  # noqa: E402
from emg_touch.models.adaptive_horizon import (  # noqa: E402
    AdaptiveHorizonHead,
    adaptive_horizon_loss,
    interpolate_at,
)
from emg_touch.models.disentangle import reversal_strength  # noqa: E402
from emg_touch.models.trajectory_intent_vae import trajectory_loss  # noqa: E402
from emg_touch.utils import choose_device, save_json, seed_everything  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_traj", Path(__file__).resolve().parent / "train_trajectory_model.py"
)
_traj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_traj)
TrajectoryModel = _traj.TrajectoryModel
make_window = _traj.make_window


def fixed_tau_error(trajectory, future, future_mask, tau_samples: float) -> float:
    """Mean error at ONE fixed tau for every row - the baseline this exists to beat."""
    tau = torch.full((trajectory.size(0),), tau_samples, device=trajectory.device)
    predicted, _ = interpolate_at(trajectory, tau)
    truth, validity = interpolate_at(future, tau, mask=future_mask)
    weight = validity / validity.sum().clamp_min(1.0)
    return float(((predicted - truth).norm(dim=-1) * weight).sum())


@torch.no_grad()
def evaluate(model, head, loader, tau_min, tau_max, minimum_prefix, dt, device) -> dict:
    model.eval()
    head.eval()
    generator = np.random.default_rng(0)
    adaptive_errors, fixed_min_errors, fixed_max_errors = [], [], []
    tau_values = []
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        made = make_window(
            batch, int(tau_max), minimum_prefix, generator, dt=dt,
            ablate=("position", "velocity"), relative=True,
        )
        if made is None:
            continue
        window, future, future_mask = made
        context, _ = model.encoder(
            window["emg"], window["imu"], window["position"], window["velocity"],
            return_modalities=True,
        )
        tau = head(context)
        outputs = model._decode(context, window, horizon=int(tau_max), strength=1.0,
                                 measure_anticipatory=False)
        trajectory = outputs["trajectory"]

        predicted, _ = interpolate_at(trajectory, tau)
        truth, validity = interpolate_at(future, tau, mask=future_mask)
        weight = validity / validity.sum().clamp_min(1.0)
        adaptive_errors.append(float(((predicted - truth).norm(dim=-1) * weight).sum()))
        fixed_min_errors.append(fixed_tau_error(trajectory, future, future_mask, tau_min))
        fixed_max_errors.append(fixed_tau_error(trajectory, future, future_mask, tau_max - 1))
        tau_values.append(tau.cpu().numpy())

    if not tau_values:
        return {}
    tau_all = np.concatenate(tau_values)
    return {
        "adaptive_error_m": float(np.mean(adaptive_errors)),
        "fixed_tau_min_error_m": float(np.mean(fixed_min_errors)),
        "fixed_tau_max_error_m": float(np.mean(fixed_max_errors)),
        "tau_mean": float(tau_all.mean()), "tau_std": float(tau_all.std()),
        "tau_min_seen": float(tau_all.min()), "tau_max_seen": float(tau_all.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory_emg_enhanced.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/adaptive_horizon")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", choices=("trajectory", "anticipatory"), default="anticipatory")
    parser.add_argument("--tau-min-ms", type=float, default=100.0)
    parser.add_argument("--tau-max-ms", type=float, default=1000.0,
                        help="Keep <=1000 unless diagnose_horizon_feasibility.py "
                        "confirms a longer value is well supported by your data.")
    parser.add_argument("--reach-weight", type=float, default=0.05,
                        help="Target value the ramp reaches by 1/3 of training. "
                        "Verified interactively (see module docstring): 0.06 in "
                        "units matched to that synthetic test separated two groups "
                        "cleanly; this dataset's real error scale may need a "
                        "different value - watch tau_std in the printed report, "
                        "0 means it collapsed and this needs adjusting.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    print(f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
          f"| test {len(test_loader.dataset)} trials")

    rate = float(config["data"]["sample_rate_hz"]) / int(config["data"]["decimation"])
    dt = 1.0 / rate
    minimum_prefix = int(config["virtual_leader"].get("minimum_prefix", 16))
    tau_min_samples = max(1, int(round(args.tau_min_ms * rate / 1000.0)))
    tau_max_samples = max(tau_min_samples + 2, int(round(args.tau_max_ms * rate / 1000.0)))
    print(f"tau range: {args.tau_min_ms:.0f}-{args.tau_max_ms:.0f} ms "
          f"({tau_min_samples}-{tau_max_samples} samples at {rate:.1f} Hz)")

    model = TrajectoryModel(config, args.model, task="wearable").to(device)
    head = AdaptiveHorizonHead(model.encoder.context_dim, tau_min_samples, tau_max_samples).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    ramp_steps = max(1, len(train_loader) * int(config["training"]["epochs"]) // 3)
    global_step = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, best_state, history = float("inf"), None, []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        head.train()
        generator = np.random.default_rng(epoch)
        running = {"base": [], "accuracy_at_tau": [], "reach_penalty": []}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            made = make_window(
                batch, tau_max_samples, minimum_prefix, generator, dt=dt,
                ablate=("position", "velocity"), relative=True,
            )
            if made is None:
                continue
            window, future, future_mask = made

            reach_weight = args.reach_weight * min(1.0, global_step / ramp_steps)
            strength = reversal_strength(global_step, ramp_steps)
            context, _ = model.encoder(
                window["emg"], window["imu"], window["position"], window["velocity"],
                return_modalities=True,
            )
            tau = head(context)
            outputs = model._decode(context, window, horizon=tau_max_samples,
                                     strength=strength, measure_anticipatory=False)
            base_losses = trajectory_loss(outputs, future, future_mask, config)
            adaptive = adaptive_horizon_loss(
                outputs["trajectory"], future, future_mask, tau, tau_max_samples, reach_weight
            )
            total = base_losses["loss"] + adaptive["loss"]

            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                float(config["training"].get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            running["base"].append(float(base_losses["loss"]))
            running["accuracy_at_tau"].append(float(adaptive["error_at_tau"]))
            running["reach_penalty"].append(float(adaptive["reach_penalty"]))

        scores = evaluate(model, head, validation_loader, tau_min_samples, tau_max_samples,
                          minimum_prefix, dt, device)
        history.append({"epoch": epoch, **{f"train_{k}": float(np.mean(v or [0]))
                                            for k, v in running.items()}, **scores})
        print(
            f"epoch={epoch} base={np.mean(running['base'] or [0]):.4f} "
            f"reach_w={reach_weight:.4f} | "
            f"val adaptive={scores.get('adaptive_error_m', float('nan'))*100:.2f}cm "
            f"fixed@min={scores.get('fixed_tau_min_error_m', float('nan'))*100:.2f}cm "
            f"fixed@max={scores.get('fixed_tau_max_error_m', float('nan'))*100:.2f}cm | "
            f"tau={scores.get('tau_mean', float('nan'))/rate*1000:.0f}"
            f"±{scores.get('tau_std', float('nan'))/rate*1000:.0f}ms"
        )
        if scores.get("adaptive_error_m", float("inf")) < best:
            best = scores["adaptive_error_m"]
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            torch.save({**best_state, "config": config, "tau_min_samples": tau_min_samples,
                       "tau_max_samples": tau_max_samples}, output / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        head.load_state_dict(best_state["head_state"])
    test = evaluate(model, head, test_loader, tau_min_samples, tau_max_samples,
                    minimum_prefix, dt, device)

    print("\n=== test ===")
    print(f"  adaptive (network's own tau) : {test.get('adaptive_error_m', float('nan'))*100:6.2f} cm")
    print(f"  fixed at tau_min ({args.tau_min_ms:.0f} ms)      : "
          f"{test.get('fixed_tau_min_error_m', float('nan'))*100:6.2f} cm")
    print(f"  fixed at tau_max ({args.tau_max_ms:.0f} ms)      : "
          f"{test.get('fixed_tau_max_error_m', float('nan'))*100:6.2f} cm")
    print(f"  tau chosen: {test.get('tau_mean', float('nan'))/rate*1000:.0f} ms "
          f"± {test.get('tau_std', float('nan'))/rate*1000:.0f} ms "
          f"(range {test.get('tau_min_seen', float('nan'))/rate*1000:.0f}-"
          f"{test.get('tau_max_seen', float('nan'))/rate*1000:.0f} ms)")

    if test.get("tau_std", 0) < 0.02 * (tau_max_samples - tau_min_samples):
        print("\n  tau_std is small relative to the tau range - the head may have")
        print("  collapsed to near-constant regardless of input. Check reach_weight;")
        print("  this is the same failure mode the module docstring's synthetic test")
        print("  demonstrated without the ramp.")
    beats_both = (
        test.get("adaptive_error_m", 1e9) < test.get("fixed_tau_min_error_m", 0)
        and test.get("adaptive_error_m", 1e9) < test.get("fixed_tau_max_error_m", 0)
    )
    print(f"\n  adaptive beats BOTH fixed extremes: "
          f"{'yes' if beats_both else 'no - the per-example choice is not earning its complexity here'}")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
