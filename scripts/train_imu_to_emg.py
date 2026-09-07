#!/usr/bin/env python3
"""Train IMU teacher, supervised EMG baseline, and distilled EMG-only student.

Isolated experiment; no existing trainer or checkpoint is changed. Uses the
existing causal preprocessing and training-only candidate normalization.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.models.imu_to_emg import WearableTaskNetwork, task_loss, transfer_loss
from scripts.train_complete_reach_model import make_complete_reach_window
from scripts import train_personalized_complete_reach as candidate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def windows(loader, config, device, seed, fixed_lead=None):
    generator = np.random.default_rng(seed)
    rate = config["data"]["sample_rate_hz"] / config["data"]["decimation"]
    settings = config["imu_to_emg"]
    for batch in loader:
        if batch is None:
            continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        window = make_complete_reach_window(
            batch, context_samples=round(settings["context_ms"] * rate / 1000),
            patch_length=1, teacher_steps=settings["steps"], generator=generator,
            trajectory_limit_m=0.8, velocity_scale_mps=1.,
            # Require recorded canvas dimensions rather than silently assume a screen.
            fallback_canvas=None,
            lead_window=tuple(round(ms * rate / 1000) for ms in settings["lead_ms"]),
            fixed_lead=None if fixed_lead is None else round(fixed_lead * rate / 1000),
            cutoffs_per_trial=settings["cutoffs"] if fixed_lead is None else 1)
        if window is not None:
            yield window


@torch.no_grad()
def evaluate(model, modality, loader, config, device):
    model.eval()
    report = {}
    total = np.zeros(3)
    count = 0
    for lead in config["imu_to_emg"]["evaluation_leads_ms"]:
        values = []
        for w in windows(loader, config, device, 0, lead):
            out = model(w[modality], w["time_mask"])
            values.append(torch.stack([
                ((out["screen"] - w["target"]) * w["canvas_size"]).norm(dim=-1),
                (out["path"] - w["trajectory_target"]).norm(dim=-1).mean(-1) * 100,
                (out["path"][:, -1] - w["endpoint_3d_target"]).norm(dim=-1) * 100,
            ], -1).cpu())
        if not values:
            raise ValueError(f"No evaluable windows at {lead} ms; inspect trial lengths")
        values = torch.cat(values).numpy()
        report[str(lead)] = dict(zip(["screen_px", "path_cm", "endpoint_cm"],
                                     map(float, values.mean(0))))
        report[str(lead)]["observations"] = len(values)
        total += values.sum(0)
        count += len(values)
    report["overall"] = dict(zip(["screen_px", "path_cm", "endpoint_cm"],
                                 map(float, total / count)))
    return report


def fit(name, model, modality, teacher, loaders, config, args, epochs):
    train, validation, _ = loaders
    settings = config["imu_to_emg"]
    # Baseline/student start from identical EMG weights and see matching RNG streams.
    seed_all(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, stale, history = float("inf"), 0, []
    checkpoint = args.output_dir / f"{name}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for w in windows(train, config, args.device, args.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            out = model(w[modality], w["time_mask"])
            loss = task_loss(out, w)
            if teacher is not None:
                with torch.no_grad():
                    target = teacher(w["imu"], w["time_mask"])
                loss = loss + transfer_loss(out, target, w,
                    args.latent_weight, args.output_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {name}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        if not losses:
            raise ValueError("No usable training windows")
        val = evaluate(model, modality, validation, config, args.device)
        metrics = val["overall"]
        score = metrics["screen_px"] + settings["joint_px_per_cm"] * metrics["path_cm"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "validation": val, "selection_score": score})
        print(f"{name} {epoch}: loss={np.mean(losses):.4f} | "
              f"val screen={metrics['screen_px']:.1f}px path={metrics['path_cm']:.2f}cm",
              flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({"format": "imu_to_emg_v1", "state_dict": model.state_dict(),
                        "model_args": settings["model_args"][modality],
                        "input_modality": modality, "config": config,
                        "epoch": epoch, "validation": val,
                        "selection_score": score}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-prefixes", nargs="+", required=True,
                        help="Real ancestor folder prefixes, including hashed session folders")
    parser.add_argument("--config", default=str(ROOT / "configs/tracked_soft_routed_complete_reach.yaml"))
    parser.add_argument("--cache-dir", default="artifacts/imu_to_emg_cache")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/imu_to_emg"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--output-weight", type=float, default=0.25)
    args = parser.parse_args()
    if min(args.epochs, args.teacher_epochs, args.patience) < 1:
        parser.error("epochs and patience must be positive")
    if min(args.latent_weight, args.output_weight) < 0:
        parser.error("distillation weights must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is not empty; choose a new directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for a smoke test")
    seed_all(args.seed)
    config = load_config(args.config)
    config["seed"] = args.seed
    config["data"]["include_session_prefixes"] = args.session_prefixes
    config["imu_to_emg"] = {
        "context_ms": 2000, "steps": 16, "cutoffs": 4,
        "lead_ms": [0, 400], "evaluation_leads_ms": [0, 50, 100, 200, 300, 400],
        "joint_px_per_cm": 5., "latent_weight": args.latent_weight,
        "output_weight": args.output_weight,
    }
    loaders = candidate.build_candidate_loaders(config, args.root, args.cache_dir)
    # Refuse ambiguous canvas size: lateral scale errors must not be hidden.
    for loader in loaders:
        for batch in loader:
            if batch is not None and ("canvas" not in batch or not torch.isfinite(batch["canvas"]).all()
                                      or (batch["canvas"] <= 0).any()):
                raise ValueError("dataset must provide valid per-trial canvas dimensions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate.save_candidate_calibration(args.output_dir)
    splits = {name: [str(p) for p in loader.dataset.trials]
              for name, loader in zip(["train", "validation", "test"], loaders)}
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not (set(splits["validation"]) & set(splits["test"]))
    (args.output_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    model_args = {modality: {"input_dim": count(config["data"])}
                  for modality, count in [("emg", emg_feature_count), ("imu", imu_feature_count)]}
    config["imu_to_emg"]["model_args"] = model_args
    teacher = WearableTaskNetwork(**model_args["imu"]).to(args.device)
    emg = WearableTaskNetwork(**model_args["emg"]).to(args.device)
    student = copy.deepcopy(emg)
    teacher = fit("imu_teacher", teacher, "imu", None, loaders, config, args, args.teacher_epochs)
    teacher.requires_grad_(False)
    baseline = fit("emg_baseline", emg, "emg", None, loaders, config, args, args.epochs)
    student = fit("emg_student", student, "emg", teacher, loaders, config, args, args.epochs)
    # Test only after all validation-based checkpoint selections are complete.
    results = {name: evaluate(model, modality, loaders[2], config, args.device)
               for name, model, modality in [("imu_teacher", teacher, "imu"),
                   ("emg_baseline", baseline, "emg"), ("emg_student", student, "emg")]}
    results["protocol"] = "within-selected-candidate trial split; not unseen-person evaluation"
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    for name in ["imu_teacher", "emg_baseline", "emg_student"]:
        print(name, results[name]["overall"])
    print("Inference checkpoint:", args.output_dir / "emg_student.pt")


if __name__ == "__main__":
    main()
