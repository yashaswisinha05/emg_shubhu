#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from emg_touch.checkpointing import load_model_state, save_checkpoint
from emg_touch.config import load_config, save_config
from emg_touch.data.loaders import build_loaders
from emg_touch.models.factory import build_model
from emg_touch.objectives import baseline_objective
from emg_touch.training import backward_step, evaluate_model, forward_model, optimizer_for, validation_huber
from emg_touch.utils import AverageMeter, choose_device, move_batch_to_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EMG-only TCN or PatchTST baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--kind", choices=["tcn", "patchtst"], required=True)
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, test_loader = build_loaders(config, args.split, args.scaler)
    model = build_model(args.kind, config).to(device)
    optimizer = optimizer_for(model, config)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = Path(config["paths"]["output_dir"]) / args.kind
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    best, stale = float("inf"), 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"{args.kind} {epoch}"):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = forward_model(model, batch, args.kind)
                loss = baseline_objective(outputs, batch)
            backward_step(
                loss, model, optimizer, scaler,
                float(config["training"]["gradient_clip_norm"]),
            )
            meter.update(float(loss.detach()), batch["emg"].size(0))
        val_loss = validation_huber(model, val_loader, args.kind, device)
        print(f"epoch={epoch} train={meter.average:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch, config,
                {"val_huber": val_loss}, args.kind,
            )
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    load_model_state(model, output_dir / "best.pt")
    metrics, _ = evaluate_model(model, test_loader, args.kind, device)
    save_json(metrics, output_dir / "test_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
