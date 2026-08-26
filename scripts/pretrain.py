#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from emg_touch.config import load_config, save_config
from emg_touch.data.loaders import build_loaders
from emg_touch.models.pretraining import MultimodalMaskedPretrainer
from emg_touch.training import backward_step, optimizer_for
from emg_touch.utils import AverageMeter, choose_device, move_batch_to_device, seed_everything


def evaluate(model, loader, device) -> float:
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch["emg"], batch["emg_mask"], batch["imu"], batch["imu_mask"])
            meter.update(float(output["loss"]), batch["emg"].size(0))
    return meter.average


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked-patch and cross-modal EMG/IMU pretraining")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, _ = build_loaders(config, args.split, args.scaler)
    model = MultimodalMaskedPretrainer(config).to(device)
    optimizer = optimizer_for(model, config)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = Path(config["paths"]["output_dir"]) / "pretraining"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    best, stale = float("inf"), 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"pretrain {epoch}"):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(
                    batch["emg"], batch["emg_mask"], batch["imu"], batch["imu_mask"], args.mask_ratio
                )
            backward_step(
                output["loss"], model, optimizer, scaler,
                float(config["training"]["gradient_clip_norm"]),
            )
            meter.update(float(output["loss"].detach()), batch["emg"].size(0))
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train={meter.average:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {
                    "emg_encoder": model.emg_encoder.state_dict(),
                    "imu_encoder": model.imu_encoder.state_dict(),
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break


if __name__ == "__main__":
    main()
