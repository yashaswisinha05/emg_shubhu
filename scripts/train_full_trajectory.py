#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from emg_touch.checkpointing import load_model_state, save_checkpoint
from emg_touch.config import load_config, save_config
from emg_touch.data.full_trajectory import build_full_trajectory_loaders
from emg_touch.models.full_trajectory import (
    build_full_trajectory_model,
    forward_full_trajectory_model,
)
from emg_touch.training import backward_step, optimizer_for
from emg_touch.trajectory_training import (
    evaluate_full_trajectory_model,
    full_trajectory_data_report,
    full_trajectory_loss,
    full_trajectory_validation_loss,
)
from emg_touch.utils import (
    AverageMeter,
    choose_device,
    load_json,
    move_batch_to_device,
    save_json,
    seed_everything,
)


MODEL_KINDS = ["emg_tcn", "emg_patch", "imu_patch", "emg_residual", "multimodal"]


def load_pretrained_encoder(
    encoder: torch.nn.Module,
    checkpoint_path: str,
    expected_kind: str,
    target_head: torch.nn.Module | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_kind = checkpoint.get("model_kind")
    if checkpoint_kind != expected_kind:
        raise ValueError(
            f"Expected a {expected_kind!r} checkpoint, found {checkpoint_kind!r}"
        )
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(f"No encoder parameters found in {checkpoint_path}")
    encoder.load_state_dict(encoder_state, strict=True)
    if target_head is not None:
        head_prefix = "head."
        head_state = {
            key[len(head_prefix) :]: value
            for key, value in checkpoint["model_state"].items()
            if key.startswith(head_prefix)
        }
        if not head_state:
            raise ValueError(f"No regression-head parameters found in {checkpoint_path}")
        target_head.load_state_dict(head_state, strict=True)
    print(f"Initialized {expected_kind} encoder from {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a deterministic model on complete variable-length trajectories"
    )
    parser.add_argument("--config", default="configs/full_trajectory.yaml")
    parser.add_argument("--kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--split", help="Trajectory-CV split JSON")
    parser.add_argument("--scaler", help="Training-fold robust scaler NPZ")
    parser.add_argument(
        "--pretrained-imu",
        help="Warm-start a fusion model's IMU encoder from an imu_patch checkpoint",
    )
    parser.add_argument(
        "--pretrained-emg",
        help="Warm-start a fusion model's EMG encoder from an emg_patch checkpoint",
    )
    parser.add_argument(
        "--freeze-pretrained-imu",
        action="store_true",
        help="Keep the warm-started IMU encoder/head fixed and learn only fusion/EMG",
    )
    parser.add_argument(
        "--output-dir",
        help="Fold output root; a model-kind subdirectory is created inside it",
    )
    parser.add_argument("--device", help="cpu, mps, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--epochs", type=int, help="Override maximum training epochs")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience")
    args = parser.parse_args()

    config = load_config(args.config)
    split_path = str(Path(args.split or config["paths"]["split_file"]).resolve())
    scaler_path = str(Path(args.scaler or config["paths"]["scaler"]).resolve())
    config["paths"]["split_file"] = split_path
    config["paths"]["scaler"] = scaler_path
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience

    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, test_loader = build_full_trajectory_loaders(
        config, split_path, scaler_path
    )
    model = build_full_trajectory_model(args.kind, config).to(device)
    fusion_kinds = {"emg_residual", "multimodal"}
    if (args.pretrained_imu or args.pretrained_emg) and args.kind not in fusion_kinds:
        raise ValueError(
            "Pretrained encoder options are valid only for emg_residual or multimodal"
        )
    if args.pretrained_imu:
        load_pretrained_encoder(
            model.imu_encoder,
            args.pretrained_imu,
            "imu_patch",
            target_head=model.imu_head,
        )
    if args.pretrained_emg:
        load_pretrained_encoder(model.emg_encoder, args.pretrained_emg, "emg_patch")
    if args.freeze_pretrained_imu:
        if args.kind not in fusion_kinds or not args.pretrained_imu:
            raise ValueError(
                "--freeze-pretrained-imu requires emg_residual/multimodal and "
                "--pretrained-imu"
            )
        for module in (model.imu_encoder, model.imu_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        print("Frozen pretrained IMU encoder and coordinate head")
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 5)),
        min_lr=float(config["training"].get("minimum_learning_rate", 1e-6)),
    )
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    beta = float(config["training"].get("huber_beta", 0.05))

    output_dir = Path(config["paths"]["output_dir"]) / args.kind
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    save_json(
        full_trajectory_data_report(train_loader, val_loader, test_loader),
        output_dir / "data_report.json",
    )
    split = load_json(split_path)
    fold = int(split["fold"]) if "fold" in split else None
    configuration = split.get("configuration", "unknown")
    print(
        f"configuration={configuration} fold={fold} kind={args.kind} device={device} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    best = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    maximum_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    gradient_clip = float(config["training"]["gradient_clip_norm"])

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        if args.freeze_pretrained_imu:
            model.imu_encoder.eval()
            model.imu_head.eval()
        meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"{configuration} fold={fold} {args.kind} {epoch}"):
            device_batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = forward_full_trajectory_model(model, device_batch, args.kind)
                loss = full_trajectory_loss(
                    outputs["prediction"], device_batch["target"], beta=beta
                )
            backward_step(loss, model, optimizer, scaler, gradient_clip)
            meter.update(float(loss.detach()), int(device_batch["target"].size(0)))

        val_loss = full_trajectory_validation_loss(
            model, val_loader, args.kind, device, beta
        )
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_huber": meter.average,
                "val_huber": val_loss,
                "learning_rate": learning_rate,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(
            f"epoch={epoch} train={meter.average:.6f} val={val_loss:.6f} "
            f"lr={learning_rate:.2e}"
        )

        if val_loss < best:
            best = val_loss
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                {"val_huber": val_loss},
                args.kind,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs; best val={best:.6f}")
                break

    load_model_state(model, output_dir / "best.pt")
    val_metrics, val_records = evaluate_full_trajectory_model(
        model, val_loader, args.kind, device, fold=fold
    )
    pd.DataFrame(val_records).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    save_json(val_metrics, output_dir / "validation_metrics.json")
    metrics, records = evaluate_full_trajectory_model(
        model, test_loader, args.kind, device, fold=fold
    )
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(metrics, output_dir / "test_metrics.json")
    print(f"validation={val_metrics}")
    print(metrics)
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
