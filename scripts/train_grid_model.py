#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from emg_touch.checkpointing import load_model_state, save_checkpoint
from emg_touch.config import load_config, save_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.grid_training import (
    continual_enabled,
    continual_prefix_batches,
    evaluate_grid_model,
    grid_data_report,
    grid_point_loss,
    grid_validation_scores,
    make_continual_training_batch,
)
from emg_touch.models.grid_point import GRID_MODEL_KINDS, build_grid_model
from emg_touch.training import backward_step, optimizer_for
from emg_touch.utils import (
    AverageMeter,
    choose_device,
    load_json,
    move_batch_to_device,
    save_json,
    seed_everything,
)


def load_grid_imu(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "grid_imu":
        raise ValueError(
            f"Expected grid_imu checkpoint, found {checkpoint.get('model_kind')!r}"
        )
    target = model.fusion if hasattr(model, "fusion") else model
    target.imu_model.load_state_dict(checkpoint["model_state"], strict=True)
    print(f"Initialized exact grid IMU model from {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a calibrated CenterNet-style grid-and-offset touch model"
    )
    parser.add_argument("--config", default="configs/grid_point.yaml")
    parser.add_argument("--kind", choices=GRID_MODEL_KINDS, required=True)
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--pretrained-imu")
    parser.add_argument("--freeze-base-imu", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
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
    if (args.pretrained_imu or args.freeze_base_imu) and args.kind not in (
        "grid_fusion",
        "grid_fusion_physics",
        "grid_fusion_physics3",
        "grid_fusion_vae",
    ):
        raise ValueError(
            "Pretrained/frozen base IMU options require --kind grid_fusion, "
            "grid_fusion_physics, grid_fusion_physics3, or grid_fusion_vae"
        )
    if args.freeze_base_imu and not args.pretrained_imu:
        raise ValueError("--freeze-base-imu requires --pretrained-imu")

    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, test_loader = build_grid_trajectory_loaders(
        config, split_path, scaler_path
    )
    model = build_grid_model(args.kind, config).to(device)
    if args.pretrained_imu:
        load_grid_imu(model, args.pretrained_imu)
    if args.freeze_base_imu:
        frozen = model.fusion if hasattr(model, "fusion") else model
        for parameter in frozen.imu_model.parameters():
            parameter.requires_grad_(False)
        print("Frozen exact pretrained grid IMU model")

    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 4)),
        min_lr=float(config["training"].get("minimum_learning_rate", 1e-6)),
    )
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = Path(config["paths"]["output_dir"]) / args.kind
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    save_json(
        grid_data_report(train_loader, val_loader, test_loader),
        output_dir / "data_report.json",
    )
    split = load_json(split_path)
    fold = int(split["fold"]) if "fold" in split else None
    configuration = str(split.get("configuration", "unknown"))
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
    selection_metric = str(
        config["training"].get("selection_metric", "total_loss")
    )
    if selection_metric not in {
        "total_loss",
        "mean_pixel_error",
        "weighted_mean_pixel_error",
        "endpoint_mean_pixel_error",
    }:
        raise ValueError(
            "training.selection_metric must be total_loss, mean_pixel_error, "
            "weighted_mean_pixel_error, or endpoint_mean_pixel_error"
        )
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        if args.freeze_base_imu:
            frozen = model.fusion if hasattr(model, "fusion") else model
            frozen.imu_model.eval()
        meters = {
            name: AverageMeter()
            for name in (
                "loss",
                "heatmap_loss",
                "offset_loss",
                "pixel_loss",
                "radial_loss",
                "transport_loss",
                "physics_loss",
                "physics_residual_loss",
                "affine_penalty",
                "nll_loss",
                "kl_loss",
            )
        }
        for batch in tqdm(
            train_loader, desc=f"{configuration} fold={fold} {args.kind} {epoch}"
        ):
            if continual_enabled(config):
                batch = make_continual_training_batch(batch, config)
            device_batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = grid_point_loss(model(device_batch), device_batch, config)
            backward_step(losses["loss"], model, optimizer, scaler, gradient_clip)
            count = int(device_batch["target"].size(0))
            for name, meter in meters.items():
                meter.update(float(losses[name].detach()), count)

        val_scores = grid_validation_scores(model, val_loader, device, config)
        selection_value = val_scores[selection_metric]
        scheduler.step(selection_value)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "val_loss": val_scores["total_loss"],
            "val_mean_pixel_error": val_scores["mean_pixel_error"],
            "val_weighted_mean_pixel_error": val_scores[
                "weighted_mean_pixel_error"
            ],
            "val_endpoint_mean_pixel_error": val_scores[
                "endpoint_mean_pixel_error"
            ],
            "selection_value": selection_value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        record.update({f"train_{name}": meter.average for name, meter in meters.items()})
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        physics_note = ""
        if meters["physics_loss"].average > 0.0 or meters["affine_penalty"].average > 0.0:
            physics_note = (
                f" physics={meters['physics_loss'].average:.4f} "
                f"affine_penalty={meters['affine_penalty'].average:.4f}"
            )
        if meters["nll_loss"].average != 0.0:
            physics_note += f" nll={meters['nll_loss'].average:.4f}"
        if meters["kl_loss"].average != 0.0:
            physics_note += f" kl={meters['kl_loss'].average:.4f}"
        print(
            f"epoch={epoch} train={meters['loss'].average:.6f} "
            f"val={val_scores['total_loss']:.6f} "
            f"val_px={val_scores['mean_pixel_error']:.2f} "
            f"select={selection_metric}:{selection_value:.2f} "
            f"lr={record['learning_rate']:.2e}"
            f"{physics_note}"
        )
        if selection_value < best:
            best = selection_value
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                {
                    "val_loss": val_scores["total_loss"],
                    "val_mean_pixel_error": val_scores["mean_pixel_error"],
                    "val_weighted_mean_pixel_error": val_scores[
                        "weighted_mean_pixel_error"
                    ],
                    "val_endpoint_mean_pixel_error": val_scores[
                        "endpoint_mean_pixel_error"
                    ],
                    "selection_metric": selection_metric,
                    "selection_value": selection_value,
                },
                args.kind,
            )
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"Early stopping after {epoch} epochs; "
                    f"best {selection_metric}={best:.6f}"
                )
                break

    load_model_state(model, output_dir / "best.pt")
    val_metrics, val_records = evaluate_grid_model(
        model, val_loader, args.kind, device, config, fold=fold
    )
    test_metrics, test_records = evaluate_grid_model(
        model, test_loader, args.kind, device, config, fold=fold
    )
    continual_metrics: dict[str, dict[str, float]] = {"touch": test_metrics}
    if continual_enabled(config):
        for cutoff in config.get("continual", {}).get(
            "evaluation_cutoffs_s", [0.0, 0.2, 0.4]
        ):
            cutoff = float(cutoff)
            try:
                cutoff_metrics, cutoff_records = evaluate_grid_model(
                    model,
                    continual_prefix_batches(test_loader, cutoff, config),
                    args.kind,
                    device,
                    config,
                    fold=fold,
                )
            except ValueError as error:
                if "Evaluation loader is empty" not in str(error):
                    raise
                print(f"Skipping {cutoff:.1f}s: no eligible test trajectories")
                continue
            label = f"{cutoff:.1f}s"
            continual_metrics[label] = cutoff_metrics
            test_records.extend(cutoff_records)
            print(f"test {label}={cutoff_metrics}")
    pd.DataFrame(val_records).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(test_records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(val_metrics, output_dir / "validation_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")
    if continual_enabled(config):
        save_json(continual_metrics, output_dir / "continual_metrics.json")
    print(f"validation={val_metrics}")
    print(f"test={test_metrics}")
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
