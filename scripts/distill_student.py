#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from emg_touch.checkpointing import load_model_state, save_checkpoint
from emg_touch.config import load_config, save_config
from emg_touch.data.loaders import build_loaders
from emg_touch.models.student import EMGStudent
from emg_touch.models.teacher import MultimodalTeacher
from emg_touch.objectives import distillation_objective
from emg_touch.training import backward_step, evaluate_model, forward_model, optimizer_for, validation_huber
from emg_touch.utils import AverageMeter, choose_device, move_batch_to_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill an EMG+IMU teacher into an EMG-only student")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
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
    teacher = MultimodalTeacher(config).to(device)
    load_model_state(teacher, args.teacher_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = EMGStudent(config).to(device)
    student.emg_encoder.load_state_dict(teacher.emg_encoder.state_dict())
    optimizer = optimizer_for(student, config)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = Path(config["paths"]["output_dir"]) / "student"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    best, stale = float("inf"), 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        student.train()
        meter = AverageMeter()
        warmup = max(1, int(config["training"]["kl_warmup_epochs"]))
        kl_beta = float(config["training"]["kl_beta"]) * min(1.0, epoch / warmup)
        for batch in tqdm(train_loader, desc=f"student {epoch}"):
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                teacher_outputs = forward_model(teacher, batch, "teacher")
            with torch.autocast(device_type=device.type, enabled=amp):
                student_outputs = forward_model(student, batch, "student", include_target=True)
                loss, _ = distillation_objective(
                    student_outputs, teacher_outputs, batch, config, kl_beta
                )
            backward_step(
                loss, student, optimizer, scaler,
                float(config["training"]["gradient_clip_norm"]),
            )
            meter.update(float(loss.detach()), batch["emg"].size(0))
        val_loss = validation_huber(student, val_loader, "student", device)
        print(f"epoch={epoch} train={meter.average:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(
                output_dir / "best.pt", student, optimizer, epoch, config,
                {"val_huber": val_loss}, "student",
            )
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    load_model_state(student, output_dir / "best.pt")
    metrics, _ = evaluate_model(student, test_loader, "student", device)
    save_json(metrics, output_dir / "test_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
