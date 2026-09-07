#!/usr/bin/env python3
"""Validation-selected causal EMG-delay correction of a frozen IMU teacher."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_imu_to_emg as base
from emg_touch.models.imu_to_emg import WearableTaskNetwork, task_loss


def delayed_windows(loader, config, device, seed, delay_ms, fixed_lead=None):
    """Fetch extra past context, never shift future samples into the input."""
    rate = config["data"]["sample_rate_hz"] / config["data"]["decimation"]
    n = round(config["imu_to_emg"]["context_ms"] * rate / 1000)
    delay = round(delay_ms * rate / 1000)
    extended = copy.deepcopy(config)
    extended["imu_to_emg"]["context_ms"] = (n + delay) * 1000 / rate
    for w in base.windows(loader, extended, device, seed, fixed_lead):
        w["emg_mask"] = w["time_mask"][:, :n]
        w["emg"] = w["emg"][:, :n]
        w["imu"] = w["imu"][:, -n:]
        w["time_mask"] = w["time_mask"][:, -n:]
        yield w


class Correction(nn.Module):
    def __init__(self, teacher, emg_dim, control=False):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.control = control
        latent = teacher.motion[0].out_features
        self.emg = nn.GRU(emg_dim, latent, batch_first=True)
        # Same trainable capacity in the control, but its sensor input is zero.
        self.head = nn.Sequential(nn.Linear(2 * latent, 64), nn.GELU(),
                                  nn.Linear(64, 2 + (teacher.steps - 1) * 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, w, ablation=None):
        with torch.no_grad():
            original = self.teacher(w["imu"], w["time_mask"])
        x, mask = w["emg"], w["emg_mask"]
        if ablation == "shuffle":
            x, mask = x.roll(1, 0), mask.roll(1, 0)
        if self.control or ablation == "zero":
            x = torch.zeros_like(x)
        # Mask state updates: unavailable early history cannot create fake evidence.
        lengths = mask.sum(1)
        compact = torch.zeros_like(x)
        for row in range(len(x)):
            compact[row, :int(lengths[row])] = x[row, mask[row]]
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        self.emg.flatten_parameters()
        _, state = self.emg(packed)
        state = state * (lengths > 0)[None, :, None]
        delta = self.head(torch.cat([original["motion"], state[0]], -1))
        screen_delta = delta[:, :2] * .1
        path_delta = delta[:, 2:].reshape(len(x), self.teacher.steps - 1, 3) * .1
        path_delta = torch.cat([path_delta.new_zeros(len(x), 1, 3), path_delta], 1)
        return {"screen": original["screen"] + screen_delta,
                "path": original["path"] + path_delta,
                "penalty": delta.square().mean()}


@torch.no_grad()
def evaluate(model, loader, config, device, delay, ablation=None):
    model.eval()
    report, pooled = {}, []
    for lead in config["imu_to_emg"]["evaluation_leads_ms"]:
        rows = []
        for w in delayed_windows(loader, config, device, 0, delay, lead):
            out = model(w, ablation)
            rows.append(torch.stack([
                ((out["screen"] - w["target"]) * w["canvas_size"]).norm(dim=-1),
                (out["path"] - w["trajectory_target"]).norm(dim=-1).mean(-1) * 100,
                (out["path"][:, -1] - w["endpoint_3d_target"]).norm(dim=-1) * 100,
            ], -1).cpu())
        if not rows:
            raise ValueError(f"No observations for lead {lead}")
        rows = torch.cat(rows)
        pooled.append(rows)
        report[str(lead)] = rows.mean(0).tolist()
    report["overall"] = dict(zip(["screen_px", "path_cm", "endpoint_cm"],
                                torch.cat(pooled).mean(0).tolist()))
    return report


def score(report):
    m = report["overall"]
    return m["screen_px"] + 5 * m["path_cm"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--cache-dir", default="artifacts/imu_to_emg_four_recordings_cache")
    p.add_argument("--output-dir", type=Path, default=Path("runs/emg_delay_correction"))
    p.add_argument("--delays-ms", type=int, nargs="+", default=[0, 25, 50, 75, 100, 150])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--correction-penalty", type=float, default=.01)
    args = p.parse_args()
    if min(args.delays_ms) < 0 or args.epochs < 1 or args.patience < 1:
        p.error("delays must be nonnegative; epochs/patience must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("use a new, empty output directory")
    ckpt = torch.load(args.teacher_checkpoint, map_location=args.device, weights_only=False)
    if ckpt.get("format") != "imu_to_emg_v1" or ckpt.get("input_modality") != "imu":
        p.error("requires imu_teacher.pt from train_imu_to_emg.py")
    config = copy.deepcopy(ckpt["config"])
    seed = config["seed"]
    base.seed_all(seed)
    loaders = base.candidate.build_candidate_loaders(config, args.root, args.cache_dir)
    saved_splits = json.loads(args.teacher_checkpoint.with_name("splits.json").read_text())
    actual = {name: [str(path) for path in loader.dataset.trials]
              for name, loader in zip(["train", "validation", "test"], loaders)}
    if actual != saved_splits:
        p.error("dataset/split differs from teacher training; use the original dataset and root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.candidate.save_candidate_calibration(args.output_dir)
    (args.output_dir / "splits.json").write_text(json.dumps(actual, indent=2))
    teacher = WearableTaskNetwork(**ckpt["model_args"]).to(args.device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.encoder.flatten_parameters()
    emg_dim = config["imu_to_emg"]["model_args"]["emg"]["input_dim"]
    summary, paths = {}, {}
    for name, delay, control in [("imu_control", 0, True)] + [
            (f"emg_{d}ms", d, False) for d in sorted(set(args.delays_ms))]:
        base.seed_all(seed)
        model = Correction(teacher, emg_dim, control).to(args.device)
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                     lr=args.lr, weight_decay=1e-4)
        # Include the zero-correction initial model in selection: validation need
        # never select a correction worse than doing nothing.
        best_report = evaluate(model, loaders[1], config, args.device, delay)
        best, stale = score(best_report), 0
        path = args.output_dir / f"{name}.pt"
        paths[name] = path

        def save(epoch):
            torch.save({"format": "emg_delay_correction_v1", "state_dict": model.state_dict(),
                        "teacher_args": ckpt["model_args"], "emg_dim": emg_dim,
                        "control": control, "delay_ms": delay, "epoch": epoch,
                        "config": config, "validation": best_report}, path)

        save(0)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for w in delayed_windows(loaders[0], config, args.device, seed + epoch, delay):
                optimizer.zero_grad(set_to_none=True)
                out = model(w)
                loss = task_loss(out, w) + args.correction_penalty * out["penalty"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite correction loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            if not losses:
                raise ValueError("no training observations")
            report = evaluate(model, loaders[1], config, args.device, delay)
            history.append({"epoch": epoch, "validation": report})
            print(name, epoch, report["overall"], flush=True)
            if score(report) < best:
                best, best_report, stale = score(report), report, 0
                save(epoch)
            else:
                stale += 1
            if stale >= args.patience:
                break
        summary[name] = best_report
        (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    selected = min((n for n in summary if n.startswith("emg_")), key=lambda n: score(summary[n]))
    results = {"validation_sweep": summary, "selected_emg": selected,
               "selection": "validation screen_px + 5 * path_cm; includes epoch zero"}
    # No test comparison of all delays: select first, then test the selected EMG
    # branch, capacity control and original teacher on identical observations.
    results["imu_baseline"] = base.evaluate(teacher, "imu", loaders[2], config, args.device)
    for name in ["imu_control", selected]:
        state = torch.load(paths[name], map_location=args.device, weights_only=False)
        model = Correction(teacher, emg_dim, state["control"]).to(args.device)
        model.load_state_dict(state["state_dict"])
        results[name] = evaluate(model, loaders[2], config, args.device, state["delay_ms"])
        if name == selected:
            results["selected_zero_emg"] = evaluate(model, loaders[2], config, args.device,
                                                     state["delay_ms"], "zero")
            results["selected_shuffled_emg"] = evaluate(model, loaders[2], config, args.device,
                                                         state["delay_ms"], "shuffle")
            torch.save(state, args.output_dir / "selected_emg.pt")
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("Selected:", selected, "| Test:", results[selected]["overall"])
    print("IMU baseline:", results["imu_baseline"]["overall"])


if __name__ == "__main__":
    main()
