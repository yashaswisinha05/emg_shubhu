#!/usr/bin/env python3
"""Read-only model diagnostics: where does EMG add to an IMU correction control?"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_imu_to_emg as base
from scripts.train_emg_delay_correction import Correction, delayed_windows
from emg_touch.models.imu_to_emg import WearableTaskNetwork

METRICS = ["screen_px", "path_cm", "endpoint_cm"]
FIELDS = ["emg", "imu", "emg_mask", "time_mask", "target", "canvas_size",
          "trajectory_target", "endpoint_3d_target"]


def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != "emg_delay_correction_v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    teacher = WearableTaskNetwork(**state["teacher_args"])
    model = Correction(teacher, state["emg_dim"], state["control"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval().requires_grad_(False)
    model.teacher.encoder.flatten_parameters()
    return model, state


def error(out, w):
    return torch.stack([
        ((out["screen"] - w["target"]) * w["canvas_size"]).norm(dim=-1),
        (out["path"] - w["trajectory_target"]).norm(dim=-1).mean(-1) * 100,
        (out["path"][:, -1] - w["endpoint_3d_target"]).norm(dim=-1) * 100,
    ], -1).cpu().numpy()


def summarize(rows, bootstrap=1000, seed=42):
    """Average repeated leads within a trial before paired trial bootstrap."""
    by_trial = defaultdict(list)
    for row in rows:
        by_trial[row["trial"]].append(row)
    report = {"trials": len(by_trial), "observations": len(rows), "errors": {}, "gains": {}}
    models = ["imu_baseline", "imu_control", "emg", "zero_emg", "shuffled_emg"]
    means = {name: np.asarray([
        np.mean([r[name] for r in group], axis=0) for group in by_trial.values()
    ]) for name in models}
    for name in models:
        report["errors"][name] = dict(zip(METRICS, means[name].mean(0).tolist()))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(by_trial), size=(bootstrap, len(by_trial)))
    for name in ["imu_baseline", "imu_control", "zero_emg", "shuffled_emg"]:
        delta = means[name] - means["emg"]
        ci = np.quantile(delta[draws].mean(1), [.025, .975], axis=0)
        report["gains"][name + "_minus_emg"] = {
            metric: {"mean": float(delta[:, i].mean()),
                     "ci95": ci[:, i].tolist() if len(by_trial) >= 2 else None}
            for i, metric in enumerate(METRICS)}
    return report


@torch.no_grad()
def inspect_group(emg, control, loader, config, device, delay, lead, recording, batch_size):
    windows = list(delayed_windows(loader, config, "cpu", 0, delay, lead))
    if not windows:
        return []
    data = {key: torch.cat([w[key] for w in windows]) for key in FIELDS}
    paths = [path for w in windows for path in w["source_paths"]]
    if len(paths) < 2:
        raise ValueError(f"Need at least two usable trials for within-recording shuffle: {recording}, {lead}ms")
    # One fixed derangement across the entire recording at the SAME lead.
    # No self-pairs, including final batches of size one. Donor masks move too.
    donor = torch.arange(len(paths)).roll(1)
    rows = []
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        w = {key: value[start:end].to(device) for key, value in data.items()}
        shuffled = dict(w)
        for key in ["emg", "emg_mask"]:
            shuffled[key] = data[key][donor[start:end]].to(device)
        control_w = dict(w)
        # Control was trained at delay zero; preserve its original mask/context.
        control_w["emg_mask"] = w["time_mask"]
        predictions = {
            "imu_baseline": emg.teacher(w["imu"], w["time_mask"]),
            "imu_control": control(control_w), "emg": emg(w),
            "zero_emg": emg(w, "zero"), "shuffled_emg": emg(shuffled),
        }
        errors = {key: error(out, w) for key, out in predictions.items()}
        for i, path in enumerate(paths[start:end]):
            x, y = w["target"][i].cpu().tolist()
            region = ("left" if x < .5 else "right") + ("_top" if y < .5 else "_bottom")
            rows.append({"trial": path, "recording": recording, "lead_ms": lead,
                         "region": region, "shuffle_donor": paths[int(donor[start + i])],
                         **{key: value[i].tolist() for key, value in errors.items()}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--run-dir", type=Path, default=Path("runs/emg_delay_correction_seed42"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/emg_increment_diagnostics"))
    p.add_argument("--cache-dir", default="artifacts/imu_to_emg_four_recordings_cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--extra-leads-ms", nargs="*", type=int, default=[])
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    if args.bootstrap < 1 or any(v < 0 for v in args.extra_leads_ms):
        p.error("positive bootstrap count and nonnegative leads required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("choose a new empty output directory")
    emg, state = load_model(args.run_dir / "selected_emg.pt", args.device)
    control, control_state = load_model(args.run_dir / "imu_control.pt", args.device)
    if state["control"] or not control_state["control"] or control_state["delay_ms"] != 0:
        p.error("expected selected EMG model and zero-delay IMU-only control")
    for key, value in emg.teacher.state_dict().items():
        if not torch.equal(value, control.teacher.state_dict()[key]):
            p.error("EMG and control do not share the identical frozen IMU baseline")
    if state["config"] != control_state["config"]:
        p.error("checkpoint configurations differ")
    config = copy.deepcopy(state["config"])
    base.seed_all(config["seed"])
    loaders = base.candidate.build_candidate_loaders(config, args.root, args.cache_dir)
    expected = json.loads((args.run_dir / "splits.json").read_text())
    actual = {name: [str(path) for path in loader.dataset.trials]
              for name, loader in zip(["train", "validation", "test"], loaders)}
    if actual != expected:
        p.error("dataset split differs from the original run; use unchanged data/root")
    selected_loader = loaders[1 if args.split == "validation" else 2]
    dataset = selected_loader.dataset
    prefixes = config["data"]["include_session_prefixes"]
    groups = defaultdict(list)
    for index, path in enumerate(dataset.trials):
        recording = base.candidate._candidate_for_path(path, prefixes)
        if recording is None:
            raise ValueError(f"No recording prefix for {path}")
        groups[recording].append(index)
    leads = sorted(set(config["imu_to_emg"]["evaluation_leads_ms"] + args.extra_leads_ms))
    rows = []
    for recording, indices in groups.items():
        loader = DataLoader(Subset(dataset, indices), batch_size=selected_loader.batch_size,
                            collate_fn=selected_loader.collate_fn, shuffle=False)
        for lead in leads:
            found = inspect_group(emg, control, loader, config, args.device,
                                  state["delay_ms"], lead, recording, selected_loader.batch_size)
            rows.extend(found)
            print(f"{recording} lead={lead}ms: {len(found)} observations", flush=True)
    low, high = config["imu_to_emg"]["lead_ms"]
    sections = defaultdict(list)
    for row in rows:
        domain = "trained_range" if low <= row["lead_ms"] <= high else "OUT_OF_TRAINING_RANGE"
        for group in ["overall", "recording/" + row["recording"],
                      "lead/" + str(row["lead_ms"]), "region/" + row["region"],
                      "recording_lead/" + row["recording"] + "/" + str(row["lead_ms"])]:
            sections[domain + "/" + group].append(row)
    report = {key: summarize(value, args.bootstrap, config["seed"]) for key, value in sections.items()}
    if not rows:
        raise ValueError("No usable diagnostic observations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"split": args.split, "delay_ms": state["delay_ms"],
                "warning": "Exploratory subgroups; trial-bootstrap intervals are not adjusted for multiple comparisons or training-seed uncertainty.",
                "metric_order": METRICS, "groups": report}
    (args.output_dir / "report.json").write_text(json.dumps(metadata, indent=2))
    (args.output_dir / "paired_trials.json").write_text(json.dumps(rows, indent=2))
    lines = ["# EMG incremental-information diagnostics", "",
             f"Split: {args.split}. Positive gain means EMG helps. Exploratory only.", "",
             "Bootstrap units are trials, not overlapping lead windows. Intervals are not multiplicity-adjusted.", "",
             "| Group | Trials | EMG vs control px [95% CI] | Shuffle cost px | EMG vs control path cm |",
             "|---|---:|---:|---:|---:|"]
    for key, value in report.items():
        gain = value["gains"]["imu_control_minus_emg"]
        pixel = gain["screen_px"]
        ci = pixel["ci95"]
        interval = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "[unavailable]"
        shuffle = value["gains"]["shuffled_emg_minus_emg"]["screen_px"]["mean"]
        lines.append(f"| {key} | {value['trials']} | {pixel['mean']:+.2f} {interval} | {shuffle:+.2f} | {gain['path_cm']['mean']:+.3f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("Read", args.output_dir / "report.md")


if __name__ == "__main__":
    main()
