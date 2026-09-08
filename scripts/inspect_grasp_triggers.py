#!/usr/bin/env python3
"""Plot complete trials and tune event-head debounce on validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp as train
from emg_touch.data.reach_grasp import preprocess
from emg_touch.data.annotation_uncertainty import soften_events
from emg_touch.models.reach_grasp import ReachGraspModel
from emg_touch.models.reach_grasp_patch_transformer import ReachGraspPatchTransformer
from emg_touch.models.reach_grasp_hybrid import ReachGraspHybrid
from emg_touch.models.reach_grasp_orientation_hybrid import ReachGraspOrientationHybrid


def model_class(state):
    if state.get("format") == "reach_grasp_annotation_v1":
        return ReachGraspModel
    if state.get("format") == "reach_grasp_patch_transformer_v1":
        return ReachGraspPatchTransformer
    if state.get("format") == "reach_grasp_hybrid_v1":
        return ReachGraspHybrid
    if state.get("format") == "reach_grasp_orientation_hybrid_v1":
        return ReachGraspOrientationHybrid
    raise ValueError("requires an annotation-aware TCN, patch-transformer, or hybrid checkpoint")


def stable_triggers(times, probabilities, valid, high, low_ratio=.5,
                    persistence_s=.06, refractory_s=.4):
    """Past-only persistence + hysteresis. Emit at confirmation, never backdate."""
    armed, since, last, previous = True, None, -np.inf, None
    found = []
    for t, p, good in zip(times, probabilities, valid):
        if previous is not None and t - previous > .05:
            since = None
        previous = t
        if not good or not np.isfinite(p):
            since = None
            continue
        if p < high * low_ratio:
            armed, since = True, None
        if p < high:
            since = None
            continue
        if not armed or t - last < refractory_s:
            since = None
            continue
        if since is None:
            since = t
        if t - since + 1e-9 >= persistence_s:
            found.append(float(t))
            armed, since, last = False, None, t
    return found


def decode(items, thresholds, parameters):
    return [dict(item, detections=[stable_triggers(item["trial"]["time"],
        item["prob"][:, e + 1], item["valid"], thresholds[e], **parameters)
        for e in range(2)]) for item in items]


def choose(items, thresholds, tolerance):
    # Include the existing decoder: validation may prefer no change at all.
    candidates = [None] + [{"low_ratio": ratio, "persistence_s": seconds,
                           "refractory_s": .4}
                          for ratio in [.5, .8] for seconds in [.03, .06, .1]]
    table = []
    for parameters in candidates:
        report = train.metrics(items if parameters is None else decode(items, thresholds, parameters),
                               thresholds, tolerance)
        table.append({"parameters": parameters, "metrics": report})
    selected = max(range(len(table)), key=lambda i: table[i]["metrics"]["event_macro_f1"])
    return table[selected]["parameters"], table


def detections(item, thresholds, parameters):
    if parameters is None:
        return [train.detect(item["trial"]["time"], item["prob"][:, e + 1], thresholds[e],
                             valid=item["valid"])
                for e in range(2)]
    return decode([item], thresholds, parameters)[0]["detections"]


def trigger_audit(items, thresholds, parameters, tolerance):
    """Distinguish duplicate-near-event, nearby-unmatched and distant triggers."""
    report = []
    for item in items:
        for e, found in enumerate(detections(item, thresholds, parameters)):
            reference = float(item["trial"]["events"][e])
            matches = [i for i, t in enumerate(found) if abs(t - reference) <= tolerance]
            matched = min(matches, key=lambda i: abs(found[i] - reference)) if matches else None
            report.append({"trial": item["trial"]["path"], "event": ["grasp", "release"][e],
                "annotation_s": reference, "missed": matched is None,
                "triggers": [{"time_s": t, "offset_ms": (t - reference) * 1000,
                    "category": "matched" if i == matched else
                        "duplicate_in_tolerance" if abs(t - reference) <= tolerance else
                        "nearby_unmatched" if abs(t - reference) <= .5 else "distant_unmatched"}
                    for i, t in enumerate(found)]})
    return report


def plot_trial(item, thresholds, parameters, holding_parameters, uncertainty, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trial = item["trial"]
    prob = np.where(item["valid"][:, None], item["prob"], np.nan)
    t = trial["time"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    fig.suptitle(Path(trial["path"]).name + " — complete recorded trial, causal predictions")
    for c, sensor in enumerate(["S0", "S4", "S8", "S12"]):
        x = np.where(trial["emg_valid"][:, c], trial["emg"][:, c], np.nan)
        axes[0].plot(t, x, label=sensor, linewidth=.8)
    axes[0].set_ylabel("EMG 20 ms RMS\nrecorded units")
    axes[0].legend(ncol=4, loc="upper right")
    for c, sensor in enumerate(["S0", "S4", "S8", "S12"]):
        x = trial["imu"][:, c * 6:c * 6 + 3]
        good = trial["imu_valid"][:, c * 6:c * 6 + 3].all(1)
        axes[1].plot(t, np.where(good, np.linalg.norm(x, axis=1), np.nan), label=sensor, linewidth=.8)
    axes[1].set_ylabel("Acceleration norm\nrecorded units")
    axes[2].step(t, trial["holding"], where="post", label="manual holding label", color="black", alpha=.6)
    axes[2].plot(t, prob[:, 0], label="predicted holding", color="purple")
    for value in [holding_parameters["low"], holding_parameters["high"]]:
        axes[2].axhline(value, color="purple", linestyle=":", alpha=.4)
    holding = train.transition_predictions([item], holding_parameters)[0]["detections"]
    for e, ts in enumerate(holding):
        axes[2].scatter(ts, np.full(len(ts), .95 if e == 0 else .05), marker="^" if e == 0 else "v",
                        label=["holding grasp trigger", "holding release trigger"][e])
    axes[2].set_ylabel("Holding probability")
    axes[2].legend(ncol=2, fontsize=8)
    original = detections(item, thresholds, None)
    improved = detections(item, thresholds, parameters)
    for e, ax in enumerate(axes[3:]):
        ax.plot(t, prob[:, e + 1], color=["tab:green", "tab:red"][e], label="event probability")
        ax.axhline(thresholds[e], color="grey", linestyle=":", label="saved threshold")
        ax.scatter(original[e], np.full(len(original[e]), 1.05), marker="x", color="grey", label="original triggers")
        ax.scatter(improved[e], np.full(len(improved[e]), 1.15), marker="v", color="blue", label="validation-selected triggers")
        ax.set_ylim(-.05, 1.25)
        ax.set_ylabel(["Grasp probability", "Release probability"][e])
        ax.legend(ncol=2, fontsize=8, loc="upper right")
    for ax in axes:
        for e, stamp in enumerate(trial["events"]):
            ax.axvline(stamp, color=["green", "red"][e], linestyle="--", alpha=.7)
            ax.axvspan(stamp - uncertainty, stamp + uncertainty, color=["green", "red"][e], alpha=.07)
        ax.fill_between(t, 0, 1, where=~item["valid"], transform=ax.get_xaxis_transform(), color="grey", alpha=.25)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Seconds from first recorded sample (including buffers); shaded bands = annotation uncertainty")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("runs/reach_grasp_annotation_seed42"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/grasp_trigger_inspection"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--plots", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.plots < 1 or args.batch_size < 1:
        p.error("plots and batch size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("choose a new empty output directory")
    splits = json.loads((args.run_dir / "splits.json").read_text())
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(splits[a]) & set(splits[b]):
            p.error("overlapping saved splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"evaluation_split": args.split,
              "warning": "Exploratory decoder development; validation gains are selection-biased. No physical timing guarantee."}
    for name in ["imu", "emg", "emg_imu"]:
        path = args.run_dir / (name + "_best.pt")
        state = torch.load(path, map_location=args.device, weights_only=False)
        model = model_class(state)(**state["model_args"]).to(args.device)
        model.load_state_dict(state["state_dict"])
        model.eval().requires_grad_(False)
        def load(which):
            return [soften_events(preprocess(Path(file), state["preprocessing"]),
                    state["preprocessing"]["annotation_uncertainty_s"]) for file in splits[which]]
        validation = train.predict(model, load("validation"), state["normalization"], args)
        saved_thresholds, tolerance = state["event_thresholds"], state["event_tolerance_s"]
        # Old checkpoints selected thresholds using zero-filled gaps. Refit the
        # same threshold grid on corrected VALIDATION outputs, not test outputs.
        thresholds = [max([.2, .35, .5, .65, .8], key=lambda high:
            train.event_summary(validation, e, high, tolerance)["f1"]) for e in range(2)]
        parameters, sweep = choose(validation, thresholds, tolerance)
        # Test is only loaded AFTER parameters have been fixed from validation.
        items = validation if args.split == "validation" else train.predict(model, load("test"), state["normalization"], args)
        decoded = items if parameters is None else decode(items, thresholds, parameters)
        holding = train.transition_predictions(items, state["holding_decoder"])
        report[name] = {"evaluation_version": "mask_aware_v2",
            "checkpoint_thresholds": saved_thresholds, "validation_thresholds": thresholds,
            "selected_parameters": parameters, "validation_sweep": sweep,
            "by_tolerance_ms": {str(ms): {
                "checkpoint_thresholds_masked": train.metrics(items, saved_thresholds, ms / 1000),
                "original_heads": train.metrics(items, thresholds, ms / 1000),
                "selected_heads": train.metrics(decoded, thresholds, ms / 1000),
                "holding_transitions": train.metrics(holding, thresholds, ms / 1000)}
                for ms in [100, 150, 200, 300]}}
        audit = {"original": trigger_audit(items, thresholds, None, tolerance),
                 "selected": trigger_audit(items, thresholds, parameters, tolerance)}
        (args.output_dir / (name + "_triggers.json")).write_text(json.dumps(audit, indent=2))
        # Exactly the same seeded trial subset for all models; not best-case selection.
        chosen = np.random.default_rng(args.seed).choice(len(items), min(args.plots, len(items)), replace=False)
        for order, index in enumerate(chosen):
            plot_trial(items[index], thresholds, parameters, state["holding_decoder"],
                state["preprocessing"]["annotation_uncertainty_s"],
                args.output_dir / f"{name}_{order + 1:02d}_{Path(items[index]['trial']['path']).stem}.png")
        print(name, "selected", parameters, report[name]["by_tolerance_ms"]["200"]["selected_heads"]["event_macro_f1"], flush=True)
    (args.output_dir / "decoder_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Trigger stability", "", f"Split: {args.split}. Parameters selected on validation only.", "",
             "| Model | Tolerance ms | Decoder | Event F1 | Grasp FP | Release FP |", "|---|---:|---|---:|---:|---:|"]
    for name in ["imu", "emg", "emg_imu"]:
        for ms, decoders in report[name]["by_tolerance_ms"].items():
            for decoder, values in decoders.items():
                lines.append(f"| {name} | {ms} | {decoder} | {values['event_macro_f1']:.3f} | {values['grasp']['fp']} | {values['release']['fp']} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("Plots and summary:", args.output_dir)


if __name__ == "__main__":
    main()
