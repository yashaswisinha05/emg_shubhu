#!/usr/bin/env python3
"""Compare the selected minimal model with the Chronos-2 encoder ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--chronos2", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())["emg+imu"]
    chronos = json.loads(args.chronos2.read_text())["emg+imu"]
    definitions = [
        ("gripper_macro_f1", baseline["gripper_macro_f1"],
         chronos["gripper_macro_f1"], True),
        ("position_cm", baseline["position_cm"], chronos["position_cm"], False),
        ("click_pixel_error", baseline["click_pixel_error"],
         chronos["click_pixel_error"], False),
    ]
    for horizon in (50, 100, 200):
        key = str(horizon)
        definitions.append((
            f"future_{horizon}ms_cm",
            baseline["future_pose_by_ms"][key]["position_cm"],
            chronos["future_pose_by_ms"][key]["position_cm"], False))
    metrics = {}
    for name, before, after, higher_better in definitions:
        improvement = after - before if higher_better else before - after
        metrics[name] = {"baseline": before, "chronos2": after,
                         "improvement": improvement}
    wins = sum(row["improvement"] > 0 for row in metrics.values())
    report = {
        "metrics": metrics, "wins": wins, "comparisons": len(metrics),
        "verdict": ("Chronos-2 improves the majority of selected metrics."
                    if wins > len(metrics) / 2 else
                    "Chronos-2 does not improve the majority of selected metrics."),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
