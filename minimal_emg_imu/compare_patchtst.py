#!/usr/bin/env python3
"""Compare the selected minimal model with its PatchTST encoder ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    ("gripper_macro_f1", True),
    ("position_cm", False),
    ("click_pixel_error", False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patchtst", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())["emg+imu"]
    patchtst = json.loads(args.patchtst.read_text())["emg+imu"]
    rows = {}
    for key, higher_better in METRICS:
        before, after = baseline[key], patchtst[key]
        improvement = after - before if higher_better else before - after
        rows[key] = {"baseline": before, "patchtst": after,
                     "improvement": improvement}
    for horizon in (50, 100, 200):
        key = str(horizon)
        before = baseline["future_pose_by_ms"][key]["position_cm"]
        after = patchtst["future_pose_by_ms"][key]["position_cm"]
        rows[f"future_{horizon}ms_cm"] = {
            "baseline": before, "patchtst": after,
            "improvement": before - after}
    wins = sum(value["improvement"] > 0 for value in rows.values())
    verdict = ("PatchTST improves the majority of selected metrics."
               if wins > len(rows) / 2 else
               "PatchTST does not improve the majority of selected metrics.")
    report = {"metrics": rows, "wins": wins, "comparisons": len(rows),
              "verdict": verdict}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

