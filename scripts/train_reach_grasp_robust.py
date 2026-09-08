#!/usr/bin/env python3
"""Train augmented causal event-horizon and uncertain 6D pose models."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp_hybrid as hybrid
from emg_touch.models.reach_grasp_robust import RobustReachGraspModel

MODEL_EXTRA_ARGS = {
    "width": 128, "patch": 16, "stride": 4,
    "layers": 4, "heads": 4, "dropout": .1,
    "event_time_bins": 6,
}


def has(name):
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def add_default(name, value=None):
    if not has(name):
        sys.argv[1:1] = [name] if value is None else [name, str(value)]


def main():
    original = hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT, hybrid.MODEL_EXTRA_ARGS
    try:
        hybrid.MODEL_CLASS = RobustReachGraspModel
        hybrid.MODEL_FORMAT = "reach_grasp_robust_v1"
        hybrid.MODEL_EXTRA_ARGS = dict(MODEL_EXTRA_ARGS)
        add_default("--annotation-aware")
        add_default("--uncertainty-ms", 200)
        add_default("--selection-tolerance-ms", 200)
        add_default("--orientation-weight", .5)
        add_default("--physiological-augmentation")
        add_default("--augmentation-strength", 1.)
        add_default("--event-time-weight", .35)
        add_default("--pose-uncertainty-weight", 1.)
        add_default("--output-dir", "runs/reach_grasp_robust_seed42")
        hybrid.main()
    finally:
        hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT, hybrid.MODEL_EXTRA_ARGS = original


if __name__ == "__main__":
    main()
