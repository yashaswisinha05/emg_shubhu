#!/usr/bin/env python3
"""Train the robust model with causal masked-EMG representation learning."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp_hybrid as hybrid
from emg_touch.models.reach_grasp_masked_reconstruction import (
    MaskedReconstructionReachGraspModel)


MODEL_EXTRA_ARGS = {
    "width": 128, "patch": 16, "stride": 4,
    "layers": 4, "heads": 4, "dropout": .1,
    "event_time_bins": 6,
}


def has(name):
    return any(value == name or value.startswith(name + "=")
               for value in sys.argv[1:])


def add_default(name, value=None):
    if not has(name):
        sys.argv[1:1] = [name] if value is None else [name, str(value)]


def main():
    original = hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT, hybrid.MODEL_EXTRA_ARGS
    try:
        hybrid.MODEL_CLASS = MaskedReconstructionReachGraspModel
        hybrid.MODEL_FORMAT = "reach_grasp_masked_reconstruction_v1"
        hybrid.MODEL_EXTRA_ARGS = dict(MODEL_EXTRA_ARGS)
        add_default("--annotation-aware")
        add_default("--uncertainty-ms", 200)
        add_default("--selection-tolerance-ms", 200)
        add_default("--orientation-weight", .5)
        add_default("--physiological-augmentation")
        add_default("--augmentation-strength", 1.)
        add_default("--event-time-weight", .35)
        add_default("--pose-uncertainty-weight", 1.)
        add_default("--masked-emg-reconstruction-weight", .1)
        add_default("--emg-reconstruction-mask-ratio", .35)
        add_default("--emg-reconstruction-mask-span", 5)
        add_default("--output-dir", "runs/reach_grasp_masked_reconstruction_seed42")
        hybrid.main()
    finally:
        hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT, hybrid.MODEL_EXTRA_ARGS = original


if __name__ == "__main__":
    main()
