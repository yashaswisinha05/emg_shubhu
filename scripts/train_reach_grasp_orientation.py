#!/usr/bin/env python3
"""Train the causal hybrid model with VIVE orientation supervision."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp_hybrid as hybrid
from emg_touch.models.reach_grasp_orientation_hybrid import ReachGraspOrientationHybrid


def has(name):
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def main():
    original = hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT
    try:
        hybrid.MODEL_CLASS = ReachGraspOrientationHybrid
        hybrid.MODEL_FORMAT = "reach_grasp_orientation_hybrid_v1"
        if not has("--orientation-weight"):
            sys.argv[1:1] = ["--orientation-weight", "0.5"]
        if not has("--output-dir"):
            sys.argv[1:1] = ["--output-dir", "runs/reach_grasp_orientation_hybrid_seed42"]
        hybrid.main()
    finally:
        hybrid.MODEL_CLASS, hybrid.MODEL_FORMAT = original


if __name__ == "__main__":
    main()
