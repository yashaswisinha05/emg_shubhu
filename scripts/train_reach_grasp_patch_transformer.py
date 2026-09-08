#!/usr/bin/env python3
"""Train the causal Chronos-style patch-transformer reach/grasp experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp as train
from emg_touch.models.reach_grasp_patch_transformer import ReachGraspPatchTransformer


def has(name):
    return any(v == name or v.startswith(name + "=") for v in sys.argv[1:])


def main():
    original = train.MODEL_CLASS, train.MODEL_FORMAT, train.MODEL_EXTRA_ARGS
    try:
        train.MODEL_CLASS = ReachGraspPatchTransformer
        train.MODEL_FORMAT = "reach_grasp_patch_transformer_v1"
        train.MODEL_EXTRA_ARGS = {"width": 128, "patch": 16, "stride": 4,
                                  "layers": 4, "heads": 4, "dropout": .1}
        if not has("--annotation-aware"):
            sys.argv[1:1] = ["--annotation-aware"]
        if not has("--uncertainty-ms"):
            sys.argv[1:1] = ["--uncertainty-ms", "200"]
        if not has("--selection-tolerance-ms"):
            sys.argv[1:1] = ["--selection-tolerance-ms", "200"]
        if not has("--output-dir"):
            sys.argv[1:1] = ["--output-dir", "runs/reach_grasp_patch_transformer_seed42"]
        train.main()
    finally:
        train.MODEL_CLASS, train.MODEL_FORMAT, train.MODEL_EXTRA_ARGS = original


if __name__ == "__main__":
    main()
