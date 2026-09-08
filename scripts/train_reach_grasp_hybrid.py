#!/usr/bin/env python3
"""Train the hybrid patch/local model and calibrate its causal event decoder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import train_reach_grasp as train
from emg_touch.data.annotation_uncertainty import soften_events
from emg_touch.data.hybrid_event_decoder import apply_decoder, calibrate_decoder
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp_hybrid import ReachGraspHybrid

MODEL_CLASS = ReachGraspHybrid
MODEL_FORMAT = "reach_grasp_hybrid_v1"
MODEL_EXTRA_ARGS = {"width": 128, "patch": 16, "stride": 4,
                    "layers": 4, "heads": 4, "dropout": .1}


def has(name):
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def wrapper_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/reach_grasp_hybrid_seed42"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["imu", "emg", "emg+imu"])
    return parser.parse_known_args()[0]


def load_trials(paths, state):
    uncertainty = state["preprocessing"]["annotation_uncertainty_s"]
    return [soften_events(preprocess(Path(path), state["preprocessing"]), uncertainty)
            for path in paths]


def calibrate_run(options):
    splits = json.loads((options.output_dir / "splits.json").read_text())
    results_path = options.output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["hybrid_event_decoding"] = {
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "description": "causal blend of local event heads and confirmed holding-transition pulses"}
    tolerance = None
    for modality in options.models:
        checkpoint = options.output_dir / (modality.replace("+", "_") + "_best.pt")
        state = torch.load(checkpoint, map_location=options.device, weights_only=False)
        model = MODEL_CLASS(**state["model_args"]).to(options.device)
        model.load_state_dict(state["state_dict"])
        model.eval().requires_grad_(False)
        validation_trials = load_trials(splits["validation"], state)
        validation = train.predict(model, validation_trials, state["normalization"], options)
        tolerance = state["event_tolerance_s"]

        def event_score(items, event):
            return train.event_summary(items, event, .5, tolerance)["f1"]

        decoder = calibrate_decoder(validation, state["holding_decoder"], event_score)
        # Decoder is frozen before loading or evaluating the test trials.
        state["hybrid_event_decoder"] = decoder
        torch.save(state, checkpoint)
        test_trials = load_trials(splits["test"], state)
        test = train.predict(model, test_trials, state["normalization"], options)
        combined = apply_decoder(test, decoder)
        transitions = train.transition_predictions(test, state["holding_decoder"])
        report = {
            "decoder": decoder,
            "dedicated_local_heads": train.metrics(
                test, state["event_thresholds"], tolerance),
            "holding_transitions": train.metrics(
                transitions, state["event_thresholds"], tolerance),
            "combined": train.metrics(combined, [.5, .5], tolerance),
            "by_tolerance_ms": {str(ms): train.metrics(combined, [.5, .5], ms / 1000)
                                for ms in [100, 150, 200, 300]}}
        results["hybrid_event_decoding"][modality] = report
        if modality == "emg+imu":
            removed = train.predict(model, test_trials, state["normalization"], options,
                                    zero_emg=True)
            results["hybrid_event_decoding"]["fusion_zero_emg"] = {
                "combined": train.metrics(apply_decoder(removed, decoder), [.5, .5], tolerance),
                "by_tolerance_ms": {str(ms): train.metrics(
                    apply_decoder(removed, decoder), [.5, .5], ms / 1000)
                    for ms in [100, 150, 200, 300]}}
        print(modality, "hybrid decoder", decoder,
              "test", report["combined"]["event_macro_f1"], flush=True)
    results_path.write_text(json.dumps(results, indent=2))
    print("Updated hybrid results:", results_path, flush=True)


def main():
    options = wrapper_arguments()
    original = train.MODEL_CLASS, train.MODEL_FORMAT, train.MODEL_EXTRA_ARGS
    try:
        train.MODEL_CLASS = MODEL_CLASS
        train.MODEL_FORMAT = MODEL_FORMAT
        train.MODEL_EXTRA_ARGS = MODEL_EXTRA_ARGS
        if not has("--annotation-aware"):
            sys.argv[1:1] = ["--annotation-aware"]
        if not has("--uncertainty-ms"):
            sys.argv[1:1] = ["--uncertainty-ms", "200"]
        if not has("--selection-tolerance-ms"):
            sys.argv[1:1] = ["--selection-tolerance-ms", "200"]
        if not has("--output-dir"):
            sys.argv[1:1] = ["--output-dir", str(options.output_dir)]
        train.main()
        # Reparse because defaults may have been inserted above.
        calibrate_run(wrapper_arguments())
    finally:
        train.MODEL_CLASS, train.MODEL_FORMAT, train.MODEL_EXTRA_ARGS = original


if __name__ == "__main__":
    main()
