#!/usr/bin/env python3
"""Train V2 motion/pixels conditioned by a frozen proven classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.models.reach_grasp_neuro_classifier_attention import (
    NeuroClassifierConditionedAttention,
)
from emg_touch.models.reach_grasp_state_attention_v2 import StateConditionedAttentionV2
from emg_touch.models.state_attention_loss import masked_mean
from scripts import train_gripper_state_attention_v2 as v2
from scripts import train_gripper_state_pose as base


active_normalization = None


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--state-distillation-weight", type=float, default=.25)
    parser.add_argument("--distillation-temperature", type=float, default=2.)
    parser.add_argument("--state-condition-weight", type=float, default=.5)
    parser.add_argument("--react-weight-v2", dest="react_weight", type=float, default=.2)
    parser.add_argument("--masked-weight-v2", dest="masked_weight", type=float, default=.15)
    parser.add_argument("--state-stability-weight", type=float, default=.03)
    parser.add_argument("--future-consistency-weight", type=float, default=.1)
    options, remaining = parser.parse_known_args()
    if options.state_distillation_weight < 0 or options.distillation_temperature <= 0:
        parser.error("distillation weight must be nonnegative and temperature positive")
    if min(options.state_condition_weight, options.react_weight, options.masked_weight,
           options.state_stability_weight, options.future_consistency_weight) < 0:
        parser.error("all auxiliary weights must be nonnegative")

    teacher_state = torch.load(
        options.classifier_checkpoint, map_location="cpu", weights_only=False)
    if teacher_state.get("format") != "gripper_neuromuscular_future_v1":
        parser.error("--classifier-checkpoint must be an uncalibrated "
                     "gripper_neuromuscular_future_v1 checkpoint")
    if teacher_state["model_args"].get("modality") != "emg+imu":
        parser.error("the classifier checkpoint must be the fused emg+imu model")

    # Reuse V2's tested task objective for its private auxiliary state head.
    v2.extra = argparse.Namespace(
        state_condition_weight=options.state_condition_weight,
        react_weight=options.react_weight,
        masked_weight=options.masked_weight,
        state_stability_weight=options.state_stability_weight,
        future_consistency_weight=options.future_consistency_weight,
    )

    sys.argv = [sys.argv[0], *remaining]
    defaults = {"--models": "emg+imu", "--pixel-architecture": "direct",
                "--pixel-weight": ".35", "--position-weight": "1.0",
                "--orientation-weight": "0", "--final-pose-weight": "0",
                "--future-pose-ms": "200", "--future-pose-weight": ".5"}
    present = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
    for option, value in defaults.items():
        if option not in present:
            sys.argv.extend((option, value))
    if "--models" in sys.argv:
        index = sys.argv.index("--models") + 1
        selected = []
        while index < len(sys.argv) and not sys.argv[index].startswith("--"):
            selected.append(sys.argv[index]); index += 1
        if selected != ["emg+imu"]:
            parser.error("this hybrid trains only --models emg+imu")

    def model_factory(**model_args):
        if active_normalization is None:
            raise RuntimeError("motion normalization was not initialized")
        teacher = NeuromuscularFutureGripperPoseModel(**teacher_state["model_args"])
        teacher.load_state_dict(teacher_state["state_dict"])
        motion = StateConditionedAttentionV2(**model_args)
        return NeuroClassifierConditionedAttention(
            teacher, motion, teacher_state["normalization"], active_normalization)

    def hybrid_loss(model, output, batch, class_weight, args):
        # V2's own classifier remains an auxiliary student. The frozen teacher
        # is the authoritative output and the actual conditioning probability.
        student = dict(output)
        student["gripper_state_logits"] = output["student_gripper_state_logits"]
        student["state_probability"] = output["student_state_probability"]
        task = v2.task_loss(model.motion, student, batch, class_weight, args)
        valid = batch["emg_usable"] & batch["imu_usable"]
        temperature = options.distillation_temperature
        teacher_probability = (output["gripper_state_logits"] / temperature).softmax(-1)
        student_log_probability = (
            output["student_gripper_state_logits"] / temperature).log_softmax(-1)
        divergence = F.kl_div(
            student_log_probability, teacher_probability, reduction="none").sum(-1)
        distillation = masked_mean(divergence, valid) * temperature ** 2
        return task + options.state_distillation_weight * distillation

    old_model, old_loss = base.GripperStatePoseModel, base.loss
    old_evaluate, old_train_one = base.evaluate, base.train_one

    def position_only(*args, **kwargs):
        report = old_evaluate(*args, **kwargs)
        report["orientation_deg"] = None
        for value in report.get("future_pose_by_ms", {}).values():
            value["orientation_deg"], value["valid_orientation_frames"] = None, 0
        return report

    def train_with_normalization(modality, args, train, validation, stats,
                                 class_weight, settings, augmenter=None):
        global active_normalization
        active_normalization = stats
        return old_train_one(modality, args, train, validation, stats,
                             class_weight, settings, augmenter)

    base.GripperStatePoseModel, base.loss = model_factory, hybrid_loss
    base.evaluate, base.train_one = position_only, train_with_normalization
    try:
        base.main()
    finally:
        base.GripperStatePoseModel, base.loss = old_model, old_loss
        base.evaluate, base.train_one = old_evaluate, old_train_one

    output_arg = next((x.split("=", 1)[1] for x in remaining
                       if x.startswith("--output-dir=")), None)
    if output_arg is None and "--output-dir" in sys.argv:
        output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg or "runs/gripper_state_pose_seed42")
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"].update({
        "architecture": "frozen-neuromuscular-classifier-conditioned-attention",
        "classifier_checkpoint": str(options.classifier_checkpoint),
        "classifier_frozen": True,
        "state_distillation_weight": options.state_distillation_weight,
        "distillation_temperature": options.distillation_temperature,
        "pixel_axes": "x/width and y/height independently; loss measured in pixels",
        "heads": ["frozen classifier open/close", "current XYZ", "pixel XY",
                  "future XYZ (training regularizer)"],
    })
    results_path.write_text(json.dumps(results, indent=2))
    for path in output_dir.glob("*_best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "format": "neuro_classifier_state_attention_v1",
            "architecture": results["protocol"]["architecture"],
            "classifier_model_args": teacher_state["model_args"],
            "classifier_normalization": teacher_state["normalization"],
            "classifier_preprocessing": teacher_state["preprocessing"],
            "classifier_source": str(options.classifier_checkpoint),
        })
        torch.save(checkpoint, path)
    print(f"hybrid protocol written to {results_path}")


if __name__ == "__main__":
    main()
