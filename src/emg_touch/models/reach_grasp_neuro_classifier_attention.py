"""Frozen proven classifier conditioning the V2 position/future/pixel model."""
from __future__ import annotations

import torch
from torch import nn


class NeuroClassifierConditionedAttention(nn.Module):
    """Use neuromuscular-future state probabilities as V2 conditioning.

    Inputs arrive normalized for the V2 training split. The affine conversion
    below reconstructs physical preprocessed features and re-normalizes them
    for the frozen classifier, avoiding an otherwise silent normalization bug.
    """

    def __init__(self, classifier, motion, classifier_normalization,
                 motion_normalization):
        super().__init__()
        self.classifier, self.motion = classifier, motion
        self.modality = motion.modality
        if self.modality != "emg+imu":
            raise ValueError("the hybrid requires the fused emg+imu motion model")
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)
        for modality, width in (("emg", 8), ("imu", 24)):
            teacher = classifier_normalization[modality]
            student = motion_normalization[modality]
            teacher_mean = torch.as_tensor(teacher["mean"], dtype=torch.float32)
            teacher_std = torch.as_tensor(teacher["std"], dtype=torch.float32)
            student_mean = torch.as_tensor(student["mean"], dtype=torch.float32)
            student_std = torch.as_tensor(student["std"], dtype=torch.float32)
            self.register_buffer(
                f"{modality}_scale", student_std / teacher_std)
            self.register_buffer(
                f"{modality}_offset", (student_mean - teacher_mean) / teacher_std)
            if getattr(self, f"{modality}_scale").numel() != width:
                raise ValueError(f"invalid {modality} normalization width")
        self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        # Dropout in a frozen teacher would make the conditioning target move.
        self.classifier.eval()
        return self

    def _classifier_inputs(self, emg, imu):
        teacher_emg, teacher_imu = emg.clone(), imu.clone()
        teacher_emg[..., :8] = (emg[..., :8] * self.emg_scale + self.emg_offset)
        teacher_imu[..., :24] = (imu[..., :24] * self.imu_scale + self.imu_offset)
        return teacher_emg, teacher_imu

    def forward(self, emg, imu):
        teacher_emg, teacher_imu = self._classifier_inputs(emg, imu)
        with torch.no_grad():
            classifier_output = self.classifier(teacher_emg, teacher_imu)
            classifier_logits = classifier_output["gripper_state_logits"]
            classifier_probability = classifier_logits.softmax(-1)
        result = self.motion(emg, imu, classifier_probability)
        result["student_gripper_state_logits"] = result["gripper_state_logits"]
        result["student_state_probability"] = result["state_probability"]
        result["gripper_state_logits"] = classifier_logits
        result["state_probability"] = classifier_probability
        result["classifier_state_probability"] = classifier_probability
        return result
