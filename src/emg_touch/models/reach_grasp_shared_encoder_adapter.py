"""Shared frozen patch encoders with causal GRU motion adapters."""
from __future__ import annotations

import math

import torch
from torch import nn


class SharedEncoderResidualGRU(nn.Module):
    """Reuse one frozen classifier encoder for state and motion.

    The classifier is evaluated exactly once. Its detached EMG/IMU temporal
    features feed small trainable recurrent adapters; no second raw-signal
    encoder is instantiated.
    """

    classifier_frozen = True

    def __init__(self, classifier, classifier_normalization, motion_normalization,
                 width=128, adapter_layers=2, dropout=.1, future_steps=20,
                 intent_horizons_steps=None):
        super().__init__()
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        horizons = tuple(intent_horizons_steps or ())
        if any(h <= 0 for h in horizons) or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("intent horizons must be positive, sorted and unique")
        self.classifier = classifier
        self.modality = "emg+imu"
        self.future_steps = future_steps
        self.intent_horizons_steps = horizons
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)
        self.classifier.eval()

        for modality, channels in (("emg", 8), ("imu", 24)):
            teacher = classifier_normalization[modality]
            student = motion_normalization[modality]
            teacher_mean = torch.as_tensor(teacher["mean"], dtype=torch.float32)
            teacher_std = torch.as_tensor(teacher["std"], dtype=torch.float32)
            student_mean = torch.as_tensor(student["mean"], dtype=torch.float32)
            student_std = torch.as_tensor(student["std"], dtype=torch.float32)
            self.register_buffer(f"{modality}_scale", student_std / teacher_std)
            self.register_buffer(
                f"{modality}_offset", (student_mean - teacher_mean) / teacher_std)
            if getattr(self, f"{modality}_scale").numel() != channels:
                raise ValueError(f"invalid {modality} normalization width")

        recurrent_dropout = dropout if adapter_layers > 1 else 0.
        self.emg_adapter = nn.GRU(width, width, adapter_layers, batch_first=True,
                                  dropout=recurrent_dropout)
        self.imu_adapter = nn.GRU(width, width, adapter_layers, batch_first=True,
                                  dropout=recurrent_dropout)
        self.emg_adapter_output = nn.Linear(width, width)
        self.imu_adapter_output = nn.Linear(width, width)
        nn.init.zeros_(self.emg_adapter_output.weight)
        nn.init.zeros_(self.emg_adapter_output.bias)
        nn.init.zeros_(self.imu_adapter_output.weight)
        nn.init.zeros_(self.imu_adapter_output.bias)

        self.state_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU())
        self.emg_correction = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width), nn.Tanh())
        self.correction_gate = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.emg_correction[-2].weight)
        nn.init.zeros_(self.emg_correction[-2].bias)
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, -2.)
        self.fused_norm = nn.LayerNorm(width)

        self.position_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.grid_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))
        self.offset_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 18))
        coordinates = torch.tensor([1 / 6, 1 / 2, 5 / 6], dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid_centers", torch.stack((xx.flatten(), yy.flatten()), -1))

        self.future_head = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 3))
        tau = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        self.register_buffer("horizon_basis", torch.stack((
            tau, tau.square(), torch.sin(math.pi * tau), torch.cos(math.pi * tau)), -1))

        self.intent_position_head = self.intent_imu_head = None
        if horizons:
            self.intent_position_head = nn.Sequential(
                nn.Linear(width * 2 + 4, width), nn.GELU(), nn.Linear(width, 3))
            # Deliberately EMG-only: this auxiliary task asks whether muscle
            # activity predicts later mechanics beyond the current IMU state.
            self.intent_imu_head = nn.Sequential(
                nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 24))
            intent_tau = torch.as_tensor(horizons, dtype=torch.float32) / max(horizons)
            intent_basis = torch.stack((intent_tau, intent_tau.square(),
                                        torch.sin(math.pi * intent_tau),
                                        torch.cos(math.pi * intent_tau)), -1)
        else:
            intent_basis = torch.empty(0, 4)
        self.register_buffer("intent_horizon_basis", intent_basis)

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self

    def _classifier_inputs(self, emg, imu):
        teacher_emg, teacher_imu = emg.clone(), imu.clone()
        teacher_emg[..., :8] = emg[..., :8] * self.emg_scale + self.emg_offset
        teacher_imu[..., :24] = imu[..., :24] * self.imu_scale + self.imu_offset
        return teacher_emg, teacher_imu

    def forward(self, emg, imu):
        teacher_emg, teacher_imu = self._classifier_inputs(emg, imu)
        with torch.no_grad():
            source = self.classifier(teacher_emg, teacher_imu)
            state_logits = source["gripper_state_logits"]
            emg_features = source["emg_context_features"]
            imu_features = source["imu_context_features"]
        state_probability = state_logits.softmax(-1)
        state = self.state_embedding(state_probability.detach())

        self.emg_adapter.flatten_parameters()
        self.imu_adapter.flatten_parameters()
        emg_delta = self.emg_adapter_output(self.emg_adapter(emg_features)[0])
        imu_delta = self.imu_adapter_output(self.imu_adapter(imu_features)[0])
        emg_motion = emg_features + emg_delta
        imu_motion = imu_features + imu_delta

        correction = self.emg_correction(torch.cat((emg_motion, state), -1))
        gate = self.correction_gate(torch.cat((imu_motion, emg_motion, state), -1)).sigmoid()
        fused = self.fused_norm(imu_motion + gate * correction)

        grid_logits = self.grid_head(fused)
        grid_offsets = self.offset_head(fused).reshape(*fused.shape[:2], 9, 2)
        grid_offsets = grid_offsets.tanh() / 6.
        candidates = (self.grid_centers.view(1, 1, 9, 2) + grid_offsets).clamp(0., 1.)
        click = (grid_logits.softmax(-1).unsqueeze(-1) * candidates).sum(-2)

        basis = self.horizon_basis.view(1, 1, self.future_steps, 4)
        future_input = torch.cat((
            fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1),
            basis.expand(*fused.shape[:2], -1, -1)), -1)
        intent_position = intent_imu = None
        if self.intent_position_head is not None:
            count = len(self.intent_horizons_steps)
            ib = self.intent_horizon_basis.view(1, 1, count, 4)
            expanded_basis = ib.expand(*fused.shape[:2], -1, -1)
            expanded_emg = emg_motion.unsqueeze(2).expand(-1, -1, count, -1)
            expanded_fused = fused.unsqueeze(2).expand(-1, -1, count, -1)
            intent_position = self.intent_position_head(torch.cat(
                (expanded_fused, expanded_emg, expanded_basis), -1))
            intent_imu = self.intent_imu_head(torch.cat(
                (expanded_emg, expanded_basis), -1))

        identity = fused.new_tensor([1., 0., 0., 0., 1., 0.])
        return {
            "gripper_state_logits": state_logits,
            "state_probability": state_probability,
            "position": self.position_head(fused),
            "click": click,
            "grid_logits": grid_logits,
            "grid_offsets": grid_offsets,
            "future_position": self.future_head(future_input),
            "future_imu_delta": None,
            "emg_reconstruction": None,
            "emg_correction": correction,
            "emg_correction_gate": gate,
            "intent_position_delta": intent_position,
            "intent_imu_delta": intent_imu,
            "intent_state_logits": None,
            "orientation_6d": identity.view(1, 1, 6).expand(*fused.shape[:2], 6),
            "future_orientation_6d": identity.view(1, 1, 1, 6).expand(
                *fused.shape[:2], self.future_steps, 6),
            "final_position": None,
            "final_orientation_6d": None,
        }
