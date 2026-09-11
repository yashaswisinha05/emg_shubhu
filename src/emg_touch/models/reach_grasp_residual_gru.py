"""Causal state-conditioned neuromuscular residual GRU."""
from __future__ import annotations

import math

import torch
from torch import nn


class NeuromuscularResidualGRU(nn.Module):
    """Separate EMG/IMU GRUs with task heads and a bounded EMG correction."""

    def __init__(self, modality="emg+imu", width=128, layers=4, dropout=.1,
                 future_steps=20, predict_click=True, predict_final_pose=False,
                 intent_horizons_steps=None,
                 **_unused):
        super().__init__()
        if modality not in {"emg", "imu", "emg+imu"}:
            raise ValueError(f"unknown modality: {modality}")
        if predict_final_pose:
            raise ValueError("residual GRU intentionally omits the endpoint head")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        self.modality, self.future_steps = modality, future_steps
        horizons = tuple(intent_horizons_steps or ())
        if any(step <= 0 for step in horizons) or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("intent_horizons_steps must be positive, sorted and unique")
        self.intent_horizons_steps = horizons
        recurrent_dropout = dropout if layers > 1 else 0.
        self.emg_input = nn.Sequential(
            nn.Linear(16, width), nn.LayerNorm(width), nn.GELU())
        self.imu_input = nn.Sequential(
            nn.Linear(48, width), nn.LayerNorm(width), nn.GELU())
        self.emg_gru = nn.GRU(width, width, layers, batch_first=True,
                              dropout=recurrent_dropout)
        self.imu_gru = nn.GRU(width, width, layers, batch_first=True,
                              dropout=recurrent_dropout)

        # Classification is deliberately an EMG-only task. Pose gradients
        # cannot alter its representation through the detached soft state.
        self.state_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.state_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU())
        self.imu_state_prior = nn.Parameter(torch.zeros(2))

        self.emg_correction = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width),
            nn.Tanh())
        self.correction_gate = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.emg_correction[-2].weight)
        nn.init.zeros_(self.emg_correction[-2].bias)
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, -2.)
        self.fused_norm = nn.LayerNorm(width)

        self.position_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.grid_head = self.offset_head = None
        if predict_click:
            self.grid_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))
            self.offset_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 18))
        coordinates = torch.tensor([1 / 6, 1 / 2, 5 / 6], dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid_centers", torch.stack((xx.flatten(), yy.flatten()), -1))

        self.future_head = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 3))
        self.future_imu_head = nn.Sequential(
            nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 24))
        self.emg_reconstruction_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 8))
        tau = torch.arange(1, future_steps + 1, dtype=torch.float32) / future_steps
        self.register_buffer("horizon_basis", torch.stack((
            tau, tau.square(), torch.sin(math.pi * tau),
            torch.cos(math.pi * tau)), -1))
        self.intent_position_head = self.intent_imu_head = self.intent_state_head = None
        if horizons:
            intent_width = width * 2 + 4
            self.intent_position_head = nn.Sequential(
                nn.Linear(intent_width, width), nn.GELU(), nn.Linear(width, 3))
            self.intent_imu_head = nn.Sequential(
                nn.Linear(intent_width, width), nn.GELU(), nn.Linear(width, 24))
            self.intent_state_head = nn.Sequential(
                nn.Linear(intent_width, width), nn.GELU(), nn.Linear(width, 2))
            intent_tau = torch.as_tensor(horizons, dtype=torch.float32) / max(horizons)
            intent_basis = torch.stack((
                intent_tau, intent_tau.square(), torch.sin(math.pi * intent_tau),
                torch.cos(math.pi * intent_tau)), -1)
        else:
            intent_basis = torch.empty(0, 4)
        self.register_buffer("intent_horizon_basis", intent_basis)

    def _encode(self, emg, imu):
        self.emg_gru.flatten_parameters()
        self.imu_gru.flatten_parameters()
        emg_features = self.emg_gru(self.emg_input(emg))[0]
        imu_features = self.imu_gru(self.imu_input(imu))[0]
        return emg_features, imu_features

    def forward(self, emg, imu):
        emg_features, imu_features = self._encode(emg, imu)
        if self.modality == "imu":
            state_logits = self.imu_state_prior.view(1, 1, 2).expand(
                emg.shape[0], emg.shape[1], -1)
        else:
            state_logits = self.state_head(emg_features)
        state_probability = state_logits.softmax(-1)
        state = self.state_embedding(state_probability.detach())

        if self.modality == "emg":
            base = torch.zeros_like(emg_features)
        else:
            base = imu_features
        correction = self.emg_correction(torch.cat((emg_features, state), -1))
        gate = self.correction_gate(torch.cat((base, emg_features, state), -1)).sigmoid()
        if self.modality == "imu":
            correction, gate = torch.zeros_like(correction), torch.zeros_like(gate)
        fused = self.fused_norm(base + gate * correction)

        grid_logits = grid_offsets = click = None
        if self.grid_head is not None:
            grid_logits = self.grid_head(fused)
            grid_offsets = self.offset_head(fused).reshape(*fused.shape[:2], 9, 2)
            grid_offsets = grid_offsets.tanh() / 6.
            candidates = (self.grid_centers.view(1, 1, 9, 2) + grid_offsets).clamp(0., 1.)
            click = (grid_logits.softmax(-1).unsqueeze(-1) * candidates).sum(-2)

        basis = self.horizon_basis.view(1, 1, self.future_steps, 4)
        future_input = torch.cat((
            fused.unsqueeze(2).expand(-1, -1, self.future_steps, -1),
            basis.expand(*fused.shape[:2], -1, -1)), -1)
        identity = fused.new_tensor([1., 0., 0., 0., 1., 0.])
        intent_position = intent_imu = intent_state = None
        if self.intent_position_head is not None:
            count = len(self.intent_horizons_steps)
            intent_basis = self.intent_horizon_basis.view(1, 1, count, 4)
            intent_input = torch.cat((
                fused.unsqueeze(2).expand(-1, -1, count, -1),
                emg_features.unsqueeze(2).expand(-1, -1, count, -1),
                intent_basis.expand(*fused.shape[:2], -1, -1)), -1)
            intent_position = self.intent_position_head(intent_input)
            intent_imu = self.intent_imu_head(intent_input)
            intent_state = self.intent_state_head(intent_input)
        return {
            "gripper_state_logits": state_logits,
            "state_probability": state_probability,
            "position": self.position_head(fused),
            "click": click,
            "grid_logits": grid_logits,
            "grid_offsets": grid_offsets,
            "future_position": self.future_head(future_input),
            "future_imu_delta": self.future_imu_head(future_input),
            "emg_reconstruction": self.emg_reconstruction_head(emg_features),
            "emg_correction": correction,
            "emg_correction_gate": gate,
            "intent_position_delta": intent_position,
            "intent_imu_delta": intent_imu,
            "intent_state_logits": intent_state,
            "orientation_6d": identity.view(1, 1, 6).expand(*fused.shape[:2], 6),
            "future_orientation_6d": identity.view(1, 1, 1, 6).expand(
                *fused.shape[:2], self.future_steps, 6),
            "final_position": None,
            "final_orientation_6d": None,
        }
