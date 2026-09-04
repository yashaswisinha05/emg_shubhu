"""One wearable goal distribution shared by screen and complete-3D heads.

This isolated successor keeps the successful temporal-EMG-residual model and
adds a small, zero-initialized bridge.  The existing screen heatmap is treated
as a soft distribution over the 8x5 target grid.  That same distribution
selects target-conditioned residual prototypes for the complete 3D path and
endpoint, while the predicted 3D endpoint supplies a bounded correction back
to the screen point.

The prototype parameters are learned only from training labels.  At
deployment ``student_forward`` still accepts causal EMG, causal IMU, and a
padding mask; no target id, screen coordinate, or VIVE value is an input.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .emg_residual_complete_reach import EMGResidualCompleteReachModel
from .soft_routed_complete_reach import scale_gradient


def _gate_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class GoalPrototypeBridge(nn.Module):
    """Couple the screen goal to target-conditioned residual 3D motion."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        settings = model.get("goal_prototype_bridge", {})
        grid_width, grid_height = map(int, model.get("grid_size", [8, 5]))
        self.cells = grid_width * grid_height
        self.steps = int(model["teacher_trajectory_steps"])
        self.prototype_limit_m = float(settings.get("prototype_limit_m", 0.15))
        self.screen_delta_limit = float(
            settings.get("screen_delta_limit_normalized", 0.25)
        )
        self.temperature = float(settings.get("goal_temperature", 1.0))
        self.goal_gradient_scale = float(
            settings.get("trajectory_to_goal_gradient_scale", 0.25)
        )
        self.geometry_gradient_scale = float(
            settings.get("screen_to_endpoint_gradient_scale", 0.10)
        )
        if self.prototype_limit_m <= 0.0:
            raise ValueError("prototype_limit_m must be positive")
        if self.screen_delta_limit <= 0.0:
            raise ValueError("screen_delta_limit_normalized must be positive")
        if self.temperature <= 0.0:
            raise ValueError("goal_temperature must be positive")
        for value in (self.goal_gradient_scale, self.geometry_gradient_scale):
            if not 0.0 <= value <= 1.0:
                raise ValueError("goal bridge gradient scales must be in [0, 1]")

        # These store the target-conditioned residual left by the already
        # trained wearable model, not an unconditional average reach.  The
        # oracle loss in the trainer makes each entry learn only from samples
        # belonging to its training target cell.
        self.path_prototypes = nn.Parameter(
            torch.zeros(self.cells, self.steps, 3)
        )
        self.endpoint_prototypes = nn.Parameter(torch.zeros(self.cells, 3))

        hidden = int(settings.get("geometry_hidden", model.get("d_model", 128)))
        dropout = float(settings.get("dropout", model.get("dropout", 0.1)))
        # The correction sees the onset-relative 3D endpoint and the existing
        # screen estimate.  Its final layer is zero-initialized, so introducing
        # this module cannot perturb a loaded checkpoint before learning.
        self.geometry_screen_correction = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.geometry_screen_correction[-1].weight)
        nn.init.zeros_(self.geometry_screen_correction[-1].bias)

        initial = float(settings.get("prototype_gate_initial", 0.15))
        screen_initial = float(settings.get("geometry_gate_initial", 0.15))
        self.path_gate_logit = nn.Parameter(torch.tensor(_gate_logit(initial)))
        self.endpoint_gate_logit = nn.Parameter(torch.tensor(_gate_logit(initial)))
        self.geometry_gate_logit = nn.Parameter(
            torch.tensor(_gate_logit(screen_initial))
        )

    @staticmethod
    def _progress(reference: torch.Tensor) -> torch.Tensor:
        progress = torch.linspace(
            0.0,
            1.0,
            reference.size(1),
            device=reference.device,
            dtype=reference.dtype,
        )
        progress = progress.square() * (3.0 - 2.0 * progress)
        return progress.view(1, -1, 1)

    def bounded_path_prototypes(self) -> torch.Tensor:
        return self.prototype_limit_m * torch.tanh(self.path_prototypes)

    def bounded_endpoint_prototypes(self) -> torch.Tensor:
        return self.prototype_limit_m * torch.tanh(self.endpoint_prototypes)

    def forward(
        self,
        heatmap_logits: torch.Tensor,
        base_prediction: torch.Tensor,
        base_trajectory: torch.Tensor,
        base_endpoint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        goal = torch.softmax(heatmap_logits / self.temperature, dim=-1)
        routed_goal = scale_gradient(goal, self.goal_gradient_scale)
        all_path = self.bounded_path_prototypes()
        all_endpoint = self.bounded_endpoint_prototypes()
        selected_path = torch.einsum("bk,ktd->btd", routed_goal, all_path)
        selected_endpoint = torch.einsum(
            "bk,kd->bd", routed_goal, all_endpoint
        )
        path_gate = torch.sigmoid(self.path_gate_logit)
        endpoint_gate = torch.sigmoid(self.endpoint_gate_logit)
        progress = self._progress(base_trajectory)
        provisional = base_trajectory + progress * path_gate * selected_path
        endpoint = base_endpoint + endpoint_gate * selected_endpoint
        trajectory = provisional + progress * (
            endpoint[:, None, :] - provisional[:, -1:]
        )

        geometry_input = torch.cat(
            [
                scale_gradient(endpoint, self.geometry_gradient_scale),
                scale_gradient(base_prediction, self.geometry_gradient_scale),
            ],
            dim=-1,
        )
        geometry_delta = self.screen_delta_limit * torch.tanh(
            self.geometry_screen_correction(geometry_input)
        )
        geometry_gate = torch.sigmoid(self.geometry_gate_logit)
        prediction = (base_prediction + geometry_gate * geometry_delta).clamp(
            0.0, 1.0
        )
        return {
            "prediction": prediction,
            "trajectory": trajectory,
            "complete_trajectory": trajectory,
            "endpoint_3d": endpoint,
            "pre_goal_prediction": base_prediction,
            "pre_goal_trajectory": base_trajectory,
            "pre_goal_endpoint": base_endpoint,
            "goal_probabilities": goal,
            "goal_predicted_cell": goal.argmax(dim=-1),
            "goal_selected_path_prototype": selected_path,
            "goal_selected_endpoint_prototype": selected_endpoint,
            "goal_all_path_prototypes": all_path,
            "goal_all_endpoint_prototypes": all_endpoint,
            "goal_geometry_delta": geometry_delta,
            "goal_path_gate": path_gate.expand(base_prediction.size(0)),
            "goal_endpoint_gate": endpoint_gate.expand(base_prediction.size(0)),
            "goal_geometry_gate": geometry_gate.expand(base_prediction.size(0)),
        }


class GoalPrototypeCompleteReachModel(EMGResidualCompleteReachModel):
    """Temporal EMG residual model plus a shared screen/3D goal bridge."""

    def __init__(
        self, config: dict[str, Any], emg_channels: int, imu_channels: int
    ) -> None:
        super().__init__(config, emg_channels, imu_channels)
        self.student.goal_prototype_bridge = GoalPrototypeBridge(config)
        self.goal_warmup = False

    def train(self, mode: bool = True) -> "GoalPrototypeCompleteReachModel":
        super().train(mode)
        if mode and self.goal_warmup:
            self.student.eval()
            self.student.goal_prototype_bridge.train()
            self.teacher.eval()
            self.decoder.eval()
            if self.guidance is not None:
                self.guidance.eval()
        return self

    def _apply_goal_bridge(
        self, outputs: dict[str, Any]
    ) -> dict[str, Any]:
        bridged = self.student.goal_prototype_bridge(
            outputs["heatmap_logits"],
            outputs["prediction"],
            outputs["trajectory"],
            outputs["endpoint_3d"],
        )
        return {**outputs, **bridged}

    def student_forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        sample: bool = False,
        noise_scale: float = 1.0,
        include_emg_only: bool = False,
        apply_imu_dropout: bool = False,
        apply_channel_dropout: bool | None = None,
    ) -> dict[str, Any]:
        outputs = super().student_forward(
            emg,
            imu,
            time_mask,
            sample=sample,
            noise_scale=noise_scale,
            include_emg_only=include_emg_only,
            apply_imu_dropout=apply_imu_dropout,
            apply_channel_dropout=apply_channel_dropout,
        )
        if include_emg_only:
            outputs["emg_only"] = self._apply_goal_bridge(outputs["emg_only"])
        return self._apply_goal_bridge(outputs)
