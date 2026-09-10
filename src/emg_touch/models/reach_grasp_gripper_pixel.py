"""Goal-consistent pixel head for the gripper-state pose model."""
from __future__ import annotations

import torch
from torch import nn

from .reach_grasp_gripper_state import GripperStatePoseModel


class GoalConsistentGripperPoseModel(GripperStatePoseModel):
    """Predict the click directly and from the predicted 3D endpoint.

    The learned blend makes screen intent and endpoint pose share a geometric
    goal without forcing either head to discard task-specific information.
    """

    def __init__(self, **kwargs):
        if not kwargs.get("predict_click") or not kwargs.get("predict_final_pose"):
            raise ValueError("goal-consistent pixels require click and final-pose heads")
        super().__init__(**kwargs)
        width = kwargs.get("width", 128)
        self.click_direct = self.click
        self.click = None
        self.endpoint_to_click = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, 2))
        self.click_blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, emg, imu):
        result = super().forward(emg, imu)
        direct = self.click_direct(result["context_features"]).sigmoid()
        # Screen supervision must not turn the metric 3D endpoint into a
        # convenient screen-coordinate code. The projector still learns the
        # 3D-to-2D association, while endpoint geometry is trained only by its
        # VIVE pose objective.
        endpoint = self.endpoint_to_click(result["final_position"].detach()).sigmoid()
        blend = self.click_blend_logit.sigmoid()
        result["click_direct"] = direct
        result["click_from_endpoint"] = endpoint
        result["click"] = torch.lerp(direct, endpoint, blend)
        result["click_endpoint_blend"] = blend
        return result
