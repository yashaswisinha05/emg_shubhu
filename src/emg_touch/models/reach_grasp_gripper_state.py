"""Causal EMG+IMU pose regression plus open/close gripper-state classification.

The task changed, the architecture did not. The dataset previously carried
grasp_onset_s/grasp_offset_s markers, from which reach_grasp_hybrid.py derived
a continuous "holding" state plus two event-pulse channels (onset/release).
It now carries a single per-timestep categorical column instead: gripper_state,
"open" or "close". There is no event/pulse structure to predict any more --
just a plain two-class state, alongside the same SE(3) pose regression.

Reused from the existing lineage, unchanged:
  * CausalPatchBranch (reach_grasp_patch_transformer.py) -- the causal
    overlapping-patch transformer used for both the EMG and IMU encoders.
  * The gated context/local fusion (_mask_and_weights / _fuse below is a
    verbatim copy of ReachGraspHybrid's, matching this codebase's existing
    convention of duplicating that small helper per model rather than
    inheriting it -- ReachGraspHybrid and ReachGraspPatchTransformer already
    do the same).
  * CausalEMGIntent (reach_grasp_react_intent.py) -- the ReactEMG-inspired
    causal transformer that aligns EMG with a 2-class-plus-MASK label. It
    already models exactly a {class0, class1, MASK} space (previously
    "not holding, holding, MASK"), so it is reused with zero changes: the
    label it aligns EMG with is now gripper_state (open=0, close=1) instead
    of holding.

Dropped, deliberately, because nothing in the new task calls for it:
  * ReachGraspHybrid's holding/event-pulse head (RobustReachGraspModel's
    event_time_bins / horizon_blend_logit machinery). Those existed to
    predict discrete onset/release EVENTS and time-until-event -- meaningful
    for a derived holding signal, not for a plain per-timestep state column.
    A fresh model class avoids instantiating (and never training) those
    now-pointless parameters, rather than inheriting and leaving them dead.
  * The future-trajectory heads from reach_grasp_future_intent.py /
    reach_grasp_react_intent.py. Nothing in the new task asked for future
    prediction; if that is wanted later it slots in the same way
    ReactFutureIntentModel added it on top of ReachGraspHybrid.

Independent implementation; not the ReactEMG authors' architecture or
pretrained weights.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .reach_grasp_patch_transformer import CausalPatchBranch
from .reach_grasp_react_intent import CausalEMGIntent


class GripperStatePoseModel(nn.Module):
    """Causal EMG+IMU -> current SE(3) pose, plus dense open/close state.

    Inputs are EMG (16-dim: 8 trailing-RMS features + 8 validity flags) and
    IMU (48-dim), exactly as every other model in this lineage expects --
    see emg_touch.data.reach_grasp.preprocess. Output is causal at every
    timestep: no future sample can affect an earlier prediction.
    """

    def __init__(self, modality="emg+imu", width=128, patch=16, stride=4,
                 layers=4, heads=4, dropout=.1, react_context=100):
        super().__init__()
        self.modality = modality
        kwargs = dict(width=width, patch=patch, stride=stride, layers=layers,
                      heads=heads, dropout=dropout)
        self.emg = CausalPatchBranch(16, **kwargs)
        self.imu = CausalPatchBranch(48, **kwargs)

        # Gated context/local fusion -- verbatim copy of ReachGraspHybrid's.
        self.context_gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                          nn.Linear(width, 2))
        self.context_fusion = nn.Sequential(
            nn.Linear(width * 4, width * 2), nn.LayerNorm(width * 2),
            nn.GELU(), nn.Dropout(dropout))
        self.local_gate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                        nn.Linear(width, 2))
        self.local_fusion = nn.Sequential(
            nn.Linear(width * 4, width), nn.LayerNorm(width), nn.GELU(),
            nn.Dropout(dropout))

        # Pose regression: same two heads RobustReachGraspModel/
        # ReachGraspHybrid already use for position/orientation.
        self.position = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                      nn.Linear(width, 3))
        self.orientation = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 6))

        # Dense open/close state: same local-catches-transitions,
        # context-stabilizes-the-sustained-state split ReachGraspHybrid used
        # for holding, now with a plain 2-class head instead of 1 holding
        # value + 2 event-pulse channels (there is no event to pulse on).
        self.gripper_local = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, padding=0, groups=width),
            nn.GELU(), nn.Conv1d(width, width, kernel_size=1), nn.GELU(),
            nn.Conv1d(width, 2, kernel_size=1))
        self.gripper_context = nn.Linear(width * 2, 2)
        # Start local-vs-context blend mostly on the context term; let
        # training decide how much the frame-resolution path should sharpen
        # transitions, matching RobustReachGraspModel's horizon_blend_logit.
        self.gripper_blend_logit = nn.Parameter(torch.tensor(-2.))

        # ReactEMG-style causal intent-alignment branch (reused unchanged),
        # now aligning EMG with gripper_state instead of holding state. It
        # corrects only the gripper_state logits, the same way
        # ReactFutureIntentModel's react_current corrects interaction logits
        # directly rather than routing through the pose heads -- pose
        # regression is an EMG+IMU fusion problem, not an intent-alignment
        # one, so there is no react_to_context term here (unlike
        # ReactFutureIntentModel, which needed one to feed its *new*
        # future-trajectory heads; this model adds none).
        self.react = CausalEMGIntent(width=width // 2, heads=heads, layers=layers,
                                     dropout=dropout, context=react_context)
        self.react_current = nn.Linear(width // 2, 2)
        # Start the correction at zero: the branch first learns from its own
        # dense holding/reconstruction supervision before it is allowed to
        # move the main gripper_state prediction.
        nn.init.zeros_(self.react_current.weight)
        nn.init.zeros_(self.react_current.bias)

    def _mask_and_weights(self, e, i, gate):
        if self.modality == "emg":
            i = torch.zeros_like(i)
        elif self.modality == "imu":
            e = torch.zeros_like(e)
        weights = gate(torch.cat([e, i], -1).detach()).softmax(-1)
        if self.modality == "emg":
            weights = torch.stack([torch.ones_like(weights[..., 0]),
                                   torch.zeros_like(weights[..., 1])], -1)
        elif self.modality == "imu":
            weights = torch.stack([torch.zeros_like(weights[..., 0]),
                                   torch.ones_like(weights[..., 1])], -1)
        return e, i, weights

    @staticmethod
    def _fuse(e, i, weights, layer):
        return layer(torch.cat([e * weights[..., 0:1], i * weights[..., 1:2],
                                e - i, e * i], -1))

    def forward(self, emg, imu):
        ef, inf = self.emg.forward_features(emg), self.imu.forward_features(imu)
        ec, ic, context_weights = self._mask_and_weights(
            ef["context"], inf["context"], self.context_gate)
        context = self._fuse(ec, ic, context_weights, self.context_fusion)
        el, il, local_weights = self._mask_and_weights(
            ef["local"], inf["local"], self.local_gate)
        local = self._fuse(el, il, local_weights, self.local_fusion)

        # Explicit left padding keeps the high-resolution conv causal.
        local_logits = self.gripper_local(F.pad(
            local.transpose(1, 2), (4, 0))).transpose(1, 2)
        context_logits = self.gripper_context(context)
        gripper_state_logits = (context_logits
                                + self.gripper_blend_logit.sigmoid() * local_logits)

        react_emg = torch.zeros_like(emg) if self.modality == "imu" else emg
        branch = self.react(react_emg)  # every gripper_state token is MASK at inference
        gripper_state_logits = gripper_state_logits + self.react_current(branch["features"])

        return {
            "position": self.position(context),
            "orientation_6d": self.orientation(context),
            "gripper_state_logits": gripper_state_logits,
            "react_gripper_state_logits": branch["holding_logits"],
            "react_reconstruction": branch["reconstruction"],
            "context_features": context,
            "local_features": local,
            "emg_context_features": ef["context"],
            "fusion_weights": context_weights,
            "local_fusion_weights": local_weights,
        }
