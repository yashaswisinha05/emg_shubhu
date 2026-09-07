"""Single-modality networks: the EMG student has no IMU input argument."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class WearableTaskNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, latent: int = 32,
                 steps: int = 16, dropout: float = 0.15):
        super().__init__()
        self.steps = steps
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU())
        self.encoder = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                              dropout=dropout)
        self.motion = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent))
        self.private = nn.Sequential(nn.Linear(hidden, latent), nn.GELU())
        self.screen = nn.Sequential(nn.Linear(latent * 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, 2))
        self.path = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                  nn.Linear(hidden, (steps - 1) * 3))

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """signal [B,T,C], mask [B,T]; history only, no timing/position labels."""
        lengths = mask.sum(1)
        if (lengths == 0).any():
            raise ValueError("each observation needs at least one valid sample")
        # Loader left-pads. Compact to right-padding for packed GRU sequences.
        compact = signal.new_zeros(signal.shape)
        for row in range(signal.shape[0]):
            valid = signal[row, mask[row]]
            compact[row, :len(valid)] = valid
        packed = pack_padded_sequence(self.input_projection(compact), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        motion = self.motion(state[-1])
        private = self.private(state[-1])
        path = self.path(motion).reshape(-1, self.steps - 1, 3)
        # Direct onset-relative positions, NOT accumulated velocity predictions.
        path = torch.cat([path.new_zeros(len(path), 1, 3), path], dim=1)
        return {"screen": self.screen(torch.cat([motion, private], -1)),
                "path": path, "motion": motion}


def task_loss(output, window, motion_scale=0.1):
    """Scale each physical task explicitly; screen outputs are normalized xy."""
    screen = F.smooth_l1_loss(
        (output["screen"] - window["target"]) * window["canvas_size"] / 100,
        torch.zeros_like(output["screen"]))
    path = F.smooth_l1_loss(output["path"] / motion_scale,
                           window["trajectory_target"] / motion_scale)
    endpoint = F.smooth_l1_loss(output["path"][:, -1] / motion_scale,
                               window["endpoint_3d_target"] / motion_scale)
    return screen + path + 0.5 * endpoint


def transfer_loss(student, teacher, window, latent_weight=0.1, output_weight=0.25):
    # Stop-gradient even if the caller accidentally passes an unfrozen teacher.
    alignment = F.mse_loss(F.normalize(student["motion"], dim=-1),
                           F.normalize(teacher["motion"].detach(), dim=-1))
    screen = F.smooth_l1_loss(
        student["screen"] * window["canvas_size"] / 100,
        teacher["screen"].detach() * window["canvas_size"] / 100)
    path = F.smooth_l1_loss(student["path"] / 0.1, teacher["path"].detach() / 0.1)
    return latent_weight * alignment + output_weight * (screen + path)
