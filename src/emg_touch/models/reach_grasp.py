"""Full-sequence causal TCN; annotation and tracker columns never enter forward."""
import torch
from torch import nn
from torch.nn import functional as F


class Branch(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.project = nn.Linear(inputs, width)
        self.layers = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d)
                                     for d in [1, 2, 4, 8, 16, 32]])

    def forward(self, x):
        x = F.gelu(self.project(x)).transpose(1, 2)
        for layer in self.layers:
            x = x + F.gelu(layer(F.pad(x, (2 * layer.dilation[0], 0))))
        return x.transpose(1, 2)


class ReachGraspModel(nn.Module):
    def __init__(self, modality="emg+imu", width=48):
        super().__init__()
        self.modality = modality
        self.emg = Branch(16, width)  # 8 RMS features + 8 availability bits
        self.imu = Branch(48, width)  # 24 measurements + 24 availability bits
        self.fusion = nn.Sequential(nn.Linear(width * 2, width * 2), nn.GELU())
        self.interaction = nn.Linear(width * 2, 3)  # holding, grasp, release logits
        self.position = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, emg, imu):
        if self.modality == "imu":
            emg = torch.zeros_like(emg)
        if self.modality == "emg":
            imu = torch.zeros_like(imu)
        context = self.fusion(torch.cat([self.emg(emg), self.imu(imu)], -1))
        return {"logits": self.interaction(context), "position": self.position(context)}
