from __future__ import annotations

from typing import Any

from torch import nn

from .baselines import CausalTCNRegressor, PatchTSTRegressor
from .student import EMGStudent
from .teacher import MultimodalTeacher


def build_model(kind: str, config: dict[str, Any]) -> nn.Module:
    kind = kind.lower()
    if kind == "teacher":
        return MultimodalTeacher(config)
    if kind == "student":
        return EMGStudent(config)
    if kind == "tcn":
        return CausalTCNRegressor(config)
    if kind == "patchtst":
        return PatchTSTRegressor(config)
    raise ValueError(f"Unknown model kind: {kind}")

