"""Focused checks for the standalone causal EMG+IMU release."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from minimal_emg_imu.model import MinimalEMGIMUModel
from minimal_emg_imu.train import minimal_loss
from emg_touch.data.click_target import add_click_target
from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import SENSORS, preprocess


def small_model():
    return MinimalEMGIMUModel(
        width=16, patch=8, stride=4, layers=1, heads=4,
        dropout=0., react_context=20, adapter_layers=1)


def test_output_shapes_and_heads():
    model = small_model().eval()
    output = model(torch.randn(2, 32, 16), torch.randn(2, 32, 48))
    assert output["gripper_state_logits"].shape == (2, 32, 2)
    assert output["position"].shape == (2, 32, 3)
    assert output["click"].shape == (2, 32, 2)
    assert output["future_position"].shape == (2, 32, 20, 3)
    assert set(output) == {
        "gripper_state_logits", "position", "click", "future_position",
        "emg_correction_gate", "fusion_weights", "local_fusion_weights",
    }


def test_predictions_are_causal():
    torch.manual_seed(3)
    model = small_model().eval()
    emg, imu = torch.randn(1, 40, 16), torch.randn(1, 40, 48)
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 25:] += 100
    changed_imu[:, 25:] -= 100
    first = model(emg, imu)
    second = model(changed_emg, changed_imu)
    for key in ("gripper_state_logits", "position", "click", "future_position"):
        torch.testing.assert_close(first[key][:, :25], second[key][:, :25])


def test_four_term_objective_backpropagates():
    model = small_model().train()
    frames = 32
    output = model(torch.randn(2, frames, 16), torch.randn(2, frames, 48))
    batch = {
        "emg_usable": torch.ones(2, frames, dtype=torch.bool),
        "imu_usable": torch.ones(2, frames, dtype=torch.bool),
        "gripper_state": torch.randint(0, 2, (2, frames)),
        "gripper_state_valid": torch.ones(2, frames, dtype=torch.bool),
        "pose": torch.randn(2, frames, 3),
        "pose_mask": torch.ones(2, frames, 1),
        "click_target": torch.rand(2, frames, 2),
        "click_valid": torch.ones(2, frames, dtype=torch.bool),
        "canvas_px": torch.tensor([[1920., 1080.], [1920., 1080.]]),
        "trial_progress": torch.linspace(0, 1, frames).repeat(2, 1),
    }
    args = SimpleNamespace(
        state_weight=1., position_weight=1., pixel_weight=.35,
        future_pose_weight=.5)
    loss = minimal_loss(output, batch, torch.ones(2), args)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in (model.gripper_context, model.position_head[-1],
                  model.pixel_head[-1], model.future_head[-1]):
        assert layer.weight.grad is not None


def test_train_and_infer_help():
    for script in ("train.py", "infer.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "minimal_emg_imu" / script), "--help"],
            check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_training_csv_preprocessing(tmp_path):
    samples, rate = 160, 1259.4
    time = np.arange(samples) / rate
    frame = pd.DataFrame({
        "time_perf_counter": time,
        "gripper_state": np.where(time < time.mean(), "open", "close"),
        "canvas_width_px": 1920,
        "canvas_height_px": 1080,
        "click_x_norm": .4,
        "click_y_norm": .6,
        "sample_rate_hz_declared": rate,
        "VIVE_T0_pos_x_m": .1 + time,
        "VIVE_T0_pos_y_m": .2 + time,
        "VIVE_T0_pos_z_m": .3 + time,
    })
    for sensor in SENSORS:
        frame[f"EMG 1_{sensor}"] = np.sin(2 * np.pi * 80 * time)
        for kind in ("ACC", "GYRO"):
            for axis in "XYZ":
                frame[f"{kind} {axis}_{sensor}"] = np.cos(2 * np.pi * 3 * time)
    path = tmp_path / "trial_001.csv"
    frame.to_csv(path, index=False)
    settings = {
        "raw_rate_hz": rate, "rate_hz": 100., "gap_s": .02,
        "event_origin": "auto", "event_pulse_s": .1,
        "require_events": False,
    }
    trial = preprocess(path, settings)
    add_gripper_state(path, trial, settings["gap_s"])
    add_click_target(path, trial)
    assert trial["emg"].shape[1] == 8
    assert trial["imu"].shape[1] == 24
    assert trial["position"].shape[1] == 3
    assert trial["gripper_state_valid"].all()
    np.testing.assert_allclose(trial["click_target"], [.4, .6])
