import torch
import json
import sys
import numpy as np

from emg_touch.models.gripper_future_loss import future_pairs, future_pose_loss
from emg_touch.models.reach_grasp_gripper_pixel import GoalConsistentGripperPoseModel
from scripts import train_gripper_state_pose as trainer
from test_reach_grasp import frame


def test_future_predictions_are_causal_and_reloadable():
    torch.manual_seed(8)
    kwargs = dict(width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
                  predict_click=True, predict_final_pose=True, future_steps=20)
    model = GoalConsistentGripperPoseModel(**kwargs).eval()
    emg, imu = torch.randn(1, 40, 16), torch.randn(1, 40, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    with torch.no_grad():
        full = model(emg, imu)
        prefix = model(emg[:, :20], imu[:, :20])
        reloaded = GoalConsistentGripperPoseModel(**kwargs).eval()
        reloaded.load_state_dict(model.state_dict())
        copy = reloaded(emg, imu)
    for key in ("future_position", "future_orientation_6d", "click", "position"):
        torch.testing.assert_close(full[key][:, :20], prefix[key])
        torch.testing.assert_close(full[key], copy[key])
    assert full["future_position"].shape == (1, 40, 20, 3)


def test_future_alignment_padding_and_missing_labels():
    batch = dict(pose=torch.arange(6.).reshape(1, 6, 1).expand(1, 6, 3),
                 orientation=torch.zeros(1, 6, 6),
                 pose_mask=torch.tensor([[[1], [1], [0], [1], [0], [0]]]),
                 orientation_mask=torch.zeros(1, 6, dtype=torch.bool),
                 emg_usable=torch.tensor([[True, True, True, True, False, False]]),
                 imu_usable=torch.tensor([[True, True, True, True, False, False]]))
    pred = torch.zeros(1, 6, 2, 3, requires_grad=True)
    rotation = torch.zeros(1, 6, 2, 6, requires_grad=True)
    out = dict(future_position=pred, future_orientation_6d=rotation)
    pairs = list(future_pairs(out, batch))
    assert pairs[0][2][0, 0, 0] == 1  # t=0 predicts t=1
    assert pairs[1][2][0, 0, 0] == 2  # t=0 predicts t=2
    assert pairs[0][3].tolist() == [[True, False, True, False, False]]
    value = future_pose_loss(out, batch)
    assert torch.isfinite(value)
    value.backward()
    assert pred.grad[0, 0, 0].abs().sum() > 0
    assert pred.grad[0, 1, 0].abs().sum() == 0  # missing label at t=2
    assert pred.grad[0, 3:].abs().sum() == 0  # padding / trial end
    assert rotation.grad.abs().sum() == 0  # no orientation labels


def test_future_loss_all_invalid_is_finite_zero():
    batch = dict(pose=torch.zeros(1, 2, 3), orientation=torch.zeros(1, 2, 6),
                 pose_mask=torch.zeros(1, 2, 1),
                 orientation_mask=torch.zeros(1, 2, dtype=torch.bool),
                 emg_usable=torch.ones(1, 2, dtype=torch.bool),
                 imu_usable=torch.ones(1, 2, dtype=torch.bool))
    out = dict(future_position=torch.randn(1, 2, 20, 3, requires_grad=True),
               future_orientation_6d=torch.randn(1, 2, 20, 6, requires_grad=True))
    value = future_pose_loss(out, batch)
    assert value.item() == 0
    value.backward()


def test_future_head_preserves_baseline_initialization():
    kwargs = dict(width=16, patch=4, stride=2, layers=1, heads=4,
                  predict_click=True, predict_final_pose=True)
    torch.manual_seed(42)
    control = GoalConsistentGripperPoseModel(**kwargs)
    torch.manual_seed(42)
    future = GoalConsistentGripperPoseModel(**kwargs, future_steps=20)
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, future.state_dict()[key])


def test_future_training_and_report(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data["gripper_state"] = np.where(data.time_perf_counter < 100.5, "open", "close")
        data["canvas_width_px"], data["canvas_height_px"] = 1440, 900
        data["click_x_norm"], data["click_y_norm"] = index / 20, .4
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    def small(**kwargs):
        kwargs.update(width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.)
        return GoalConsistentGripperPoseModel(**kwargs)
    monkeypatch.setattr(trainer, "GoalConsistentGripperPoseModel", small)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir", str(output),
        "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000",
        "--models", "emg+imu", "--pixel-architecture", "goal-consistent",
        "--future-pose-weight", ".1"])
    trainer.main()
    report = json.loads((output / "results.json").read_text())
    future = report["emg+imu"]["future_pose_by_ms"]["200"]
    assert future["valid_position_frames"] > 0
    assert np.isfinite(future["position_cm"])
    assert future["orientation_deg"] is None  # fixture has no quaternions
    saved = torch.load(output / "emg_imu_best.pt", weights_only=False)
    assert saved["model_args"]["future_steps"] == 20
    assert saved["format"].endswith("_future_pose_v1")
