import pytest
import torch

from emg_touch.models.reach_grasp_gripper_pixel import GoalConsistentGripperPoseModel


def make_inputs():
    emg, imu = torch.randn(2, 30, 16), torch.randn(2, 30, 48)
    emg[..., 8:] = 1
    imu[..., 24:] = 1
    return emg, imu


def test_goal_consistent_pixel_shapes_bounds_and_causality():
    torch.manual_seed(12)
    model = GoalConsistentGripperPoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
        predict_click=True, predict_final_pose=True).eval()
    emg, imu = make_inputs()
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 20:, :8] += 50
    changed_imu[:, 20:, :24] += 50
    with torch.no_grad():
        output = model(emg, imu)
        changed = model(changed_emg, changed_imu)
    for key in ("click", "click_direct", "click_from_endpoint"):
        assert output[key].shape == (2, 30, 2)
        assert ((0 <= output[key]) & (output[key] <= 1)).all()
        torch.testing.assert_close(output[key][:, :20], changed[key][:, :20])
    assert output["click_endpoint_blend"].shape == ()


def test_goal_consistent_pixel_requires_both_targets():
    with pytest.raises(ValueError, match="click and final-pose"):
        GoalConsistentGripperPoseModel(predict_click=True, predict_final_pose=False)


def test_click_loss_cannot_distort_final_3d_head():
    model = GoalConsistentGripperPoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
        predict_click=True, predict_final_pose=True)
    emg, imu = make_inputs()
    model(emg, imu)["click"].sum().backward()
    assert all(parameter.grad is None for parameter in model.final_position.parameters())
    assert any(parameter.grad is not None for parameter in model.endpoint_to_click.parameters())
