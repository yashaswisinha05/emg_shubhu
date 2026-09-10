import torch

from emg_touch.models.neuromuscular_future_loss import future_imu_delta_loss
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)


def inputs(length=24):
    emg, imu = torch.randn(1, length, 16), torch.randn(1, length, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    return emg, imu


def test_neuromuscular_future_is_causal_and_has_horizon_gate():
    model = NeuromuscularFutureGripperPoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
        predict_click=True, predict_final_pose=True, future_steps=5).eval()
    emg, imu = inputs()
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 16:, :8] += 100
    changed_imu[:, 16:, :24] -= 100
    with torch.no_grad():
        output = model(emg, imu)
        changed = model(changed_emg, changed_imu)
    assert output["future_position"].shape == (1, 24, 5, 3)
    assert output["future_imu_delta"].shape == (1, 24, 5, 24)
    assert output["emg_innovation_gate"].shape == (1, 24, 5, 1)
    torch.testing.assert_close(
        output["future_position"][:, :16], changed["future_position"][:, :16])


def test_imu_model_disables_emg_innovation():
    model = NeuromuscularFutureGripperPoseModel(
        modality="imu", width=16, patch=4, stride=2, layers=1, heads=4,
        predict_click=True, predict_final_pose=True, future_steps=3).eval()
    emg, imu = inputs()
    with torch.no_grad():
        output = model(emg, imu)
    assert output["emg_innovation_gate"].count_nonzero() == 0


def test_future_motion_loss_uses_future_delta():
    prediction = torch.zeros(1, 5, 2, 24, requires_grad=True)
    imu = torch.zeros(1, 5, 48)
    imu[..., 24:] = 1
    imu[:, 1:, :24] = 2
    batch = {"imu": imu, "imu_usable": torch.ones(1, 5, dtype=torch.bool)}
    value = future_imu_delta_loss({"future_imu_delta": prediction}, batch)
    assert value > 0
    value.backward()
    assert prediction.grad.abs().sum() > 0
