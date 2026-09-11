import torch
from torch import nn

from emg_touch.models.reach_grasp_shared_encoder_adapter import SharedEncoderResidualGRU


class DummySharedClassifier(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.emg = nn.Linear(16, width)
        self.imu = nn.Linear(48, width)
        self.state = nn.Linear(width * 2, 2)

    def forward(self, emg, imu):
        e, i = self.emg(emg), self.imu(imu)
        return {
            "emg_context_features": e,
            "imu_context_features": i,
            "gripper_state_logits": self.state(torch.cat((e, i), -1)),
        }


def stats():
    return {
        "emg": {"mean": [0.] * 8, "std": [1.] * 8},
        "imu": {"mean": [0.] * 24, "std": [1.] * 24},
        "position": {"mean": [0.] * 3, "std": [1.] * 3},
    }


def make_model():
    torch.manual_seed(3)
    return SharedEncoderResidualGRU(
        DummySharedClassifier(), stats(), stats(), width=16,
        adapter_layers=1, dropout=0., future_steps=3,
        intent_horizons_steps=(10, 20))


def test_shared_classifier_is_frozen_and_shapes_are_complete():
    model = make_model()
    assert not any(parameter.requires_grad for parameter in model.classifier.parameters())
    output = model(torch.randn(2, 30, 16), torch.randn(2, 30, 48))
    assert output["gripper_state_logits"].shape == (2, 30, 2)
    assert output["position"].shape == (2, 30, 3)
    assert output["click"].shape == (2, 30, 2)
    assert output["future_position"].shape == (2, 30, 3, 3)
    assert output["intent_position_delta"].shape == (2, 30, 2, 3)
    assert output["intent_imu_delta"].shape == (2, 30, 2, 24)


def test_model_is_causal():
    model = make_model().eval()
    emg, imu = torch.randn(1, 35, 16), torch.randn(1, 35, 48)
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 20:] = torch.randn_like(changed_emg[:, 20:])
    changed_imu[:, 20:] = torch.randn_like(changed_imu[:, 20:])
    first, second = model(emg, imu), model(changed_emg, changed_imu)
    for key in ("gripper_state_logits", "position", "click", "future_position",
                "intent_position_delta", "intent_imu_delta"):
        torch.testing.assert_close(first[key][:, :20], second[key][:, :20])


def test_future_imu_auxiliary_has_no_direct_imu_input():
    model = make_model().eval()
    emg = torch.randn(1, 30, 16)
    first = model(emg, torch.randn(1, 30, 48))["intent_imu_delta"]
    second = model(emg, torch.randn(1, 30, 48))["intent_imu_delta"]
    torch.testing.assert_close(first, second)
