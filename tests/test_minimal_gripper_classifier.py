from types import SimpleNamespace

import torch

from scripts.train_gripper_classifier_minimal import classification_loss


def test_stage_one_loss_is_only_classification_cross_entropy():
    logits = torch.tensor([[[2., -1.], [-1., 2.]]])
    batch = {
        "emg_usable": torch.ones(1, 2, dtype=torch.bool),
        "imu_usable": torch.ones(1, 2, dtype=torch.bool),
        "gripper_state_valid": torch.ones(1, 2, dtype=torch.bool),
        "gripper_state": torch.tensor([[0, 1]]),
    }
    output = {
        "gripper_state_logits": logits,
        "position": torch.full((1, 2, 3), 1e9),
        "click": torch.full((1, 2, 2), 1e9),
    }
    model = SimpleNamespace(modality="emg+imu")
    first = classification_loss(model, output, batch, torch.ones(2), SimpleNamespace())
    output["position"].zero_()
    output["click"].zero_()
    second = classification_loss(model, output, batch, torch.ones(2), SimpleNamespace())
    torch.testing.assert_close(first, second)
