import torch

from minimal_emg_imu.patchtst_model import (
    CausalPatchTSTBranch, PatchTSTEMGIMUModel)


def test_patchtst_branch_is_frame_aligned_and_causal():
    torch.manual_seed(7)
    branch = CausalPatchTSTBranch(
        4, width=16, patch=8, stride=4, layers=1, heads=4,
        dropout=0., context_frames=32, chunk_frames=16).eval()
    values = torch.randn(1, 28, 4)
    changed = values.clone()
    changed[:, 20:] = torch.randn_like(changed[:, 20:])
    first = branch(values)
    second = branch(changed)
    assert first.shape == (1, 28, 16)
    assert torch.allclose(first[:, :20], second[:, :20], atol=1e-5)


def test_patchtst_ablation_keeps_the_four_outputs():
    model = PatchTSTEMGIMUModel(
        width=16, patch=8, stride=4, layers=1, heads=4,
        dropout=0., react_context=20, adapter_layers=1,
        context_frames=32, chunk_frames=16).eval()
    output = model(torch.randn(2, 28, 16), torch.randn(2, 28, 48))
    assert output["gripper_state_logits"].shape == (2, 28, 2)
    assert output["position"].shape == (2, 28, 3)
    assert output["click"].shape == (2, 28, 2)
    assert output["future_position"].shape == (2, 28, 20, 3)

